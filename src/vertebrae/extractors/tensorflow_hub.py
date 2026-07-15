"""Optional TensorFlow Hub extractor."""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vertebrae.extractors._identity import (
    cache_identity_fields,
    validate_cache_identity,
    validate_extractor_name,
)
from vertebrae.extractors._utils import (
    callable_name,
    materialize_named_outputs,
    materialize_named_structured_outputs,
    optional_dependency_versions,
    resolve_output_specs,
    resolve_structured_output_specs,
    snapshot_mapping,
    spec_to_recipe,
    structured_spec_to_recipe,
    validate_batch_size,
    validate_nonblank_string,
    validate_optional_nonblank_string,
)
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.extractors.structured import StructuredEmbeddingOutput, StructuredOutputSpec


class TFHubExtractor:
    """Wrap a TensorFlow Hub module as a vertebrae extractor."""

    def __init__(
        self,
        name: str,
        handle: str,
        input_fn: Optional[Callable[[Any], Any]] = None,
        output_fn: Optional[Callable[[Any], Any]] = None,
        outputs: Optional[Sequence[Dict[str, Any]]] = None,
        structured_outputs: Optional[Sequence[Dict[str, Any]]] = None,
        batch_size: int = 32,
        modality: str = "unknown",
        model_kwargs: Optional[Dict[str, Any]] = None,
        call_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_paths: Optional[Sequence[str]] = None,
        profiling_device: Optional[str] = None,
        cache_identity: Optional[str] = None,
    ) -> None:
        batch_size = validate_batch_size(batch_size)
        if input_fn is not None and not callable(input_fn):
            raise TypeError("input_fn must be callable when provided.")
        if output_fn is not None and not callable(output_fn):
            raise TypeError("output_fn must be callable when provided.")
        self.name = validate_extractor_name(name)
        self.handle = validate_nonblank_string(handle, "handle")
        self.input_fn = input_fn or np.asarray
        self.output_fn = output_fn
        self._output_specs = resolve_output_specs(outputs)
        self._structured_output_specs = resolve_structured_output_specs(structured_outputs)
        self.batch_size = batch_size
        self.modality = validate_nonblank_string(modality, "modality")
        self.model_kwargs = snapshot_mapping(model_kwargs, "model_kwargs")
        self.call_kwargs = snapshot_mapping(call_kwargs, "call_kwargs")
        self.checkpoint_paths = tuple(checkpoint_paths or ())
        self.profiling_device = validate_optional_nonblank_string(
            profiling_device, "profiling_device"
        )
        self.extractor_type = "tensorflow_hub"
        self.streaming_safe = True
        self._hub: Any = None
        self._model: Any = None
        self.cache_identity = validate_cache_identity(cache_identity)

    def fit(self, X: Any, y: Any = None) -> "TFHubExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "TFHubExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return outputs[0].embeddings

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        model = self._load_model()
        collected: Dict[str, List[np.ndarray]] = {spec.name: [] for spec in self._output_specs}
        for batch in _iter_input_batches(X, self.batch_size):
            batch_length = _input_batch_length(batch)
            raw_inputs = self.input_fn(batch)
            raw_output = _call_hub_model(model, raw_inputs, self.call_kwargs)
            projected = self.output_fn(raw_output) if self.output_fn is not None else raw_output
            outputs = materialize_named_outputs(
                projected,
                self._output_specs,
                owner=f"TFHubExtractor '{self.name}'",
                allow_sparse=False,
                fallback_output=raw_output,
            )
            for output in outputs:
                matrix = np.asarray(output.embeddings, dtype=np.float32)
                if matrix.shape[0] != batch_length:
                    raise ValueError(
                        f"TFHubExtractor '{self.name}' output '{output.name}' returned "
                        f"{matrix.shape[0]} rows for a batch of {batch_length}."
                    )
                collected[output.name].append(matrix)
        return [
            EmbeddingOutput(
                name=spec.name,
                embeddings=(
                    np.vstack(collected[spec.name]).astype(np.float32, copy=False)
                    if collected[spec.name]
                    else np.empty((0, 0), dtype=np.float32)
                ),
                recipe=spec_to_recipe(spec),
                metadata=dict(spec.metadata),
            )
            for spec in self._output_specs
        ]

    def structured_output_specs(self) -> List[StructuredOutputSpec]:
        return list(self._structured_output_specs)

    def transform_structured(self, X: Any) -> List[StructuredEmbeddingOutput]:
        if not self._structured_output_specs:
            raise ValueError("TFHubExtractor was not configured with structured_outputs.")
        model = self._load_model()
        collected: Dict[str, List[np.ndarray]] = {
            spec.name: [] for spec in self._structured_output_specs
        }
        for batch in _iter_input_batches(X, self.batch_size):
            batch_length = _input_batch_length(batch)
            raw_inputs = self.input_fn(batch)
            raw_output = _call_hub_model(model, raw_inputs, self.call_kwargs)
            projected = self.output_fn(raw_output) if self.output_fn is not None else raw_output
            outputs = materialize_named_structured_outputs(
                projected,
                self._structured_output_specs,
                owner=f"TFHubExtractor '{self.name}'",
                raw_output=raw_output,
                expected_parents=batch_length,
            )
            for output in outputs:
                collected[output.name].extend(
                    np.asarray(item, dtype=np.float32) for item in output.embeddings
                )
        return [
            StructuredEmbeddingOutput(
                name=spec.name,
                embeddings=collected[spec.name],
                unit_type=spec.unit_type,
                recipe=structured_spec_to_recipe(spec),
                metadata=dict(spec.metadata),
            )
            for spec in self._structured_output_specs
        ]

    def recipe(self) -> Dict[str, Any]:
        recipe = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "handle": self.handle,
            "input_fn": callable_name(self.input_fn),
            "output_fn": callable_name(self.output_fn) if self.output_fn is not None else None,
            "outputs": [spec_to_recipe(spec) for spec in self._output_specs],
            "batch_size": self.batch_size,
            "model_kwargs": self.model_kwargs,
            "call_kwargs": self.call_kwargs,
            "dependency_versions": optional_dependency_versions("tensorflow", "tensorflow-hub"),
            "streaming_safe": self.streaming_safe,
        }
        if self._structured_output_specs:
            recipe["structured_outputs"] = [
                structured_spec_to_recipe(spec) for spec in self._structured_output_specs
            ]
        local_paths = (self.handle,) if _is_existing_path(self.handle) else ()
        recipe.update(
            cache_identity_fields(
                explicit=self.cache_identity,
                callables=(("input_fn", self.input_fn), ("output_fn", self.output_fn)),
                paths=local_paths,
                require_pinned_revision=True,
                revision_identifiers=(self.handle,),
            )
        )
        return recipe

    def get_resource_profile_adapter(self) -> Any:
        from vertebrae.profiling import TensorFlowResourceProfileAdapter

        return TensorFlowResourceProfileAdapter(
            self,
            self.checkpoint_paths,
            model_getter=lambda: self._model,
            backend="tensorflow_hub",
            profiling_device=self.profiling_device,
        )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import tensorflow_hub as hub
            except ImportError as exc:
                raise ImportError(
                    "TFHubExtractor requires optional TensorFlow Hub dependencies. "
                    "Install with `poetry install -E tensorflow-hub`."
                ) from exc
            self._hub = hub
            self._model = hub.load(self.handle, **self.model_kwargs)
        return self._model


def _call_hub_model(model: Any, raw_inputs: Any, call_kwargs: Dict[str, Any]) -> Any:
    if callable(model):
        return model(raw_inputs, **call_kwargs)
    return model.signatures["default"](raw_inputs)


def _iter_input_batches(value: Any, batch_size: int) -> Any:
    if isinstance(value, dict):
        lengths = {key: len(items) for key, items in value.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"TFHubExtractor mapping inputs must align; got {lengths}.")
        total = next(iter(lengths.values()), 0)
        for start in range(0, total, batch_size):
            stop = start + batch_size
            yield {
                key: items.iloc[start:stop] if hasattr(items, "iloc") else items[start:stop]
                for key, items in value.items()
            }
        return
    items = list(value)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _input_batch_length(value: Any) -> int:
    if isinstance(value, dict):
        return len(next(iter(value.values()))) if value else 0
    return len(value)


def _is_existing_path(value: str) -> bool:
    from pathlib import Path

    try:
        return Path(value).expanduser().exists()
    except (OSError, ValueError):
        return False
