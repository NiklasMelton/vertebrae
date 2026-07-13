"""Optional ONNX runtime feature extractor."""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from vertebrae.extractors._utils import (
    callable_name,
    infer_batch_size,
    materialize_named_outputs,
    materialize_structured_parent_matrices,
    resolve_output_specs,
    resolve_structured_output_specs,
    spec_to_recipe,
    structured_spec_to_recipe,
)
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.extractors.structured import StructuredEmbeddingOutput, StructuredOutputSpec
from vertebrae.utils.validation import ensure_numeric_matrix


class ONNXExtractor:
    """Wrap a local ONNX model as a feature extractor.

    Args:
        name: User-facing extractor name.
        model_path: Path to a local ONNX model file.
        input_fn: Optional callable that converts raw inputs into ONNX inputs.
        output_fn: Optional callable that converts raw ONNX outputs into embeddings.
        input_names: Optional ONNX input names to feed. Defaults to the model inputs.
        output_names: Optional ONNX output names to fetch. Defaults to the model outputs.
        providers: Optional ONNX Runtime execution providers.
        provider_options: Optional provider-specific configuration dictionaries.
        modality: Input modality metadata.
        extractor_type: Extractor family metadata.
        recipe_data: Extra serializable metadata for reproducibility.
        allow_sparse: Whether sparse embedding outputs are allowed.
        streaming_safe: Whether independent batches can be embedded without full-context state.
    """

    def __init__(
        self,
        name: str,
        model_path: Union[str, Path],
        input_fn: Optional[Callable[[Any], Any]] = None,
        output_fn: Optional[Callable[[Sequence[Any]], Any]] = None,
        outputs: Optional[List[Dict[str, Any]]] = None,
        structured_outputs: Optional[List[Dict[str, Any]]] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict[str, Any]]] = None,
        modality: str = "unknown",
        extractor_type: str = "custom_onnx",
        recipe_data: Optional[Dict[str, Any]] = None,
        allow_sparse: bool = False,
        streaming_safe: bool = True,
        external_data_paths: Optional[Iterable[str]] = None,
    ) -> None:
        self.name = name
        self.model_path = Path(model_path)
        self.input_fn = input_fn
        self.output_fn = output_fn
        self._output_specs = resolve_output_specs(outputs)
        self._structured_output_specs = resolve_structured_output_specs(structured_outputs)
        self.input_names = input_names
        self.output_names = output_names
        self.providers = providers
        self.provider_options = provider_options
        self.modality = modality
        self.extractor_type = extractor_type
        self.recipe_data = recipe_data or {}
        self.allow_sparse = allow_sparse
        self.streaming_safe = streaming_safe
        self.external_data_paths = tuple(external_data_paths or ())
        self._session: Any = None
        self._ort: Any = None

    def fit(self, X: Any, y: Any = None) -> "ONNXExtractor":
        """No-op fit for frozen ONNX models."""

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Run the ONNX model and validate the resulting embeddings."""

        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "ONNXExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return ensure_numeric_matrix(
            outputs[0].embeddings,
            f"ONNXExtractor '{self.name}' output",
            allow_sparse=self.allow_sparse,
        )

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Run the ONNX model and return validated embeddings."""

        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        session = self._load_session()
        projected, named_outputs = self._run_session(session, X)
        outputs = materialize_named_outputs(
            projected,
            self._output_specs,
            owner=f"ONNXExtractor '{self.name}'",
            allow_sparse=self.allow_sparse,
            fallback_output=named_outputs,
        )
        materialized: List[EmbeddingOutput] = []
        for output, spec in zip(outputs, self._output_specs):
            materialized.append(
                EmbeddingOutput(
                    name=output.name,
                    embeddings=output.embeddings,
                    recipe=spec_to_recipe(spec),
                    metadata=dict(spec.metadata),
                )
            )
        return materialized

    def structured_output_specs(self) -> List[StructuredOutputSpec]:
        return list(self._structured_output_specs)

    def transform_structured(self, X: Any) -> List[StructuredEmbeddingOutput]:
        if not self._structured_output_specs:
            raise ValueError("ONNXExtractor was not configured with structured_outputs.")
        session = self._load_session()
        projected, named_outputs = self._run_session(session, X)
        expected_parents = self._infer_batch_size(X, session)
        outputs: List[StructuredEmbeddingOutput] = []
        for spec in self._structured_output_specs:
            value = None
            selector = spec.metadata.get("selector")
            if selector is not None:
                value = _resolve_selector(projected, str(selector))
                if value is None and named_outputs is not projected:
                    value = _resolve_selector(named_outputs, str(selector))
            elif isinstance(projected, dict) and spec.name in projected:
                value = projected[spec.name]
            elif isinstance(named_outputs, dict) and spec.name in named_outputs:
                value = named_outputs[spec.name]
            elif len(self._structured_output_specs) == 1:
                value = projected
            if value is None:
                raise ValueError(
                    f"ONNXExtractor structured output '{spec.name}' could not be resolved."
                )
            embeddings = materialize_structured_parent_matrices(
                value,
                f"ONNXExtractor structured output '{spec.name}'",
                expected_parents=expected_parents,
            )
            outputs.append(
                StructuredEmbeddingOutput(
                    name=spec.name,
                    embeddings=[np.asarray(item, dtype=np.float32) for item in embeddings],
                    unit_type=spec.unit_type,
                    recipe=structured_spec_to_recipe(spec),
                    metadata=dict(spec.metadata),
                )
            )
        return outputs

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable ONNX extractor recipe."""

        recipe = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_path": str(self.model_path),
            "input_fn": callable_name(self.input_fn) if self.input_fn is not None else None,
            "output_fn": callable_name(self.output_fn) if self.output_fn is not None else None,
            "outputs": [spec_to_recipe(spec) for spec in self._output_specs],
            "input_names": self.input_names,
            "output_names": self.output_names,
            "providers": self.providers,
            "provider_options": self.provider_options,
            "recipe_data": self.recipe_data,
            "allow_sparse": self.allow_sparse,
            "streaming_safe": self.streaming_safe,
            "external_data_paths": list(self.external_data_paths),
        }
        if self._structured_output_specs:
            recipe["structured_outputs"] = [
                structured_spec_to_recipe(spec) for spec in self._structured_output_specs
            ]
        return recipe

    def get_resource_profile_adapter(self) -> Any:
        """Return ONNX Runtime model-artifact profiling hooks."""

        from vertebrae.profiling import ONNXResourceProfileAdapter

        return ONNXResourceProfileAdapter(self, self.external_data_paths)

    def _load_session(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "ONNXExtractor requires optional ONNX Runtime support. "
                    'Install with `poetry install -E onnx` or `pip install "vertebrae[onnx]"`.'
                ) from exc
            session_kwargs: Dict[str, Any] = {}
            if self.providers is not None:
                session_kwargs["providers"] = self.providers
            if self.provider_options is not None:
                session_kwargs["provider_options"] = self.provider_options
            self._ort = ort
            self._session = ort.InferenceSession(str(self.model_path), **session_kwargs)
        return self._session

    def _prepare_inputs(self, session: Any, X: Any) -> Dict[str, Any]:
        if self.input_fn is not None:
            prepared = self.input_fn(X)
            if isinstance(prepared, dict):
                inputs: Dict[str, Any] = {}
                for key, value in prepared.items():
                    inputs[str(key)] = self._coerce_input_value(value)
                return inputs
            input_names = self._resolve_input_names(session, allow_multiple=False)
            return {input_names[0]: self._coerce_input_value(prepared)}

        input_names = self._resolve_input_names(session)
        if len(input_names) != 1:
            raise ValueError(
                "ONNXExtractor requires an input_fn when the model has multiple inputs."
            )
        return {input_names[0]: self._coerce_input_value(X)}

    def _resolve_input_names(self, session: Any, allow_multiple: bool = True) -> List[str]:
        if self.input_names is not None:
            if not self.input_names:
                raise ValueError("input_names must not be empty.")
            if not allow_multiple and len(self.input_names) != 1:
                raise ValueError(
                    "ONNXExtractor requires exactly one input when input_fn returns a single "
                    "array-like value."
                )
            return list(self.input_names)
        names = [str(item.name) for item in session.get_inputs()]
        if not names:
            raise ValueError("The ONNX model exposes no inputs.")
        if not allow_multiple and len(names) != 1:
            raise ValueError(
                "ONNXExtractor requires exactly one input when no input_names are provided."
            )
        return names

    def _resolve_output_names(self, session: Any) -> Optional[List[str]]:
        if self.output_names is not None:
            if not self.output_names:
                raise ValueError("output_names must not be empty.")
            return list(self.output_names)
        names = [str(item.name) for item in session.get_outputs()]
        if not names:
            raise ValueError("The ONNX model exposes no outputs.")
        if (
            self.output_fn is not None
            or len(self._output_specs) > 1
            or self._structured_output_specs
        ):
            return names
        if len(names) != 1:
            raise ValueError(
                "ONNXExtractor requires an output_fn when the model has multiple outputs."
            )
        return names

    def _run_session(self, session: Any, X: Any) -> Any:
        inputs = self._prepare_inputs(session, X)
        output_names = self._resolve_output_names(session)
        raw_outputs = session.run(output_names, inputs)
        named_outputs = dict(zip(output_names or [], raw_outputs))
        if self.output_fn is not None:
            projected = self.output_fn(raw_outputs)
        elif len(self._output_specs) > 1 or self._structured_output_specs:
            projected = named_outputs
        else:
            if len(raw_outputs) != 1:
                raise ValueError(
                    "ONNXExtractor received multiple outputs but no output_fn was provided."
                )
            projected = raw_outputs[0]
        return projected, named_outputs

    def _infer_batch_size(self, X: Any, session: Any) -> int:
        inputs = self._prepare_inputs(session, X)
        return infer_batch_size(inputs)

    def _coerce_input_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)


def _resolve_selector(value: Any, selector: str) -> Any:
    current = value
    for part in selector.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current
