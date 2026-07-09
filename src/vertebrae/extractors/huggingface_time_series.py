"""Optional Hugging Face time-series embedding extractor."""

from typing import Any, Dict, List, Optional, cast

import numpy as np

from vertebrae.extractors._utils import (
    materialize_structured_parent_matrices,
    resolve_output_value,
    resolve_structured_output_specs,
    structured_spec_to_recipe,
)
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.extractors.structured import StructuredEmbeddingOutput, StructuredOutputSpec


class HFTimeSeriesExtractor:
    """Hugging Face time-series backbone extractor with explicit pooling.

    Args:
        name: User-facing extractor name.
        model_id: Hugging Face model identifier or local path.
        pooling: Pooling mode: `"mean"`, `"last"`, or `"flatten"`.
        hidden_layer: Optional hidden-state layer index to pool from. Defaults to
            the model's final sequence output.
        batch_size: Number of series encoded per batch.
        device: Optional device string.
        revision: Optional model revision.
        trust_remote_code: Whether to allow remote model code.
        input_kwargs: Extra keyword arguments merged into every model call.
        model_kwargs: Extra keyword arguments for `AutoModel`.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        pooling: str = "mean",
        hidden_layer: Optional[int] = None,
        outputs: Optional[List[Dict[str, Any]]] = None,
        structured_outputs: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        input_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if pooling not in {"mean", "last", "flatten"}:
            raise ValueError("pooling must be one of: mean, last, flatten.")
        self.name = name
        self.model_id = model_id
        self.pooling = pooling
        self.hidden_layer = hidden_layer
        self._output_specs = _resolve_output_specs(
            outputs=outputs,
            default_pooling=pooling,
            default_hidden_layer=hidden_layer,
        )
        self._structured_output_specs = resolve_structured_output_specs(structured_outputs)
        self.batch_size = batch_size
        self.device = device
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.input_kwargs = input_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.modality = "time_series"
        self.extractor_type = "frozen_pretrained"
        self.streaming_safe = True
        self._model: Any = None
        self._torch: Any = None

    def fit(self, X: Any, y: Any = None) -> "HFTimeSeriesExtractor":
        """No-op fit for frozen Hugging Face time-series models."""

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Encode time-series inputs into dense embeddings."""
        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "HFTimeSeriesExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return outputs[0].embeddings

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Encode time-series inputs into dense embeddings."""

        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        model, torch = self._load_model()
        series_inputs = _normalize_time_series_inputs(X, owner="HFTimeSeriesExtractor")
        collected: Dict[str, List[np.ndarray]] = {spec.name: [] for spec in self._output_specs}
        model.eval()
        need_hidden_states = any(spec.hidden_layer is not None for spec in self._output_specs)
        with torch.no_grad():
            for batch in _iter_chunks(series_inputs, self.batch_size):
                model_output = self._forward_batch(
                    batch,
                    torch,
                    output_hidden_states=need_hidden_states,
                )
                for spec in self._output_specs:
                    hidden = self._select_hidden_state(model_output, spec.hidden_layer)
                    pooled = self._pool(hidden, cast(str, spec.pooling))
                    collected[spec.name].append(
                        pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
        outputs: List[EmbeddingOutput] = []
        for spec in self._output_specs:
            arrays = collected[spec.name]
            embeddings = (
                np.vstack(arrays).astype(np.float32, copy=False) if arrays else np.empty((0, 0))
            )
            outputs.append(
                EmbeddingOutput(
                    name=spec.name,
                    embeddings=embeddings,
                    recipe={
                        "pooling": spec.pooling,
                        "hidden_layer": spec.hidden_layer,
                    },
                    metadata={
                        "pooling": spec.pooling,
                        "hidden_layer": spec.hidden_layer,
                    },
                )
            )
        return outputs

    def structured_output_specs(self) -> List[StructuredOutputSpec]:
        return list(self._structured_output_specs)

    def transform_structured(self, X: Any) -> List[StructuredEmbeddingOutput]:
        if not self._structured_output_specs:
            raise ValueError("HFTimeSeriesExtractor was not configured with structured_outputs.")
        model, torch = self._load_model()
        series_inputs = _normalize_time_series_inputs(X, owner="HFTimeSeriesExtractor")
        collected: Dict[str, List[np.ndarray]] = {
            spec.name: [] for spec in self._structured_output_specs
        }
        model.eval()
        with torch.no_grad():
            for batch in _iter_chunks(series_inputs, self.batch_size):
                model_output = self._forward_batch(batch, torch, output_hidden_states=True)
                for spec in self._structured_output_specs:
                    value = self._resolve_structured_value(model_output, spec)
                    parents = materialize_structured_parent_matrices(
                        value,
                        f"HFTimeSeriesExtractor structured output '{spec.name}'",
                        expected_parents=len(batch),
                    )
                    collected[spec.name].extend(
                        np.asarray(parent, dtype=np.float32) for parent in parents
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
        """Return a serializable Hugging Face time-series recipe."""

        recipe: Dict[str, Any] = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_id": self.model_id,
            "pooling": self.pooling,
            "hidden_layer": self.hidden_layer,
            "batch_size": self.batch_size,
            "device": self.device,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "input_kwargs": self.input_kwargs,
            "model_kwargs": self.model_kwargs,
            "streaming_safe": self.streaming_safe,
        }
        if len(self._output_specs) > 1:
            recipe["outputs"] = [_spec_to_dict(spec) for spec in self._output_specs]
        if self._structured_output_specs:
            recipe["structured_outputs"] = [
                _structured_spec_to_dict(spec) for spec in self._structured_output_specs
            ]
        return recipe

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "HFTimeSeriesExtractor requires optional Hugging Face time-series "
                    "dependencies. Install with the documented extras."
                ) from exc
            common_kwargs = {
                "revision": self.revision,
                "trust_remote_code": self.trust_remote_code,
            }
            common_kwargs = {
                key: value for key, value in common_kwargs.items() if value is not None
            }
            self._model = AutoModel.from_pretrained(
                self.model_id,
                **common_kwargs,
                **self.model_kwargs,
            )
            self._torch = torch
            assert self._model is not None
            self._model.to(self._device(torch))
        return self._model, self._torch

    def _device(self, torch: Any) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _encode_batch(self, batch: List[Dict[str, Any]], torch: Any) -> Dict[str, Any]:
        series = np.asarray([sample["series"] for sample in batch], dtype=np.float32)
        encoded: Dict[str, Any] = {"past_values": torch.as_tensor(series).to(self._device(torch))}
        observed_mask = [sample.get("observed_mask") for sample in batch]
        if all(mask is not None for mask in observed_mask):
            encoded["past_observed_mask"] = torch.as_tensor(
                np.asarray(observed_mask, dtype=np.float32)
            ).to(self._device(torch))
        time_features = [sample.get("time_features") for sample in batch]
        if all(features is not None for features in time_features):
            encoded["past_time_features"] = torch.as_tensor(
                np.asarray(time_features, dtype=np.float32)
            ).to(self._device(torch))
        for key, value in self.input_kwargs.items():
            encoded[key] = value.to(self._device(torch)) if hasattr(value, "to") else value
        return encoded

    def _forward_batch(
        self,
        batch: List[Dict[str, Any]],
        torch: Any,
        output_hidden_states: bool,
    ) -> Any:
        model, _ = self._load_model()
        encoded = self._encode_batch(batch, torch)
        return model(
            **encoded,
            output_hidden_states=output_hidden_states,
        )

    def _select_hidden_state(self, output: Any, hidden_layer: Optional[int]) -> Any:
        if hidden_layer is not None:
            hidden_states = getattr(output, "hidden_states", None)
            if hidden_states is None:
                raise ValueError(
                    "hidden_layer was requested, but model output has no hidden_states. "
                    "This model may not support output_hidden_states."
                )
            try:
                return hidden_states[hidden_layer]
            except IndexError as exc:
                raise ValueError(
                    f"hidden_layer index {hidden_layer} is out of range for "
                    f"{len(hidden_states)} hidden states."
                ) from exc
        for attr in ("last_hidden_state", "encoder_last_hidden_state"):
            hidden = getattr(output, attr, None)
            if hidden is not None:
                return hidden
        raise ValueError("Model output has no last_hidden_state or encoder_last_hidden_state.")

    def _pool(self, hidden: Any, pooling: str) -> Any:
        if pooling == "last":
            return hidden[:, -1, :]
        if pooling == "flatten":
            return hidden.flatten(start_dim=1)
        return hidden.mean(dim=1)

    def _resolve_structured_value(self, output: Any, spec: StructuredOutputSpec) -> Any:
        selector = spec.metadata.get("selector")
        if selector:
            value = resolve_output_value(output, str(selector))
            if value is None:
                raise ValueError(
                    f"HFTimeSeriesExtractor structured output '{spec.name}' could not resolve "
                    f"selector='{selector}'."
                )
            if spec.hidden_layer is not None:
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        f"HFTimeSeriesExtractor structured output '{spec.name}' requested "
                        "hidden_layer but selector does not resolve to hidden states."
                    )
                try:
                    return value[spec.hidden_layer]
                except IndexError as exc:
                    raise ValueError(
                        f"HFTimeSeriesExtractor structured output '{spec.name}' hidden_layer "
                        f"{spec.hidden_layer} is out of range."
                    ) from exc
            return value
        return self._select_hidden_state(output, spec.hidden_layer)


def _normalize_time_series_inputs(value: Any, owner: str) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        return _normalize_time_series_mapping(value, owner)
    array = np.asarray(value)
    if array.ndim not in {2, 3}:
        raise ValueError(
            f"{owner} expects a 2D or 3D array, or a structured mapping of time-series inputs."
        )
    return [{"series": sample} for sample in array]


def _normalize_time_series_mapping(value: Dict[str, Any], owner: str) -> List[Dict[str, Any]]:
    series = np.asarray(value.get("series"))
    if series.ndim not in {2, 3}:
        raise ValueError(f"{owner} expects 'series' with shape (n, time) or (n, time, channels).")
    size = int(series.shape[0])
    observed_mask = _optional_aligned_sequence(value.get("observed_mask"), size, "observed_mask")
    time_features = _optional_aligned_sequence(value.get("time_features"), size, "time_features")
    return [
        {
            "series": series[index],
            "observed_mask": observed_mask[index],
            "time_features": time_features[index],
        }
        for index in range(size)
    ]


def _optional_aligned_sequence(value: Any, size: int, name: str) -> List[Optional[Any]]:
    if value is None:
        return [None] * size
    items = list(value)
    if len(items) != size:
        raise ValueError(
            f"Structured time-series field '{name}' must match the number of samples; "
            f"got {len(items)} and {size}."
        )
    return items


def _iter_chunks(items: List[Dict[str, Any]], batch_size: int) -> Any:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _resolve_output_specs(
    outputs: Optional[List[Dict[str, Any]]],
    default_pooling: str,
    default_hidden_layer: Optional[int],
) -> List[EmbeddingOutputSpec]:
    if outputs is None:
        return [
            EmbeddingOutputSpec(
                name="default",
                pooling=default_pooling,
                hidden_layer=default_hidden_layer,
            )
        ]
    specs = []
    for raw in outputs:
        if "name" not in raw:
            raise ValueError("HFTimeSeriesExtractor output specs must include a name.")
        pooling = raw.get("pooling", default_pooling)
        if pooling not in {"mean", "last", "flatten"}:
            raise ValueError(
                "HFTimeSeriesExtractor output pooling must be one of: mean, last, flatten."
            )
        specs.append(
            EmbeddingOutputSpec(
                name=str(raw["name"]),
                pooling=pooling,
                hidden_layer=raw.get("hidden_layer"),
            )
        )
    _ensure_unique_names(specs)
    return specs


def _ensure_unique_names(specs: List[EmbeddingOutputSpec]) -> None:
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("HFTimeSeriesExtractor output names must be unique.")


def _spec_to_dict(spec: EmbeddingOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "pooling": spec.pooling,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }


def _structured_spec_to_dict(spec: StructuredOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "unit_type": spec.unit_type,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }
