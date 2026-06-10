"""Optional Hugging Face time-series embedding extractor."""

from typing import Any, Dict, List, Optional

import numpy as np


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

        model, torch = self._load_model()
        series_inputs = _normalize_time_series_inputs(X, owner="HFTimeSeriesExtractor")
        outputs: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for batch in _iter_chunks(series_inputs, self.batch_size):
                encoded = self._encode_batch(batch, torch)
                model_output = model(
                    **encoded,
                    output_hidden_states=self.hidden_layer is not None,
                )
                hidden = self._select_hidden_state(model_output)
                pooled = self._pool(hidden)
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(outputs).astype(np.float32, copy=False) if outputs else np.empty((0, 0))

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Encode time-series inputs into dense embeddings."""

        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable Hugging Face time-series recipe."""

        return {
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
        encoded: Dict[str, Any] = {
            "past_values": torch.as_tensor(series).to(self._device(torch))
        }
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

    def _select_hidden_state(self, output: Any) -> Any:
        if self.hidden_layer is not None:
            hidden_states = getattr(output, "hidden_states", None)
            if hidden_states is None:
                raise ValueError(
                    "hidden_layer was requested, but model output has no hidden_states. "
                    "This model may not support output_hidden_states."
                )
            try:
                return hidden_states[self.hidden_layer]
            except IndexError as exc:
                raise ValueError(
                    f"hidden_layer index {self.hidden_layer} is out of range for "
                    f"{len(hidden_states)} hidden states."
                ) from exc
        for attr in ("last_hidden_state", "encoder_last_hidden_state"):
            hidden = getattr(output, attr, None)
            if hidden is not None:
                return hidden
        raise ValueError(
            "Model output has no last_hidden_state or encoder_last_hidden_state."
        )

    def _pool(self, hidden: Any) -> Any:
        if self.pooling == "last":
            return hidden[:, -1, :]
        if self.pooling == "flatten":
            return hidden.flatten(start_dim=1)
        return hidden.mean(dim=1)


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
