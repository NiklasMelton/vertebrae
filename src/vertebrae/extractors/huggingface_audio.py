"""Optional Hugging Face audio embedding extractor."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class HFAudioExtractor:
    """Hugging Face audio backbone extractor with explicit pooling.

    Args:
        name: User-facing extractor name.
        model_id: Hugging Face model identifier or local path.
        processor_id: Optional Hugging Face processor identifier or local path.
            Defaults to `model_id`.
        pooling: Pooling mode: `"mean"`, `"cls"`, or `"pooler"`.
        hidden_layer: Optional hidden-state layer index to pool from. Defaults to
            the model's final output.
        batch_size: Number of audio clips encoded per batch.
        sampling_rate: Default sampling rate for array inputs.
        device: Optional device string.
        revision: Optional model revision.
        trust_remote_code: Whether to allow remote model code.
        processor_kwargs: Extra keyword arguments for the processor.
        model_kwargs: Extra keyword arguments for `AutoModel`.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        processor_id: Optional[str] = None,
        pooling: str = "mean",
        hidden_layer: Optional[int] = None,
        batch_size: int = 16,
        sampling_rate: Optional[int] = None,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if pooling not in {"mean", "cls", "pooler"}:
            raise ValueError("pooling must be one of: mean, cls, pooler.")
        self.name = name
        self.model_id = model_id
        self.processor_id = processor_id or model_id
        self.pooling = pooling
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.device = device
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.processor_kwargs = processor_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.modality = "audio"
        self.extractor_type = "frozen_pretrained"
        self.streaming_safe = True
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def fit(self, X: Any, y: Any = None) -> "HFAudioExtractor":
        """No-op fit for frozen Hugging Face audio models."""

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Encode audio inputs into dense embeddings."""

        processor, model, torch = self._load_model()
        samples = _normalize_audio_inputs(X, owner="HFAudioExtractor")
        outputs: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for batch in _iter_chunks(samples, self.batch_size):
                arrays, sampling_rate = self._resolve_batch(batch)
                encoded = processor(
                    arrays,
                    sampling_rate=sampling_rate,
                    padding=True,
                    return_tensors="pt",
                    **self.processor_kwargs,
                )
                encoded = {key: value.to(self._device(torch)) for key, value in encoded.items()}
                model_output = model(
                    **encoded,
                    output_hidden_states=self.hidden_layer is not None,
                )
                hidden = self._select_hidden_state(model_output)
                pooled = self._pool(model_output, hidden, encoded.get("attention_mask"))
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(outputs).astype(np.float32, copy=False) if outputs else np.empty((0, 0))

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Encode audio inputs into dense embeddings."""

        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable Hugging Face audio recipe."""

        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_id": self.model_id,
            "processor_id": self.processor_id,
            "pooling": self.pooling,
            "hidden_layer": self.hidden_layer,
            "batch_size": self.batch_size,
            "sampling_rate": self.sampling_rate,
            "device": self.device,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "processor_kwargs": self.processor_kwargs,
            "model_kwargs": self.model_kwargs,
            "streaming_safe": self.streaming_safe,
        }

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoProcessor
            except ImportError:
                try:
                    import torch
                    from transformers import AutoFeatureExtractor, AutoModel
                except ImportError as exc:
                    raise ImportError(
                        "HFAudioExtractor requires optional Hugging Face audio dependencies. "
                        "Install with the documented extras."
                    ) from exc
                self._processor = AutoFeatureExtractor.from_pretrained(
                    self.processor_id,
                    **self._common_kwargs(),
                    **self.processor_kwargs,
                )
            else:
                self._processor = AutoProcessor.from_pretrained(
                    self.processor_id,
                    **self._common_kwargs(),
                    **self.processor_kwargs,
                )
            self._model = AutoModel.from_pretrained(
                self.model_id,
                **self._common_kwargs(),
                **self.model_kwargs,
            )
            self._torch = torch
            assert self._model is not None
            self._model.to(self._device(torch))
        return self._processor, self._model, self._torch

    def _common_kwargs(self) -> Dict[str, Any]:
        common_kwargs = {
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
        }
        return {key: value for key, value in common_kwargs.items() if value is not None}

    def _device(self, torch: Any) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_batch(self, batch: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], int]:
        arrays: List[np.ndarray] = []
        sampling_rates = set()
        for sample in batch:
            array, inferred_rate = _load_audio_input(sample["audio"])
            arrays.append(array)
            resolved_rate = sample["sampling_rate"] or inferred_rate or self.sampling_rate
            if resolved_rate is None:
                raise ValueError(
                    "HFAudioExtractor requires a sampling rate for array inputs. "
                    "Pass sampling_rate to the extractor or dataset."
                )
            sampling_rates.add(int(resolved_rate))
        if len(sampling_rates) != 1:
            raise ValueError(
                "HFAudioExtractor requires a single sampling rate per batch; "
                f"got {sorted(sampling_rates)}."
            )
        return arrays, sampling_rates.pop()

    def _select_hidden_state(self, output: Any) -> Any:
        if self.hidden_layer is None:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = getattr(output, "extract_features", None)
            if hidden is None:
                raise ValueError("Model output has no last_hidden_state or extract_features.")
            return hidden
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

    def _pool(self, output: Any, hidden: Any, mask: Any) -> Any:
        if self.pooling == "cls":
            return hidden[:, 0, :]
        if self.pooling == "pooler":
            pooler_output = getattr(output, "pooler_output", None)
            if pooler_output is None:
                raise ValueError("pooler pooling requested, but model output has no pooler_output.")
            return pooler_output
        if mask is None:
            return hidden.mean(dim=1)
        expanded_mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        masked_hidden = hidden * expanded_mask
        lengths = expanded_mask.sum(dim=1).clamp(min=1e-9)
        return masked_hidden.sum(dim=1) / lengths


def _normalize_audio_inputs(value: Any, owner: str) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        return _normalize_audio_mapping(value, owner)
    if isinstance(value, (str, Path)):
        return [{"audio": value, "sampling_rate": None}]
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(
            f"{owner} expects audio paths, waveform arrays, or structured inputs."
        ) from exc
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(_normalize_audio_sample(item, owner))
        else:
            normalized.append({"audio": item, "sampling_rate": None})
    return normalized


def _normalize_audio_mapping(value: Dict[str, Any], owner: str) -> List[Dict[str, Any]]:
    if "array" in value:
        arrays = list(value["array"])
        rates = _broadcast_optional_sequence(value.get("sampling_rate"), len(arrays))
        return [
            {"audio": array, "sampling_rate": rates[index]} for index, array in enumerate(arrays)
        ]
    if "path" in value:
        paths = list(value["path"])
        rates = _broadcast_optional_sequence(value.get("sampling_rate"), len(paths))
        return [{"audio": path, "sampling_rate": rates[index]} for index, path in enumerate(paths)]
    raise ValueError(f"{owner} expects structured audio inputs with 'array' or 'path'.")


def _normalize_audio_sample(value: Dict[str, Any], owner: str) -> Dict[str, Any]:
    if "array" in value:
        return {"audio": value["array"], "sampling_rate": value.get("sampling_rate")}
    if "path" in value:
        return {"audio": value["path"], "sampling_rate": value.get("sampling_rate")}
    raise ValueError(f"{owner} audio sample dictionaries must contain 'array' or 'path'.")


def _broadcast_optional_sequence(value: Any, size: int) -> List[Optional[int]]:
    if value is None:
        return [None] * size
    if np.isscalar(value):
        return [_coerce_optional_int(value)] * size
    values = list(value)
    if len(values) != size:
        raise ValueError(
            "Structured audio sampling_rate must match the number of samples; "
            f"got {len(values)} and {size}."
        )
    return [_coerce_optional_int(item) for item in values]


def _iter_chunks(items: List[Dict[str, Any]], batch_size: int) -> Any:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_audio_input(value: Any) -> Tuple[np.ndarray, Optional[int]]:
    if isinstance(value, (str, Path)):
        return _read_audio_path(value)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array, None
    if array.ndim == 2:
        # Average channels to keep the extractor output contract simple.
        return array.mean(axis=1).astype(np.float32, copy=False), None
    raise ValueError(
        "Audio arrays must be 1D (samples) or 2D (samples, channels); " f"got shape {array.shape}."
    )


def _read_audio_path(path: Any) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Audio path inputs require optional soundfile support. "
            "Install with the documented audio or hf extras."
        ) from exc
    audio, sampling_rate = sf.read(str(path), always_2d=False)
    array, _ = _load_audio_input(audio)
    return array, int(sampling_rate)


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("sampling_rate must be an integer, not a boolean.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sampling_rate must be integer-like; got {value!r}.") from exc
