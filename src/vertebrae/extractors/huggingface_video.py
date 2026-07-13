"""Optional Hugging Face video embedding extractor."""

from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np

from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.extractors.structured import StructuredEmbeddingOutput, StructuredOutputSpec


class HFVideoExtractor:
    """Hugging Face video backbone extractor with explicit pooling."""

    def __init__(
        self,
        name: str,
        model_id: str,
        processor_id: Optional[str] = None,
        pooling: str = "mean",
        hidden_layer: Optional[int] = None,
        outputs: Optional[List[Dict[str, Any]]] = None,
        structured_outputs: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 4,
        num_frames: int = 16,
        clip_duration_sec: Optional[float] = None,
        clip_start_sec: Optional[float] = None,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_paths: Optional[List[str]] = None,
    ) -> None:
        if pooling not in {"mean", "cls", "pooler"}:
            raise ValueError("pooling must be one of: mean, cls, pooler.")
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1.")
        if clip_duration_sec is not None and clip_duration_sec <= 0:
            raise ValueError("clip_duration_sec must be > 0 when provided.")
        if clip_start_sec is not None and clip_start_sec < 0:
            raise ValueError("clip_start_sec must be >= 0 when provided.")
        self.name = name
        self.model_id = model_id
        self.processor_id = processor_id or model_id
        self.pooling = pooling
        self.hidden_layer = hidden_layer
        self._output_specs = _resolve_output_specs(
            outputs=outputs,
            default_pooling=pooling,
            default_hidden_layer=hidden_layer,
        )
        self._structured_output_specs = _resolve_structured_output_specs(structured_outputs)
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.clip_duration_sec = clip_duration_sec
        self.clip_start_sec = clip_start_sec
        self.device = device
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.processor_kwargs = processor_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.checkpoint_paths = tuple(checkpoint_paths or ())
        self.modality = "video"
        self.extractor_type = "frozen_pretrained"
        self.streaming_safe = True
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def fit(self, X: Any, y: Any = None) -> "HFVideoExtractor":
        """No-op fit for frozen Hugging Face video models."""

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Encode video inputs into dense embeddings."""

        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "HFVideoExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return outputs[0].embeddings

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Encode video inputs into dense embeddings."""

        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        processor, model, torch = self._load_model()
        samples = _normalize_video_inputs(X, owner="HFVideoExtractor")
        collected: Dict[str, List[np.ndarray]] = {spec.name: [] for spec in self._output_specs}
        model.eval()
        need_hidden_states = any(spec.hidden_layer is not None for spec in self._output_specs)
        with torch.no_grad():
            for batch in _iter_chunks(samples, self.batch_size):
                clips = [self._prepare_clip(sample) for sample in batch]
                encoded = self._encode_batch(clips, processor, torch)
                model_output = model(
                    **encoded,
                    output_hidden_states=need_hidden_states,
                )
                for spec in self._output_specs:
                    pooled = self._pool(model_output, spec)
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
            raise ValueError("HFVideoExtractor was not configured with structured_outputs.")
        processor, model, torch = self._load_model()
        samples = _normalize_video_inputs(X, owner="HFVideoExtractor")
        collected: Dict[str, List[np.ndarray]] = {
            spec.name: [] for spec in self._structured_output_specs
        }
        model.eval()
        with torch.no_grad():
            for batch in _iter_chunks(samples, self.batch_size):
                clips = [self._prepare_clip(sample) for sample in batch]
                encoded = self._encode_batch(clips, processor, torch)
                model_output = model(**encoded, output_hidden_states=True)
                for spec in self._structured_output_specs:
                    hidden = self._select_hidden_state(model_output, spec.hidden_layer)
                    hidden = _flatten_sequence_axes(hidden)
                    values = hidden.detach().cpu().numpy().astype(np.float32, copy=False)
                    special_tokens = int(spec.metadata.get("special_tokens", 1))
                    for index in range(values.shape[0]):
                        collected[spec.name].append(values[index, special_tokens:])
        return [
            StructuredEmbeddingOutput(
                name=spec.name,
                embeddings=collected[spec.name],
                unit_type=spec.unit_type,
                recipe={"hidden_layer": spec.hidden_layer},
                metadata=dict(spec.metadata),
            )
            for spec in self._structured_output_specs
        ]

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable Hugging Face video recipe."""

        recipe: Dict[str, Any] = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_id": self.model_id,
            "processor_id": self.processor_id,
            "pooling": self.pooling,
            "hidden_layer": self.hidden_layer,
            "batch_size": self.batch_size,
            "num_frames": self.num_frames,
            "clip_duration_sec": self.clip_duration_sec,
            "clip_start_sec": self.clip_start_sec,
            "device": self.device,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "processor_kwargs": self.processor_kwargs,
            "model_kwargs": self.model_kwargs,
            "checkpoint_paths": list(self.checkpoint_paths),
            "streaming_safe": self.streaming_safe,
        }
        if len(self._output_specs) > 1:
            recipe["outputs"] = [_spec_to_dict(spec) for spec in self._output_specs]
        if self._structured_output_specs:
            recipe["structured_outputs"] = [
                _structured_spec_to_dict(spec) for spec in self._structured_output_specs
            ]
        return recipe

    def get_resource_profile_adapter(self) -> Any:
        from vertebrae.profiling import TorchResourceProfileAdapter

        return TorchResourceProfileAdapter(
            self,
            self.checkpoint_paths,
            model_getter=lambda: self._model,
            device_resolver=self._device,
        )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "HFVideoExtractor requires optional Hugging Face video dependencies. "
                    "Install with the documented extras."
                ) from exc
            processor_loader: Any
            try:
                from transformers import AutoVideoProcessor

                processor_loader = AutoVideoProcessor
            except ImportError:
                try:
                    from transformers import AutoImageProcessor

                    processor_loader = AutoImageProcessor
                except ImportError:
                    from transformers import AutoProcessor

                    processor_loader = AutoProcessor

            self._processor = processor_loader.from_pretrained(
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

    def _encode_batch(self, clips: List[np.ndarray], processor: Any, torch: Any) -> Dict[str, Any]:
        try:
            encoded = processor(videos=clips, return_tensors="pt", **self.processor_kwargs)
        except TypeError:
            encoded = processor(clips, return_tensors="pt", **self.processor_kwargs)
        return {key: value.to(self._device(torch)) for key, value in encoded.items()}

    def _prepare_clip(self, sample: Dict[str, Any]) -> np.ndarray:
        if "frames" in sample:
            clip = _coerce_clip_array(sample["frames"])
        else:
            clip = self._decode_video_path(Path(sample["path"]))
        return _sample_video_frames(clip, self.num_frames)

    def _decode_video_path(self, path: Path) -> np.ndarray:
        try:
            from pytorchvideo.data.encoded_video import EncodedVideo
        except ImportError as exc:
            raise ImportError(
                "HFVideoExtractor requires optional video decoding dependencies for video paths. "
                "Install the documented video extras or pass predecoded frame arrays."
            ) from exc

        video = EncodedVideo.from_path(str(path))
        start_sec = float(self.clip_start_sec or 0.0)
        end_sec = self.clip_duration_sec + start_sec if self.clip_duration_sec is not None else None
        if end_sec is None:
            duration = getattr(video, "duration", None)
            end_sec = float(duration) if duration is not None else start_sec
        clip = video.get_clip(start_sec=start_sec, end_sec=float(end_sec))
        frames = clip.get("video") if isinstance(clip, dict) else None
        if frames is None:
            raise ValueError(f"Could not decode video frames from '{path}'.")
        return _coerce_decoded_video(frames)

    def _pool(self, model_output: Any, spec: EmbeddingOutputSpec) -> Any:
        pooling = cast(str, spec.pooling)
        if pooling == "pooler":
            if spec.hidden_layer is not None:
                raise ValueError("pooler pooling cannot be combined with hidden_layer.")
            pooled = getattr(model_output, "pooler_output", None)
            if pooled is None:
                raise ValueError("Model output has no pooler_output.")
            return _flatten_feature_axes(pooled)

        hidden = self._select_hidden_state(model_output, spec.hidden_layer)
        hidden = _flatten_sequence_axes(hidden)
        if pooling == "cls":
            return hidden[:, 0, :]
        return hidden.mean(dim=1)

    def _select_hidden_state(self, output: Any, hidden_layer: Optional[int]) -> Any:
        if hidden_layer is None:
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
            return hidden_states[hidden_layer]
        except IndexError as exc:
            raise ValueError(
                f"hidden_layer index {hidden_layer} is out of range for "
                f"{len(hidden_states)} hidden states."
            ) from exc


def _normalize_video_inputs(value: Any, owner: str) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        return _normalize_video_mapping(value, owner)
    if isinstance(value, (str, Path)):
        return [{"path": str(value)}]
    if isinstance(value, np.ndarray) and value.ndim == 4:
        return [{"frames": value}]
    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(_normalize_video_sample(item, owner))
            elif isinstance(item, (str, Path)):
                normalized.append({"path": str(item)})
            else:
                normalized.append({"frames": item})
        return normalized
    raise ValueError(
        f"{owner} expects video paths, frame arrays, or structured inputs containing "
        "'frames' or 'path'."
    )


def _normalize_video_mapping(value: Dict[str, Any], owner: str) -> List[Dict[str, Any]]:
    if "frames" in value:
        frames = _coerce_object_sequence(value["frames"])
        frame_rates = _optional_aligned_sequence(value.get("frame_rate"), len(frames), "frame_rate")
        return [
            {
                "frames": frames[index],
                "frame_rate": frame_rates[index],
            }
            for index in range(len(frames))
        ]
    if "path" in value:
        paths = np.asarray(value["path"], dtype=object)
        return [{"path": str(paths[index])} for index in range(len(paths))]
    raise ValueError(f"{owner} expects structured video inputs with 'frames' or 'path'.")


def _normalize_video_sample(value: Dict[str, Any], owner: str) -> Dict[str, Any]:
    if "frames" in value:
        return {"frames": value["frames"], "frame_rate": value.get("frame_rate")}
    if "path" in value:
        return {"path": str(value["path"])}
    raise ValueError(f"{owner} video sample dictionaries must contain 'frames' or 'path'.")


def _optional_aligned_sequence(value: Any, size: int, name: str) -> List[Optional[Any]]:
    if value is None:
        return [None] * size
    if np.isscalar(value):
        return [value] * size
    items = list(value)
    if len(items) != size:
        raise ValueError(
            f"Structured video field '{name}' must match the number of samples; "
            f"got {len(items)} and {size}."
        )
    return items


def _coerce_clip_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 4:
        raise ValueError(
            "HFVideoExtractor expects frame arrays with shape (time, height, width, channels)."
        )
    if array.shape[-1] not in {1, 3, 4}:
        raise ValueError("HFVideoExtractor frame arrays must have 1, 3, or 4 channels.")
    if array.shape[0] < 1:
        raise ValueError("HFVideoExtractor frame arrays must contain at least one frame.")
    return array


def _coerce_decoded_video(value: Any) -> np.ndarray:
    shape = getattr(value, "shape", None)
    if shape is None:
        array = np.asarray(value)
        return _coerce_clip_array(array)
    if len(shape) != 4:
        raise ValueError("Decoded video tensor must have four dimensions.")

    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 3, 0))
    elif array.shape[1] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (0, 2, 3, 1))
    return _coerce_clip_array(array)


def _sample_video_frames(clip: np.ndarray, num_frames: int) -> np.ndarray:
    if clip.shape[0] == num_frames:
        return clip
    if clip.shape[0] > num_frames:
        indices = np.linspace(0, clip.shape[0] - 1, num_frames, dtype=int)
        return clip[indices]

    pad_indices = np.linspace(0, clip.shape[0] - 1, num_frames, dtype=int)
    return clip[pad_indices]


def _flatten_sequence_axes(hidden: Any) -> Any:
    shape = hidden.shape
    if len(shape) <= 3:
        return hidden
    return hidden.reshape(shape[0], int(np.prod(shape[1:-1])), shape[-1])


def _flatten_feature_axes(value: Any) -> Any:
    shape = value.shape
    if len(shape) <= 2:
        return value
    return value.reshape(shape[0], int(np.prod(shape[1:])))


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
            raise ValueError("HFVideoExtractor output specs must include a name.")
        pooling = raw.get("pooling", default_pooling)
        if pooling not in {"mean", "cls", "pooler"}:
            raise ValueError("HFVideoExtractor output pooling must be one of: mean, cls, pooler.")
        specs.append(
            EmbeddingOutputSpec(
                name=str(raw["name"]),
                pooling=pooling,
                hidden_layer=raw.get("hidden_layer"),
            )
        )
    _ensure_unique_names(specs)
    return specs


def _ensure_unique_names(specs: List[Any]) -> None:
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("HFVideoExtractor output names must be unique.")


def _spec_to_dict(spec: EmbeddingOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "pooling": spec.pooling,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }


def _resolve_structured_output_specs(
    outputs: Optional[List[Dict[str, Any]]],
) -> List[StructuredOutputSpec]:
    specs = []
    for raw in outputs or []:
        if "name" not in raw:
            raise ValueError("HFVideoExtractor structured outputs must include a name.")
        specs.append(
            StructuredOutputSpec(
                name=str(raw["name"]),
                unit_type=str(raw.get("unit_type", "frame")),
                hidden_layer=raw.get("hidden_layer"),
                metadata={
                    "special_tokens": int(raw.get("special_tokens", 1)),
                    **dict(raw.get("metadata", {})),
                },
            )
        )
    _ensure_unique_names(specs)
    return specs


def _structured_spec_to_dict(spec: StructuredOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "unit_type": spec.unit_type,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }


def _coerce_object_sequence(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == object:
        return value
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError("Expected a sequence of per-sample values.") from exc
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result
