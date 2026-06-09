"""Optional Hugging Face vision embedding extractor."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class HFVisionExtractor:
    """Hugging Face vision backbone extractor with explicit pooling.

    Args:
        name: User-facing extractor name.
        model_id: Hugging Face model identifier or local path.
        processor_id: Optional Hugging Face processor identifier or local path.
            Defaults to `model_id`.
        pooling: Pooling mode: `"cls"`, `"mean"`, or `"pooler"`.
        hidden_layer: Optional hidden-state layer index to pool from. Defaults to
            the model's final output.
        batch_size: Number of images encoded per batch.
        image_mode: Image representation to pass to the processor:
            `"auto"`, `"rgb"`, `"grayscale"`, or `"preserve"`.
        alpha_mode: How alpha channels are handled when converting:
            `"drop"`, `"white_background"`, or `"black_background"`.
        device: Optional device string.
        revision: Optional model revision.
        trust_remote_code: Whether to allow remote model code.
        processor_kwargs: Extra keyword arguments for `AutoImageProcessor`.
        model_kwargs: Extra keyword arguments for `AutoModel`.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        processor_id: Optional[str] = None,
        pooling: str = "cls",
        hidden_layer: Optional[int] = None,
        batch_size: int = 16,
        image_mode: str = "auto",
        alpha_mode: str = "drop",
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if pooling not in {"cls", "mean", "pooler"}:
            raise ValueError("pooling must be one of: cls, mean, pooler.")
        if image_mode not in _IMAGE_MODES:
            raise ValueError(f"image_mode must be one of: {', '.join(sorted(_IMAGE_MODES))}.")
        if alpha_mode not in _ALPHA_MODES:
            raise ValueError(f"alpha_mode must be one of: {', '.join(sorted(_ALPHA_MODES))}.")
        self.name = name
        self.model_id = model_id
        self.processor_id = processor_id or model_id
        self.pooling = pooling
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.image_mode = image_mode
        self.alpha_mode = alpha_mode
        self.device = device
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.processor_kwargs = processor_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.modality = "image"
        self.extractor_type = "frozen_pretrained"
        self.streaming_safe = True
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._image_module: Any = None

    def fit(self, X: Any, y: Any = None) -> "HFVisionExtractor":
        """No-op fit for frozen Hugging Face vision models.

        Args:
            X: Image inputs.
            y: Optional labels.

        Returns:
            This extractor.
        """

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Encode image inputs into dense embeddings.

        Args:
            X: PIL images, NumPy image arrays, image paths, or a sequence of them.

        Returns:
            Dense float32 embedding matrix.

        Raises:
            ImportError: If optional Hugging Face vision dependencies are missing.
            ValueError: If pooling is invalid for the model output.
        """

        processor, model, torch, image_module = self._load_model()
        outputs: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for items in _iter_chunks(_as_iterable(X), self.batch_size):
                batch = [
                    _coerce_image(item, image_module, self.image_mode, self.alpha_mode)
                    for item in items
                ]
                encoded = processor(images=batch, return_tensors="pt", **self.processor_kwargs)
                encoded = {key: value.to(self._device(torch)) for key, value in encoded.items()}
                model_output = model(
                    **encoded,
                    output_hidden_states=self.hidden_layer is not None,
                )
                pooled = self._pool(model_output)
                if len(pooled.shape) > 2:
                    pooled = pooled.flatten(start_dim=1)
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(outputs).astype(np.float32, copy=False) if outputs else np.empty((0, 0))

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Encode image inputs into dense embeddings.

        Args:
            X: Image inputs.
            y: Optional labels.

        Returns:
            Dense float32 embedding matrix.
        """

        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable Hugging Face vision recipe.

        Returns:
            JSON-compatible recipe dictionary.
        """

        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_id": self.model_id,
            "processor_id": self.processor_id,
            "pooling": self.pooling,
            "hidden_layer": self.hidden_layer,
            "batch_size": self.batch_size,
            "image_mode": self.image_mode,
            "alpha_mode": self.alpha_mode,
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
                from PIL import Image
                from transformers import AutoImageProcessor, AutoModel
            except ImportError as exc:
                raise ImportError(
                    "HFVisionExtractor requires optional Hugging Face vision "
                    "dependencies. Install with the documented Hugging Face extra or "
                    "Poetry group."
                ) from exc
            common_kwargs = {
                "revision": self.revision,
                "trust_remote_code": self.trust_remote_code,
            }
            common_kwargs = {
                key: value for key, value in common_kwargs.items() if value is not None
            }
            self._processor = AutoImageProcessor.from_pretrained(
                self.processor_id,
                **common_kwargs,
                **self.processor_kwargs,
            )
            self._model = AutoModel.from_pretrained(
                self.model_id,
                **common_kwargs,
                **self.model_kwargs,
            )
            self._torch = torch
            self._image_module = Image
            assert self._model is not None
            self._model.to(self._device(torch))
        return self._processor, self._model, self._torch, self._image_module

    def _device(self, torch: Any) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _pool(self, output: Any) -> Any:
        if self.hidden_layer is not None:
            if self.pooling == "pooler":
                raise ValueError("pooler pooling cannot be used with hidden_layer.")
            hidden = self._select_hidden_state(output)
            if self.pooling == "cls":
                return hidden[:, 0, :]
            return hidden.mean(dim=1)
        if self.pooling == "pooler":
            pooler_output = getattr(output, "pooler_output", None)
            if pooler_output is None:
                raise ValueError("pooler pooling requested, but model output has no pooler_output.")
            return pooler_output
        hidden = output.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0, :]
        return hidden.mean(dim=1)

    def _select_hidden_state(self, output: Any) -> Any:
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


def _as_iterable(value: Any) -> Any:
    if isinstance(value, (str, Path)):
        return iter([value])
    try:
        return iter(value)
    except TypeError as exc:
        raise ValueError("HFVisionExtractor expects images, image arrays, or image paths.") from exc


def _iter_chunks(items: Any, batch_size: int) -> Any:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


_IMAGE_MODES = {"auto", "rgb", "grayscale", "preserve"}
_ALPHA_MODES = {"drop", "white_background", "black_background"}


def _coerce_image(value: Any, image_module: Any, image_mode: str, alpha_mode: str) -> Any:
    if image_mode not in _IMAGE_MODES:
        raise ValueError(f"image_mode must be one of: {', '.join(sorted(_IMAGE_MODES))}.")
    if alpha_mode not in _ALPHA_MODES:
        raise ValueError(f"alpha_mode must be one of: {', '.join(sorted(_ALPHA_MODES))}.")

    is_path = isinstance(value, (str, Path))
    if image_mode == "preserve" and not is_path:
        return value

    if isinstance(value, (str, Path)):
        image = image_module.open(value)
        if image_mode == "preserve":
            return image
        if image_mode == "auto":
            return image.convert("RGB")
        return _convert_image(image, image_module, image_mode, alpha_mode)

    if isinstance(value, np.ndarray):
        image = _array_to_image(value, image_module)
        if image_mode == "auto":
            return image
        return _convert_image(image, image_module, image_mode, alpha_mode)
    if image_mode in {"rgb", "grayscale"}:
        return _convert_image(value, image_module, image_mode, alpha_mode)
    return value


def _array_to_image(value: np.ndarray, image_module: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 2:
        return image_module.fromarray(array)
    if array.ndim != 3:
        raise ValueError(
            "NumPy image arrays must have shape (height, width), "
            "(height, width, 1), (height, width, 3), or (height, width, 4)."
        )
    channels = array.shape[2]
    if channels == 1:
        return image_module.fromarray(array[:, :, 0])
    if channels in {3, 4}:
        return image_module.fromarray(array)
    raise ValueError(
        "NumPy image arrays must have 1, 3, or 4 channels in the last dimension; "
        f"got {channels}."
    )


def _convert_image(image: Any, image_module: Any, image_mode: str, alpha_mode: str) -> Any:
    image = _apply_alpha_mode(image, image_module, alpha_mode)
    if image_mode == "rgb":
        return image.convert("RGB")
    if image_mode == "grayscale":
        grayscale = np.asarray(image.convert("L"))
        return grayscale[:, :, np.newaxis]
    raise ValueError("image_mode must be 'rgb' or 'grayscale' for conversion.")


def _apply_alpha_mode(image: Any, image_module: Any, alpha_mode: str) -> Any:
    if alpha_mode == "drop" or not _has_alpha(image):
        return image
    color = (255, 255, 255, 255) if alpha_mode == "white_background" else (0, 0, 0, 255)
    rgba = image.convert("RGBA")
    background = image_module.new("RGBA", rgba.size, color)
    return image_module.alpha_composite(background, rgba).convert("RGB")


def _has_alpha(image: Any) -> bool:
    mode = getattr(image, "mode", "")
    return mode in {"RGBA", "LA"} or ("transparency" in getattr(image, "info", {}))
