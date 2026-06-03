"""Optional Hugging Face vision embedding extractor."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class HFVisionExtractor:
    """Hugging Face vision backbone extractor with explicit pooling.

    Args:
        name: User-facing extractor name.
        model_id: Hugging Face model identifier or local path.
        pooling: Pooling mode: `"cls"`, `"mean"`, or `"pooler"`.
        batch_size: Number of images encoded per batch.
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
        pooling: str = "cls",
        batch_size: int = 16,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if pooling not in {"cls", "mean", "pooler"}:
            raise ValueError("pooling must be one of: cls, mean, pooler.")
        self.name = name
        self.model_id = model_id
        self.pooling = pooling
        self.batch_size = batch_size
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
                batch = [_coerce_image(item, image_module) for item in items]
                encoded = processor(images=batch, return_tensors="pt", **self.processor_kwargs)
                encoded = {key: value.to(self._device(torch)) for key, value in encoded.items()}
                model_output = model(**encoded)
                pooled = self._pool(model_output)
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
            "pooling": self.pooling,
            "batch_size": self.batch_size,
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
                self.model_id,
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
        if self.pooling == "pooler":
            pooler_output = getattr(output, "pooler_output", None)
            if pooler_output is None:
                raise ValueError("pooler pooling requested, but model output has no pooler_output.")
            return pooler_output
        hidden = output.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0, :]
        return hidden.mean(dim=1)


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


def _coerce_image(value: Any, image_module: Any) -> Any:
    if isinstance(value, (str, Path)):
        return image_module.open(value).convert("RGB")
    if isinstance(value, np.ndarray):
        return image_module.fromarray(value)
    return value
