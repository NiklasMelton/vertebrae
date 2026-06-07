"""Optional Torch module extractor for local user-supplied models."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from vertebrae.utils.validation import ensure_numeric_matrix


class TorchExtractor:
    """Wrap a locally loaded PyTorch module as a feature extractor.

    Args:
        name: User-facing extractor name.
        model: A locally loaded ``torch.nn.Module`` or compatible callable object.
        collate_fn: Callable that converts a batch of raw inputs into model inputs.
        output_fn: Optional callable that converts raw model output into embeddings.
        device: Optional device string passed to ``model.to(...)`` and batch items.
        modality: Input modality metadata.
        extractor_type: Extractor family metadata.
        recipe_data: Extra serializable metadata for reproducibility.
        allow_sparse: Whether sparse embedding outputs are accepted.
        streaming_safe: Whether independent batches can be embedded without full-context state.
        move_batch_to_device: Whether tensor-like batch items should be moved to ``device``.
        move_model_to_device: Whether the model should be moved to ``device`` once on first use.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        collate_fn: Callable[[Any], Any],
        output_fn: Optional[Callable[[Any], Any]] = None,
        device: Optional[str] = None,
        modality: str = "unknown",
        extractor_type: str = "custom_torch",
        recipe_data: Optional[Dict[str, Any]] = None,
        allow_sparse: bool = False,
        streaming_safe: bool = True,
        move_batch_to_device: bool = True,
        move_model_to_device: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.collate_fn = collate_fn
        self.output_fn = output_fn
        self.device = device
        self.modality = modality
        self.extractor_type = extractor_type
        self.recipe_data = recipe_data or {}
        self.allow_sparse = allow_sparse
        self.streaming_safe = streaming_safe
        self.move_batch_to_device = move_batch_to_device
        self.move_model_to_device = move_model_to_device
        self._torch: Any = None
        self._model_moved = False

    def fit(self, X: Any, y: Any = None) -> "TorchExtractor":
        """No-op fit for local Torch models."""

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Apply the collate function, run the model, and validate embeddings."""

        torch_module = self._load_torch()
        self._maybe_move_model(torch_module)
        batch = self.collate_fn(X)
        if self.device is not None and self.move_batch_to_device:
            batch = self._move_to_device(batch, torch_module)
        model_output = self._call_model(batch)
        embeddings = self.output_fn(model_output) if self.output_fn is not None else model_output
        embeddings = self._tensor_to_numpy(embeddings)
        return ensure_numeric_matrix(
            embeddings,
            f"TorchExtractor '{self.name}' output",
            allow_sparse=self.allow_sparse,
        )

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the extractor and transform inputs."""

        return self.fit(X, y).transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable recipe for this extractor."""

        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_class": self.model.__class__.__module__ + "." + self.model.__class__.__name__,
            "collate_fn": _callable_name(self.collate_fn),
            "output_fn": _callable_name(self.output_fn) if self.output_fn is not None else None,
            "device": self.device,
            "recipe_data": self.recipe_data,
            "allow_sparse": self.allow_sparse,
            "streaming_safe": self.streaming_safe,
            "move_batch_to_device": self.move_batch_to_device,
            "move_model_to_device": self.move_model_to_device,
        }

    def _load_torch(self) -> Any:
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "TorchExtractor requires optional PyTorch support. Install with "
                    "`poetry install -E torch` or `poetry install -E hf`."
                ) from exc
            self._torch = torch
        return self._torch

    def _maybe_move_model(self, torch_module: Any) -> None:
        if self.device is None or not self.move_model_to_device or self._model_moved:
            return
        if not hasattr(self.model, "to"):
            raise TypeError(
                "TorchExtractor received a device but the wrapped model does not expose to()."
            )
        moved_model = self.model.to(self.device)
        if moved_model is not None:
            self.model = moved_model
        self._model_moved = True

    def _move_to_device(self, value: Any, torch_module: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._move_to_device(item, torch_module) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item, torch_module) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item, torch_module) for item in value]
        tensor_type = getattr(torch_module, "Tensor", None)
        if tensor_type is not None and isinstance(value, tensor_type):
            moved = value.to(self.device)
            return value if moved is None else moved
        if hasattr(value, "to") and callable(value.to):
            moved = value.to(self.device)
            return value if moved is None else moved
        return value

    def _call_model(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return self.model(**batch)
        if isinstance(batch, tuple):
            return self.model(*batch)
        return self.model(batch)

    def _tensor_to_numpy(self, value: Any) -> Any:
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.detach().cpu().numpy()
        return value


def _callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"
