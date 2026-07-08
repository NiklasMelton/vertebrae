"""Optional graph-model extractor for PyG and DGL workflows."""

from typing import Any, Callable, Dict, Optional

from vertebrae.extractors._utils import (
    call_model,
    callable_name,
    maybe_move_to_device,
    tensor_to_numpy,
)
from vertebrae.utils.validation import ensure_numeric_matrix


class GraphModelExtractor:
    """Wrap a graph model that returns one embedding row per graph sample."""

    def __init__(
        self,
        name: str,
        model: Any,
        collate_fn: Callable[[Any], Any],
        output_fn: Optional[Callable[[Any], Any]] = None,
        device: Optional[str] = None,
        framework: Optional[str] = None,
        recipe_data: Optional[Dict[str, Any]] = None,
        allow_sparse: bool = False,
        streaming_safe: bool = True,
        move_batch_to_device: bool = True,
        move_model_to_device: bool = True,
    ) -> None:
        if framework not in {None, "dgl", "pyg"}:
            raise ValueError("framework must be one of: None, 'pyg', 'dgl'.")
        self.name = name
        self.model = model
        self.collate_fn = collate_fn
        self.output_fn = output_fn
        self.device = device
        self.framework = framework
        self.recipe_data = recipe_data or {}
        self.allow_sparse = allow_sparse
        self.streaming_safe = streaming_safe
        self.move_batch_to_device = move_batch_to_device
        self.move_model_to_device = move_model_to_device
        self.modality = "graph"
        self.extractor_type = "graph_model"
        self._torch: Any = None
        self._model_moved = False

    def fit(self, X: Any, y: Any = None) -> "GraphModelExtractor":
        return self

    def transform(self, X: Any) -> Any:
        torch_module = self._load_dependencies()
        self._maybe_move_model(torch_module)
        if hasattr(self.model, "eval"):
            self.model.eval()
        batch = self.collate_fn(X)
        if self.move_batch_to_device:
            batch = maybe_move_to_device(batch, device=self.device, torch_module=torch_module)
        with torch_module.no_grad():
            raw_output = call_model(self.model, batch)
        embeddings = self.output_fn(raw_output) if self.output_fn is not None else raw_output
        embeddings = tensor_to_numpy(embeddings)
        return ensure_numeric_matrix(
            embeddings,
            f"GraphModelExtractor '{self.name}' output",
            allow_sparse=self.allow_sparse,
        )

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X, y).transform(X)

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "framework": self.framework,
            "model_class": self.model.__class__.__module__ + "." + self.model.__class__.__name__,
            "collate_fn": callable_name(self.collate_fn),
            "output_fn": callable_name(self.output_fn) if self.output_fn is not None else None,
            "device": self.device,
            "recipe_data": self.recipe_data,
            "allow_sparse": self.allow_sparse,
            "streaming_safe": self.streaming_safe,
            "move_batch_to_device": self.move_batch_to_device,
            "move_model_to_device": self.move_model_to_device,
        }

    def _load_dependencies(self) -> Any:
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "GraphModelExtractor requires optional PyTorch graph support. "
                    "Install with `poetry install -E graph`."
                ) from exc
            if self.framework == "pyg":
                try:
                    import torch_geometric  # noqa: F401
                except ImportError as exc:
                    raise ImportError(
                        "GraphModelExtractor with framework='pyg' requires the optional "
                        "'graph' extra including torch-geometric."
                    ) from exc
            if self.framework == "dgl":
                try:
                    import dgl  # noqa: F401
                except ImportError as exc:
                    raise ImportError(
                        "GraphModelExtractor with framework='dgl' requires the optional "
                        "'graph' extra including DGL."
                    ) from exc
            self._torch = torch
        return self._torch

    def _maybe_move_model(self, torch_module: Any) -> None:
        if self.device is None or not self.move_model_to_device or self._model_moved:
            return
        if not hasattr(self.model, "to"):
            raise TypeError(
                "GraphModelExtractor received a device but the wrapped model does not expose to()."
            )
        moved = self.model.to(self.device)
        if moved is not None:
            self.model = moved
        self._model_moved = True
