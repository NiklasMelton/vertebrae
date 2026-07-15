"""Optional Torch module extractor for local user-supplied models."""

from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from vertebrae.extractors._identity import (
    cache_identity_fields,
    validate_cache_identity,
    validate_extractor_name,
)
from vertebrae.extractors._outputs import validate_named_output_mapping
from vertebrae.extractors._utils import (
    optional_dependency_versions,
    snapshot_mapping,
    validate_bool,
    validate_nonblank_string,
    validate_optional_nonblank_string,
)
from vertebrae.extractors.spatial import (
    SpatialEmbeddingOutput,
    SpatialOutputSpec,
    _per_image_values,
)
from vertebrae.extractors.structured import (
    StructuredEmbeddingOutput,
    StructuredOutputSpec,
    _per_parent_structured_values,
)
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
        spatial_output_fn: Optional[Callable[[Any], Any]] = None,
        spatial_output_specs: Optional[Iterable[SpatialOutputSpec]] = None,
        structured_output_fn: Optional[Callable[[Any], Any]] = None,
        structured_output_specs: Optional[Iterable[StructuredOutputSpec]] = None,
        checkpoint_paths: Optional[Iterable[str]] = None,
        eval_mode: bool = True,
        inference_mode: bool = True,
        restore_model_mode: bool = True,
        cache_identity: Optional[str] = None,
    ) -> None:
        for option_name, option_value in (
            ("allow_sparse", allow_sparse),
            ("streaming_safe", streaming_safe),
            ("move_batch_to_device", move_batch_to_device),
            ("move_model_to_device", move_model_to_device),
            ("eval_mode", eval_mode),
            ("inference_mode", inference_mode),
            ("restore_model_mode", restore_model_mode),
        ):
            validate_bool(option_value, option_name)
        if not callable(collate_fn):
            raise TypeError("collate_fn must be callable.")
        for callback_name, callback_value in (
            ("output_fn", output_fn),
            ("spatial_output_fn", spatial_output_fn),
            ("structured_output_fn", structured_output_fn),
        ):
            if callback_value is not None and not callable(callback_value):
                raise TypeError(f"{callback_name} must be callable when provided.")
        self.name = validate_extractor_name(name)
        self.model = model
        self.collate_fn = collate_fn
        self.output_fn = output_fn
        self.device = validate_optional_nonblank_string(device, "device")
        self.modality = validate_nonblank_string(modality, "modality")
        self.extractor_type = validate_nonblank_string(extractor_type, "extractor_type")
        self.recipe_data = snapshot_mapping(recipe_data, "recipe_data")
        self.allow_sparse = allow_sparse
        self.streaming_safe = streaming_safe
        self.move_batch_to_device = move_batch_to_device
        self.move_model_to_device = move_model_to_device
        self.spatial_output_fn = spatial_output_fn
        self._spatial_output_specs = list(spatial_output_specs or [])
        self.structured_output_fn = structured_output_fn
        self._structured_output_specs = list(structured_output_specs or [])
        for owner, specs in (
            ("spatial", self._spatial_output_specs),
            ("structured", self._structured_output_specs),
        ):
            names = [spec.name for spec in specs]
            if len(names) != len(set(names)):
                raise ValueError(f"TorchExtractor {owner} output names must be unique.")
        if (self.spatial_output_fn is None) != (not self._spatial_output_specs):
            raise ValueError(
                "spatial_output_fn and spatial_output_specs must be provided together."
            )
        if (self.structured_output_fn is None) != (not self._structured_output_specs):
            raise ValueError(
                "structured_output_fn and structured_output_specs must be provided together."
            )
        self._torch: Any = None
        self._model_moved = False
        self.checkpoint_paths = tuple(checkpoint_paths or ())
        self.eval_mode = eval_mode
        self.inference_mode = inference_mode
        self.restore_model_mode = restore_model_mode
        self.cache_identity = validate_cache_identity(cache_identity)

    def get_resource_profile_adapter(self) -> Any:
        """Return Torch-specific synchronization and footprint hooks."""

        from vertebrae.profiling import TorchResourceProfileAdapter

        return TorchResourceProfileAdapter(self, self.checkpoint_paths)

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
        model_output = self._call_model_for_extraction(batch, torch_module)
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

    def spatial_output_specs(self) -> List[SpatialOutputSpec]:
        return list(self._spatial_output_specs)

    def transform_spatial(self, X: Any) -> List[SpatialEmbeddingOutput]:
        if self.spatial_output_fn is None:
            raise ValueError("TorchExtractor was not configured with spatial outputs.")
        torch_module = self._load_torch()
        self._maybe_move_model(torch_module)
        batch = self.collate_fn(X)
        if self.device is not None and self.move_batch_to_device:
            batch = self._move_to_device(batch, torch_module)
        values = self._to_numpy_nested(
            self.spatial_output_fn(self._call_model_for_extraction(batch, torch_module))
        )
        if not isinstance(values, dict) and len(self._spatial_output_specs) == 1:
            values = {self._spatial_output_specs[0].name: values}
        if not isinstance(values, dict):
            raise ValueError("Multi-output Torch spatial adapters must return a mapping.")
        values = validate_named_output_mapping(
            values,
            [spec.name for spec in self._spatial_output_specs],
            "TorchExtractor spatial adapter",
        )
        return [
            SpatialEmbeddingOutput(
                name=spec.name,
                embeddings=_per_image_values(values[spec.name], spec.layout),
                layout=spec.layout,
                recipe={"hidden_layer": spec.hidden_layer},
                metadata=dict(spec.metadata),
                annotation_transform=spec.annotation_transform,
            )
            for spec in self._spatial_output_specs
        ]

    def structured_output_specs(self) -> List[StructuredOutputSpec]:
        return list(self._structured_output_specs)

    def transform_structured(self, X: Any) -> List[StructuredEmbeddingOutput]:
        if self.structured_output_fn is None:
            raise ValueError("TorchExtractor was not configured with structured outputs.")
        torch_module = self._load_torch()
        self._maybe_move_model(torch_module)
        batch = self.collate_fn(X)
        if self.device is not None and self.move_batch_to_device:
            batch = self._move_to_device(batch, torch_module)
        values = self._to_numpy_nested(
            self.structured_output_fn(self._call_model_for_extraction(batch, torch_module))
        )
        if not isinstance(values, dict) and len(self._structured_output_specs) == 1:
            values = {self._structured_output_specs[0].name: values}
        if not isinstance(values, dict):
            raise ValueError("Multi-output Torch structured adapters must return a mapping.")
        values = validate_named_output_mapping(
            values,
            [spec.name for spec in self._structured_output_specs],
            "TorchExtractor structured adapter",
        )
        return [
            StructuredEmbeddingOutput(
                name=spec.name,
                embeddings=_per_parent_structured_values(values[spec.name], spec.unit_type),
                unit_type=spec.unit_type,
                recipe={"hidden_layer": spec.hidden_layer},
                metadata=dict(spec.metadata),
            )
            for spec in self._structured_output_specs
        ]

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable recipe for this extractor."""

        recipe = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_class": self.model.__class__.__module__ + "." + self.model.__class__.__name__,
            "collate_fn": _callable_name(self.collate_fn),
            "output_fn": _callable_name(self.output_fn) if self.output_fn is not None else None,
            "device": self.device,
            "recipe_data": self.recipe_data,
            "allow_sparse": self.allow_sparse,
            "dependency_versions": optional_dependency_versions("torch"),
            "streaming_safe": self.streaming_safe,
            "move_batch_to_device": self.move_batch_to_device,
            "move_model_to_device": self.move_model_to_device,
            "eval_mode": self.eval_mode,
            "inference_mode": self.inference_mode,
            "restore_model_mode": self.restore_model_mode,
            "spatial_output_fn": (
                _callable_name(self.spatial_output_fn)
                if self.spatial_output_fn is not None
                else None
            ),
            "spatial_outputs": [
                {
                    "name": spec.name,
                    "layout": spec.layout.__dict__,
                    "hidden_layer": spec.hidden_layer,
                    "metadata": spec.metadata,
                }
                for spec in self._spatial_output_specs
            ],
            "structured_output_fn": (
                _callable_name(self.structured_output_fn)
                if self.structured_output_fn is not None
                else None
            ),
            "structured_outputs": [
                {
                    "name": spec.name,
                    "unit_type": spec.unit_type,
                    "hidden_layer": spec.hidden_layer,
                    "metadata": spec.metadata,
                }
                for spec in self._structured_output_specs
            ],
        }
        identity_callables = [
            ("collate_fn", self.collate_fn),
            ("output_fn", self.output_fn),
            ("spatial_output_fn", self.spatial_output_fn),
            ("structured_output_fn", self.structured_output_fn),
        ]
        identity_callables.extend(
            (f"annotation_transform:{spec.name}", spec.annotation_transform)
            for spec in self._spatial_output_specs
            if spec.annotation_transform is not None
        )
        recipe.update(
            cache_identity_fields(
                explicit=self.cache_identity,
                callables=identity_callables,
                paths=self.checkpoint_paths,
                state_required=True,
                paths_authoritative=False,
            )
        )
        return recipe

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

    def _call_model_for_extraction(self, batch: Any, torch_module: Any) -> Any:
        evaluator = None
        training_states: List[Tuple[Any, bool]] = []
        if self.eval_mode:
            evaluator = getattr(self.model, "eval", None)
            if not callable(evaluator):
                raise TypeError("TorchExtractor eval_mode=True requires model.eval().")
            if self.restore_model_mode:
                training_states = self._snapshot_training_states()
        try:
            if evaluator is not None:
                evaluator()
            if self.inference_mode:
                inference_mode = getattr(torch_module, "inference_mode", None)
                if not callable(inference_mode):
                    raise ImportError(
                        "TorchExtractor inference_mode=True requires torch.inference_mode(); "
                        "install PyTorch >= 1.9 or set inference_mode=False."
                    )
                context = inference_mode()
            else:
                context = nullcontext()
            with context:
                return self._call_model(batch)
        finally:
            self._restore_training_states(training_states)

    def _snapshot_training_states(self) -> List[Tuple[Any, bool]]:
        """Capture every declared module mode before ``eval()`` mutates the tree."""

        modules_fn = getattr(self.model, "modules", None)
        declared_modules = list(modules_fn()) if callable(modules_fn) else []
        modules = [self.model, *declared_modules]
        states: List[Tuple[Any, bool]] = []
        seen: set[int] = set()
        for module in modules:
            identifier = id(module)
            if identifier in seen:
                continue
            seen.add(identifier)
            training = getattr(module, "training", None)
            if not isinstance(training, bool):
                continue
            trainer = getattr(module, "train", None)
            if not callable(trainer):
                raise TypeError(
                    "TorchExtractor restore_model_mode=True requires every module with a "
                    "training state to expose train()."
                )
            states.append((module, training))
        return states

    @staticmethod
    def _restore_training_states(states: List[Tuple[Any, bool]]) -> None:
        """Restore modes top-down so child-specific states override parent recursion."""

        for module, training in states:
            module.train(training)

    def _tensor_to_numpy(self, value: Any) -> Any:
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.detach().cpu().numpy()
        return value

    def _to_numpy_nested(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._to_numpy_nested(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._to_numpy_nested(item) for item in value)
        if isinstance(value, list):
            return [self._to_numpy_nested(item) for item in value]
        return self._tensor_to_numpy(value)


def _callable_name(fn: Callable[..., Any]) -> str:
    value_type = type(fn)
    module = getattr(fn, "__module__", value_type.__module__)
    qualname = getattr(fn, "__qualname__", value_type.__qualname__)
    return f"{module}.{qualname}"
