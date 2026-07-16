"""Optional Hugging Face vision embedding extractor."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from vertebrae.extractors._identity import (
    cache_identity_fields,
    local_model_paths,
    validate_cache_identity,
    validate_extractor_name,
)
from vertebrae.extractors._utils import (
    coerce_image,
    optional_dependency_versions,
    snapshot_mapping,
    validate_batch_size,
    validate_bool,
    validate_choice,
    validate_nonblank_string,
    validate_optional_nonblank_string,
)
from vertebrae.extractors.base import (
    EmbeddingOutput,
    EmbeddingOutputSpec,
    normalize_optional_output_integer,
)
from vertebrae.extractors.spatial import (
    SpatialEmbeddingOutput,
    SpatialLayout,
    SpatialOutputSpec,
)
from vertebrae.extractors.structured import StructuredEmbeddingOutput, StructuredOutputSpec


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
        outputs: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 16,
        image_mode: str = "auto",
        alpha_mode: str = "drop",
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        spatial_outputs: Optional[List[Dict[str, Any]]] = None,
        structured_outputs: Optional[List[Dict[str, Any]]] = None,
        checkpoint_paths: Optional[List[str]] = None,
        cache_identity: Optional[str] = None,
    ) -> None:
        pooling = validate_choice(pooling, "pooling", {"cls", "mean", "pooler"})
        image_mode = validate_choice(image_mode, "image_mode", _IMAGE_MODES)
        alpha_mode = validate_choice(alpha_mode, "alpha_mode", _ALPHA_MODES)
        batch_size = validate_batch_size(batch_size)
        self.name = validate_extractor_name(name)
        self.model_id = validate_nonblank_string(model_id, "model_id")
        self.processor_id = (
            validate_optional_nonblank_string(processor_id, "processor_id") or self.model_id
        )
        self.pooling = pooling
        self.hidden_layer = hidden_layer
        self._output_specs = _resolve_output_specs(
            outputs=outputs,
            default_pooling=pooling,
            default_hidden_layer=hidden_layer,
        )
        self.batch_size = batch_size
        self.image_mode = image_mode
        self.alpha_mode = alpha_mode
        self.device = validate_optional_nonblank_string(device, "device")
        self.revision = validate_optional_nonblank_string(revision, "revision")
        self.trust_remote_code = validate_bool(trust_remote_code, "trust_remote_code")
        self.processor_kwargs = snapshot_mapping(processor_kwargs, "processor_kwargs")
        self.model_kwargs = snapshot_mapping(model_kwargs, "model_kwargs")
        self._spatial_output_specs = _resolve_spatial_output_specs(spatial_outputs)
        self._structured_output_specs = _resolve_structured_output_specs(structured_outputs)
        self.checkpoint_paths = tuple(checkpoint_paths or ())
        self.modality = "image"
        self.extractor_type = "frozen_pretrained"
        self.streaming_safe = True
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._image_module: Any = None
        self.cache_identity = validate_cache_identity(cache_identity)

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

        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "HFVisionExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return outputs[0].embeddings

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Encode image inputs into dense embeddings.

        Args:
            X: Image inputs.
            y: Optional labels.

        Returns:
            Dense float32 embedding matrix.
        """

        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        processor, model, torch, image_module = self._load_model()
        collected: Dict[str, List[np.ndarray]] = {spec.name: [] for spec in self._output_specs}
        model.eval()
        need_hidden_states = any(spec.hidden_layer is not None for spec in self._output_specs)
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
                    output_hidden_states=need_hidden_states,
                )
                for spec in self._output_specs:
                    pooled = self._pool(model_output, spec)
                    if len(pooled.shape) > 2:
                        pooled = pooled.flatten(start_dim=1)
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
                    recipe=_spec_to_dict(spec),
                    metadata={
                        **dict(spec.metadata),
                        "pooling": spec.pooling,
                        "hidden_layer": spec.hidden_layer,
                    },
                )
            )
        return outputs

    def spatial_output_specs(self) -> List[SpatialOutputSpec]:
        return list(self._spatial_output_specs)

    def transform_spatial(self, X: Any) -> List[SpatialEmbeddingOutput]:
        if not self._spatial_output_specs:
            raise ValueError("HFVisionExtractor was not configured with spatial_outputs.")
        processor, model, torch, image_module = self._load_model()
        collected: Dict[str, List[np.ndarray]] = {
            spec.name: [] for spec in self._spatial_output_specs
        }
        model.eval()
        need_hidden_states = any(
            spec.hidden_layer is not None for spec in self._spatial_output_specs
        )
        with torch.no_grad():
            for items in _iter_chunks(_as_iterable(X), self.batch_size):
                batch = [
                    _coerce_image(item, image_module, self.image_mode, self.alpha_mode)
                    for item in items
                ]
                encoded = processor(images=batch, return_tensors="pt", **self.processor_kwargs)
                encoded = {key: value.to(self._device(torch)) for key, value in encoded.items()}
                model_output = model(**encoded, output_hidden_states=need_hidden_states)
                for spec in self._spatial_output_specs:
                    hidden = (
                        self._select_hidden_state(model_output, spec.hidden_layer)
                        if spec.hidden_layer is not None
                        else model_output.last_hidden_state
                    )
                    values = hidden.detach().cpu().numpy().astype(np.float32, copy=False)
                    collected[spec.name].extend(values[index] for index in range(values.shape[0]))
        return [
            SpatialEmbeddingOutput(
                name=spec.name,
                embeddings=collected[spec.name],
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
        if not self._structured_output_specs:
            raise ValueError("HFVisionExtractor was not configured with structured_outputs.")
        processor, model, torch, image_module = self._load_model()
        collected: Dict[str, List[np.ndarray]] = {
            spec.name: [] for spec in self._structured_output_specs
        }
        model.eval()
        with torch.no_grad():
            for items in _iter_chunks(_as_iterable(X), self.batch_size):
                batch = [
                    _coerce_image(item, image_module, self.image_mode, self.alpha_mode)
                    for item in items
                ]
                encoded = processor(images=batch, return_tensors="pt", **self.processor_kwargs)
                encoded = {key: value.to(self._device(torch)) for key, value in encoded.items()}
                model_output = model(**encoded, output_hidden_states=True)
                for spec in self._structured_output_specs:
                    layer = -1 if spec.hidden_layer is None else spec.hidden_layer
                    hidden = self._select_hidden_state(model_output, layer)
                    values = hidden.detach().cpu().numpy().astype(np.float32, copy=False)
                    special_tokens = int(spec.metadata.get("special_tokens", 1))
                    for index in range(values.shape[0]):
                        collected[spec.name].append(values[index, special_tokens:])
        return [
            StructuredEmbeddingOutput(
                name=spec.name,
                embeddings=collected[spec.name],
                unit_type=spec.unit_type,
                recipe=_structured_spec_to_dict(spec),
                metadata=dict(spec.metadata),
            )
            for spec in self._structured_output_specs
        ]

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable Hugging Face vision recipe.

        Returns:
            JSON-compatible recipe dictionary.
        """

        recipe: Dict[str, Any] = {
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
            "dependency_versions": optional_dependency_versions("Pillow", "torch", "transformers"),
            "streaming_safe": self.streaming_safe,
        }
        recipe["outputs"] = [_spec_to_dict(spec) for spec in self._output_specs]
        if self._spatial_output_specs:
            recipe["spatial_outputs"] = [
                {
                    "name": spec.name,
                    "hidden_layer": spec.hidden_layer,
                    "layout": {
                        "grid_height": spec.layout.grid_height,
                        "grid_width": spec.layout.grid_width,
                        "special_tokens": spec.layout.special_tokens,
                        "channel_axis": spec.layout.channel_axis,
                    },
                    "metadata": dict(spec.metadata),
                    "annotation_transform": (
                        f"{spec.annotation_transform.__module__}."
                        f"{spec.annotation_transform.__qualname__}"
                        if spec.annotation_transform is not None
                        else None
                    ),
                }
                for spec in self._spatial_output_specs
            ]
        if self._structured_output_specs:
            recipe["structured_outputs"] = [
                _structured_spec_to_dict(spec) for spec in self._structured_output_specs
            ]
        identity_callables = [
            (f"annotation_transform:{spec.name}", spec.annotation_transform)
            for spec in self._spatial_output_specs
            if spec.annotation_transform is not None
        ]
        recipe.update(
            cache_identity_fields(
                explicit=self.cache_identity,
                callables=identity_callables,
                paths=local_model_paths(
                    self.model_id,
                    self.checkpoint_paths,
                    additional_identifiers=(self.processor_id,),
                ),
                require_pinned_revision=True,
                revision=self.revision,
                revision_identifiers=(self.model_id, self.processor_id),
            )
        )
        return recipe

    def get_resource_profile_adapter(self) -> Any:
        """Return Torch profiling hooks without forcing model loading."""

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

    def _pool(self, output: Any, spec: EmbeddingOutputSpec) -> Any:
        if spec.hidden_layer is not None:
            if spec.pooling == "pooler":
                raise ValueError("pooler pooling cannot be used with hidden_layer.")
            hidden = self._select_hidden_state(output, spec.hidden_layer)
            if spec.pooling == "cls":
                return hidden[:, 0, :]
            return hidden.mean(dim=1)
        if spec.pooling == "pooler":
            pooler_output = getattr(output, "pooler_output", None)
            if pooler_output is None:
                raise ValueError("pooler pooling requested, but model output has no pooler_output.")
            return pooler_output
        hidden = output.last_hidden_state
        if spec.pooling == "cls":
            return hidden[:, 0, :]
        return hidden.mean(dim=1)

    def _select_hidden_state(self, output: Any, hidden_layer: int) -> Any:
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
        if not isinstance(raw, Mapping):
            raise TypeError("HFVisionExtractor output specs must be mappings.")
        if "name" not in raw:
            raise ValueError("HFVisionExtractor output specs must include a name.")
        pooling = raw.get("pooling", default_pooling)
        if pooling not in {"cls", "mean", "pooler"}:
            raise ValueError("HFVisionExtractor output pooling must be one of: cls, mean, pooler.")
        hidden_layer = raw.get("hidden_layer")
        if pooling == "pooler" and hidden_layer is not None:
            raise ValueError("pooler pooling cannot be used with hidden_layer.")
        specs.append(
            EmbeddingOutputSpec(
                name=raw["name"],
                pooling=pooling,
                hidden_layer=hidden_layer,
                metadata=raw.get("metadata", {}),
            )
        )
    _ensure_unique_names(specs)
    return specs


def _ensure_unique_names(specs: List[EmbeddingOutputSpec]) -> None:
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("HFVisionExtractor output names must be unique.")


def _spec_to_dict(spec: EmbeddingOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "pooling": spec.pooling,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }


def _resolve_spatial_output_specs(
    outputs: Optional[List[Dict[str, Any]]],
) -> List[SpatialOutputSpec]:
    specs = []
    for raw in outputs or []:
        if not isinstance(raw, Mapping):
            raise TypeError("HFVisionExtractor spatial outputs must be mappings.")
        if "name" not in raw or "grid_shape" not in raw:
            raise ValueError("HFVisionExtractor spatial outputs require name and grid_shape.")
        if isinstance(raw["grid_shape"], (str, bytes)):
            raise TypeError("grid_shape must contain [height, width].")
        grid_shape = tuple(raw["grid_shape"])
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must contain [height, width].")
        specs.append(
            SpatialOutputSpec(
                name=raw["name"],
                hidden_layer=raw.get("hidden_layer"),
                layout=SpatialLayout(
                    grid_height=grid_shape[0],
                    grid_width=grid_shape[1],
                    special_tokens=raw.get("special_tokens", 1),
                    channel_axis=raw.get("channel_axis", -1),
                ),
                metadata=raw.get("metadata", {}),
                annotation_transform=raw.get("annotation_transform"),
            )
        )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("HFVisionExtractor spatial output names must be unique.")
    return specs


def _resolve_structured_output_specs(
    outputs: Optional[List[Dict[str, Any]]],
) -> List[StructuredOutputSpec]:
    specs = []
    for raw in outputs or []:
        if not isinstance(raw, Mapping):
            raise TypeError("HFVisionExtractor structured outputs must be mappings.")
        if "name" not in raw:
            raise ValueError("HFVisionExtractor structured outputs must include a name.")
        special_tokens = normalize_optional_output_integer(
            raw.get("special_tokens", 1), "HFVisionExtractor special_tokens"
        )
        assert special_tokens is not None
        if special_tokens < 0:
            raise ValueError("HFVisionExtractor special_tokens must be non-negative.")
        raw_metadata = raw.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("HFVisionExtractor structured output metadata must be a mapping.")
        specs.append(
            StructuredOutputSpec(
                name=raw["name"],
                unit_type=raw.get("unit_type", "region"),
                hidden_layer=raw.get("hidden_layer"),
                metadata={
                    **dict(raw_metadata),
                    "special_tokens": special_tokens,
                },
            )
        )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("HFVisionExtractor structured output names must be unique.")
    return specs


def _structured_spec_to_dict(spec: StructuredOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "unit_type": spec.unit_type,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }


_IMAGE_MODES = {"auto", "rgb", "grayscale", "preserve"}
_ALPHA_MODES = {"drop", "white_background", "black_background"}


def _coerce_image(value: Any, image_module: Any, image_mode: str, alpha_mode: str) -> Any:
    return coerce_image(
        value,
        image_module=image_module,
        image_mode=image_mode,
        alpha_mode=alpha_mode,
    )
