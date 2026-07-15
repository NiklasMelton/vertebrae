"""Explicit spatial extractor contracts and adapters."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from vertebrae.extractors._identity import (
    cache_identity_fields,
    validate_cache_identity,
    validate_extractor_name,
)
from vertebrae.extractors._outputs import validate_named_output_mapping
from vertebrae.extractors.base import (
    normalize_optional_output_integer,
    normalize_output_metadata,
)


@dataclass(frozen=True)
class SpatialLayout:
    """Mapping from a spatial feature tensor to a uniform image grid."""

    grid_height: int
    grid_width: int
    special_tokens: int = 0
    channel_axis: int = -1

    def __post_init__(self) -> None:
        grid_height = _exact_layout_integer(self.grid_height, "grid_height")
        grid_width = _exact_layout_integer(self.grid_width, "grid_width")
        special_tokens = _exact_layout_integer(self.special_tokens, "special_tokens")
        channel_axis = _exact_layout_integer(self.channel_axis, "channel_axis")
        if grid_height < 1 or grid_width < 1:
            raise ValueError("Spatial grid dimensions must be positive.")
        if special_tokens < 0:
            raise ValueError("special_tokens must be >= 0.")
        object.__setattr__(self, "grid_height", grid_height)
        object.__setattr__(self, "grid_width", grid_width)
        object.__setattr__(self, "special_tokens", special_tokens)
        object.__setattr__(self, "channel_axis", channel_axis)


@dataclass(frozen=True)
class SpatialOutputSpec:
    """Declarative description of one spatial extractor output."""

    name: str
    layout: SpatialLayout
    hidden_layer: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotation_transform: Optional[Callable[[Any], Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_extractor_name(self.name))
        if not isinstance(self.layout, SpatialLayout):
            raise TypeError("SpatialOutputSpec.layout must be a SpatialLayout.")
        object.__setattr__(
            self,
            "hidden_layer",
            normalize_optional_output_integer(self.hidden_layer, "SpatialOutputSpec.hidden_layer"),
        )
        object.__setattr__(
            self,
            "metadata",
            normalize_output_metadata(self.metadata, "SpatialOutputSpec.metadata"),
        )
        if self.annotation_transform is not None and not callable(self.annotation_transform):
            raise TypeError("SpatialOutputSpec.annotation_transform must be callable.")


@dataclass(frozen=True)
class SpatialEmbeddingOutput:
    """Per-image spatial feature tensors from one named output."""

    name: str
    embeddings: List[Any]
    layout: SpatialLayout
    recipe: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotation_transform: Optional[Callable[[Any], Any]] = None


class CallableSpatialExtractor:
    """Wrap a callable returning explicit spatial outputs."""

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Any], Any],
        output_specs: Iterable[SpatialOutputSpec],
        recipe_data: Optional[Dict[str, Any]] = None,
        streaming_safe: bool = True,
        cache_identity: Optional[str] = None,
    ) -> None:
        self.name = validate_extractor_name(name)
        self.transform_fn = transform_fn
        self._output_specs = list(output_specs)
        if not self._output_specs:
            raise ValueError("At least one spatial output spec is required.")
        names = [spec.name for spec in self._output_specs]
        if len(names) != len(set(names)):
            raise ValueError("Spatial output names must be unique.")
        self.recipe_data = recipe_data or {}
        self.streaming_safe = streaming_safe
        self.modality = "image"
        self.extractor_type = "custom_spatial"
        self.cache_identity = validate_cache_identity(cache_identity)
        self._precomputed = False

    def fit(self, X: Any, y: Any = None) -> "CallableSpatialExtractor":
        return self

    def spatial_output_specs(self) -> List[SpatialOutputSpec]:
        return list(self._output_specs)

    def transform_spatial(self, X: Any) -> List[SpatialEmbeddingOutput]:
        raw = self.transform_fn(X)
        if isinstance(raw, dict):
            values = validate_named_output_mapping(
                raw,
                [spec.name for spec in self._output_specs],
                "CallableSpatialExtractor",
            )
        elif len(self._output_specs) == 1:
            values = {self._output_specs[0].name: raw}
        else:
            raise ValueError("Multi-output spatial callables must return a name-to-output mapping.")
        outputs = []
        for spec in self._output_specs:
            embeddings = _per_image_values(values[spec.name], spec.layout)
            outputs.append(
                SpatialEmbeddingOutput(
                    name=spec.name,
                    embeddings=embeddings,
                    layout=spec.layout,
                    recipe={"hidden_layer": spec.hidden_layer},
                    metadata=dict(spec.metadata),
                    annotation_transform=spec.annotation_transform,
                )
            )
        return outputs

    def recipe(self) -> Dict[str, Any]:
        recipe = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "transform_fn": _callable_name(self.transform_fn),
            "outputs": [_spec_dict(spec) for spec in self._output_specs],
            "recipe_data": self.recipe_data,
            "streaming_safe": self.streaming_safe,
        }
        callables = [
            (f"annotation_transform:{spec.name}", spec.annotation_transform)
            for spec in self._output_specs
            if spec.annotation_transform is not None
        ]
        if not self._precomputed:
            callables.insert(0, ("transform_fn", self.transform_fn))
        recipe.update(cache_identity_fields(explicit=self.cache_identity, callables=callables))
        return recipe


class PrecomputedSpatialExtractor(CallableSpatialExtractor):
    """Spatial extractor for tensors already stored in the dataset inputs."""

    def __init__(
        self,
        output_specs: Iterable[SpatialOutputSpec],
        name: str = "precomputed_spatial",
    ) -> None:
        super().__init__(
            name=name,
            transform_fn=lambda value: value,
            output_specs=output_specs,
            streaming_safe=True,
        )
        self.extractor_type = "precomputed_spatial"
        self._precomputed = True


def _per_image_values(value: Any, layout: SpatialLayout) -> List[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim not in {3, 4}:
            raise ValueError("Spatial outputs must be batched token grids or feature maps.")
        return [value[index] for index in range(value.shape[0])]
    return list(value)


def _callable_name(fn: Callable[..., Any]) -> str:
    value_type = type(fn)
    module = getattr(fn, "__module__", value_type.__module__)
    qualname = getattr(fn, "__qualname__", value_type.__qualname__)
    return f"{module}.{qualname}"


def _spec_dict(spec: SpatialOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "layout": {
            "grid_height": spec.layout.grid_height,
            "grid_width": spec.layout.grid_width,
            "special_tokens": spec.layout.special_tokens,
            "channel_axis": spec.layout.channel_axis,
        },
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
        "annotation_transform": (
            _callable_name(spec.annotation_transform)
            if spec.annotation_transform is not None
            else None
        ),
    }


def _exact_layout_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"SpatialLayout.{name} must be an integer.")
    return int(value)
