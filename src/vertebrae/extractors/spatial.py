"""Explicit spatial extractor contracts and adapters."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class SpatialLayout:
    """Mapping from a spatial feature tensor to a uniform image grid."""

    grid_height: int
    grid_width: int
    special_tokens: int = 0
    channel_axis: int = -1

    def __post_init__(self) -> None:
        if self.grid_height < 1 or self.grid_width < 1:
            raise ValueError("Spatial grid dimensions must be positive.")
        if self.special_tokens < 0:
            raise ValueError("special_tokens must be >= 0.")


@dataclass(frozen=True)
class SpatialOutputSpec:
    """Declarative description of one spatial extractor output."""

    name: str
    layout: SpatialLayout
    hidden_layer: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotation_transform: Optional[Callable[[Any], Any]] = None


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
    ) -> None:
        self.name = name
        self.transform_fn = transform_fn
        self._output_specs = list(output_specs)
        if not self._output_specs:
            raise ValueError("At least one spatial output spec is required.")
        self.recipe_data = recipe_data or {}
        self.streaming_safe = streaming_safe
        self.modality = "image"
        self.extractor_type = "custom_spatial"

    def fit(self, X: Any, y: Any = None) -> "CallableSpatialExtractor":
        return self

    def spatial_output_specs(self) -> List[SpatialOutputSpec]:
        return list(self._output_specs)

    def transform_spatial(self, X: Any) -> List[SpatialEmbeddingOutput]:
        raw = self.transform_fn(X)
        if isinstance(raw, dict):
            values = raw
        elif len(self._output_specs) == 1:
            values = {self._output_specs[0].name: raw}
        else:
            raise ValueError("Multi-output spatial callables must return a name-to-output mapping.")
        outputs = []
        for spec in self._output_specs:
            if spec.name not in values:
                raise ValueError(f"Missing spatial output {spec.name!r}.")
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
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "transform_fn": _callable_name(self.transform_fn),
            "outputs": [_spec_dict(spec) for spec in self._output_specs],
            "recipe_data": self.recipe_data,
            "streaming_safe": self.streaming_safe,
        }


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


def _per_image_values(value: Any, layout: SpatialLayout) -> List[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim not in {3, 4}:
            raise ValueError("Spatial outputs must be batched token grids or feature maps.")
        return [value[index] for index in range(value.shape[0])]
    return list(value)


def _callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"


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
