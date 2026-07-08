"""Explicit structured-output extractor contracts and adapters."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from vertebrae.utils.validation import ensure_numeric_matrix


@dataclass(frozen=True)
class StructuredOutputSpec:
    """Declarative description of one structured extractor output."""

    name: str
    unit_type: str
    hidden_layer: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredEmbeddingOutput:
    """Per-parent structured unit embeddings from one named output."""

    name: str
    embeddings: List[Any]
    unit_type: str
    recipe: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CallableStructuredExtractor:
    """Wrap a callable returning explicit structured outputs."""

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Any], Any],
        output_specs: Iterable[StructuredOutputSpec],
        recipe_data: Optional[Dict[str, Any]] = None,
        streaming_safe: bool = True,
        modality: str = "unknown",
        extractor_type: str = "custom_structured",
    ) -> None:
        self.name = name
        self.transform_fn = transform_fn
        self._output_specs = list(output_specs)
        if not self._output_specs:
            raise ValueError("At least one structured output spec is required.")
        _ensure_unique_names(self._output_specs)
        self.recipe_data = recipe_data or {}
        self.streaming_safe = streaming_safe
        self.modality = modality
        self.extractor_type = extractor_type

    def fit(self, X: Any, y: Any = None) -> "CallableStructuredExtractor":
        return self

    def structured_output_specs(self) -> List[StructuredOutputSpec]:
        return list(self._output_specs)

    def transform_structured(self, X: Any) -> List[StructuredEmbeddingOutput]:
        raw = self.transform_fn(X)
        if isinstance(raw, dict):
            values = raw
        elif len(self._output_specs) == 1:
            values = {self._output_specs[0].name: raw}
        else:
            raise ValueError(
                "Multi-output structured callables must return a name-to-output mapping."
            )
        outputs = []
        for spec in self._output_specs:
            if spec.name not in values:
                raise ValueError(f"Missing structured output {spec.name!r}.")
            embeddings = _per_parent_structured_values(values[spec.name], spec.unit_type)
            outputs.append(
                StructuredEmbeddingOutput(
                    name=spec.name,
                    embeddings=embeddings,
                    unit_type=spec.unit_type,
                    recipe={"hidden_layer": spec.hidden_layer},
                    metadata=dict(spec.metadata),
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


class PrecomputedStructuredExtractor(CallableStructuredExtractor):
    """Structured extractor for per-parent matrices already stored in dataset inputs."""

    def __init__(
        self,
        output_specs: Iterable[StructuredOutputSpec],
        name: str = "precomputed_structured",
        modality: str = "unknown",
    ) -> None:
        super().__init__(
            name=name,
            transform_fn=lambda value: value,
            output_specs=output_specs,
            streaming_safe=True,
            modality=modality,
            extractor_type="precomputed_structured",
        )


def _per_parent_structured_values(value: Any, unit_type: str) -> List[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 3:
            return [
                ensure_numeric_matrix(value[index], f"{unit_type} embeddings", allow_sparse=True)
                for index in range(value.shape[0])
            ]
        if value.ndim == 2:
            return [ensure_numeric_matrix(value, f"{unit_type} embeddings", allow_sparse=True)]
        if value.ndim == 1 and value.dtype == object:
            return [
                ensure_numeric_matrix(item, f"{unit_type} embeddings", allow_sparse=True)
                for item in value.tolist()
            ]
        raise ValueError(
            "Structured outputs must be a batched 3D array or a sequence of per-parent 2D arrays."
        )
    values = list(value)
    return [
        ensure_numeric_matrix(item, f"{unit_type} embeddings", allow_sparse=True)
        for item in values
    ]


def _ensure_unique_names(specs: List[StructuredOutputSpec]) -> None:
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Structured output names must be unique.")


def _callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"


def _spec_dict(spec: StructuredOutputSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "unit_type": spec.unit_type,
        "hidden_layer": spec.hidden_layer,
        "metadata": dict(spec.metadata),
    }
