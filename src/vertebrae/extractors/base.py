"""Feature extractor protocols and shared multi-output types."""

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np

from vertebrae.cache.fingerprint import hash_json_exact
from vertebrae.extractors._identity import validate_extractor_name
from vertebrae.profiling import (
    AdapterOperationResult,
    DeploymentArtifact,
    DeviceMemoryMeasurement,
    ModelFootprintMeasurement,
    ResourceAdapterMetadata,
)


@dataclass(frozen=True)
class EmbeddingOutputSpec:
    """Declarative description of one named extractor output."""

    name: str
    pooling: Optional[str] = None
    hidden_layer: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_extractor_name(self.name))
        if self.pooling is not None:
            if not isinstance(self.pooling, str) or not self.pooling.strip():
                raise ValueError("EmbeddingOutputSpec.pooling must be a non-empty string.")
            object.__setattr__(self, "pooling", self.pooling.strip())
        object.__setattr__(
            self,
            "hidden_layer",
            normalize_optional_output_integer(
                self.hidden_layer, "EmbeddingOutputSpec.hidden_layer"
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            normalize_output_metadata(self.metadata, "EmbeddingOutputSpec.metadata"),
        )


@dataclass(frozen=True)
class EmbeddingOutput:
    """Materialized embedding output from one extractor pass."""

    name: str
    embeddings: Any
    recipe: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractor(Protocol):
    """Protocol implemented by all vertebrae feature extractors.

    Attributes:
        name: User-facing extractor name.
        modality: Input modality handled by the extractor.
        extractor_type: Extractor family metadata.
    """

    name: str
    modality: str
    extractor_type: str

    def fit(self, X: Any, y: Any = None) -> "FeatureExtractor":
        """Fit extractor state when applicable.

        Args:
            X: Input samples.
            y: Optional labels.

        Returns:
            The fitted extractor.
        """

        ...

    def transform(self, X: Any) -> np.ndarray:
        """Transform inputs into embeddings.

        Args:
            X: Input samples.

        Returns:
            Dense or sparse numeric embedding matrix.
        """

        ...

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the extractor and transform inputs into embeddings.

        Args:
            X: Input samples.
            y: Optional labels.

        Returns:
            Dense or sparse numeric embedding matrix.
        """

        ...

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable extractor recipe.

        Returns:
            JSON-compatible recipe used for cache keys and reports.
        """

        ...


class MultiOutputFeatureExtractor(Protocol):
    """Optional protocol for extractors that can emit multiple embedding matrices."""

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        """Return the named outputs this extractor can materialize."""

        ...

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        """Transform inputs into multiple named embedding outputs."""

        ...


class RetrievalCapableExtractor(Protocol):
    """Optional protocol for independent query/gallery branch encoding."""

    def encode_retrieval(self, X: Any, *, branch: str, modality: str) -> Any:
        """Encode one declared endpoint branch into a numeric embedding matrix."""

        ...


class ResourceProfileAdapter(Protocol):
    """Optional extractor-owned hooks for framework-specific resource data."""

    def metadata(self) -> ResourceAdapterMetadata:
        """Return backend, device, dtype, and synchronization metadata."""

        ...

    def synchronize(self) -> AdapterOperationResult:
        """Synchronize asynchronous device work when supported."""

        ...

    def reset_peak_device_memory(self) -> AdapterOperationResult:
        """Reset allocator peak counters when supported."""

        ...

    def peak_device_memory(self) -> DeviceMemoryMeasurement:
        """Return allocator memory and availability metadata."""

        ...

    def model_footprint(self) -> ModelFootprintMeasurement:
        """Return parameter and buffer counts/bytes when available."""

        ...

    def deployment_artifacts(self) -> Sequence[DeploymentArtifact]:
        """Return explicit local model/checkpoint artifact paths."""

        ...


def normalize_optional_output_integer(value: Any, name: str) -> Optional[int]:
    """Validate an optional signed integer without lossy coercion."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer when provided.")
    return int(value)


def normalize_output_metadata(value: Any, name: str) -> Dict[str, Any]:
    """Copy deterministic, finite metadata without coercing its typed content."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    try:
        copied = deepcopy(dict(value))
        _validate_finite_metadata(copied, name, set())
        hash_json_exact(copied)
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain deterministic, finite, exactly serializable values."
        ) from exc
    return copied


def _validate_finite_metadata(value: Any, path: str, active: set[int]) -> None:
    if isinstance(value, np.generic):
        _validate_finite_metadata(value.item(), path, active)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{path} contains a non-finite Decimal.")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            for array_index in np.ndindex(value.shape):
                _validate_finite_metadata(value[array_index], f"{path}[{array_index}]", active)
        elif np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{path} contains a non-finite array value.")
        return
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - scipy is a core dependency
        sparse = None
    if sparse is not None and sparse.issparse(value):
        _validate_finite_metadata(np.asarray(value.data), f"{path}.data", active)
        return
    recurse = isinstance(value, (Mapping, list, tuple, set, frozenset)) or (
        is_dataclass(value) and not isinstance(value, type)
    )
    identity = id(value)
    if recurse:
        if identity in active:
            raise ValueError(f"{path} contains a cycle.")
        active.add(identity)
    try:
        if isinstance(value, Mapping):
            for key, item in value.items():
                _validate_finite_metadata(key, f"{path}.<key>", active)
                _validate_finite_metadata(item, f"{path}[{key!r}]", active)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item_index, item in enumerate(value):
                _validate_finite_metadata(item, f"{path}[{item_index}]", active)
        elif is_dataclass(value) and not isinstance(value, type):
            for declared in fields(value):
                _validate_finite_metadata(
                    getattr(value, declared.name), f"{path}.{declared.name}", active
                )
        elif hasattr(value, "to_numpy"):
            _validate_finite_metadata(value.to_numpy(), f"{path}.to_numpy()", active)
    finally:
        if recurse:
            active.remove(identity)
