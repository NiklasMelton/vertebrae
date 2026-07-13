"""Feature extractor protocols and shared multi-output types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class EmbeddingOutputSpec:
    """Declarative description of one named extractor output."""

    name: str
    pooling: Optional[str] = None
    hidden_layer: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    def metadata(self) -> Dict[str, Any]:
        """Return backend, device, precision, and synchronization metadata."""

        ...

    def synchronize(self) -> bool:
        """Synchronize asynchronous device work and return whether it succeeded."""

        ...

    def reset_peak_device_memory(self) -> bool:
        """Reset allocator peak counters when supported."""

        ...

    def peak_device_memory(self) -> Dict[str, Any]:
        """Return peak allocated/reserved device bytes and availability metadata."""

        ...

    def model_footprint(self) -> Dict[str, Any]:
        """Return parameter and buffer counts/bytes when available."""

        ...

    def deployment_artifacts(self) -> Sequence[str]:
        """Return explicit local model/checkpoint artifact paths."""

        ...
