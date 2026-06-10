"""Feature extractor protocols and shared multi-output types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

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
