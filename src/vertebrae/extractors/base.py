"""Feature extractor protocol."""

from typing import Any, Dict, Protocol

import numpy as np


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
