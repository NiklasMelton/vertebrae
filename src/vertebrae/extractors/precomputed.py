"""Extractor for precomputed embeddings."""

from typing import Any, Dict

import numpy as np

from vertebrae.utils.validation import ensure_numeric_matrix


class PrecomputedExtractor:
    """Extractor for already-computed dense or sparse embeddings.

    Args:
        name: User-facing extractor name.
    """

    def __init__(self, name: str = "precomputed") -> None:
        self.name = name
        self.modality = "embeddings"
        self.extractor_type = "precomputed"
        self.streaming_safe = True

    def fit(self, X: Any, y: Any = None) -> "PrecomputedExtractor":
        """No-op fit for precomputed embeddings.

        Args:
            X: Embedding matrix.
            y: Optional labels.

        Returns:
            This extractor.
        """

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Validate and return precomputed embeddings.

        Args:
            X: Dense or sparse embedding matrix.

        Returns:
            Validated dense or sparse numeric embedding matrix.
        """

        return ensure_numeric_matrix(X, f"PrecomputedExtractor '{self.name}' output")

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Return validated precomputed embeddings.

        Args:
            X: Dense or sparse embedding matrix.
            y: Optional labels.

        Returns:
            Validated dense or sparse numeric embedding matrix.
        """

        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable recipe for this extractor.

        Returns:
            JSON-compatible recipe dictionary.
        """

        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "streaming_safe": self.streaming_safe,
        }
