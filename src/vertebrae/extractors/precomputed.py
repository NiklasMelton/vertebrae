"""Extractor for precomputed embeddings."""

from typing import Any, Dict

import numpy as np


class PrecomputedExtractor:
    def __init__(self, name: str = "precomputed") -> None:
        self.name = name
        self.modality = "embeddings"
        self.extractor_type = "precomputed"

    def fit(self, X: Any, y: Any = None) -> "PrecomputedExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
        }
