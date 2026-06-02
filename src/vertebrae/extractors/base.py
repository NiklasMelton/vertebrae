"""Feature extractor protocol."""

from typing import Any, Dict, Protocol

import numpy as np


class FeatureExtractor(Protocol):
    name: str
    modality: str
    extractor_type: str

    def fit(self, X: Any, y: Any = None) -> "FeatureExtractor":
        ...

    def transform(self, X: Any) -> np.ndarray:
        ...

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        ...

    def recipe(self) -> Dict[str, Any]:
        ...
