"""Custom callable feature extractor."""

from typing import Any, Callable, Dict, Optional

import numpy as np


class CallableExtractor:
    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Any], Any],
        fit_fn: Optional[Callable[[Any, Any], Any]] = None,
        modality: str = "unknown",
        extractor_type: str = "custom_callable",
    ) -> None:
        self.name = name
        self.transform_fn = transform_fn
        self.fit_fn = fit_fn
        self.modality = modality
        self.extractor_type = extractor_type

    def fit(self, X: Any, y: Any = None) -> "CallableExtractor":
        if self.fit_fn is not None:
            self.fit_fn(X, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(self.transform_fn(X))

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "transform_fn": _callable_name(self.transform_fn),
            "fit_fn": _callable_name(self.fit_fn) if self.fit_fn is not None else None,
        }


def _callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"
