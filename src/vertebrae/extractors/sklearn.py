"""Scikit-learn transformer and pipeline extractors."""

from typing import Any, Dict

import numpy as np


class SklearnExtractor:
    def __init__(
        self,
        name: str,
        pipeline: Any,
        already_fitted: bool = False,
        extractor_type: str = "unsupervised_fitted",
        max_dense_bytes: int = 2_000_000_000,
    ) -> None:
        self.name = name
        self.pipeline = pipeline
        self.already_fitted = already_fitted
        self.extractor_type = extractor_type
        self.max_dense_bytes = max_dense_bytes
        self.modality = "unknown"

    def fit(self, X: Any, y: Any = None) -> "SklearnExtractor":
        if not self.already_fitted:
            if not hasattr(self.pipeline, "fit"):
                raise TypeError(
                    "pipeline must expose fit or use already_fitted=True with transform."
                )
            self.pipeline.fit(X, y)
            self.already_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self.pipeline, "transform"):
            raise TypeError("pipeline must expose transform.")
        return self._to_dense_array(self.pipeline.transform(X))

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        if self.already_fitted:
            return self.transform(X)
        if hasattr(self.pipeline, "fit_transform"):
            output = self.pipeline.fit_transform(X, y)
            self.already_fitted = True
            return self._to_dense_array(output)
        self.fit(X, y)
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if hasattr(self.pipeline, "get_params"):
            params = {
                key: repr(value)
                for key, value in self.pipeline.get_params(deep=True).items()
                if _is_simple_param(value)
            }
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "pipeline_class": (
                self.pipeline.__class__.__module__ + "." + self.pipeline.__class__.__name__
            ),
            "already_fitted": self.already_fitted,
            "max_dense_bytes": self.max_dense_bytes,
            "params": params,
        }

    def _to_dense_array(self, output: Any) -> np.ndarray:
        if hasattr(output, "toarray"):
            dtype = getattr(output, "dtype", np.dtype(float))
            itemsize = np.dtype(dtype).itemsize
            rows, cols = output.shape
            dense_bytes = int(rows) * int(cols) * int(itemsize)
            if dense_bytes > self.max_dense_bytes:
                raise ValueError(
                    "Sparse extractor output would require "
                    f"{dense_bytes} bytes when densified, exceeding max_dense_bytes="
                    f"{self.max_dense_bytes}."
                )
            output = output.toarray()
        return np.asarray(output)


def _is_simple_param(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), tuple))
