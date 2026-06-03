"""Scikit-learn transformer and pipeline extractors."""

from typing import Any, Dict

import numpy as np

from vertebrae.utils.validation import (
    ensure_dense_numeric_2d,
    ensure_sparse_numeric_2d,
    estimate_dense_nbytes,
)


class SklearnExtractor:
    """Wrap a scikit-learn transformer or pipeline as an extractor.

    Args:
        name: User-facing extractor name.
        pipeline: Object exposing `fit`, `transform`, or `fit_transform`.
        already_fitted: Whether to skip fitting and call only `transform`.
        extractor_type: Extractor family metadata.
        max_dense_bytes: Maximum allowed sparse-to-dense conversion size.
        allow_sparse: Whether sparse pipeline outputs should be preserved.
    """

    def __init__(
        self,
        name: str,
        pipeline: Any,
        already_fitted: bool = False,
        extractor_type: str = "unsupervised_fitted",
        max_dense_bytes: int = 2_000_000_000,
        allow_sparse: bool = False,
    ) -> None:
        self.name = name
        self.pipeline = pipeline
        self.already_fitted = already_fitted
        self.extractor_type = extractor_type
        self.max_dense_bytes = max_dense_bytes
        self.allow_sparse = allow_sparse
        self.modality = "unknown"

    def fit(self, X: Any, y: Any = None) -> "SklearnExtractor":
        """Fit the wrapped scikit-learn object when needed.

        Args:
            X: Input samples.
            y: Optional labels.

        Returns:
            This extractor.
        """

        if not self.already_fitted:
            if not hasattr(self.pipeline, "fit"):
                raise TypeError(
                    "pipeline must expose fit or use already_fitted=True with transform."
                )
            self.pipeline.fit(X, y)
            self.already_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform inputs with the wrapped scikit-learn object.

        Args:
            X: Input samples.

        Returns:
            Dense or sparse numeric embedding matrix.
        """

        if not hasattr(self.pipeline, "transform"):
            raise TypeError("pipeline must expose transform.")
        return self._prepare_output(self.pipeline.transform(X))

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the pipeline if needed and transform inputs.

        Args:
            X: Input samples.
            y: Optional labels.

        Returns:
            Dense or sparse numeric embedding matrix.
        """

        if self.already_fitted:
            return self.transform(X)
        if hasattr(self.pipeline, "fit_transform"):
            output = self.pipeline.fit_transform(X, y)
            self.already_fitted = True
            return self._prepare_output(output)
        self.fit(X, y)
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable scikit-learn extractor recipe.

        Returns:
            JSON-compatible recipe dictionary.
        """

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
            "allow_sparse": self.allow_sparse,
            "params": params,
        }

    def _prepare_output(self, output: Any) -> np.ndarray:
        if hasattr(output, "toarray"):
            if self.allow_sparse:
                return ensure_sparse_numeric_2d(output, f"SklearnExtractor '{self.name}' output")
            dense_bytes = estimate_dense_nbytes(output)
            if dense_bytes > self.max_dense_bytes:
                raise ValueError(
                    "Sparse extractor output would require "
                    f"{dense_bytes} bytes when densified, exceeding max_dense_bytes="
                    f"{self.max_dense_bytes}."
                )
            output = output.toarray()
        return ensure_dense_numeric_2d(output, f"SklearnExtractor '{self.name}' output")


def _is_simple_param(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), tuple))
