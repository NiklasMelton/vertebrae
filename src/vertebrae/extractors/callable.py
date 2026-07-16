"""Custom callable feature extractor."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from vertebrae.extractors._identity import (
    cache_identity_fields,
    validate_cache_identity,
    validate_extractor_name,
)
from vertebrae.utils.validation import ensure_numeric_matrix


class CallableExtractor:
    """Wrap a custom Python callable as a feature extractor.

    Args:
        name: User-facing extractor name.
        transform_fn: Callable that converts inputs into dense or sparse embeddings.
        fit_fn: Optional callable invoked during `fit`.
        modality: Input modality metadata.
        extractor_type: Extractor family metadata.
        recipe_data: Extra serializable data to include in `recipe()`.
        allow_sparse: Whether sparse transform outputs are allowed.
        streaming_safe: Whether independent batches can be transformed without
            access to the full dataset.
    """

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Any], Any],
        fit_fn: Optional[Callable[[Any, Any], Any]] = None,
        modality: str = "unknown",
        extractor_type: str = "custom_callable",
        recipe_data: Optional[Dict[str, Any]] = None,
        allow_sparse: bool = True,
        streaming_safe: bool = False,
        resource_profile_adapter: Optional[Any] = None,
        cache_identity: Optional[str] = None,
    ) -> None:
        self.name = validate_extractor_name(name)
        self.transform_fn = transform_fn
        self.fit_fn = fit_fn
        self.modality = modality
        self.extractor_type = extractor_type
        self.recipe_data = recipe_data or {}
        self.allow_sparse = allow_sparse
        self.streaming_safe = streaming_safe
        self._resource_profile_adapter = resource_profile_adapter
        self.cache_identity = validate_cache_identity(cache_identity)

    def fit(self, X: Any, y: Any = None) -> "CallableExtractor":
        """Fit the callable extractor when a fit function is supplied.

        Args:
            X: Input samples.
            y: Optional labels.

        Returns:
            This extractor.
        """

        if self.fit_fn is not None:
            self.fit_fn(X, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform inputs with `transform_fn` and validate embeddings.

        Args:
            X: Input samples.

        Returns:
            Dense or sparse numeric embedding matrix.
        """

        return ensure_numeric_matrix(
            self.transform_fn(X),
            f"CallableExtractor '{self.name}' output",
            allow_sparse=self.allow_sparse,
        )

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the extractor and transform inputs.

        Args:
            X: Input samples.
            y: Optional labels.

        Returns:
            Dense or sparse numeric embedding matrix.
        """

        self.fit(X, y)
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        """Return a serializable callable extractor recipe.

        Returns:
            JSON-compatible recipe dictionary.
        """

        recipe = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "transform_fn": _callable_name(self.transform_fn),
            "fit_fn": _callable_name(self.fit_fn) if self.fit_fn is not None else None,
            "recipe_data": self.recipe_data,
            "allow_sparse": self.allow_sparse,
            "streaming_safe": self.streaming_safe,
        }
        recipe.update(
            cache_identity_fields(
                explicit=self.cache_identity,
                callables=(("transform_fn", self.transform_fn), ("fit_fn", self.fit_fn)),
                state_required=self.fit_fn is not None,
            )
        )
        return recipe

    def get_resource_profile_adapter(self) -> Any:
        """Return the optional user-supplied resource profiling adapter."""

        return self._resource_profile_adapter


def _callable_name(fn: Callable[..., Any]) -> str:
    value_type = type(fn)
    module = getattr(fn, "__module__", value_type.__module__)
    qualname = getattr(fn, "__qualname__", value_type.__qualname__)
    return f"{module}.{qualname}"
