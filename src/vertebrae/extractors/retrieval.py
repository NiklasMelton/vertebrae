"""Explicit adapter for independent query and gallery embedding branches."""

from typing import Any, Callable, Dict, Optional

from vertebrae.extractors._identity import (
    cache_identity_fields,
    validate_cache_identity,
    validate_extractor_name,
)
from vertebrae.utils.validation import ensure_numeric_matrix


class CallableRetrievalExtractor:
    """Wrap explicit frozen query/gallery encoding functions for retrieval."""

    def __init__(
        self,
        name: str,
        query_fn: Callable[[Any], Any],
        gallery_fn: Optional[Callable[[Any], Any]] = None,
        *,
        query_modality: str = "embeddings",
        gallery_modality: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cache_identity: Optional[str] = None,
    ) -> None:
        if not callable(query_fn) or not callable(gallery_fn or query_fn):
            raise TypeError("query_fn and gallery_fn must be callable.")
        self.name = validate_extractor_name(name)
        self.query_fn = query_fn
        self.gallery_fn = gallery_fn or query_fn
        self.query_modality = query_modality
        self.gallery_modality = gallery_modality or query_modality
        self.metadata = dict(metadata or {})
        self.cache_identity = validate_cache_identity(cache_identity)
        self.modality = "retrieval"
        self.extractor_type = "custom_retrieval"
        self.streaming_safe = True

    def encode_retrieval(self, values: Any, *, branch: str, modality: str) -> Any:
        if branch not in {"query", "gallery"}:
            raise ValueError("CallableRetrievalExtractor branch must be 'query' or 'gallery'.")
        expected = self.query_modality if branch == "query" else self.gallery_modality
        if modality != expected:
            raise ValueError(f"Branch {branch!r} expects modality {expected!r}, got {modality!r}.")
        fn = self.query_fn if branch == "query" else self.gallery_fn
        return ensure_numeric_matrix(
            fn(values), f"CallableRetrievalExtractor '{self.name}' {branch}"
        )

    def recipe(self) -> Dict[str, Any]:
        recipe = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "query_modality": self.query_modality,
            "gallery_modality": self.gallery_modality,
            "metadata": self.metadata,
            "query_callable": _callable_name(self.query_fn),
            "gallery_callable": _callable_name(self.gallery_fn),
        }
        recipe.update(
            cache_identity_fields(
                explicit=self.cache_identity,
                callables=(("query_fn", self.query_fn), ("gallery_fn", self.gallery_fn)),
            )
        )
        return recipe


def _callable_name(function: Callable[[Any], Any]) -> str:
    value_type = type(function)
    module = getattr(function, "__module__", value_type.__module__)
    qualname = getattr(function, "__qualname__", value_type.__qualname__)
    return f"{module}.{qualname}"
