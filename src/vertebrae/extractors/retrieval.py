"""Explicit adapter for independent query and gallery embedding branches."""

import importlib
from typing import Any, Callable, Dict, Optional

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
        self.name = name
        self.query_fn = query_fn
        self.gallery_fn = gallery_fn or query_fn
        self.query_modality = query_modality
        self.gallery_modality = gallery_modality or query_modality
        self.metadata = dict(metadata or {})
        if cache_identity is not None and (
            not isinstance(cache_identity, str) or not cache_identity
        ):
            raise ValueError("cache_identity must be a non-empty string when provided.")
        self.cache_identity = cache_identity
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
        query_path = _callable_path(self.query_fn)
        gallery_path = _callable_path(self.gallery_fn)
        cache_safe = bool(self.cache_identity or (query_path and gallery_path))
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "query_modality": self.query_modality,
            "gallery_modality": self.gallery_modality,
            "metadata": self.metadata,
            "query_callable": query_path,
            "gallery_callable": gallery_path,
            "cache_identity": self.cache_identity,
            "cache_safe": cache_safe,
        }


def _callable_path(function: Callable[[Any], Any]) -> Optional[str]:
    """Return a stable import-style path when a callable has one.

    Nested functions and lambdas intentionally have no portable identity: their
    process-local implementation cannot safely participate in artifact cache keys.
    """

    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str) or module == "__main__":
        return None
    if "<locals>" in qualname or "<lambda>" in qualname:
        return None
    try:
        resolved: Any = importlib.import_module(module)
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except (AttributeError, ImportError, ValueError):
        return None
    expected = getattr(function, "__func__", function)
    actual = getattr(resolved, "__func__", resolved)
    if actual is not expected:
        return None
    return f"{module}:{qualname}"
