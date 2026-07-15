"""Hosted embedding API extractor."""

import time
from numbers import Real
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vertebrae.extractors._identity import cache_identity_fields, validate_cache_identity
from vertebrae.extractors._utils import (
    callable_name,
    iter_chunks,
    snapshot_mapping,
    validate_batch_size,
    validate_nonblank_string,
)
from vertebrae.utils.validation import ensure_numeric_matrix


class HostedEmbeddingExtractor:
    """Wrap a hosted embedding API behind a batch-oriented callable."""

    def __init__(
        self,
        name: str,
        provider: str,
        model: str,
        embed_fn: Callable[[Any], Any],
        batch_size: int = 64,
        modality: str = "text",
        input_fn: Optional[Callable[[Any], Any]] = None,
        output_fn: Optional[Callable[[Any], Any]] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        recipe_data: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.0,
        cache_embeddings: bool = False,
        cache_identity: Optional[str] = None,
    ) -> None:
        name = validate_nonblank_string(name, "name")
        provider = validate_nonblank_string(provider, "provider")
        model = validate_nonblank_string(model, "model")
        if not callable(embed_fn):
            raise TypeError("embed_fn must be callable.")
        if input_fn is not None and not callable(input_fn):
            raise TypeError("input_fn must be callable when provided.")
        if output_fn is not None and not callable(output_fn):
            raise TypeError("output_fn must be callable when provided.")
        batch_size = validate_batch_size(batch_size)
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an integer.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if isinstance(retry_backoff_seconds, bool) or not isinstance(retry_backoff_seconds, Real):
            raise TypeError("retry_backoff_seconds must be a finite number.")
        retry_backoff_seconds = float(retry_backoff_seconds)
        if not np.isfinite(retry_backoff_seconds) or retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be finite and >= 0.")
        if not isinstance(cache_embeddings, bool):
            raise TypeError("cache_embeddings must be a bool.")
        self.name: str = name
        self.provider: str = provider
        self.model: str = model
        self.embed_fn: Callable[[Any], Any] = embed_fn
        self.batch_size: int = batch_size
        self.modality: str = validate_nonblank_string(modality, "modality")
        self.input_fn: Callable[[Any], Any] = input_fn or _identity_batch
        self.output_fn: Optional[Callable[[Any], Any]] = output_fn
        self.request_metadata: Dict[str, Any] = snapshot_mapping(
            request_metadata, "request_metadata"
        )
        self.recipe_data: Dict[str, Any] = snapshot_mapping(recipe_data, "recipe_data")
        self.max_retries: int = max_retries
        self.retry_backoff_seconds: float = retry_backoff_seconds
        self.cache_embeddings: bool = cache_embeddings
        self.extractor_type: str = "hosted_embeddings"
        self.streaming_safe: bool = True
        self.cache_identity: Optional[str] = validate_cache_identity(cache_identity)

    def fit(self, X: Any, y: Any = None) -> "HostedEmbeddingExtractor":
        return self

    def transform(self, X: Any) -> Any:
        collected: List[np.ndarray] = []
        for batch in iter_chunks(list(X), self.batch_size):
            payload = self.input_fn(batch)
            response = self._call_with_retries(payload)
            embeddings = self.output_fn(response) if self.output_fn is not None else response
            matrix = ensure_numeric_matrix(
                np.asarray(embeddings, dtype=np.float32),
                f"HostedEmbeddingExtractor '{self.name}' output",
                allow_sparse=False,
            )
            if matrix.shape[0] != len(batch):
                raise ValueError(
                    f"HostedEmbeddingExtractor '{self.name}' returned {matrix.shape[0]} rows "
                    f"for a request batch of {len(batch)}."
                )
            if collected and matrix.shape[1] != collected[0].shape[1]:
                raise ValueError(
                    f"HostedEmbeddingExtractor '{self.name}' changed embedding width across "
                    f"batches: expected {collected[0].shape[1]}, got {matrix.shape[1]}."
                )
            collected.append(matrix)
        if not collected:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(collected).astype(np.float32, copy=False)

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        recipe = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "provider": self.provider,
            "model": self.model,
            "embed_fn": callable_name(self.embed_fn),
            "input_fn": callable_name(self.input_fn),
            "output_fn": callable_name(self.output_fn) if self.output_fn is not None else None,
            "batch_size": self.batch_size,
            "request_metadata": dict(self.request_metadata),
            "recipe_data": dict(self.recipe_data),
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "cache_embeddings": self.cache_embeddings,
            "streaming_safe": self.streaming_safe,
        }
        recipe.update(
            cache_identity_fields(
                explicit=self.cache_identity,
                callables=(
                    ("embed_fn", self.embed_fn),
                    ("input_fn", self.input_fn),
                    ("output_fn", self.output_fn),
                ),
                require_pinned_revision=True,
            )
        )
        return recipe

    def _call_with_retries(self, payload: Any) -> Any:
        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return self.embed_fn(payload)
            except Exception as exc:  # pragma: no cover - exercised via tests
                last_error = exc
                if attempt == attempts - 1:
                    break
                if self.retry_backoff_seconds > 0.0:
                    time.sleep(self.retry_backoff_seconds)
        assert last_error is not None
        raise last_error


def _identity_batch(batch: Any) -> List[Any]:
    return list(batch)
