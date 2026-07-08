"""Hosted embedding API extractor."""

import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from vertebrae.extractors._utils import callable_name, iter_chunks
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
    ) -> None:
        self.name = name
        self.provider = provider
        self.model = model
        self.embed_fn = embed_fn
        self.batch_size = batch_size
        self.modality = modality
        self.input_fn = input_fn or (lambda batch: list(batch))
        self.output_fn = output_fn
        self.request_metadata = request_metadata or {}
        self.recipe_data = recipe_data or {}
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.cache_embeddings = cache_embeddings
        self.extractor_type = "hosted_embeddings"
        self.streaming_safe = True

    def fit(self, X: Any, y: Any = None) -> "HostedEmbeddingExtractor":
        return self

    def transform(self, X: Any) -> Any:
        collected = []
        for batch in iter_chunks(list(X), self.batch_size):
            payload = self.input_fn(batch)
            response = self._call_with_retries(payload)
            embeddings = self.output_fn(response) if self.output_fn is not None else response
            matrix = ensure_numeric_matrix(
                np.asarray(embeddings, dtype=np.float32),
                f"HostedEmbeddingExtractor '{self.name}' output",
                allow_sparse=False,
            )
            collected.append(matrix)
        if not collected:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(collected).astype(np.float32, copy=False)

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.transform(X)

    def recipe(self) -> Dict[str, Any]:
        return {
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
