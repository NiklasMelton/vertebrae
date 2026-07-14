"""Dataset contracts for exact frozen-embedding retrieval evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from vertebrae.datasets.identity import DatasetIdentity
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


@dataclass
class RetrievalDataset:
    """Independent query and gallery collections with explicit graded relevance."""

    queries: Any
    gallery: Any
    query_ids: Sequence[Any]
    gallery_ids: Sequence[Any]
    relevance: Any
    identity: DatasetIdentity
    query_modality: str = "embeddings"
    gallery_modality: str = "embeddings"
    exclusions: Optional[Iterable[Tuple[Any, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _normalized: bool = field(default=False, init=False, repr=False)
    _identity_key_cache: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.validated()

    @classmethod
    def from_embeddings(
        cls,
        queries: Any,
        gallery: Any,
        relevance: Any,
        *,
        identity: DatasetIdentity,
        query_ids: Optional[Sequence[Any]] = None,
        gallery_ids: Optional[Sequence[Any]] = None,
        exclusions: Optional[Iterable[Tuple[Any, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RetrievalDataset":
        query_matrix = ensure_numeric_matrix(queries, "query embeddings", allow_sparse=True)
        gallery_matrix = ensure_numeric_matrix(gallery, "gallery embeddings", allow_sparse=True)
        return cls(
            queries=query_matrix,
            gallery=gallery_matrix,
            query_ids=list(query_ids)
            if query_ids is not None
            else list(range(query_matrix.shape[0])),
            gallery_ids=list(gallery_ids)
            if gallery_ids is not None
            else list(range(gallery_matrix.shape[0])),
            relevance=relevance,
            identity=identity,
            exclusions=exclusions,
            metadata={"precomputed_embeddings": True, **(metadata or {})},
        )

    @classmethod
    def from_arrays(
        cls,
        queries: Any,
        gallery: Any,
        relevance: Any,
        *,
        identity: DatasetIdentity,
        query_modality: str,
        gallery_modality: str,
        query_ids: Optional[Sequence[Any]] = None,
        gallery_ids: Optional[Sequence[Any]] = None,
        exclusions: Optional[Iterable[Tuple[Any, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RetrievalDataset":
        n_queries, n_gallery = len(queries), len(gallery)
        return cls(
            queries=queries,
            gallery=gallery,
            query_ids=list(query_ids) if query_ids is not None else list(range(n_queries)),
            gallery_ids=list(gallery_ids) if gallery_ids is not None else list(range(n_gallery)),
            relevance=relevance,
            identity=identity,
            query_modality=query_modality,
            gallery_modality=gallery_modality,
            exclusions=exclusions,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_relevance_matrix(
        cls, queries: Any, gallery: Any, relevance: Any, **kwargs: Any
    ) -> "RetrievalDataset":
        """Alias that makes a dense relevance-matrix workflow explicit."""
        return cls.from_arrays(queries, gallery, np.asarray(relevance), **kwargs)

    def validated(self) -> "RetrievalDataset":
        if not isinstance(self.identity, DatasetIdentity):
            raise TypeError("identity must be a DatasetIdentity.")
        if self._normalized:
            return self
        self._validate_ids(self.query_ids, "query_ids")
        self._validate_ids(self.gallery_ids, "gallery_ids")
        if len(self.query_ids) != len(self.queries) or len(self.gallery_ids) != len(self.gallery):
            raise ValueError("query/gallery IDs must align with their input collections.")
        self.relevance = self._normalize_relevance(self.relevance)
        self.exclusions = self._normalize_exclusions(self.exclusions)
        eligible = self.eligible_relevance()
        if any(not values for values in eligible.values()):
            raise ValueError(
                "Every query must retain at least one positive relevance after exclusions."
            )
        self._normalized = True
        return self

    def _validate_ids(self, values: Sequence[Any], name: str) -> None:
        if not values:
            raise ValueError(f"{name} must not be empty.")
        normalized = [self._hashable(value, name) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must be unique.")

    @staticmethod
    def _hashable(value: Any, name: str) -> Any:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} entries must be hashable.") from exc
        return value

    def _normalize_relevance(self, value: Any) -> Dict[int, Dict[int, float]]:
        n_queries, n_gallery = len(self.query_ids), len(self.gallery_ids)
        if isinstance(value, dict):
            normalized_mapping = {
                int(query): {int(gallery): float(grade) for gallery, grade in values.items()}
                for query, values in value.items()
            }
            if set(normalized_mapping) != set(range(n_queries)):
                raise ValueError("Normalized relevance must contain one entry for every query.")
            for _query, values in normalized_mapping.items():
                for gallery, grade in values.items():
                    if not 0 <= gallery < n_gallery or not np.isfinite(grade) or grade <= 0:
                        raise ValueError("Normalized relevance contains an invalid positive grade.")
            return normalized_mapping
        query_index = {value: index for index, value in enumerate(self.query_ids)}
        gallery_index = {value: index for index, value in enumerate(self.gallery_ids)}
        normalized: Dict[int, Dict[int, float]] = {index: {} for index in range(n_queries)}
        is_dense_list = (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(row, list) for row in value)
            and np.asarray(value).shape == (n_queries, n_gallery)
        )
        if is_sparse_matrix(value) or isinstance(value, np.ndarray) or is_dense_list:
            array = value
            if tuple(array.shape) != (n_queries, n_gallery):
                raise ValueError("Dense relevance must have shape (n_queries, n_gallery).")
            dense = array.toarray() if is_sparse_matrix(array) else np.asarray(array)
            if not np.all(np.isfinite(dense)) or np.any(dense < 0):
                raise ValueError("Relevance grades must be finite and non-negative.")
            for query_row, gallery_row in zip(*np.nonzero(dense > 0)):
                normalized[int(query_row)][int(gallery_row)] = float(dense[query_row, gallery_row])
            return normalized
        seen = set()
        try:
            records = list(value)
        except TypeError as exc:
            raise ValueError(
                "Relevance must be a dense matrix or (query_id, gallery_id, grade) records."
            ) from exc
        for record in records:
            if len(record) != 3:
                raise ValueError("Relevance records must be (query_id, gallery_id, grade).")
            query_id, gallery_id, grade = record
            if query_id not in query_index or gallery_id not in gallery_index:
                raise ValueError("Relevance record references an unknown query or gallery ID.")
            pair = (query_id, gallery_id)
            if pair in seen:
                raise ValueError("Duplicate query/gallery relevance records are not allowed.")
            seen.add(pair)
            grade = float(grade)
            if not np.isfinite(grade) or grade < 0:
                raise ValueError("Relevance grades must be finite and non-negative.")
            if grade > 0:
                normalized[query_index[query_id]][gallery_index[gallery_id]] = grade
        return normalized

    def _normalize_exclusions(
        self, values: Optional[Iterable[Tuple[Any, Any]]]
    ) -> set[Tuple[int, int]]:
        if values is None:
            return set()
        query_index = {value: index for index, value in enumerate(self.query_ids)}
        gallery_index = {value: index for index, value in enumerate(self.gallery_ids)}
        normalized = set()
        for query_id, gallery_id in values:
            if query_id not in query_index or gallery_id not in gallery_index:
                raise ValueError("Exclusion references an unknown query or gallery ID.")
            normalized.add((query_index[query_id], gallery_index[gallery_id]))
        return normalized

    def eligible_relevance(self) -> Dict[int, Dict[int, float]]:
        exclusions = set(self.exclusions or ())
        return {
            query: {
                gallery: grade
                for gallery, grade in values.items()
                if (query, gallery) not in exclusions
            }
            for query, values in self.relevance.items()
        }

    def summary(self) -> Dict[str, Any]:
        relevance = self.eligible_relevance()
        return {
            "modality": "retrieval",
            "query_modality": self.query_modality,
            "gallery_modality": self.gallery_modality,
            "n_queries": len(self.query_ids),
            "n_gallery": len(self.gallery_ids),
            "n_relevance_pairs": sum(len(values) for values in relevance.values()),
            "n_exclusions": sum(1 for _ in (self.exclusions or ())),
            "identity": self.identity.descriptor(self.identity_key()),
            "metadata": make_json_safe(self.metadata),
        }

    def identity_key(self) -> str:
        if self._identity_key_cache is None:
            self._identity_key_cache = self.identity.resolve(
                {
                    "query_ids": list(self.query_ids),
                    "gallery_ids": list(self.gallery_ids),
                    "queries": self.queries,
                    "gallery": self.gallery,
                    "relevance": self.relevance,
                    "exclusions": sorted(self.exclusions or ()),
                    "query_modality": self.query_modality,
                    "gallery_modality": self.gallery_modality,
                    "metadata": self.metadata,
                }
            )
        return self._identity_key_cache
