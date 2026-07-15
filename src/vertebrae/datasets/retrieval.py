"""Dataset contracts for exact frozen-embedding retrieval evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from vertebrae.cache.fingerprint import canonical_json_exact
from vertebrae.datasets._snapshots import (
    ReadOnlyMapping,
    immutable_value,
    outward_copy,
    snapshot_value,
)
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
    _query_ids_snapshot: Tuple[Any, ...] = field(default=(), init=False, repr=False)
    _gallery_ids_snapshot: Tuple[Any, ...] = field(default=(), init=False, repr=False)
    _queries_snapshot: Any = field(default=None, init=False, repr=False)
    _gallery_snapshot: Any = field(default=None, init=False, repr=False)
    _relevance_snapshot: Dict[int, Dict[int, float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _exclusions_snapshot: frozenset[Tuple[int, int]] = field(
        default_factory=frozenset, init=False, repr=False
    )
    _metadata_snapshot: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _query_modality_snapshot: str = field(default="", init=False, repr=False)
    _gallery_modality_snapshot: str = field(default="", init=False, repr=False)

    _IMMUTABLE_PROTOCOL_FIELDS = frozenset(
        {
            "queries",
            "gallery",
            "query_ids",
            "gallery_ids",
            "relevance",
            "identity",
            "query_modality",
            "gallery_modality",
            "exclusions",
            "metadata",
        }
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._IMMUTABLE_PROTOCOL_FIELDS and self.__dict__.get("_normalized", False):
            raise AttributeError("RetrievalDataset protocol fields are immutable.")
        super().__setattr__(name, value)

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
        if query_matrix.shape[1] != gallery_matrix.shape[1]:
            raise ValueError(
                "query and gallery embeddings must have the same width; "
                f"got {query_matrix.shape[1]} and {gallery_matrix.shape[1]}."
            )
        if metadata is not None and "precomputed_embeddings" in metadata:
            raise ValueError(
                "metadata cannot override constructor-owned key 'precomputed_embeddings'."
            )
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
        matrix = relevance if is_sparse_matrix(relevance) else np.asarray(relevance)
        return cls.from_arrays(queries, gallery, matrix, **kwargs)

    def validated(self) -> "RetrievalDataset":
        if not isinstance(self.identity, DatasetIdentity):
            raise TypeError("identity must be a DatasetIdentity.")
        if self._normalized:
            return self
        self.query_ids = list(self.query_ids)
        self.gallery_ids = list(self.gallery_ids)
        self._validate_ids(self.query_ids, "query_ids")
        self._validate_ids(self.gallery_ids, "gallery_ids")
        queries_snapshot = snapshot_value(self.queries)
        gallery_snapshot = snapshot_value(self.gallery)
        if len(self.query_ids) != len(queries_snapshot) or len(self.gallery_ids) != len(
            gallery_snapshot
        ):
            raise ValueError("query/gallery IDs must align with their input collections.")
        self.relevance = self._normalize_relevance(self.relevance)
        self.exclusions = self._normalize_exclusions(self.exclusions)
        try:
            metadata_snapshot = make_json_safe(self.metadata)
        except TypeError as exc:
            raise ValueError(
                "Retrieval metadata must be deterministically JSON-serializable."
            ) from exc
        self._queries_snapshot = queries_snapshot
        self._gallery_snapshot = gallery_snapshot
        self._query_ids_snapshot = tuple(snapshot_value(self.query_ids))
        self._gallery_ids_snapshot = tuple(snapshot_value(self.gallery_ids))
        self._relevance_snapshot = {query: dict(values) for query, values in self.relevance.items()}
        self._exclusions_snapshot = frozenset(self.exclusions)
        self._metadata_snapshot = metadata_snapshot
        self._query_modality_snapshot = str(self.query_modality)
        self._gallery_modality_snapshot = str(self.gallery_modality)
        eligible = self.eligible_relevance()
        if any(not values for values in eligible.values()):
            raise ValueError(
                "Every query must retain at least one positive relevance after exclusions."
            )
        self._identity_key_cache = self.identity.resolve(
            {
                "query_ids": list(self._query_ids_snapshot),
                "gallery_ids": list(self._gallery_ids_snapshot),
                "queries": self._queries_snapshot,
                "gallery": self._gallery_snapshot,
                "relevance": self._relevance_snapshot,
                "exclusions": sorted(self._exclusions_snapshot),
                "query_modality": self._query_modality_snapshot,
                "gallery_modality": self._gallery_modality_snapshot,
                "metadata": self._metadata_snapshot,
            }
        )
        self.queries = immutable_value(self._queries_snapshot)
        self.gallery = immutable_value(self._gallery_snapshot)
        self.query_ids = immutable_value(self._query_ids_snapshot)
        self.gallery_ids = immutable_value(self._gallery_ids_snapshot)
        self.relevance = ReadOnlyMapping(
            {
                query: ReadOnlyMapping(dict(values))
                for query, values in self._relevance_snapshot.items()
            }
        )
        self.exclusions = self._exclusions_snapshot
        self.metadata = immutable_value(self._metadata_snapshot)
        self.query_modality = self._query_modality_snapshot
        self.gallery_modality = self._gallery_modality_snapshot
        self._normalized = True
        return self

    def _validate_ids(self, values: Sequence[Any], name: str) -> None:
        if len(values) == 0:
            raise ValueError(f"{name} must not be empty.")
        normalized = [self._exact_id(value, name) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must be unique under exact typed identity.")

    @staticmethod
    def _exact_id(value: Any, name: str) -> str:
        normalized = value.item() if hasattr(value, "item") else value
        if normalized is None or (
            isinstance(normalized, (float, np.floating)) and np.isnan(normalized)
        ):
            raise ValueError(f"{name} entries must be non-missing.")
        try:
            hash(normalized)
            return canonical_json_exact(normalized)
        except TypeError as exc:
            raise ValueError(
                f"{name} entries must be hashable and have deterministic exact identities."
            ) from exc

    def _normalize_relevance(self, value: Any) -> Dict[int, Dict[int, float]]:
        n_queries, n_gallery = len(self.query_ids), len(self.gallery_ids)
        if isinstance(value, dict):
            normalized_mapping: Dict[int, Dict[int, float]] = {}
            for query, query_values in value.items():
                query_index_value = _exact_nonnegative_index(query, "relevance query index")
                normalized_mapping[query_index_value] = {}
                for gallery, grade in query_values.items():
                    gallery_index_value = _exact_nonnegative_index(
                        gallery, "relevance gallery index"
                    )
                    normalized_mapping[query_index_value][gallery_index_value] = float(grade)
            if set(normalized_mapping) != set(range(n_queries)):
                raise ValueError("Normalized relevance must contain one entry for every query.")
            for _query, values in normalized_mapping.items():
                for gallery, grade in values.items():
                    if not 0 <= gallery < n_gallery or not np.isfinite(grade) or grade <= 0:
                        raise ValueError("Normalized relevance contains an invalid positive grade.")
            return normalized_mapping
        query_index = {
            self._exact_id(value, "query_ids"): index for index, value in enumerate(self.query_ids)
        }
        gallery_index = {
            self._exact_id(value, "gallery_ids"): index
            for index, value in enumerate(self.gallery_ids)
        }
        normalized: Dict[int, Dict[int, float]] = {index: {} for index in range(n_queries)}
        if is_sparse_matrix(value):
            if tuple(value.shape) != (n_queries, n_gallery):
                raise ValueError("Sparse relevance must have shape (n_queries, n_gallery).")
            sparse = value.tocoo(copy=True)
            sum_duplicates = getattr(sparse, "sum_duplicates", None)
            if callable(sum_duplicates):
                sum_duplicates()
            try:
                grades = np.asarray(sparse.data, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Relevance grades must be numeric.") from exc
            if not np.all(np.isfinite(grades)) or np.any(grades < 0):
                raise ValueError("Relevance grades must be finite and non-negative.")
            for query_row, gallery_row, grade in zip(sparse.row, sparse.col, grades):
                if grade > 0:
                    normalized[int(query_row)][int(gallery_row)] = float(grade)
            return normalized
        if isinstance(value, np.ndarray):
            if tuple(value.shape) != (n_queries, n_gallery):
                raise ValueError("Dense relevance must have shape (n_queries, n_gallery).")
            try:
                dense = np.asarray(value, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Relevance grades must be numeric.") from exc
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
                "Relevance must be an ndarray/sparse matrix or "
                "(query_id, gallery_id, grade) records. Use from_relevance_matrix() for "
                "nested Python matrix values."
            ) from exc
        for record in records:
            if len(record) != 3:
                raise ValueError("Relevance records must be (query_id, gallery_id, grade).")
            query_id, gallery_id, grade = record
            query_key = self._exact_id(query_id, "relevance query IDs")
            gallery_key = self._exact_id(gallery_id, "relevance gallery IDs")
            if query_key not in query_index or gallery_key not in gallery_index:
                raise ValueError("Relevance record references an unknown query or gallery ID.")
            pair = (query_key, gallery_key)
            if pair in seen:
                raise ValueError("Duplicate query/gallery relevance records are not allowed.")
            seen.add(pair)
            grade = float(grade)
            if not np.isfinite(grade) or grade < 0:
                raise ValueError("Relevance grades must be finite and non-negative.")
            if grade > 0:
                normalized[query_index[query_key]][gallery_index[gallery_key]] = grade
        return normalized

    def _normalize_exclusions(
        self, values: Optional[Iterable[Tuple[Any, Any]]]
    ) -> set[Tuple[int, int]]:
        if values is None:
            return set()
        query_index = {
            self._exact_id(value, "query_ids"): index for index, value in enumerate(self.query_ids)
        }
        gallery_index = {
            self._exact_id(value, "gallery_ids"): index
            for index, value in enumerate(self.gallery_ids)
        }
        normalized = set()
        for query_id, gallery_id in values:
            query_key = self._exact_id(query_id, "exclusion query IDs")
            gallery_key = self._exact_id(gallery_id, "exclusion gallery IDs")
            if query_key not in query_index or gallery_key not in gallery_index:
                raise ValueError("Exclusion references an unknown query or gallery ID.")
            normalized.add((query_index[query_key], gallery_index[gallery_key]))
        return normalized

    def eligible_relevance(self) -> Dict[int, Dict[int, float]]:
        exclusions = self._exclusions_snapshot
        return {
            query: {
                gallery: grade
                for gallery, grade in values.items()
                if (query, gallery) not in exclusions
            }
            for query, values in self._relevance_snapshot.items()
        }

    def query_id_values(self) -> Tuple[Any, ...]:
        """Return the validated query-ID snapshot used by the protocol."""

        return outward_copy(self._query_ids_snapshot)

    def gallery_id_values(self) -> Tuple[Any, ...]:
        """Return the validated gallery-ID snapshot used by the protocol."""

        return outward_copy(self._gallery_ids_snapshot)

    def query_values(self) -> Any:
        """Return a detached copy of the validated query inputs used for evaluation."""

        return outward_copy(self._queries_snapshot)

    def gallery_values(self) -> Any:
        """Return a detached copy of the validated gallery inputs used for evaluation."""

        return outward_copy(self._gallery_snapshot)

    def normalized_relevance(self) -> Dict[int, Dict[int, float]]:
        """Return a copy of the validated relevance snapshot."""

        return {query: dict(values) for query, values in self._relevance_snapshot.items()}

    def normalized_exclusions(self) -> set[Tuple[int, int]]:
        """Return a copy of the validated exclusion snapshot."""

        return set(self._exclusions_snapshot)

    def protocol_modalities(self) -> Tuple[str, str]:
        """Return the validated query and gallery modalities."""

        return self._query_modality_snapshot, self._gallery_modality_snapshot

    def summary(self) -> Dict[str, Any]:
        relevance = self.eligible_relevance()
        return {
            "modality": "retrieval",
            "query_modality": self._query_modality_snapshot,
            "gallery_modality": self._gallery_modality_snapshot,
            "n_queries": len(self._query_ids_snapshot),
            "n_gallery": len(self._gallery_ids_snapshot),
            "n_relevance_pairs": sum(len(values) for values in relevance.values()),
            "n_exclusions": len(self._exclusions_snapshot),
            "identity": self.identity.descriptor(self.identity_key()),
            "metadata": outward_copy(self._metadata_snapshot),
        }

    def identity_key(self) -> str:
        if self._identity_key_cache is None:  # pragma: no cover - old pickle defense
            self.validated()
        assert self._identity_key_cache is not None
        return self._identity_key_cache


def _exact_nonnegative_index(value: Any, name: str) -> int:
    from numbers import Integral

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative.")
    return normalized
