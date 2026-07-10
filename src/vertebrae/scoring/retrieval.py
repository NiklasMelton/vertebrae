"""Exact, memory-bounded scoring for frozen query--gallery embeddings."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vertebrae.config import RetrievalConfig
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


@dataclass
class RetrievalScoreResult:
    """Aggregate metrics and diagnostics for one retrieval direction."""

    score: float
    primary_metric: str
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "primary_metric": self.primary_metric,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class RetrievalScorer:
    """Score explicit relevance with exhaustive exact ranking and bounded memory."""

    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        self.config = config or RetrievalConfig()

    def score(
        self,
        queries: Any,
        gallery: Any,
        relevance: Dict[int, Dict[int, float]],
        *,
        query_ids: Optional[List[Any]] = None,
        gallery_ids: Optional[List[Any]] = None,
        exclusions: Optional[set[Tuple[int, int]]] = None,
    ) -> RetrievalScoreResult:
        query_matrix = ensure_numeric_matrix(queries, "query embeddings", allow_sparse=True)
        gallery_matrix = ensure_numeric_matrix(gallery, "gallery embeddings", allow_sparse=True)
        if query_matrix.shape[1] != gallery_matrix.shape[1]:
            raise ValueError("Query and gallery embedding dimensions must match.")
        n_queries, n_gallery = query_matrix.shape[0], gallery_matrix.shape[0]
        if set(relevance) != set(range(n_queries)):
            raise ValueError("Relevance must contain one entry for each query.")
        if query_ids is not None and len(query_ids) != n_queries:
            raise ValueError("query_ids must align with query embeddings.")
        if gallery_ids is not None and len(gallery_ids) != n_gallery:
            raise ValueError("gallery_ids must align with gallery embeddings.")
        exclusions = exclusions or set()
        for _query_index, values in relevance.items():
            for gallery_index, grade in values.items():
                if not 0 <= gallery_index < n_gallery or not np.isfinite(grade) or grade <= 0:
                    raise ValueError(
                        "Relevance contains an invalid positive gallery index or grade."
                    )
        if any(
            not 0 <= excluded_query < n_queries or not 0 <= excluded_gallery < n_gallery
            for excluded_query, excluded_gallery in exclusions
        ):
            raise ValueError("Exclusions contain an invalid query or gallery index.")
        comparisons = int(n_queries) * int(n_gallery)
        if (
            self.config.max_pairwise_comparisons is not None
            and comparisons > self.config.max_pairwise_comparisons
        ):
            raise MemoryError(
                "Exact retrieval would exceed max_pairwise_comparisons; reduce the declared "
                "candidate set or raise the explicit limit."
            )
        required = _dense_bytes(query_matrix) + _dense_bytes(gallery_matrix)
        required += min(self.config.gallery_batch_size, n_gallery) * 8
        if required > self.config.max_dense_bytes:
            raise MemoryError(
                "Retrieval inputs and one gallery score block exceed "
                "RetrievalConfig.max_dense_bytes."
            )
        query_dense = _to_dense(query_matrix)
        gallery_dense = _to_dense(gallery_matrix)
        if self.config.similarity == "cosine":
            query_dense = _l2_normalize(query_dense)
            gallery_dense = _l2_normalize(gallery_dense)
        metric_rows: List[Dict[str, float]] = []
        positive_sims: List[float] = []
        negative_sims: List[float] = []
        margins: List[float] = []
        worst: List[Tuple[float, int, Dict[str, float]]] = []
        warnings: List[str] = []
        for q_index in range(n_queries):
            positives = {
                index: grade
                for index, grade in relevance[q_index].items()
                if (q_index, index) not in exclusions and grade > 0
            }
            if not positives:
                query_id = query_ids[q_index] if query_ids else q_index
                warnings.append(f"Query {query_id!r} has no eligible positive.")
                continue
            eligible_count = n_gallery - sum(
                1 for index in range(n_gallery) if (q_index, index) in exclusions
            )
            if not eligible_count:
                query_id = query_ids[q_index] if query_ids else q_index
                warnings.append(f"Query {query_id!r} has no eligible candidates.")
                continue
            row, query_diagnostics = self._score_query_blockwise(
                query_dense[q_index], gallery_dense, q_index, positives, exclusions
            )
            metric_rows.append(row)
            positive_sims.extend(query_diagnostics["positive_scores"])
            nearest_negative = query_diagnostics["nearest_negative"]
            if nearest_negative is not None:
                negative_sims.append(nearest_negative)
                margins.append(max(query_diagnostics["positive_scores"]) - nearest_negative)
            else:
                query_id = query_ids[q_index] if query_ids else q_index
                warnings.append(f"Query {query_id!r} has no negative candidates.")
            worst.append((row[self.config.primary_metric], q_index, row))
        if not metric_rows:
            raise ValueError("Retrieval scoring has no eligible queries.")
        metrics = {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]}
        worst.sort(key=lambda item: item[0])
        worst_rows = [
            {"query_id": (query_ids[index] if query_ids else index), **row}
            for _, index, row in worst[: self.config.worst_queries]
        ]
        diagnostics = {
            "positive_similarity": _summary(positive_sims),
            "nearest_negative_similarity": _summary(negative_sims),
            "retrieval_margin": _summary(margins),
            "eligible_queries": len(metric_rows),
            "worst_queries": worst_rows,
        }
        return RetrievalScoreResult(
            score=metrics[self.config.primary_metric],
            primary_metric=self.config.primary_metric,
            metrics=metrics,
            diagnostics=diagnostics,
            warnings=sorted(set(warnings)),
            metadata={
                "config": asdict(self.config),
                "n_queries": n_queries,
                "n_gallery": n_gallery,
                "similarity": self.config.similarity,
                "exact": True,
            },
        )

    def _scores_for_query(self, query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
        if self.config.similarity in {"cosine", "dot"}:
            return gallery @ query
        difference = gallery - query
        return -np.einsum("ij,ij->i", difference, difference)

    def _score_query_blockwise(
        self,
        query: np.ndarray,
        gallery: np.ndarray,
        query_index: int,
        positives: Dict[int, float],
        exclusions: set[Tuple[int, int]],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        positive_indices = np.asarray(sorted(positives), dtype=int)
        positive_scores = self._scores_for_query(query, gallery[positive_indices])
        ranks = np.ones(len(positive_indices), dtype=int)
        nearest_negative: Optional[float] = None
        for start in range(0, len(gallery), self.config.gallery_batch_size):
            stop = min(start + self.config.gallery_batch_size, len(gallery))
            indices = np.arange(start, stop)
            scores = self._scores_for_query(query, gallery[start:stop])
            eligible = np.asarray(
                [(query_index, int(index)) not in exclusions for index in indices], dtype=bool
            )
            if not np.any(eligible):
                continue
            eligible_indices = indices[eligible]
            eligible_scores = scores[eligible]
            for position, positive_index in enumerate(positive_indices):
                better = eligible_scores > positive_scores[position]
                tied_before = (eligible_scores == positive_scores[position]) & (
                    eligible_indices < positive_index
                )
                ranks[position] += int(np.count_nonzero(better | tied_before))
            negative_mask = np.asarray([int(index) not in positives for index in eligible_indices])
            if np.any(negative_mask):
                candidate = float(np.max(eligible_scores[negative_mask]))
                nearest_negative = (
                    candidate if nearest_negative is None else max(nearest_negative, candidate)
                )
        grades = np.asarray([positives[int(index)] for index in positive_indices], dtype=float)
        return _query_metrics_from_ranks(ranks, grades, self.config.ks), {
            "positive_scores": positive_scores.tolist(),
            "nearest_negative": nearest_negative,
        }


def _query_metrics_from_ranks(
    ranks: np.ndarray, grades: np.ndarray, ks: Tuple[int, ...]
) -> Dict[str, float]:
    n_positive = len(ranks)
    row: Dict[str, float] = {}
    ordered_ranks = np.sort(ranks)
    row["mrr"] = float(1.0 / ordered_ranks[0])
    row["map"] = float(np.mean(np.arange(1, n_positive + 1, dtype=float) / ordered_ranks))
    for k in ks:
        found = float(np.count_nonzero(ranks <= k))
        row[f"precision@{k}"] = found / float(k)
        row[f"recall@{k}"] = found / float(n_positive)
        row[f"hit_rate@{k}"] = float(found > 0)
        in_top_k = ranks <= k
        dcg = float(np.sum((np.power(2.0, grades[in_top_k]) - 1.0) / np.log2(ranks[in_top_k] + 1)))
        ideal = np.sort(grades)[::-1][:k]
        idcg = float(np.sum((np.power(2.0, ideal) - 1.0) / np.log2(np.arange(2, len(ideal) + 2))))
        row[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return row


def _to_dense(value: Any) -> np.ndarray:
    return np.asarray(value.toarray() if is_sparse_matrix(value) else value, dtype=np.float64)


def _dense_bytes(value: Any) -> int:
    return int(value.shape[0]) * int(value.shape[1]) * np.dtype(np.float64).itemsize


def _l2_normalize(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return value / norms


def _summary(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
