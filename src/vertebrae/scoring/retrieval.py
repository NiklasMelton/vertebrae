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
        queries_without_positives = [
            query_index
            for query_index, values in relevance.items()
            if not any((query_index, gallery_index) not in exclusions for gallery_index in values)
        ]
        if queries_without_positives:
            display_ids = [
                query_ids[index] if query_ids is not None else index
                for index in queries_without_positives[:10]
            ]
            suffix = ", ..." if len(queries_without_positives) > len(display_ids) else ""
            raise ValueError(
                "Every query must retain at least one eligible positive relevance after "
                f"exclusions; missing for {display_ids!r}{suffix}."
            )
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
        required += (
            min(self.config.query_batch_size, n_queries)
            * min(self.config.gallery_batch_size, n_gallery)
            * np.dtype(np.float64).itemsize
        )
        if required > self.config.max_dense_bytes:
            raise MemoryError(
                "Retrieval inputs and one gallery score block exceed "
                "RetrievalConfig.max_dense_bytes."
            )
        query_dense = _to_dense(query_matrix)
        gallery_dense = _to_dense(gallery_matrix)
        if self.config.similarity == "cosine":
            query_dense = _l2_normalize(
                query_dense,
                endpoint="Query embeddings",
                row_ids=query_ids,
            )
            gallery_dense = _l2_normalize(
                gallery_dense,
                endpoint="Gallery embeddings",
                row_ids=gallery_ids,
            )
        metric_rows: List[Dict[str, float]] = []
        positive_sims: List[float] = []
        negative_sims: List[float] = []
        margins: List[float] = []
        worst: List[Tuple[float, int, Dict[str, float]]] = []
        warnings: List[str] = []
        for query_start in range(0, n_queries, self.config.query_batch_size):
            query_stop = min(query_start + self.config.query_batch_size, n_queries)
            batch_indices = np.arange(query_start, query_stop)
            batch_results = self._score_query_batch_blockwise(
                query_dense[query_start:query_stop],
                gallery_dense,
                batch_indices,
                relevance,
                exclusions,
            )
            for q_index, row, query_diagnostics, warning in batch_results:
                if warning is not None:
                    query_id = query_ids[q_index] if query_ids else q_index
                    raise ValueError(f"Query {query_id!r} {warning}")
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

    def _scores_for_queries(self, queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
        if self.config.similarity in {"cosine", "dot"}:
            return gallery @ queries.T
        gallery_norms = np.einsum("ij,ij->i", gallery, gallery)[:, None]
        query_norms = np.einsum("ij,ij->i", queries, queries)[None, :]
        return -(gallery_norms + query_norms - 2.0 * (gallery @ queries.T))

    def _score_query_batch_blockwise(
        self,
        queries: np.ndarray,
        gallery: np.ndarray,
        query_indices: np.ndarray,
        relevance: Dict[int, Dict[int, float]],
        exclusions: set[Tuple[int, int]],
    ) -> List[Tuple[int, Dict[str, float], Dict[str, Any], Optional[str]]]:
        states: List[Optional[Dict[str, Any]]] = []
        for query_index, query in zip(query_indices, queries):
            positives = {
                index: grade
                for index, grade in relevance[int(query_index)].items()
                if (int(query_index), index) not in exclusions and grade > 0
            }
            if not positives:
                states.append(None)
                continue
            eligible_count = len(gallery) - sum(
                1 for index in range(len(gallery)) if (int(query_index), index) in exclusions
            )
            if not eligible_count:
                states.append({"error": "has no eligible candidates."})
                continue
            positive_indices = np.asarray(sorted(positives), dtype=int)
            positive_scores = self._scores_for_query(query, gallery[positive_indices])
            states.append(
                {
                    "positives": positives,
                    "positive_indices": positive_indices,
                    "positive_scores": positive_scores,
                    "ranks": np.ones(len(positive_indices), dtype=int),
                    "nearest_negative": None,
                }
            )
        for start in range(0, len(gallery), self.config.gallery_batch_size):
            stop = min(start + self.config.gallery_batch_size, len(gallery))
            gallery_indices = np.arange(start, stop)
            score_block = self._scores_for_queries(queries, gallery[start:stop])
            for position, (query_index, state) in enumerate(zip(query_indices, states)):
                if state is None or "error" in state:
                    continue
                eligible = np.asarray(
                    [(int(query_index), int(index)) not in exclusions for index in gallery_indices],
                    dtype=bool,
                )
                if not np.any(eligible):
                    continue
                eligible_indices = gallery_indices[eligible]
                eligible_scores = score_block[eligible, position]
                for positive_position, positive_index in enumerate(state["positive_indices"]):
                    better = eligible_scores > state["positive_scores"][positive_position]
                    tied_before = (
                        eligible_scores == state["positive_scores"][positive_position]
                    ) & (eligible_indices < positive_index)
                    state["ranks"][positive_position] += int(np.count_nonzero(better | tied_before))
                negative_mask = np.asarray(
                    [int(index) not in state["positives"] for index in eligible_indices]
                )
                if np.any(negative_mask):
                    candidate = float(np.max(eligible_scores[negative_mask]))
                    nearest = state["nearest_negative"]
                    state["nearest_negative"] = (
                        candidate if nearest is None else max(nearest, candidate)
                    )
        output: List[Tuple[int, Dict[str, float], Dict[str, Any], Optional[str]]] = []
        for query_index, state in zip(query_indices, states):
            if state is None:
                output.append((int(query_index), {}, {}, "has no eligible positive."))
                continue
            if "error" in state:
                output.append((int(query_index), {}, {}, state["error"]))
                continue
            grades = np.asarray(
                [state["positives"][int(index)] for index in state["positive_indices"]], dtype=float
            )
            output.append(
                (
                    int(query_index),
                    _query_metrics_from_ranks(state["ranks"], grades, self.config.ks),
                    {
                        "positive_scores": state["positive_scores"].tolist(),
                        "nearest_negative": state["nearest_negative"],
                    },
                    None,
                )
            )
        return output

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
        max_grade = float(np.max(grades))
        max_log_gain = float(_log_relevance_gain(np.asarray([max_grade]))[0])
        retrieved_gains = np.exp(_log_relevance_gain(grades[in_top_k]) - max_log_gain)
        dcg = float(np.sum(retrieved_gains / np.log2(ranks[in_top_k] + 1)))
        ideal = np.sort(grades)[::-1][:k]
        ideal_gains = np.exp(_log_relevance_gain(ideal) - max_log_gain)
        idcg = float(np.sum(ideal_gains / np.log2(np.arange(2, len(ideal) + 2))))
        ndcg = dcg / idcg if idcg else 0.0
        row[f"ndcg@{k}"] = float(min(1.0, max(0.0, ndcg)))
    return row


def _log_relevance_gain(grades: np.ndarray) -> np.ndarray:
    """Compute ``log(2**grade - 1)`` without cancellation or overflow."""

    values = np.asarray(grades, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("NDCG relevance grades must be finite and positive.")
    scaled = values * np.log(2.0)
    result = np.empty_like(values)
    small = scaled < 1e-4
    large = scaled > 50.0
    middle = ~(small | large)
    # expm1(x) = x * (1 + x/2 + x**2/6 + ...). Keeping x factored
    # avoids underflow for the smallest positive float grades.
    result[small] = (
        np.log(values[small])
        + np.log(np.log(2.0))
        + np.log1p(scaled[small] / 2.0 + scaled[small] ** 2 / 6.0)
    )
    result[middle] = np.log(np.expm1(scaled[middle]))
    result[large] = scaled[large] + np.log1p(-np.exp(-scaled[large]))
    return result


def _to_dense(value: Any) -> np.ndarray:
    return np.asarray(value.toarray() if is_sparse_matrix(value) else value, dtype=np.float64)


def _dense_bytes(value: Any) -> int:
    return int(value.shape[0]) * int(value.shape[1]) * np.dtype(np.float64).itemsize


def _l2_normalize(
    value: np.ndarray,
    *,
    endpoint: str,
    row_ids: Optional[List[Any]] = None,
) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms[:, 0] == 0.0)
    if len(zero_rows):
        preview = []
        for index in zero_rows[:10]:
            row = int(index)
            identity = f"index {row}"
            if row_ids is not None:
                identity += f" (ID {row_ids[row]!r})"
            preview.append(identity)
        suffix = "" if len(zero_rows) <= len(preview) else ", ..."
        raise ValueError(
            f"{endpoint} contain {len(zero_rows)} zero-norm row(s): "
            f"{', '.join(preview)}{suffix}. Cosine similarity is undefined for zero-norm "
            "embeddings."
        )
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
