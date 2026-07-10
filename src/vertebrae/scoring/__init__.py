"""Scoring helpers."""

from vertebrae.scoring.metrics import (
    CallableMetric,
    EmbeddingMetric,
    LabelRetrievalMetric,
    MetricResult,
    OverlapMetric,
    as_embedding_metric,
    load_metric_callable,
)
from vertebrae.scoring.overlap import (
    OverlapIndexScorer,
    OverlapScoreResult,
    auto_k_for_class,
    resolve_kmeans_k,
)
from vertebrae.scoring.retrieval import RetrievalScorer, RetrievalScoreResult
from vertebrae.scoring.separatix import SeparatixResult, SeparatixScorer
from vertebrae.scoring.stability import run_stability_analysis

__all__ = [
    "OverlapIndexScorer",
    "OverlapScoreResult",
    "CallableMetric",
    "EmbeddingMetric",
    "LabelRetrievalMetric",
    "MetricResult",
    "OverlapMetric",
    "as_embedding_metric",
    "load_metric_callable",
    "SeparatixResult",
    "SeparatixScorer",
    "auto_k_for_class",
    "resolve_kmeans_k",
    "run_stability_analysis",
    "RetrievalScoreResult",
    "RetrievalScorer",
]
