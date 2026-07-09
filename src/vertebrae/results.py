"""Structured benchmark results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from vertebrae.scoring.metrics import MetricResult
from vertebrae.scoring.separatix import SeparatixResult
from vertebrae.utils.serialization import make_json_safe


@dataclass
class ExtractorResult:
    """Result data for one evaluated extractor.

    Attributes:
        name: Extractor name.
        extractor_type: Extractor family/type metadata.
        metrics: All normalized metric results, including ``overlap`` when enabled.
        primary_metric_name: Name of the metric used for ranking.
        stability: Optional stability-analysis summary.
        separatix: Optional Separatix diagnostic summary.
        embedding_metadata: Metadata for the embedding artifact.
        runtime: Runtime timing metadata by benchmark stage.
        warnings: Warnings produced during evaluation.
        recommendation: Recommendation label for this extractor.
        label_view: Label-view metadata for the scoring target.
        target_view: Target-view metadata for the scoring target.
        weakest_class: Class with the lowest per-class score when available.
        weakest_class_score: Score for `weakest_class` when available.
    """

    name: str
    extractor_type: str
    stability: Optional[Dict[str, Any]]
    separatix: Optional[SeparatixResult]
    embedding_metadata: Dict[str, Any]
    compression_metadata: Dict[str, Any]
    runtime: Dict[str, Any]
    warnings: List[str]
    recommendation: str
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    primary_metric_name: str = "overlap"
    label_view: Optional[Dict[str, Any]] = None
    target_view: Optional[Dict[str, Any]] = None
    weakest_class: Optional[str] = None
    weakest_class_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the extractor result to a JSON-safe dictionary.

        Returns:
            JSON-compatible result data.
        """

        return make_json_safe(self)

    @property
    def overlap(self) -> Optional[MetricResult]:
        """Return the overlap metric result without duplicating serialized state."""

        metric = self.metrics.get("overlap")
        return metric if metric and metric.kind == "overlap_index" else None

    @property
    def primary_score(self) -> float:
        """Return the aggregate score used for ranking."""

        return self.metrics[self.primary_metric_name].score


@dataclass
class BenchmarkResult:
    """Aggregated result for a benchmark run.

    Attributes:
        dataset_summary: Summary of the labeled dataset.
        extractor_results: Per-extractor results.
        recommendations: Practitioner-facing benchmark recommendations.
        metadata: Reproducibility metadata for the run.
    """

    dataset_summary: Dict[str, Any]
    extractor_results: List[ExtractorResult]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the benchmark result to a JSON-safe dictionary.

        Returns:
            JSON-compatible benchmark data.
        """

        return make_json_safe(self)

    def ranked_results(self) -> List[ExtractorResult]:
        """Return extractor results sorted by the selected aggregate metric.

        Returns:
            Sorted extractor results.
        """

        return sorted(
            self.extractor_results,
            key=lambda item: (
                bool(_primary_metric(item).metadata.get("aggregate_valid", True)),
                _rankable_score(item),
            ),
            reverse=True,
        )

    def to_dataframe(self) -> Any:
        """Convert benchmark rankings into a pandas DataFrame.

        Returns:
            A pandas DataFrame with one row per extractor.
        """

        import pandas as pd

        rows = []
        for rank, item in enumerate(self.ranked_results(), start=1):
            rows.append(
                {
                    "rank": rank,
                    "extractor": item.name,
                    "extractor_type": item.extractor_type,
                    "primary_metric": item.primary_metric_name,
                    "primary_score": item.primary_score,
                    "primary_higher_is_better": _primary_metric(item).higher_is_better,
                    "overlap_score": item.overlap.score if item.overlap else None,
                    "overlap_macro": item.overlap.macro_score if item.overlap else None,
                    "overlap_weighted": item.overlap.weighted_score if item.overlap else None,
                    "target_type": _result_target_metadata(item).get("target_type", "single_label"),
                    "target_names": _result_target_metadata(item).get("target_names"),
                    "target_view": (item.target_view or {}).get("name"),
                    "label_view": (item.label_view or {}).get("name"),
                    "weakest_class": item.weakest_class,
                    "weakest_class_score": item.weakest_class_score,
                    "probe_metric": _separatix_probe_metric_name(item.separatix),
                    "probe_score": _separatix_probe_accuracy(item.separatix),
                    "probe_accuracy": _separatix_probe_accuracy(item.separatix),
                    "embedding_dim": item.embedding_metadata.get("embedding_dim"),
                    "task_family": (item.embedding_metadata.get("structured", {}) or {}).get(
                        "task_family"
                    ),
                    "alignment_mode": (item.embedding_metadata.get("structured", {}) or {}).get(
                        "alignment_mode"
                    ),
                    "alignment_recipe": (item.embedding_metadata.get("structured", {}) or {}).get(
                        "alignment_recipe"
                    ),
                    "compression_method": item.compression_metadata.get("method", "none"),
                    "compression_precision": item.compression_metadata.get("precision"),
                    "compressed_dim": item.compression_metadata.get(
                        "compressed_dim",
                        item.embedding_metadata.get("embedding_dim"),
                    ),
                    "recommendation": item.recommendation,
                    "separatix_ran": bool(item.separatix and item.separatix.ran),
                    "separatix_recommendation": (
                        item.separatix.recommendation if item.separatix else None
                    ),
                    "separatix_confidence": (item.separatix.confidence if item.separatix else None),
                }
            )
        return pd.DataFrame(rows)

    def save_json(self, path: str) -> None:
        """Save the benchmark result as JSON."""

        from vertebrae.reports.json_report import save_json_report

        save_json_report(self, str(Path(path)))

    def save_markdown(self, path: str) -> None:
        """Save the benchmark result as a Markdown report."""

        from vertebrae.reports.markdown_report import save_markdown_report

        save_markdown_report(self, str(Path(path)))


def _primary_metric(item: ExtractorResult) -> MetricResult:
    return item.metrics[item.primary_metric_name]


def _rankable_score(item: ExtractorResult) -> float:
    metric = _primary_metric(item)
    return metric.score if metric.higher_is_better else -metric.score


def _result_target_metadata(item: ExtractorResult) -> Dict[str, Any]:
    if item.overlap is not None:
        return item.overlap.metadata
    return _primary_metric(item).metadata


def _separatix_probe_accuracy(separatix: Optional[SeparatixResult]) -> Optional[float]:
    if not separatix or not separatix.ran:
        return None
    report = separatix.report or {}
    metrics = report.get("metrics", {})
    baseline = metrics.get("baseline", {})
    probes = metrics.get("probes", {})
    best_probe = baseline.get("best_probe")
    if not best_probe:
        return None
    best_probe_metrics = probes.get(best_probe, {})
    accuracy = best_probe_metrics.get("accuracy")
    if accuracy is None:
        return None
    return float(accuracy)


def _separatix_probe_metric_name(separatix: Optional[SeparatixResult]) -> Optional[str]:
    if _separatix_probe_accuracy(separatix) is None:
        return None
    return "accuracy"
