"""Structured benchmark results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from vertebrae.scoring.overlap import OverlapScoreResult
from vertebrae.scoring.separatix import SeparatixResult
from vertebrae.utils.serialization import make_json_safe


@dataclass
class ExtractorResult:
    """Result data for one evaluated extractor.

    Attributes:
        name: Extractor name.
        extractor_type: Extractor family/type metadata.
        overlap: OverlapIndex scoring result.
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
    overlap: OverlapScoreResult
    stability: Optional[Dict[str, Any]]
    separatix: Optional[SeparatixResult]
    embedding_metadata: Dict[str, Any]
    compression_metadata: Dict[str, Any]
    runtime: Dict[str, Any]
    warnings: List[str]
    recommendation: str
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
        """Return extractor results sorted by descending macro score.

        Returns:
            Sorted extractor results.
        """

        return sorted(
            self.extractor_results,
            key=lambda item: (
                bool(item.overlap.metadata.get("aggregate_valid", True)),
                item.overlap.score,
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
                    "overlap_score": item.overlap.score,
                    "overlap_macro": item.overlap.macro_score,
                    "overlap_weighted": item.overlap.weighted_score,
                    "target_type": item.overlap.metadata.get("target_type", "single_label"),
                    "target_names": item.overlap.metadata.get("target_names"),
                    "target_view": (item.target_view or {}).get("name"),
                    "label_view": (item.label_view or {}).get("name"),
                    "weakest_class": item.weakest_class,
                    "weakest_class_score": item.weakest_class_score,
                    "probe_metric": _separatix_probe_metric_name(item.separatix),
                    "probe_score": _separatix_probe_accuracy(item.separatix),
                    "probe_accuracy": _separatix_probe_accuracy(item.separatix),
                    "embedding_dim": item.embedding_metadata.get("embedding_dim"),
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
        """Save the benchmark result as JSON.

        Args:
            path: Destination file path.
        """

        from vertebrae.reports.json_report import save_json_report

        save_json_report(self, str(Path(path)))

    def save_markdown(self, path: str) -> None:
        """Save the benchmark result as a Markdown report.

        Args:
            path: Destination file path.
        """

        from vertebrae.reports.markdown_report import save_markdown_report

        save_markdown_report(self, str(Path(path)))


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
