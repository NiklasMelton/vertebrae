"""Structured benchmark results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from vertebrae.scoring.overlap import OverlapScoreResult
from vertebrae.utils.serialization import make_json_safe


@dataclass
class ExtractorResult:
    """Result data for one evaluated extractor.

    Attributes:
        name: Extractor name.
        extractor_type: Extractor family/type metadata.
        overlap: OverlapIndex scoring result.
        stability: Optional stability-analysis summary.
        probes: Optional probe-classifier summary.
        embedding_metadata: Metadata for the embedding artifact.
        runtime: Runtime timing metadata by benchmark stage.
        warnings: Warnings produced during evaluation.
        recommendation: Recommendation label for this extractor.
        weakest_class: Class with the lowest per-class score when available.
        weakest_class_score: Score for `weakest_class` when available.
    """

    name: str
    extractor_type: str
    overlap: OverlapScoreResult
    stability: Optional[Dict[str, Any]]
    probes: Optional[Dict[str, Any]]
    embedding_metadata: Dict[str, Any]
    compression_metadata: Dict[str, Any]
    runtime: Dict[str, Any]
    warnings: List[str]
    recommendation: str
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
            key=lambda item: item.overlap.macro_score,
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
                    "overlap_macro": item.overlap.macro_score,
                    "weakest_class": item.weakest_class,
                    "weakest_class_score": item.weakest_class_score,
                    "embedding_dim": item.embedding_metadata.get("embedding_dim"),
                    "compression_method": item.compression_metadata.get("method", "none"),
                    "compression_precision": item.compression_metadata.get("precision"),
                    "compressed_dim": item.compression_metadata.get(
                        "compressed_dim",
                        item.embedding_metadata.get("embedding_dim"),
                    ),
                    "recommendation": item.recommendation,
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
