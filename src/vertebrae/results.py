"""Structured benchmark results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from vertebrae.scoring.overlap import OverlapScoreResult
from vertebrae.utils.serialization import make_json_safe


@dataclass
class ExtractorResult:
    name: str
    extractor_type: str
    overlap: OverlapScoreResult
    stability: Optional[Dict[str, Any]]
    probes: Optional[Dict[str, Any]]
    embedding_metadata: Dict[str, Any]
    runtime: Dict[str, Any]
    warnings: List[str]
    recommendation: str
    weakest_class: Optional[str] = None
    weakest_class_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return make_json_safe(self)


@dataclass
class BenchmarkResult:
    dataset_summary: Dict[str, Any]
    extractor_results: List[ExtractorResult]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return make_json_safe(self)

    def ranked_results(self) -> List[ExtractorResult]:
        return sorted(
            self.extractor_results,
            key=lambda item: item.overlap.macro_score,
            reverse=True,
        )

    def to_dataframe(self) -> Any:
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
                    "recommendation": item.recommendation,
                }
            )
        return pd.DataFrame(rows)

    def save_json(self, path: str) -> None:
        from vertebrae.reports.json_report import save_json_report

        save_json_report(self, str(Path(path)))

    def save_markdown(self, path: str) -> None:
        from vertebrae.reports.markdown_report import save_markdown_report

        save_markdown_report(self, str(Path(path)))
