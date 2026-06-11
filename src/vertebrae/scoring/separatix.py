"""Internal Separatix adapter."""

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional

import numpy as np

from vertebrae.config import OverlapScoringConfig, SeparatixConfig
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix, l2_normalize_rows


@dataclass
class SeparatixResult:
    """Structured result from Separatix complexity diagnostics."""

    ran: bool
    recommendation: Optional[str] = None
    recommendation_text: Optional[str] = None
    confidence: Optional[str] = None
    scores: Dict[str, Any] = field(default_factory=dict)
    decision_path: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    skipped_reason: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the diagnostic result to a JSON-safe dictionary."""

        return make_json_safe(self)


class SeparatixScorer:
    """Internal adapter for Separatix complexity diagnostics."""

    def __init__(
        self,
        config: Optional[SeparatixConfig] = None,
        overlap_config: Optional[OverlapScoringConfig] = None,
    ) -> None:
        self.config = config or SeparatixConfig()
        self.overlap_config = overlap_config or OverlapScoringConfig()

    def score(self, Z: Any, y: Any) -> SeparatixResult:
        """Run Separatix on dense or sparse embeddings and labels."""

        embeddings = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
        labels = np.asarray(y)
        if embeddings.shape[0] != len(labels):
            raise ValueError(
                "embeddings and labels must have the same length; "
                f"got {embeddings.shape[0]} and {len(labels)}."
            )

        sparse_input = is_sparse_matrix(embeddings)
        normalize_embeddings = self.overlap_config.normalize_embeddings
        if normalize_embeddings:
            embeddings = _l2_normalize_for_separatix(embeddings)

        report = self._run_separatix(embeddings, labels)
        report_dict = make_json_safe(report.to_dict())
        return SeparatixResult(
            ran=True,
            recommendation=report_dict.get("recommendation"),
            recommendation_text=report_dict.get("recommendation_text"),
            confidence=report_dict.get("confidence"),
            scores=report_dict.get("scores", {}),
            decision_path=report_dict.get("decision_path", []),
            warnings=report_dict.get("warnings", []),
            skipped_diagnostics=report_dict.get("skipped_diagnostics", []),
            report=report_dict,
            metadata={
                "normalized_embeddings": normalize_embeddings,
                "sparse_input": sparse_input,
                "budget": self.config.budget or "standard",
                "max_samples": self.config.max_samples,
                "max_dense_mb": self._max_dense_mb(),
                "n_jobs": self.config.n_jobs,
            },
        )

    def skipped_result(self, reason: str, macro_score: float) -> SeparatixResult:
        """Return a structured skipped result for gated runs."""

        return SeparatixResult(
            ran=False,
            skipped_reason=reason,
            metadata={
                "normalized_embeddings": self.overlap_config.normalize_embeddings,
                "overlap_macro_score": float(macro_score),
                "overlap_threshold": self.config.overlap_threshold,
            },
        )

    def _run_separatix(self, embeddings: Any, labels: np.ndarray) -> Any:
        separatix = _load_separatix()
        kwargs = {
            "return_report": True,
            "random_state": self.config.random_state,
            "budget": self.config.budget or "standard",
            "max_dense_mb": self._max_dense_mb(),
            "max_samples": self.config.max_samples,
        }
        if self.config.n_jobs is None:
            return separatix.diagnose(embeddings, labels, **kwargs)

        profiler = separatix.ComplexityProfiler(
            budget=kwargs["budget"],
            max_dense_mb=kwargs["max_dense_mb"],
            max_samples=kwargs["max_samples"],
            random_state=kwargs["random_state"],
            n_jobs=self.config.n_jobs,
        )
        return profiler.fit(embeddings, labels).report()

    def _max_dense_mb(self) -> int:
        max_dense_bytes = self.config.max_dense_bytes
        if max_dense_bytes is None:
            max_dense_bytes = self.overlap_config.max_dense_bytes
        return max(1, int(ceil(max_dense_bytes / (1024**2))))


def _load_separatix() -> Any:
    try:
        import separatix
    except ImportError as exc:
        raise ImportError(
            "separatix is required for complexity diagnostics. Install dependencies with "
            "Poetry or install separatix directly."
        ) from exc
    return separatix


def _l2_normalize_for_separatix(value: Any) -> Any:
    if is_sparse_matrix(value):
        squared = value.multiply(value)
        norms = np.sqrt(np.asarray(squared.sum(axis=1)).reshape(-1))
        norms[norms == 0.0] = 1.0
        return value.multiply(1.0 / norms[:, None])
    return l2_normalize_rows(value)
