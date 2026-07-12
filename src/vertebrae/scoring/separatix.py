"""Internal Separatix adapter."""

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional, Union

import numpy as np

from vertebrae.config import (
    ContinuousOverlapScoringConfig,
    OverlapScoringConfig,
    SeparatixConfig,
)
from vertebrae.utils.labels import REGRESSION_TARGET, metric_labels, target_summary
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix, l2_normalize_rows


@dataclass
class SeparatixResult:
    """Structured result from Separatix complexity diagnostics."""

    ran: bool
    probe_summary: Dict[str, Any]
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
        overlap_config: Optional[
            Union[OverlapScoringConfig, ContinuousOverlapScoringConfig]
        ] = None,
    ) -> None:
        self.config = config or SeparatixConfig()
        self.overlap_config = overlap_config or OverlapScoringConfig()

    def score(
        self,
        Z: Any,
        y: Any,
        label_names: Optional[Any] = None,
        groups: Optional[Any] = None,
        target_type: str = "auto",
        target_names: Optional[Any] = None,
    ) -> SeparatixResult:
        """Run Separatix on dense or sparse embeddings and labels."""

        embeddings = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
        labels, label_metadata = metric_labels(
            y,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        if embeddings.shape[0] != len(labels):
            raise ValueError(
                "embeddings and labels must have the same length; "
                f"got {embeddings.shape[0]} and {len(labels)}."
            )
        normalized_groups = _validate_groups(groups, len(labels))

        sparse_input = is_sparse_matrix(embeddings)
        normalize_embeddings = self.overlap_config.normalize_embeddings
        if normalize_embeddings:
            embeddings = _l2_normalize_for_separatix(embeddings)

        report = self._run_separatix(embeddings, labels, label_metadata, normalized_groups)
        report_dict = make_json_safe(report.to_dict())
        summary = target_summary(
            y,
            label_names=label_metadata.get("label_names"),
            target_type=label_metadata["target_type"],
            target_names=label_metadata.get("target_names"),
        )
        grouped = normalized_groups is not None
        n_groups = (
            int(len(np.unique(normalized_groups))) if normalized_groups is not None else None
        )
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
            probe_summary=summarize_probe_diagnostics(
                report_dict,
                target_type=label_metadata["target_type"],
                grouped=grouped,
                n_groups=n_groups,
            ),
            metadata={
                "normalized_embeddings": normalize_embeddings,
                "sparse_input": sparse_input,
                "budget": self.config.budget or "standard",
                "max_samples": self.config.max_samples,
                "max_dense_mb": self._max_dense_mb(),
                "n_jobs": self.config.n_jobs,
                "target_type": label_metadata["target_type"],
                "label_names": label_metadata.get("label_names"),
                "target_names": label_metadata.get("target_names"),
                "target_summary": summary,
                "grouped": grouped,
                "n_groups": n_groups,
            },
        )

    def skipped_result(
        self,
        reason: str,
        overlap_score: float,
        threshold: float,
    ) -> SeparatixResult:
        """Return a structured skipped result for gated runs."""

        return SeparatixResult(
            ran=False,
            skipped_reason=reason,
            probe_summary={
                "status": "skipped",
                "best_probe": None,
                "primary_metric": None,
                "metrics": {},
                "comparison": None,
                "evaluation": {},
                "skip_reason": reason,
            },
            metadata={
                "normalized_embeddings": self.overlap_config.normalize_embeddings,
                "overlap_score": float(overlap_score),
                "overlap_threshold": float(threshold),
            },
        )

    def _run_separatix(
        self,
        embeddings: Any,
        labels: np.ndarray,
        label_metadata: Dict[str, Any],
        groups: Optional[np.ndarray],
    ) -> Any:
        separatix = _load_separatix()
        kwargs: Dict[str, Any] = {
            "return_report": True,
            "random_state": self.config.random_state,
            "budget": self.config.budget or "standard",
            "max_dense_mb": self._max_dense_mb(),
            "max_samples": self.config.max_samples,
            "mlp_probes": self.config.mlp_probes,
            "mlp_device": self.config.mlp_device,
            "mlp_trigger_skill_threshold": self.config.mlp_trigger_skill_threshold,
            "mlp_min_improvement": self.config.mlp_min_improvement,
            "mlp_max_parameters": self.config.mlp_max_parameters,
        }
        if label_metadata["target_type"] == "multi_label":
            kwargs["target_mode"] = "multilabel"
        elif label_metadata["target_type"] == REGRESSION_TARGET:
            kwargs["target_mode"] = "regression"
        else:
            kwargs["target_mode"] = "singlelabel"
        if groups is not None:
            kwargs["groups"] = groups
        if self.config.n_jobs is None:
            return separatix.diagnose(embeddings, labels, **kwargs)

        profiler = separatix.ComplexityProfiler(
            target_mode=kwargs["target_mode"],
            budget=kwargs["budget"],
            max_dense_mb=kwargs["max_dense_mb"],
            max_samples=kwargs["max_samples"],
            random_state=kwargs["random_state"],
            n_jobs=self.config.n_jobs,
            mlp_probes=self.config.mlp_probes,
            mlp_device=self.config.mlp_device,
            mlp_trigger_skill_threshold=self.config.mlp_trigger_skill_threshold,
            mlp_min_improvement=self.config.mlp_min_improvement,
            mlp_max_parameters=self.config.mlp_max_parameters,
        )
        fit_kwargs: Dict[str, Any] = {}
        if groups is not None:
            fit_kwargs["groups"] = groups
        fitted = profiler.fit(embeddings, labels, **fit_kwargs)
        return fitted.report()

    def _max_dense_mb(self) -> int:
        max_dense_bytes = self.config.max_dense_bytes
        if max_dense_bytes is None:
            max_dense_bytes = self.overlap_config.max_dense_bytes
        return max(1, int(ceil(max_dense_bytes / (1024**2))))


_PROBE_METRICS = {
    "single_label": ("balanced_accuracy", "accuracy", "macro_f1"),
    "multi_label": (
        "micro_f1",
        "macro_f1",
        "sample_jaccard",
        "samples_jaccard",
        "jaccard_samples",
        "subset_accuracy",
    ),
    "regression": (
        "r2",
        "mae",
        "rmse",
        "mean_absolute_error",
        "root_mean_squared_error",
    ),
}

_NONLINEAR_PROBES = ("smooth_poly", "kernel_approx", "knn")


def probe_summary_for_result(result: Optional[SeparatixResult]) -> Dict[str, Any]:
    """Return the required probe summary or a disabled diagnostic summary."""

    if result is None:
        return {
            "status": "disabled",
            "best_probe": None,
            "primary_metric": None,
            "metrics": {},
            "comparison": None,
            "evaluation": {},
            "skip_reason": "Separatix diagnostics were disabled.",
        }
    return result.probe_summary


def summarize_probe_diagnostics(
    report: Dict[str, Any],
    *,
    target_type: str,
    grouped: bool,
    n_groups: Optional[int],
) -> Dict[str, Any]:
    """Normalize existing Separatix probe evidence without fitting new models."""

    report_metrics = report.get("metrics", {}) or {}
    baseline = report_metrics.get("baseline", {}) or {}
    probes = report_metrics.get("probes", {}) or {}
    best_probe = baseline.get("best_probe") or baseline.get("recommended_family")
    best_metrics = probes.get(best_probe, {}) if best_probe else {}
    metric_names = _PROBE_METRICS.get(target_type, ())
    metric_map = {
        name: float(best_metrics[name])
        for name in metric_names
        if _is_number(best_metrics.get(name))
    }

    primary_name = (
        baseline.get("best_probe_metric")
        or baseline.get("primary_metric")
        or baseline.get("metric_name")
    )
    if primary_name is None and target_type == "single_label":
        primary_name = "balanced_accuracy" if "balanced_accuracy" in metric_map else None
    primary_value = None
    if primary_name and _is_number(best_metrics.get(primary_name)):
        primary_value = float(best_metrics[primary_name])
    elif primary_name and _is_number(baseline.get("best_probe_score")):
        primary_value = float(baseline["best_probe_score"])
    primary_metric = (
        {"name": str(primary_name), "value": primary_value}
        if primary_name is not None and primary_value is not None
        else None
    )

    skip_reason = best_metrics.get("skipped_reason") if best_probe else None
    if best_probe is None:
        skip_reason = baseline.get("skipped_reason") or "Separatix did not identify a best probe."
    status = "executed" if best_probe and metric_map else "unavailable"
    evaluation = {
        "mode": best_metrics.get("evaluation_mode"),
        "sampling": best_metrics.get("sample_info") or (report.get("sampling", {}) or {}).get(
            "probe"
        ),
        "grouped": grouped,
        "n_groups": n_groups,
    }
    return {
        "status": status,
        "target_type": target_type,
        "best_probe": best_probe,
        "primary_metric": primary_metric,
        "metrics": metric_map,
        "comparison": _probe_comparison(report_metrics, probes, primary_metric),
        "evaluation": evaluation,
        "skip_reason": skip_reason,
    }


def _probe_comparison(
    metrics: Dict[str, Any],
    probes: Dict[str, Any],
    primary_metric: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    explicit = (metrics.get("baseline", {}) or {}).get("linear_nonlinear_comparison")
    if not explicit:
        explicit = metrics.get("linear_nonlinear_comparison")
    if explicit:
        return make_json_safe(explicit)
    if not primary_metric or "linear" not in probes:
        return None
    metric_name = primary_metric["name"]
    linear = probes.get("linear", {}) or {}
    nonlinear_candidates = [
        name
        for name in _NONLINEAR_PROBES
        if _is_number((probes.get(name, {}) or {}).get(metric_name))
    ]
    nonlinear_name = (
        max(nonlinear_candidates, key=lambda name: float(probes[name][metric_name]))
        if nonlinear_candidates
        else None
    )
    if nonlinear_name is None or not _is_number(linear.get(metric_name)):
        return None
    nonlinear = probes[nonlinear_name]
    if linear.get("evaluation_mode") != nonlinear.get("evaluation_mode"):
        return None
    linear_value = float(linear[metric_name])
    nonlinear_value = float(nonlinear[metric_name])
    delta = nonlinear_value - linear_value
    return {
        "linear_probe": "linear",
        "nonlinear_probe": nonlinear_name,
        "metric": metric_name,
        "linear_value": linear_value,
        "nonlinear_value": nonlinear_value,
        "delta": delta,
        "favored_family": "nonlinear" if delta > 0 else "linear" if delta < 0 else "tie",
        "confidence": None,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _load_separatix() -> Any:
    try:
        import separatix
    except ImportError as exc:
        raise ImportError(
            "separatix>=0.1.0a3 is required for complexity diagnostics. Install dependencies with "
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


def _validate_groups(groups: Optional[Any], n_samples: int) -> Optional[np.ndarray]:
    if groups is None:
        return None
    array = np.asarray(groups)
    if array.ndim != 1:
        raise ValueError("groups must be one-dimensional.")
    if len(array) != n_samples:
        raise ValueError(
            f"groups and labels must have the same length; got {len(array)} and {n_samples}."
        )
    return array
