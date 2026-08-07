"""Internal Separatix adapter."""

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

from vertebrae.config import (
    ContinuousOverlapScoringConfig,
    OverlapScoringConfig,
    SeparatixConfig,
)
from vertebrae.utils.labels import REGRESSION_TARGET, metric_labels, target_summary
from vertebrae.utils.semantic_labels import semantic_label_key
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix, l2_normalize_rows


@dataclass
class SeparatixResult:
    """Structured result from Separatix complexity diagnostics."""

    ran: bool
    probe_summary: Dict[str, Any]
    family_guidance: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None
    recommendation_text: Optional[str] = None
    confidence: Optional[str] = None
    scores: Dict[str, Any] = field(default_factory=dict)
    decision_path: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    densification_events: List[Dict[str, Any]] = field(default_factory=list)
    skipped_reason: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the diagnostic result to a JSON-safe dictionary."""

        return make_json_safe(self)

    def probe_recipe(self, probe_name_or_recipe_id: str) -> Optional[Dict[str, Any]]:
        """Return a retained Separatix probe recipe by name or stable id.

        Separatix 0.1.1 emits recipes for the core probes as well as optional
        aligned comparators and MLP architectures.  Vertebrae keeps the raw
        report untouched, so this lookup intentionally searches those nested
        report payloads rather than trying to reconstruct estimators locally.
        The returned recipe is a JSON-safe copy and can be passed directly to
        Separatix' ``make_probe_estimator`` factory.
        """

        if not probe_name_or_recipe_id or not self.report:
            return None
        wanted = str(probe_name_or_recipe_id)
        for recipe in _iter_probe_recipes(self.report):
            recipe_id = recipe.get("recipe_id")
            probe = recipe.get("probe") or {}
            probe_name = probe.get("name") if isinstance(probe, dict) else None
            if wanted in {recipe_id, probe_name}:
                return make_json_safe(recipe)
        return None

    def selected_probe_recipe(self) -> Optional[Dict[str, Any]]:
        """Return the recipe used by the normalized family guidance."""

        recipe_id = self.family_guidance.get("selected_recipe_id")
        if recipe_id:
            recipe = self.probe_recipe(str(recipe_id))
            if recipe is not None:
                return recipe
        probe_name = self.family_guidance.get("selected_probe")
        return self.probe_recipe(str(probe_name)) if probe_name else None


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
        label_rows = int(labels.shape[0])
        if embeddings.shape[0] != label_rows:
            raise ValueError(
                "embeddings and labels must have the same length; "
                f"got {embeddings.shape[0]} and {label_rows}."
            )
        normalized_groups = _validate_groups(groups, label_rows)

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
        n_groups = int(len(np.unique(normalized_groups))) if normalized_groups is not None else None
        return SeparatixResult(
            ran=True,
            recommendation=report_dict.get("recommendation"),
            recommendation_text=report_dict.get("recommendation_text"),
            confidence=report_dict.get("confidence"),
            scores=report_dict.get("scores", {}),
            decision_path=report_dict.get("decision_path", []),
            warnings=report_dict.get("warnings", []),
            skipped_diagnostics=report_dict.get("skipped_diagnostics", []),
            preprocessing=report_dict.get("preprocessing", {}),
            densification_events=report_dict.get("densification_events", []),
            report=report_dict,
            probe_summary=summarize_probe_diagnostics(
                report_dict,
                target_type=label_metadata["target_type"],
                grouped=grouped,
                n_groups=n_groups,
            ),
            family_guidance=normalize_family_guidance(
                report_dict,
                target_type=label_metadata["target_type"],
            ),
            metadata={
                "normalized_embeddings": normalize_embeddings,
                "sparse_input": sparse_input,
                "budget": self.config.budget or "standard",
                "max_samples": self.config.max_samples,
                "densify_policy": self.config.densify_policy,
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
            family_guidance=normalize_family_guidance(
                {},
                target_type="unknown",
                status="skipped",
                reason=reason,
            ),
            metadata={
                "normalized_embeddings": self.overlap_config.normalize_embeddings,
                "overlap_score": float(overlap_score),
                "overlap_threshold": float(threshold),
            },
        )

    def _run_separatix(
        self,
        embeddings: Any,
        labels: Any,
        label_metadata: Dict[str, Any],
        groups: Optional[np.ndarray],
    ) -> Any:
        separatix = _load_separatix()
        kwargs: Dict[str, Any] = {
            "return_report": True,
            "random_state": self.config.random_state,
            "budget": self.config.budget or "standard",
            "densify_policy": self.config.densify_policy,
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
            densify_policy=kwargs["densify_policy"],
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
        "r2_variance_weighted",
        "r2_uniform_average",
        "normalized_rmse_mean",
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
    best_probe = baseline.get("best_probe")
    best_by_metric = baseline.get("best_by_metric", {}) or {}
    selected_baseline_metric = None
    if best_probe is None and isinstance(best_by_metric, dict):
        for metric_name in _PROBE_METRICS.get(target_type, ()):
            candidate = best_by_metric.get(metric_name, {})
            if isinstance(candidate, dict) and candidate.get("probe"):
                best_probe = candidate["probe"]
                selected_baseline_metric = metric_name
                break
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
        or selected_baseline_metric
    )
    if primary_name is None and target_type == "single_label":
        primary_name = "balanced_accuracy" if "balanced_accuracy" in metric_map else None
    primary_value = None
    if primary_name and _is_number(best_metrics.get(primary_name)):
        primary_value = float(best_metrics[primary_name])
    elif primary_name and _is_number(baseline.get("best_probe_score")):
        primary_value = float(baseline["best_probe_score"])
    elif primary_name and isinstance(best_by_metric.get(primary_name), dict):
        candidate_score = best_by_metric[primary_name].get("score")
        if _is_number(candidate_score):
            primary_value = float(candidate_score)
    primary_metric = (
        {"name": str(primary_name), "value": primary_value}
        if primary_name is not None and primary_value is not None
        else None
    )

    skip_reason = best_metrics.get("skipped_reason") if best_probe else None
    if best_probe is None:
        skip_reason = baseline.get("skipped_reason") or "Separatix did not identify a best probe."
    status = "executed" if best_probe and metric_map else "unavailable"
    evaluation_report = report_metrics.get("probe_evaluation", {}) or {}
    sampling = best_metrics.get("sample_info") or (report.get("sampling", {}) or {}).get(
        "probe"
    )
    if not isinstance(sampling, dict):
        sampling = {}
    evaluation_plan_id = evaluation_report.get("evaluation_plan_id") or best_metrics.get(
        "evaluation_plan_id"
    )
    alignment_status = evaluation_report.get("alignment_status") or best_metrics.get(
        "alignment_status"
    )
    if alignment_status is None:
        alignment_status = "aligned" if evaluation_plan_id else "unavailable"
    cohort_size = evaluation_report.get("n_samples")
    if cohort_size is None:
        cohort_size = sampling.get("n_used") or sampling.get("n_original")
    cohort_size_int = _integer_or_none(cohort_size)
    evaluation = {
        "mode": best_metrics.get("evaluation_mode")
        or evaluation_report.get("evaluation_mode"),
        "sampling": sampling,
        "grouped": grouped,
        "n_groups": n_groups,
        "alignment_status": alignment_status,
        "evaluation_plan_id": evaluation_plan_id,
        "cv_method": evaluation_report.get("cv_method")
        or best_metrics.get("cv_stratification_method"),
        "cohort_size": cohort_size_int,
        "n_samples": cohort_size_int,
        "n_splits": evaluation_report.get("n_splits"),
        "group_aware": evaluation_report.get("group_aware"),
        "effective_train_size_summary": evaluation_report.get(
            "effective_train_size_summary"
        ),
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


def normalize_family_guidance(
    report: Dict[str, Any],
    *,
    target_type: str,
    status: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize Separatix' target-specific evidence into one stable shape.

    Separatix uses separate evidence keys for single-label, multilabel, and
    regression reports.  Their family meanings are shared, so integrations
    should consume this compact normalized frontier instead of branching on
    target mode.  Unknown or older reports produce an explicit unavailable
    object rather than a guessed recommendation.
    """

    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    canonical_target_type = {
        "singlelabel": "single_label",
        "multilabel": "multi_label",
    }.get(target_type, target_type)
    evidence_keys = {
        "single_label": "recommendation_evidence",
        "singlelabel": "recommendation_evidence",
        "multi_label": "multilabel_recommendation_evidence",
        "multilabel": "multilabel_recommendation_evidence",
        "regression": "regression_recommendation_evidence",
    }
    evidence = metrics.get(
        evidence_keys.get(canonical_target_type, "recommendation_evidence"), {}
    )
    if not isinstance(evidence, dict):
        evidence = {}
    family_set = evidence.get("plausible_family_set", {})
    if not isinstance(family_set, dict):
        family_set = {}
    mlp_payload = metrics.get("mlp_recommendation_evidence", {})
    if not isinstance(mlp_payload, dict):
        mlp_payload = {}
    mlp_active = bool(mlp_payload.get("recommendation_override"))
    best_architecture = mlp_payload.get("best_architecture")
    if not isinstance(best_architecture, dict):
        best_architecture = {}

    minimum_family = family_set.get("minimum_recommended_family")
    if minimum_family is None:
        minimum_family = evidence.get("recommended_family")
    plausible = family_set.get("plausible_families", [])
    if not isinstance(plausible, list):
        plausible = list(plausible) if isinstance(plausible, (tuple, set)) else []
    selected_family = evidence.get("selected_family") or evidence.get("recommended_family")
    selected_probe = None
    if selected_family and selected_family != "mlp":
        family_payload = (evidence.get("families", {}) or {}).get(selected_family, {})
        if isinstance(family_payload, dict):
            selected_probe = family_payload.get("best_probe")
            if selected_probe is None:
                # Multilabel and regression evidence stores one probe choice
                # per primary metric instead of a flat family summary.
                metric_order = (
                    "balanced_accuracy",
                    "micro_f1",
                    "macro_f1",
                    "sample_jaccard",
                    "r2_variance_weighted",
                    "r2_uniform_average",
                )
                for metric_name in metric_order:
                    metric_payload = family_payload.get(metric_name, {})
                    if isinstance(metric_payload, dict) and metric_payload.get("probe"):
                        selected_probe = metric_payload["probe"]
                        break
    selected_recipe_id = None
    if selected_probe:
        selected_probe_payload = (metrics.get("probes", {}) or {}).get(selected_probe, {})
        if isinstance(selected_probe_payload, dict):
            selected_recipe = selected_probe_payload.get("probe_recipe")
            if isinstance(selected_recipe, dict):
                selected_recipe_id = selected_recipe.get("recipe_id")
    if mlp_active:
        selected_family = "mlp"
        selected_probe = best_architecture.get("probe_name") or selected_probe
        selected_recipe_id = best_architecture.get("probe_recipe_id")
    paired = metrics.get("paired_probe_comparisons", {})
    if not isinstance(paired, dict):
        paired = {}
    paired_status = paired.get("status")
    paired_method = paired.get("method")
    if paired_status is None and evidence.get("family_comparisons"):
        paired_status = (
            "available"
            if evidence.get("decision_method") == "paired_oof_bootstrap"
            else "unavailable"
        )
    inferred_status = status or family_set.get("status") or (
        "available" if evidence else "unavailable"
    )
    return {
        "status": inferred_status,
        "target_type": canonical_target_type,
        "scope": family_set.get("scope", "core_probe_families"),
        "minimum_recommended_family": minimum_family,
        "plausible_families": [str(value) for value in plausible],
        "decision_method": family_set.get("decision_method") or evidence.get("decision_method"),
        "selected_family": selected_family,
        "selected_probe": selected_probe,
        "selected_recipe_id": selected_recipe_id if selected_family else None,
        "mlp_override": mlp_active,
        "mlp_override_details": {
            "active": mlp_active,
            "recommendation_override": mlp_active,
            "probe_name": best_architecture.get("probe_name"),
            "probe_recipe_id": best_architecture.get("probe_recipe_id"),
            "status": mlp_payload.get("status"),
            "reason": mlp_payload.get("override_reason"),
        },
        "paired": {
            "status": paired_status,
            "method": paired_method,
            "evaluation_plan_id": paired.get("evaluation_plan_id"),
            "resamples_used": paired.get("resamples_used"),
        },
        "paired_status": paired_status,
        "paired_method": paired_method,
        "reason": reason or family_set.get("reason"),
    }


def _iter_probe_recipes(value: Any) -> Iterator[Dict[str, Any]]:
    """Yield nested probe recipes from a serialized Separatix report."""

    if isinstance(value, dict):
        recipe = value.get("probe_recipe")
        if isinstance(recipe, dict):
            yield recipe
        for child in value.values():
            yield from _iter_probe_recipes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_probe_recipes(child)


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
    higher_is_better = _probe_metric_higher_is_better(str(metric_name))
    linear = probes.get("linear", {}) or {}
    nonlinear_candidates = [
        name
        for name in _NONLINEAR_PROBES
        if _is_number((probes.get(name, {}) or {}).get(metric_name))
    ]
    selector = max if higher_is_better else min
    nonlinear_name = (
        selector(nonlinear_candidates, key=lambda name: float(probes[name][metric_name]))
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
    raw_delta = nonlinear_value - linear_value
    improvement = raw_delta if higher_is_better else -raw_delta
    return {
        "linear_probe": "linear",
        "nonlinear_probe": nonlinear_name,
        "metric": metric_name,
        "linear_value": linear_value,
        "nonlinear_value": nonlinear_value,
        "higher_is_better": higher_is_better,
        "raw_delta": raw_delta,
        "delta": improvement,
        "favored_family": (
            "nonlinear" if improvement > 0 else "linear" if improvement < 0 else "tie"
        ),
        "confidence": None,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _integer_or_none(value: Any) -> Optional[int]:
    """Convert a finite numeric value to an integer for compact metadata."""

    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return int(value)
    return None


def _probe_metric_higher_is_better(metric_name: str) -> bool:
    return metric_name.lower() not in {
        "mae",
        "rmse",
        "mean_absolute_error",
        "root_mean_squared_error",
    }


def _load_separatix() -> Any:
    try:
        import separatix
    except ImportError as exc:
        raise ImportError(
            "separatix>=0.1.1 is required for complexity diagnostics. Install dependencies with "
            "Poetry or install separatix directly."
        ) from exc
    return separatix


def _l2_normalize_for_separatix(value: Any) -> Any:
    return l2_normalize_rows(value)


def _validate_groups(groups: Optional[Any], n_samples: int) -> Optional[np.ndarray]:
    if groups is None:
        return None
    array = np.asarray(groups, dtype=object)
    if array.ndim != 1:
        raise ValueError("groups must be one-dimensional.")
    if len(array) != n_samples:
        raise ValueError(
            f"groups and labels must have the same length; got {len(array)} and {n_samples}."
        )
    keys = []
    for value in array.tolist():
        normalized = value.item() if hasattr(value, "item") else value
        if normalized is None or (
            isinstance(normalized, (float, complex, np.floating, np.complexfloating))
            and not bool(np.isfinite(normalized))
        ):
            raise ValueError("groups values must be non-missing and finite.")
        if hasattr(normalized, "is_finite") and callable(normalized.is_finite):
            if not bool(normalized.is_finite()):
                raise ValueError("groups values must be non-missing and finite.")
        try:
            keys.append(semantic_label_key(normalized))
        except TypeError as exc:
            raise ValueError(
                "groups values must have deterministic exact semantic identities."
            ) from exc
    return np.asarray(keys, dtype=object)
