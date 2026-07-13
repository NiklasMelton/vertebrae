"""Simple recommendation rules for benchmark reports."""

from typing import Any, Dict, List, Optional


def recommendation_for_extractor(
    score: float,
    stability: Optional[Dict[str, Any]],
    weakest_class_score: Optional[float],
    target_type: str = "single_label",
) -> str:
    """Compute a recommendation label for one extractor.

    Args:
        score: Primary overlap score.
        stability: Optional stability summary.
        weakest_class_score: Lowest per-class score when available.

    Returns:
        Recommendation label string.
    """

    width = _stability_width(stability)
    if target_type == "regression":
        if _crosses_null_reference(stability, null_reference=0.5):
            return "continuous_overlap_null_indeterminate"
        if score >= 0.5:
            return "continuous_structure_above_null"
        return "continuous_overlap_below_null"
    if score >= 0.9 and width <= 0.05:
        recommendation = "strong_candidate"
    elif score >= 0.80:
        recommendation = "promising_inspect_weak_classes"
    elif score >= 0.75:
        recommendation = "moderate_overlap_fine_tuning_likely"
    else:
        recommendation = "poor_frozen_representation"

    if weakest_class_score is not None and weakest_class_score < 0.7:
        recommendation += "_weak_class_attention"
    if width > 0.15:
        recommendation += "_unstable"
    return recommendation


def recommendations_for_benchmark(
    extractor_results: List[Any],
    quality_tolerance: Optional[float] = None,
) -> List[str]:
    """Compute practitioner-facing recommendations for a benchmark.

    Args:
        extractor_results: Evaluated extractor results.

    Returns:
        Recommendation messages for the report executive summary.
    """

    if not extractor_results:
        return ["No extractors were evaluated."]
    ranked = sorted(
        extractor_results,
        key=lambda item: (
            item.primary_score
            if item.metrics.get(item.primary_metric_name, None) is None
            or item.metrics[item.primary_metric_name].higher_is_better
            else -item.primary_score
        ),
        reverse=True,
    )
    top = ranked[0]
    top_overlap = top.overlap
    top_target_type = (top_overlap.metadata if top_overlap else {}).get(
        "target_type",
        "single_label",
    )
    top_score_label = (
        "continuous overlap"
        if top.primary_metric_name == "overlap" and top_target_type == "regression"
        else "overlap macro"
        if top.primary_metric_name == "overlap"
        else top.primary_metric_name
    )
    messages = [
        f"Top representation under this protocol: {top.name} "
        f"({top_score_label} {float(top.primary_score):.3f})."
    ]
    if quality_tolerance is not None:
        top_rankable = _rankable_primary_score(top)
        cohort = [
            item
            for item in ranked
            if top_rankable - _rankable_primary_score(item) <= quality_tolerance
        ]
        if len(cohort) > 1:
            messages.append(
                f"{len(cohort)} representations are within {quality_tolerance:.3f} of the "
                "best primary score; compare their resource profiles before selection."
            )
    weak = [
        result
        for result in ranked
        if result.weakest_class_score is not None and result.weakest_class_score < 0.60
    ]
    if weak:
        messages.append("Inspect weak classes before treating the ranking as deployment-ready.")
    if any(_stability_width(result.stability) > 0.15 for result in ranked):
        messages.append(
            "One or more extractors has a wide stability interval; rerun or inspect seeds."
        )
    extractor_types = {result.extractor_type for result in ranked}
    if "frozen_pretrained" in extractor_types and (
        "unsupervised_fitted" in extractor_types or "supervised_fitted" in extractor_types
    ):
        messages.append(
            "This comparison includes both frozen pretrained extractors and fitted "
            "pipelines. Scores are comparable as representation diagnostics, but fitted "
            "pipelines may have used the benchmark data during feature construction."
        )
    return messages


def _rankable_primary_score(item: Any) -> float:
    metric = item.metrics.get(item.primary_metric_name)
    return (
        float(item.primary_score)
        if metric is None or metric.higher_is_better
        else -float(item.primary_score)
    )


def _stability_width(stability: Optional[Dict[str, Any]]) -> float:
    if not stability:
        return 0.0
    summary = stability.get("summary", {})
    return float(summary.get("width", 0.0))


def _crosses_null_reference(
    stability: Optional[Dict[str, Any]],
    null_reference: float,
) -> bool:
    if not stability:
        return False
    summary = stability.get("summary", {})
    lower = summary.get("lower")
    upper = summary.get("upper")
    if lower is None or upper is None:
        return False
    return float(lower) <= null_reference <= float(upper)
