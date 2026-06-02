"""Simple recommendation rules for benchmark reports."""

from typing import Any, Dict, List, Optional


def recommendation_for_extractor(
    macro_score: float,
    stability: Optional[Dict[str, Any]],
    weakest_class_score: Optional[float],
) -> str:
    width = _stability_width(stability)
    if macro_score >= 0.85 and width <= 0.05:
        recommendation = "strong_candidate"
    elif macro_score >= 0.70:
        recommendation = "promising_inspect_weak_classes"
    elif macro_score >= 0.55:
        recommendation = "moderate_overlap_fine_tuning_likely"
    else:
        recommendation = "poor_frozen_representation"

    if weakest_class_score is not None and weakest_class_score < 0.60:
        recommendation += "_weak_class_attention"
    if width > 0.15:
        recommendation += "_unstable"
    return recommendation


def recommendations_for_benchmark(extractor_results: List[Any]) -> List[str]:
    if not extractor_results:
        return ["No extractors were evaluated."]
    ranked = sorted(extractor_results, key=lambda item: item.overlap.macro_score, reverse=True)
    top = ranked[0]
    messages = [
        f"Top representation under this protocol: {top.name} "
        f"(overlap macro {top.overlap.macro_score:.3f})."
    ]
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


def _stability_width(stability: Optional[Dict[str, Any]]) -> float:
    if not stability:
        return 0.0
    summary = stability.get("summary", {})
    return float(summary.get("width", 0.0))
