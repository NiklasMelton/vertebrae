"""Prototype and target-aware subsample stability analysis."""

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np

from vertebrae.config import (
    ContinuousOverlapScoringConfig,
    OverlapScoringConfig,
    StabilityConfig,
)
from vertebrae.scoring.overlap import OverlapIndexScorer
from vertebrae.utils.labels import (
    MULTI_LABEL_TARGET,
    REGRESSION_TARGET,
    display_label,
    normalize_targets,
    regression_subsample_indices,
    stratified_label_indices,
)
from vertebrae.utils.validation import ensure_numeric_matrix


def run_stability_analysis(
    Z: Any,
    y: Any,
    scoring_config: Union[OverlapScoringConfig, ContinuousOverlapScoringConfig],
    stability_config: Optional[StabilityConfig] = None,
    label_names: Optional[Any] = None,
    target_type: str = "auto",
    target_names: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Run prototype or subsample stability analysis.

    Args:
        Z: Dense or sparse embedding matrix.
        y: Class labels.
        scoring_config: OverlapIndex scoring configuration.
        stability_config: Stability-analysis configuration.

    Returns:
        Stability summary dictionary, or `None` when disabled.
    """

    config = stability_config or StabilityConfig()
    if not config.enabled or config.mode == "none":
        return None

    embeddings = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
    labels, label_metadata = normalize_targets(
        y,
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    scoring_rng = np.random.default_rng(config.random_state)
    seeds = scoring_rng.integers(0, np.iinfo(np.int32).max, size=config.repeats).tolist()
    sampling_seeds: List[int] = []
    if config.mode == "subsample":
        _validate_subsample_feasibility(labels, config, label_metadata)
        sampling_rng = np.random.default_rng(np.random.SeedSequence([config.random_state, 1]))
        sampling_seeds = sampling_rng.integers(
            0, np.iinfo(np.int32).max, size=config.repeats
        ).tolist()
    scorer = OverlapIndexScorer(scoring_config)

    scores: List[float] = []
    per_class_values: Dict[str, List[float]] = {}
    warnings: List[str] = []
    effective_sample_counts: List[int] = []
    effective_subsample_fractions: List[float] = []
    for repeat_index, seed in enumerate(seeds):
        if config.mode == "prototype":
            repeat_Z, repeat_y = embeddings, labels
        elif config.mode == "subsample":
            indices = _subsample_indices(
                labels,
                config,
                sampling_seed=sampling_seeds[repeat_index],
                label_metadata=label_metadata,
            )
            repeat_Z, repeat_y = embeddings[indices], labels[indices]
            _validate_subsample_target(repeat_y, label_metadata)
            effective_sample_counts.append(int(len(indices)))
            effective_subsample_fractions.append(float(len(indices) / len(labels)))
        else:
            raise ValueError(f"Unsupported stability mode: {config.mode}")

        result = scorer.score(
            repeat_Z,
            repeat_y,
            seed=int(seed),
            label_names=label_metadata.get("label_names"),
            target_type=label_metadata["target_type"],
            target_names=label_metadata.get("target_names"),
        )
        scores.append(result.score)
        warnings.extend(result.warnings)
        for label, value in result.per_class_scores.items():
            if isinstance(value, (int, float, np.number)):
                per_class_values.setdefault(str(label), []).append(float(value))

    payload = {
        "mode": config.mode,
        "repeats": config.repeats,
        "interval_level": config.interval_level,
        "scores": scores,
        "summary": _summary(scores, config.interval_level),
        "per_class_summary": {
            label: _summary(values, config.interval_level)
            for label, values in per_class_values.items()
            if values
        },
        "warnings": sorted(set(warnings)),
        "seeds": [int(seed) for seed in seeds],
    }
    if config.mode == "subsample":
        payload.update(
            {
                "requested_subsample_fraction": config.subsample_fraction,
                "sampling_seeds": [int(seed) for seed in sampling_seeds],
                "effective_sample_counts": effective_sample_counts,
                "effective_subsample_fractions": effective_subsample_fractions,
            }
        )
    return payload


def _summary(scores: List[float], interval_level: float) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    alpha = 1.0 - interval_level
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "lower": float(np.percentile(arr, lower_q)),
        "upper": float(np.percentile(arr, upper_q)),
        "width": float(np.percentile(arr, upper_q) - np.percentile(arr, lower_q)),
    }


def _subsample_indices(
    labels: np.ndarray,
    config: StabilityConfig,
    sampling_seed: int,
    label_metadata: Dict[str, Any],
) -> np.ndarray:
    if label_metadata["target_type"] != REGRESSION_TARGET:
        return stratified_label_indices(
            labels,
            rate=config.subsample_fraction,
            random_state=sampling_seed,
            min_samples_per_class=2,
            label_names=label_metadata.get("label_names"),
            target_type=label_metadata["target_type"],
            target_names=label_metadata.get("target_names"),
        )
    return regression_subsample_indices(
        labels,
        n_take=math.floor(len(labels) * config.subsample_fraction),
        random_state=sampling_seed,
    )


def _validate_subsample_feasibility(
    labels: np.ndarray,
    config: StabilityConfig,
    label_metadata: Dict[str, Any],
) -> None:
    """Reject subsample requests that cannot satisfy scoring target invariants."""

    fraction = config.subsample_fraction
    if len(labels) == 0:
        raise ValueError("Subsample stability requires at least one target row.")
    if label_metadata["target_type"] == REGRESSION_TARGET:
        n_take = math.floor(len(labels) * fraction)
        if n_take < 3:
            minimum = 3 / len(labels)
            raise ValueError(
                f"StabilityConfig.subsample_fraction={fraction} retains {n_take} regression "
                "rows, but regression stability requires at least 3. Increase "
                f"subsample_fraction to at least {minimum:.6g} or use mode='prototype'."
            )
        if not label_metadata.get("nonconstant_targets"):
            raise ValueError(
                "Regression stability requires at least one non-constant target. "
                "Use a valid regression target or disable stability."
            )
        return

    counts = label_metadata["class_counts"]
    if not counts:
        raise ValueError(
            "Categorical subsample stability requires at least one observed class or label."
        )
    insufficient = {
        display_label(label): int(count)
        for label, count in counts.items()
        if math.floor(count * fraction) < 2
    }
    if insufficient:
        minimum = max(2 / count if count else math.inf for count in counts.values())
        unit = "active label" if label_metadata["target_type"] == MULTI_LABEL_TARGET else "class"
        minimum_guidance = (
            "No fraction in (0, 1] can satisfy the source target counts"
            if not math.isfinite(minimum) or minimum > 1.0
            else f"Increase subsample_fraction to at least {minimum:.6g}"
        )
        raise ValueError(
            f"StabilityConfig.subsample_fraction={fraction} cannot retain at least 2 "
            f"samples for every {unit}; insufficient counts: {insufficient}. "
            f"{minimum_guidance} or use mode='prototype'."
        )


def _validate_subsample_target(
    labels: np.ndarray,
    source_metadata: Dict[str, Any],
) -> None:
    """Validate one generated subset immediately before scoring it."""

    _, subset_metadata = normalize_targets(
        labels,
        label_names=source_metadata.get("label_names"),
        target_type=source_metadata["target_type"],
        target_names=source_metadata.get("target_names"),
    )
    if source_metadata["target_type"] == REGRESSION_TARGET:
        if len(labels) < 3 or not subset_metadata.get("nonconstant_targets"):
            raise ValueError(
                "Generated regression stability subset is invalid: expected at least "
                "3 rows and one non-constant target. Increase subsample_fraction or "
                "use mode='prototype'."
            )
        return

    expected_labels = set(source_metadata["class_counts"])
    subset_counts = subset_metadata["class_counts"]
    invalid = {
        display_label(label): int(subset_counts.get(label, 0))
        for label in expected_labels
        if subset_counts.get(label, 0) < 2
    }
    if invalid:
        unit = (
            "active labels" if source_metadata["target_type"] == MULTI_LABEL_TARGET else "classes"
        )
        raise ValueError(
            "Generated stability subset is invalid: expected at least 2 samples for "
            f"all original {unit}; found {invalid}. Increase subsample_fraction or "
            "use mode='prototype'."
        )
