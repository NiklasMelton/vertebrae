"""Prototype and subsample stability analysis."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from vertebrae.config import (
    ContinuousOverlapScoringConfig,
    OverlapScoringConfig,
    StabilityConfig,
)
from vertebrae.scoring.overlap import OverlapIndexScorer
from vertebrae.utils.labels import (
    REGRESSION_TARGET,
    normalize_targets,
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
    if label_metadata["target_type"] == REGRESSION_TARGET and config.stratified:
        raise ValueError(
            "StabilityConfig(stratified=True) is not supported for regression targets."
        )
    rng = np.random.default_rng(config.random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=config.repeats).tolist()
    scorer = OverlapIndexScorer(scoring_config)

    scores: List[float] = []
    per_class_values: Dict[str, List[float]] = {}
    warnings: List[str] = []
    for seed in seeds:
        if config.mode == "prototype":
            repeat_Z, repeat_y = embeddings, labels
        elif config.mode == "subsample":
            indices = _subsample_indices(labels, config, rng, label_metadata)
            repeat_Z, repeat_y = embeddings[indices], labels[indices]
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

    return {
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
    rng: np.random.Generator,
    label_metadata: Dict[str, Any],
) -> np.ndarray:
    n_samples = len(labels)
    if config.stratified:
        return stratified_label_indices(
            labels,
            rate=config.subsample_fraction,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            min_samples_per_class=2,
            label_names=label_metadata.get("label_names"),
            target_type=label_metadata["target_type"],
            target_names=label_metadata.get("target_names"),
        )

    n_take = max(2, int(round(n_samples * config.subsample_fraction)))
    n_take = min(n_samples, n_take)
    return rng.choice(np.arange(n_samples), size=n_take, replace=False)
