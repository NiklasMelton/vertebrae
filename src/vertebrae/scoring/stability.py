"""Prototype and subsample stability analysis."""

from typing import Any, Dict, List, Optional

import numpy as np

from vertebrae.config import OverlapScoringConfig, StabilityConfig
from vertebrae.scoring.overlap import OverlapIndexScorer
from vertebrae.utils.labels import class_counts


def run_stability_analysis(
    Z: Any,
    y: Any,
    scoring_config: OverlapScoringConfig,
    stability_config: Optional[StabilityConfig] = None,
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

    embeddings = np.asarray(Z)
    labels = np.asarray(y)
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
            indices = _subsample_indices(labels, config, rng)
            repeat_Z, repeat_y = embeddings[indices], labels[indices]
        else:
            raise ValueError(f"Unsupported stability mode: {config.mode}")

        result = scorer.score(repeat_Z, repeat_y, seed=int(seed))
        scores.append(result.macro_score)
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
) -> np.ndarray:
    n_samples = len(labels)
    if config.stratified:
        indices: List[int] = []
        for label in class_counts(labels):
            class_indices = np.flatnonzero(labels == label)
            n_take = max(2, int(round(len(class_indices) * config.subsample_fraction)))
            n_take = min(len(class_indices), n_take)
            indices.extend(rng.choice(class_indices, size=n_take, replace=False).tolist())
        return np.asarray(indices, dtype=int)

    n_take = max(2, int(round(n_samples * config.subsample_fraction)))
    n_take = min(n_samples, n_take)
    return rng.choice(np.arange(n_samples), size=n_take, replace=False)
