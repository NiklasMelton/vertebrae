"""Lightweight probe classifier evaluation."""

from typing import Any, Dict, List, Optional

import numpy as np

from vertebrae.config import ProbeConfig
from vertebrae.utils.labels import class_counts


def run_probes(Z: Any, y: Any, config: Optional[ProbeConfig] = None) -> Optional[Dict[str, Any]]:
    """Evaluate lightweight probe classifiers on embeddings.

    Args:
        Z: Dense or sparse embedding matrix.
        y: Class labels.
        config: Probe configuration.

    Returns:
        Probe result dictionary, or `None` when disabled.
    """

    probe_config = config or ProbeConfig()
    if not probe_config.enabled:
        return None

    embeddings = np.asarray(Z)
    labels = np.asarray(y)
    counts = class_counts(labels)
    warnings: List[str] = []
    if min(counts.values()) < 2:
        return {
            "enabled": False,
            "warnings": ["Probe evaluation requires at least 2 samples per class."],
            "results": {},
        }

    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError("Probe evaluation requires scikit-learn.") from exc

    stratify = labels if _can_stratify(labels, probe_config.test_size) else None
    if stratify is None:
        warnings.append(
            "Probe split is non-stratified because classes are too small "
            "for the requested test_size."
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=probe_config.test_size,
            random_state=probe_config.random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        return {
            "enabled": False,
            "warnings": [f"Probe evaluation disabled: {exc}"],
            "results": {},
        }

    results: Dict[str, Dict[str, float]] = {}
    for method in probe_config.methods:
        model = _build_probe(method, y_train)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[method] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        }

    return {
        "enabled": True,
        "test_size": probe_config.test_size,
        "methods": list(probe_config.methods),
        "warnings": warnings,
        "results": results,
    }


def _can_stratify(labels: np.ndarray, test_size: float) -> bool:
    counts = class_counts(labels)
    n_classes = len(counts)
    n_test = int(np.ceil(len(labels) * test_size))
    n_train = len(labels) - n_test
    return min(counts.values()) >= 2 and n_test >= n_classes and n_train >= n_classes


def _build_probe(method: str, y_train: np.ndarray) -> Any:
    if method == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        min_class_count = min(class_counts(y_train).values())
        return KNeighborsClassifier(n_neighbors=max(1, min(5, min_class_count)))
    if method == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1_000)
    if method == "nearest_centroid":
        from sklearn.neighbors import NearestCentroid

        return NearestCentroid()
    raise ValueError(f"Unknown probe method: {method}")
