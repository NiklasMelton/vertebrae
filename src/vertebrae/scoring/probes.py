"""Optional lightweight native probe classifier evaluation."""

from typing import Any, Dict, List, Optional

import numpy as np

from vertebrae.config import ProbeConfig
from vertebrae.utils.labels import REGRESSION_TARGET, class_counts, normalize_targets


def run_probes(
    Z: Any,
    y: Any,
    config: Optional[ProbeConfig] = None,
    groups: Optional[Any] = None,
    target_type: str = "auto",
    target_names: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
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

    normalized_labels, label_metadata = normalize_targets(
        y,
        target_type=target_type,
        target_names=target_names,
    )
    if label_metadata["target_type"] == "multi_label":
        return {
            "enabled": False,
            "target_type": "multi_label",
            "warnings": [
                "Native vertebrae probes are currently single-label only; "
                "probe evaluation was skipped for this multi-label dataset."
            ],
            "results": {},
        }
    if label_metadata["target_type"] == REGRESSION_TARGET:
        return {
            "enabled": False,
            "target_type": REGRESSION_TARGET,
            "warnings": [
                "Native vertebrae probes are currently classification-only; "
                "probe evaluation was skipped for this regression dataset."
            ],
            "results": {},
        }

    embeddings = np.asarray(Z)
    labels = normalized_labels
    counts = class_counts(labels, target_type=label_metadata["target_type"])
    warnings: List[str] = []
    if not counts:
        return {
            "enabled": False,
            "warnings": ["Probe evaluation has no non-excluded classes to evaluate."],
            "results": {},
        }
    if min(counts.values()) < 2:
        return {
            "enabled": False,
            "warnings": ["Probe evaluation requires at least 2 samples per class."],
            "results": {},
        }

    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.model_selection import GroupShuffleSplit, train_test_split
    except ImportError as exc:
        raise ImportError("Probe evaluation requires scikit-learn.") from exc

    stratify = labels if _can_stratify(labels, probe_config.test_size) else None
    if stratify is None:
        warnings.append(
            "Probe split is non-stratified because classes are too small "
            "for the requested test_size."
        )

    try:
        if groups is not None:
            group_array = np.asarray(groups)
            if group_array.ndim != 1 or len(group_array) != len(labels):
                raise ValueError("groups must be one-dimensional and aligned to labels.")
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=probe_config.test_size,
                random_state=probe_config.random_state,
            )
            train_indices, test_indices = next(splitter.split(embeddings, labels, group_array))
            X_train, X_test = embeddings[train_indices], embeddings[test_indices]
            y_train, y_test = labels[train_indices], labels[test_indices]
        else:
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
    if len(class_counts(y_train)) < 2:
        return {
            "enabled": False,
            "grouped": groups is not None,
            "warnings": [
                "Probe evaluation disabled because the training split contains fewer "
                "than two classes."
            ],
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
        "grouped": groups is not None,
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
