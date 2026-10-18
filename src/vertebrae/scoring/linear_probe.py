"""Fixed-capacity out-of-fold linear-probe scoring."""

import warnings as python_warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vertebrae.config import LinearProbeConfig
from vertebrae.utils.labels import (
    MULTI_LABEL_TARGET,
    REGRESSION_TARGET,
    SINGLE_LABEL_TARGET,
    multilabel_indicator,
    normalize_targets,
)
from vertebrae.utils.semantic_labels import (
    canonical_semantic_array,
    semantic_label_key,
    semantic_label_keys,
)
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_numeric_matrix, l2_normalize_rows

_TARGET_METRICS = {
    SINGLE_LABEL_TARGET: ("balanced_accuracy", "accuracy", "macro_f1"),
    MULTI_LABEL_TARGET: (
        "macro_f1",
        "micro_f1",
        "sample_jaccard",
        "subset_accuracy",
    ),
    REGRESSION_TARGET: (
        "r2_variance_weighted",
        "r2_uniform_average",
        "mae",
        "rmse",
        "normalized_rmse_mean",
    ),
}
_DEFAULT_PRIMARY_METRIC = {
    SINGLE_LABEL_TARGET: "balanced_accuracy",
    MULTI_LABEL_TARGET: "macro_f1",
    REGRESSION_TARGET: "r2_variance_weighted",
}


@dataclass
class LinearProbeScoreResult:
    """Structured result from one out-of-fold linear-probe evaluation."""

    score: float
    primary_metric: str
    metrics: Dict[str, float]
    folds: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    per_class_scores: Dict[Any, float] = field(default_factory=dict)
    per_target_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the score result to JSON-compatible data."""

        return make_json_safe(self)


class LinearProbeScorer:
    """Evaluate a fixed linear decoder through deterministic OOF predictions."""

    def __init__(self, config: Optional[LinearProbeConfig] = None) -> None:
        if config is not None and not isinstance(config, LinearProbeConfig):
            raise TypeError("config must be a LinearProbeConfig.")
        self.config = config or LinearProbeConfig()

    def score(
        self,
        embeddings: Any,
        labels: Any,
        *,
        target_metadata: Optional[Dict[str, Any]] = None,
        groups: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> LinearProbeScoreResult:
        """Return target-aware aggregate metrics from OOF predictions."""

        metadata = dict(target_metadata or {})
        matrix = ensure_numeric_matrix(embeddings, "embeddings", allow_sparse=True)
        targets, normalized_metadata = normalize_targets(
            labels,
            label_names=metadata.get("label_names"),
            target_type=metadata.get("target_type", "auto"),
            target_names=metadata.get("target_names"),
        )
        if int(matrix.shape[0]) != len(targets):
            raise ValueError(
                "embeddings and labels must have the same length; "
                f"got {matrix.shape[0]} and {len(targets)}."
            )
        target_type = str(normalized_metadata["target_type"])
        if target_type == SINGLE_LABEL_TARGET:
            targets = np.asarray(semantic_label_keys(targets.tolist()), dtype=object)
        primary_metric = self._primary_metric(target_type)
        resolved_seed = self.config.random_state if seed is None else int(seed)
        normalized_groups = _validated_groups(groups, len(targets))
        if self.config.normalize_embeddings:
            matrix = l2_normalize_rows(matrix)

        split_target = _split_target(targets, normalized_metadata)
        splits, cv_method, split_warnings = _cross_validation_splits(
            split_target,
            target_type=target_type,
            groups=normalized_groups,
            requested_splits=self.config.n_splits,
            seed=resolved_seed,
        )
        exclusions = _excluded_target_keys(metadata.get("exclude_classes"))
        if target_type == SINGLE_LABEL_TARGET:
            result = self._score_single_label(
                matrix,
                targets,
                splits,
                exclusions=exclusions,
                seed=resolved_seed,
            )
        elif target_type == MULTI_LABEL_TARGET:
            result = self._score_multi_label(
                matrix,
                targets,
                normalized_metadata,
                splits,
                exclusions=exclusions,
                seed=resolved_seed,
            )
        else:
            if exclusions:
                raise ValueError("exclude_classes is not supported for regression probes.")
            result = self._score_regression(
                matrix,
                targets,
                normalized_metadata,
                splits,
            )

        import sklearn

        result.primary_metric = primary_metric
        result.score = float(result.metrics[primary_metric])
        result.warnings = sorted(set(split_warnings + result.warnings))
        result.metadata = {
            **normalized_metadata,
            **result.metadata,
            "aggregate_valid": True,
            "evaluation_mode": "out_of_fold",
            "cv_method": cv_method,
            "group_aware": normalized_groups is not None,
            "n_groups": (
                None if normalized_groups is None else int(len(np.unique(normalized_groups)))
            ),
            "n_splits_requested": int(self.config.n_splits),
            "n_splits": int(len(splits)),
            "random_state": resolved_seed,
            "normalize_embeddings": self.config.normalize_embeddings,
            "exclude_classes": sorted(exclusions),
            "estimator": _estimator_name(target_type),
            "probe_config": make_json_safe(asdict(self.config)),
            "scikit_learn_version": sklearn.__version__,
        }
        return result

    def _primary_metric(self, target_type: str) -> str:
        primary = self.config.primary_metric
        if primary == "auto":
            return _DEFAULT_PRIMARY_METRIC[target_type]
        if primary not in _TARGET_METRICS[target_type]:
            raise ValueError(
                f"Linear probe metric {primary!r} is not valid for target type "
                f"{target_type!r}; choose one of {list(_TARGET_METRICS[target_type])}."
            )
        return primary

    def _score_single_label(
        self,
        matrix: Any,
        targets: np.ndarray,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        *,
        exclusions: set[str],
        seed: Optional[int],
    ) -> LinearProbeScoreResult:
        predictions = np.empty(len(targets), dtype=object)
        fold_rows: List[Dict[str, Any]] = []
        warnings: List[str] = []
        all_classes = np.unique(targets)
        for fold, (train, holdout) in enumerate(splits):
            train_classes = np.unique(targets[train])
            if len(train_classes) != len(all_classes):
                raise ValueError(
                    "Linear-probe cross-validation produced a training fold without every "
                    "class. Provide more independent groups or use a dataset with broader "
                    "per-class support."
                )
            estimator = _classification_estimator(self.config, seed, fold)
            predicted, fold_warnings = _fit_predict(
                estimator,
                matrix[train],
                targets[train],
                matrix[holdout],
                fold,
            )
            predictions[holdout] = predicted
            warnings.extend(fold_warnings)
            fold_metrics, _ = _single_label_metrics(
                targets[holdout],
                np.asarray(predicted, dtype=object),
                exclusions,
            )
            fold_rows.append(_fold_row(fold, train, holdout, fold_metrics))
        metrics, per_class = _single_label_metrics(targets, predictions, exclusions)
        return LinearProbeScoreResult(
            score=0.0,
            primary_metric="",
            metrics=metrics,
            folds=fold_rows,
            warnings=warnings,
            per_class_scores=per_class,
        )

    def _score_multi_label(
        self,
        matrix: Any,
        targets: Any,
        target_metadata: Dict[str, Any],
        splits: List[Tuple[np.ndarray, np.ndarray]],
        *,
        exclusions: set[str],
        seed: Optional[int],
    ) -> LinearProbeScoreResult:
        label_names = tuple(target_metadata["label_names"])
        _assert_dense_allocation(
            (len(targets), len(label_names)),
            np.dtype(np.uint8),
            self.config.max_dense_bytes,
            "multi-label targets",
        )
        indicator = multilabel_indicator(targets, label_names)
        _assert_dense_allocation(
            indicator.shape,
            np.dtype(np.uint8),
            self.config.max_dense_bytes,
            "multi-label OOF predictions",
        )
        predictions = np.zeros(indicator.shape, dtype=np.uint8)
        fold_rows: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for fold, (train, holdout) in enumerate(splits):
            estimator = _multilabel_estimator(self.config, seed, fold)
            predicted, fold_warnings = _fit_predict(
                estimator,
                matrix[train],
                indicator[train],
                matrix[holdout],
                fold,
            )
            predictions[holdout] = np.asarray(predicted, dtype=np.uint8)
            warnings.extend(fold_warnings)
            fold_metrics = _multi_label_metrics(
                indicator[holdout],
                predictions[holdout],
                label_names,
                exclusions,
            )
            fold_rows.append(_fold_row(fold, train, holdout, fold_metrics))
        metrics = _multi_label_metrics(
            indicator,
            predictions,
            label_names,
            exclusions,
        )
        return LinearProbeScoreResult(
            score=0.0,
            primary_metric="",
            metrics=metrics,
            folds=fold_rows,
            warnings=warnings,
        )

    def _score_regression(
        self,
        matrix: Any,
        targets: Any,
        target_metadata: Dict[str, Any],
        splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> LinearProbeScoreResult:
        values = np.asarray(targets, dtype=float)
        target_matrix = values.reshape(-1, 1) if values.ndim == 1 else values
        _assert_dense_allocation(
            target_matrix.shape,
            np.dtype(float),
            self.config.max_dense_bytes,
            "regression OOF predictions",
        )
        predictions = np.empty_like(target_matrix, dtype=float)
        fold_rows: List[Dict[str, Any]] = []
        target_names = tuple(target_metadata["target_names"])
        active = np.flatnonzero(np.var(target_matrix, axis=0) > 0.0)
        if not len(active):
            raise ValueError("Linear-probe regression requires a non-constant target.")
        for fold, (train, holdout) in enumerate(splits):
            estimator = _regression_estimator(self.config)
            fit_target: Any = target_matrix[train]
            if fit_target.shape[1] == 1:
                fit_target = fit_target[:, 0]
            predicted, _ = _fit_predict(
                estimator,
                matrix[train],
                fit_target,
                matrix[holdout],
                fold,
            )
            predicted_matrix = np.asarray(predicted, dtype=float)
            if predicted_matrix.ndim == 1:
                predicted_matrix = predicted_matrix.reshape(-1, 1)
            predictions[holdout] = predicted_matrix
            fold_metrics, _ = _regression_metrics(
                target_matrix[holdout],
                predicted_matrix,
                target_names,
                active,
                allow_r2=len(holdout) >= 2,
            )
            fold_rows.append(_fold_row(fold, train, holdout, fold_metrics))
        metrics, per_target = _regression_metrics(
            target_matrix,
            predictions,
            target_names,
            active,
            allow_r2=True,
        )
        return LinearProbeScoreResult(
            score=0.0,
            primary_metric="",
            metrics=metrics,
            folds=fold_rows,
            per_target_scores=per_target,
            metadata={
                "evaluated_target_names": [target_names[index] for index in active],
                "excluded_constant_targets": [
                    target_names[index]
                    for index in range(target_matrix.shape[1])
                    if index not in set(active.tolist())
                ],
            },
        )


def _validated_groups(groups: Optional[Any], n_samples: int) -> Optional[np.ndarray]:
    if groups is None:
        return None
    values = canonical_semantic_array(groups)
    if values.ndim != 1 or len(values) != n_samples:
        raise ValueError("groups must be one-dimensional and aligned with embeddings.")
    if len(np.unique(values)) < 2:
        raise ValueError("Grouped linear-probe evaluation requires at least two groups.")
    return values


def _split_target(targets: Any, metadata: Dict[str, Any]) -> Any:
    if metadata["target_type"] == SINGLE_LABEL_TARGET:
        return np.asarray(targets, dtype=object)
    return np.zeros(len(targets), dtype=np.uint8)


def _cross_validation_splits(
    split_target: Any,
    *,
    target_type: str,
    groups: Optional[np.ndarray],
    requested_splits: int,
    seed: Optional[int],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str, List[str]]:
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    warnings: List[str] = []
    n_samples = len(split_target)
    if target_type == SINGLE_LABEL_TARGET:
        _, counts = np.unique(split_target, return_counts=True)
        maximum = int(np.min(counts))
        if groups is None:
            resolved = min(int(requested_splits), maximum)
            splitter = StratifiedKFold(
                n_splits=resolved,
                shuffle=True,
                random_state=seed,
            )
            raw_splits = splitter.split(np.zeros(n_samples, dtype=np.uint8), split_target)
            method = "stratified_kfold"
        else:
            per_class_groups = [
                len(np.unique(groups[split_target == label])) for label in np.unique(split_target)
            ]
            maximum = min(maximum, int(np.min(per_class_groups)), len(np.unique(groups)))
            resolved = min(int(requested_splits), maximum)
            splitter = StratifiedGroupKFold(
                n_splits=resolved,
                shuffle=True,
                random_state=seed,
            )
            raw_splits = splitter.split(
                np.zeros(n_samples, dtype=np.uint8),
                split_target,
                groups,
            )
            method = "stratified_group_kfold"
    elif groups is None:
        resolved = min(int(requested_splits), n_samples)
        splitter = KFold(n_splits=resolved, shuffle=True, random_state=seed)
        raw_splits = splitter.split(np.zeros(n_samples, dtype=np.uint8))
        method = "kfold"
    else:
        resolved = min(int(requested_splits), len(np.unique(groups)))
        splitter = GroupKFold(n_splits=resolved)
        raw_splits = splitter.split(
            np.zeros(n_samples, dtype=np.uint8),
            groups=groups,
        )
        method = "group_kfold"
    if resolved < 2:
        raise ValueError(
            "Linear-probe evaluation requires at least two feasible cross-validation folds."
        )
    if resolved < int(requested_splits):
        warnings.append(
            "Reduced linear-probe folds from "
            f"{int(requested_splits)} to {resolved} because target or group support is limited."
        )
    return [(np.asarray(train), np.asarray(test)) for train, test in raw_splits], method, warnings


def _classification_estimator(
    config: LinearProbeConfig,
    seed: Optional[int],
    fold: int,
) -> Any:
    from sklearn.linear_model import LogisticRegression

    fold_seed = None if seed is None else int(seed) + int(fold)
    return LogisticRegression(
        penalty="l2",
        C=float(config.classification_c),
        solver="lbfgs",
        max_iter=int(config.max_iter),
        class_weight=config.class_weight,
        random_state=fold_seed,
    )


def _multilabel_estimator(
    config: LinearProbeConfig,
    seed: Optional[int],
    fold: int,
) -> Any:
    from sklearn.multiclass import OneVsRestClassifier

    return OneVsRestClassifier(
        _classification_estimator(config, seed, fold),
        n_jobs=config.n_jobs,
    )


def _regression_estimator(config: LinearProbeConfig) -> Any:
    from sklearn.linear_model import Ridge

    return Ridge(alpha=float(config.regression_alpha))


def _fit_predict(
    estimator: Any,
    train_embeddings: Any,
    train_targets: Any,
    holdout_embeddings: Any,
    fold: int,
) -> Tuple[Any, List[str]]:
    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always")
        estimator.fit(train_embeddings, train_targets)
        predictions = estimator.predict(holdout_embeddings)
    warnings = [f"Fold {fold}: {item.category.__name__}: {item.message}" for item in caught]
    return predictions, warnings


def _single_label_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    exclusions: set[str],
) -> Tuple[Dict[str, float], Dict[Any, float]]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    classes = np.unique(targets)
    included = [label for label in classes if semantic_label_key(label) not in exclusions]
    if not included:
        raise ValueError("Linear-probe aggregation excluded every class.")
    mask = np.asarray([semantic_label_key(value) not in exclusions for value in targets])
    recalls = recall_score(
        targets,
        predictions,
        labels=classes.tolist(),
        average=None,
        zero_division=0,
    )
    recall_by_class = {label: float(value) for label, value in zip(classes, recalls)}
    metrics = {
        "balanced_accuracy": float(np.mean([recall_by_class[label] for label in included])),
        "accuracy": float(accuracy_score(targets[mask], predictions[mask])),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=included,
                average="macro",
                zero_division=0,
            )
        ),
    }
    return metrics, recall_by_class


def _multi_label_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    label_names: Tuple[Any, ...],
    exclusions: set[str],
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, jaccard_score

    keep = [
        index
        for index, label in enumerate(label_names)
        if semantic_label_key(label) not in exclusions
    ]
    if not keep:
        raise ValueError("Linear-probe aggregation excluded every label.")
    selected_targets = targets[:, keep]
    selected_predictions = predictions[:, keep]
    return {
        "macro_f1": float(
            f1_score(selected_targets, selected_predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(selected_targets, selected_predictions, average="micro", zero_division=0)
        ),
        "sample_jaccard": float(
            jaccard_score(
                selected_targets,
                selected_predictions,
                average="samples",
                zero_division=0,
            )
        ),
        "subset_accuracy": float(accuracy_score(selected_targets, selected_predictions)),
    }


def _regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    target_names: Tuple[str, ...],
    active: np.ndarray,
    *,
    allow_r2: bool,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    selected_targets = targets[:, active]
    selected_predictions = predictions[:, active]
    mae_raw = np.atleast_1d(
        mean_absolute_error(selected_targets, selected_predictions, multioutput="raw_values")
    )
    rmse_raw = np.sqrt(
        np.atleast_1d(
            mean_squared_error(selected_targets, selected_predictions, multioutput="raw_values")
        )
    )
    scales = np.std(selected_targets, axis=0)
    valid_scales = scales > 0.0
    normalized_rmse = np.divide(
        rmse_raw,
        scales,
        out=np.full_like(rmse_raw, np.nan, dtype=float),
        where=valid_scales,
    )
    metrics = {
        "mae": float(np.mean(mae_raw)),
        "rmse": float(np.mean(rmse_raw)),
    }
    if np.any(valid_scales):
        metrics["normalized_rmse_mean"] = float(np.mean(normalized_rmse[valid_scales]))
    if allow_r2:
        r2_raw = np.atleast_1d(
            r2_score(selected_targets, selected_predictions, multioutput="raw_values")
        )
        metrics.update(
            {
                "r2_variance_weighted": float(
                    r2_score(
                        selected_targets,
                        selected_predictions,
                        multioutput="variance_weighted",
                    )
                ),
                "r2_uniform_average": float(
                    r2_score(
                        selected_targets,
                        selected_predictions,
                        multioutput="uniform_average",
                    )
                ),
            }
        )
    else:
        r2_raw = np.full(len(active), np.nan)
    per_target = {
        str(target_names[index]): {
            "r2": float(r2_raw[position]),
            "mae": float(mae_raw[position]),
            "rmse": float(rmse_raw[position]),
            "normalized_rmse": float(normalized_rmse[position]),
        }
        for position, index in enumerate(active)
        if allow_r2
    }
    return metrics, per_target


def _fold_row(
    fold: int,
    train: np.ndarray,
    holdout: np.ndarray,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "fold": int(fold),
        "train_size": int(len(train)),
        "holdout_size": int(len(holdout)),
        "metrics": metrics,
    }


def _excluded_target_keys(value: Any) -> set[str]:
    if value is None:
        return set()
    values = [value] if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__") else value
    return {semantic_label_key(item) for item in values}


def _assert_dense_allocation(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    max_dense_bytes: int,
    operation: str,
) -> None:
    required = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    if required > int(max_dense_bytes):
        raise ValueError(
            f"{operation} requires {required} dense bytes, exceeding the configured "
            f"limit of {int(max_dense_bytes)} bytes."
        )


def _estimator_name(target_type: str) -> str:
    if target_type == SINGLE_LABEL_TARGET:
        return "l2_logistic_regression"
    if target_type == MULTI_LABEL_TARGET:
        return "one_vs_rest_l2_logistic_regression"
    return "ridge_regression"
