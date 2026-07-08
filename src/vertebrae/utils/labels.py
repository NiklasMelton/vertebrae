"""Label helpers."""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

LABEL_PATH_DELIMITER = " > "
LABEL_SET_DELIMITER = " + "
SINGLE_LABEL_TARGET = "single_label"
MULTI_LABEL_TARGET = "multi_label"
REGRESSION_TARGET = "regression"


def coerce_label_input(labels: Any) -> np.ndarray:
    """Coerce user-provided labels without breaking ragged multi-label rows."""

    if isinstance(labels, np.ndarray):
        return labels
    try:
        return np.asarray(labels)
    except ValueError:
        items = list(labels)
        result = np.empty(len(items), dtype=object)
        result[:] = items
        return result


def normalize_targets(
    y: Any,
    label_names: Optional[Iterable[Any]] = None,
    target_type: str = "auto",
    target_names: Optional[Iterable[Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize single-label, multi-label, or explicit regression targets.

    Single-label targets are returned as a one-dimensional array. Multi-label
    targets are returned as a one-dimensional object array where each element is
    a tuple of labels ordered by the resolved label names. Regression targets
    are returned as a one- or two-dimensional float array.
    """

    labels = coerce_label_input(y)
    names = normalize_label_names(label_names)
    resolved_target_names = _normalize_target_names(target_names)
    _validate_target_mode(
        target_type,
        label_names=names,
        target_names=resolved_target_names,
    )
    if target_type == REGRESSION_TARGET:
        normalized, metadata = _normalize_regression_targets(
            labels,
            target_names=resolved_target_names,
        )
        return normalized, metadata
    if _is_indicator_matrix(labels):
        normalized = _labels_from_indicator(labels, names)
        resolved_names = names if names is not None else tuple(range(labels.shape[1]))
        return normalized, _target_metadata(
            MULTI_LABEL_TARGET,
            normalized,
            label_names=resolved_names,
        )
    if labels.ndim == 2 and names is not None and labels.dtype.kind in {"b", "i", "u", "f"}:
        raise ValueError("Indicator labels must contain only 0/1 or boolean values.")
    if labels.ndim == 2:
        if target_type == SINGLE_LABEL_TARGET:
            raise ValueError("single_label targets must be one-dimensional.")
        normalized, resolved_names = _normalize_label_sequences(
            [row for row in labels],
            label_names=names,
        )
        return normalized, _target_metadata(
            MULTI_LABEL_TARGET,
            normalized,
            label_names=resolved_names,
        )
    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional or a two-dimensional multilabel target.")
    if _is_sequence_label_array(labels):
        if target_type == SINGLE_LABEL_TARGET:
            raise ValueError(
                "single_label targets must contain scalar label values, not label sequences."
            )
        normalized, resolved_names = _normalize_label_sequences(
            list(labels),
            label_names=names,
        )
        return normalized, _target_metadata(
            MULTI_LABEL_TARGET,
            normalized,
            label_names=resolved_names,
        )
    if names is not None:
        raise ValueError("label_names can only be provided for multi-label targets.")
    if target_type == MULTI_LABEL_TARGET:
        raise ValueError(
            "multi_label targets must be a 2D indicator matrix or a 1D sequence of label sets."
        )
    normalized_single = np.asarray([_normalize_scalar(label) for label in labels], dtype=object)
    if _has_missing_single_labels(normalized_single):
        raise ValueError("Labels must be non-missing.")
    return normalized_single, _target_metadata(SINGLE_LABEL_TARGET, normalized_single)


def target_type(
    y: Any,
    label_names: Optional[Iterable[Any]] = None,
    target_type: str = "auto",
    target_names: Optional[Iterable[Any]] = None,
) -> str:
    """Return the target type for labels."""

    _, metadata = normalize_targets(
        y,
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    return str(metadata["target_type"])


def class_counts(
    y: Any,
    label_names: Optional[Iterable[Any]] = None,
    target_type: str = "auto",
    target_names: Optional[Iterable[Any]] = None,
) -> Dict[Any, int]:
    """Count labels while preserving scalar label values.

    For multi-label targets, counts are per-label occurrence counts. Regression
    targets do not define classes and return an empty mapping.
    """

    labels, metadata = normalize_targets(
        y,
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    if metadata["target_type"] == REGRESSION_TARGET:
        return {}
    if metadata["target_type"] == MULTI_LABEL_TARGET:
        return _multi_label_counts(labels, tuple(metadata["label_names"]))

    unique, counts = np.unique(labels, return_counts=True)
    return {
        label.item() if hasattr(label, "item") else label: int(count)
        for label, count in zip(unique, counts)
    }


def labelset_counts(
    y: Any,
    label_names: Optional[Iterable[Any]] = None,
    target_type: str = "auto",
    target_names: Optional[Iterable[Any]] = None,
) -> Dict[str, int]:
    """Count exact label combinations for a multi-label target."""

    labels, metadata = normalize_targets(
        y,
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    if metadata["target_type"] != MULTI_LABEL_TARGET:
        return {}
    return _labelset_counts_normalized(labels)


def target_summary(
    y: Any,
    label_names: Optional[Iterable[Any]] = None,
    target_type: str = "auto",
    target_names: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Return JSON-friendly target summary metadata."""

    labels, metadata = normalize_targets(
        y,
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    summary = {
        "target_type": metadata["target_type"],
        "n_classes": metadata["n_classes"],
        "class_counts": class_counts(
            labels,
            label_names=metadata.get("label_names"),
            target_type=metadata["target_type"],
            target_names=metadata.get("target_names"),
        ),
    }
    if metadata["target_type"] == MULTI_LABEL_TARGET:
        summary.update(
            {
                "label_names": list(metadata["label_names"]),
                "labelset_counts": labelset_counts(
                    labels,
                    label_names=metadata["label_names"],
                ),
                "mean_label_cardinality": metadata["mean_label_cardinality"],
                "label_density": metadata["label_density"],
            }
        )
    if metadata["target_type"] == REGRESSION_TARGET:
        summary.update(
            {
                "n_targets": metadata["n_targets"],
                "target_names": list(metadata["target_names"]),
                "target_means": metadata["target_means"],
                "target_variances": metadata["target_variances"],
                "constant_targets": list(metadata["constant_targets"]),
                "nonconstant_targets": list(metadata["nonconstant_targets"]),
            }
        )
    return summary


def metric_labels(
    y: Any,
    label_names: Optional[Iterable[Any]] = None,
    target_type: str = "auto",
    target_names: Optional[Iterable[Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return labels in the shape expected by metric libraries."""

    labels, metadata = normalize_targets(
        y,
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    if metadata["target_type"] == MULTI_LABEL_TARGET:
        return multilabel_indicator(labels, metadata["label_names"]), metadata
    if metadata["target_type"] == REGRESSION_TARGET:
        return np.asarray(labels, dtype=float), metadata
    return labels, metadata


def labels_to_jsonable(
    y: Any,
    label_names: Optional[Iterable[Any]] = None,
    target_type: str = "auto",
    target_names: Optional[Iterable[Any]] = None,
) -> Any:
    """Serialize labels in the canonical artifact shape."""

    labels, metadata = normalize_targets(
        y,
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    if metadata["target_type"] == MULTI_LABEL_TARGET:
        return [list(labelset) for labelset in labels]
    if metadata["target_type"] == REGRESSION_TARGET:
        return np.asarray(labels, dtype=float).tolist()
    return labels.tolist()


def labels_from_jsonable(
    payload: Any,
    label_names: Optional[Iterable[Any]] = None,
    target_type: str = "auto",
    target_names: Optional[Iterable[Any]] = None,
) -> np.ndarray:
    """Load labels from an artifact JSON payload."""

    if target_type == REGRESSION_TARGET:
        labels, _ = normalize_targets(
            payload,
            target_type=target_type,
            target_names=target_names,
        )
        return labels
    if label_names is not None and isinstance(payload, list):
        rows = np.empty(len(payload), dtype=object)
        rows[:] = payload
        labels, _ = normalize_targets(
            rows,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        return labels
    labels, _ = normalize_targets(
        payload,
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    return labels


def multilabel_indicator(y: Any, label_names: Optional[Iterable[Any]] = None) -> np.ndarray:
    """Convert a multi-label target to a dense binary indicator matrix."""

    labels, metadata = normalize_targets(y, label_names=label_names)
    if metadata["target_type"] != MULTI_LABEL_TARGET:
        raise ValueError("multilabel_indicator requires a multi-label target.")
    resolved_names = tuple(metadata["label_names"])
    positions = {label: index for index, label in enumerate(resolved_names)}
    indicator = np.zeros((len(labels), len(resolved_names)), dtype=int)
    for row_index, labelset in enumerate(labels):
        for label in labelset:
            indicator[row_index, positions[label]] = 1
    return indicator


def stratified_label_indices(
    y: Any,
    rate: float,
    random_state: int = 42,
    min_samples_per_class: int = 2,
    label_names: Optional[Iterable[Any]] = None,
    target_type: str = "auto",
    target_names: Optional[Iterable[Any]] = None,
) -> np.ndarray:
    """Select deterministic label-aware sample indices."""

    if not 0.0 < rate <= 1.0:
        raise ValueError("subsample rate must be in (0, 1].")
    labels, metadata = normalize_targets(
        y,
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    if rate >= 1.0:
        return np.arange(len(labels), dtype=int)
    if metadata["target_type"] == REGRESSION_TARGET:
        raise ValueError("stratified_label_indices does not support regression targets.")
    rng = np.random.default_rng(random_state)
    if metadata["target_type"] != MULTI_LABEL_TARGET:
        return _single_label_subsample_indices(
            labels,
            rate=rate,
            rng=rng,
            min_samples_per_class=min_samples_per_class,
        )
    indicator = multilabel_indicator(labels, metadata["label_names"])
    selected: set[int] = set()
    for column in range(indicator.shape[1]):
        class_indices = np.flatnonzero(indicator[:, column] == 1)
        if len(class_indices) == 0:
            continue
        target = int(np.floor(len(class_indices) * rate))
        if len(class_indices) >= min_samples_per_class:
            target = max(min_samples_per_class, target)
        target = max(1, min(len(class_indices), target))
        already = [index for index in class_indices.tolist() if index in selected]
        if len(already) >= target:
            continue
        remaining = np.asarray(
            [index for index in class_indices.tolist() if index not in selected],
            dtype=int,
        )
        needed = min(len(remaining), target - len(already))
        selected.update(rng.choice(remaining, size=needed, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def display_label(label: Any) -> str:
    """Convert a label value to display text."""

    if hasattr(label, "item"):
        label = label.item()
    return str(label)


def display_labelset(labelset: Any) -> str:
    """Convert a multi-label labelset to display text."""

    return LABEL_SET_DELIMITER.join(display_label(label) for label in tuple(labelset))


def normalize_label_paths(label_paths: Any, n_samples: int) -> Tuple[Tuple[Any, ...], ...]:
    """Validate and normalize one hierarchy path per sample."""

    raw_paths = list(label_paths)
    if len(raw_paths) != n_samples:
        raise ValueError(
            "label_paths must have the same length as the dataset; "
            f"got {len(raw_paths)} and {n_samples}."
        )
    normalized = []
    for index, path in enumerate(raw_paths):
        if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
            raise ValueError(
                "Each label path must be a non-string sequence of category values; "
                f"sample {index} has {type(path).__name__}."
            )
        values = tuple(_normalize_scalar(value) for value in path)
        if not values:
            raise ValueError(
                "Each label path must contain at least one label; " f"sample {index} is empty."
            )
        if any(_is_missing_label(value) for value in values):
            raise ValueError(
                "Label paths must be non-missing; " f"sample {index} contains a missing value."
            )
        normalized.append(values)
    return tuple(normalized)


def normalize_level_names(
    level_names: Optional[Iterable[Any]],
    max_depth: int,
) -> Optional[Tuple[str, ...]]:
    """Validate optional hierarchy level names."""

    if level_names is None:
        return None
    normalized = tuple(str(name) for name in level_names)
    if len(normalized) != max_depth:
        raise ValueError(
            "level_names must match the hierarchy depth; "
            f"got {len(normalized)} names for depth {max_depth}."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("level_names must be unique.")
    return normalized


def normalize_label_names(label_names: Optional[Iterable[Any]]) -> Optional[Tuple[Any, ...]]:
    """Validate optional multi-label names."""

    if label_names is None:
        return None
    normalized = tuple(_normalize_scalar(name) for name in label_names)
    if not normalized:
        raise ValueError("label_names must not be empty.")
    if any(_is_missing_label(name) for name in normalized):
        raise ValueError("label_names must be non-missing.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("label_names must be unique.")
    return normalized


def hierarchy_depth(label_paths: Sequence[Sequence[Any]]) -> int:
    """Return the maximum hierarchy depth."""

    if not label_paths:
        raise ValueError("At least one label path is required.")
    return max(len(path) for path in label_paths)


def label_view_from_paths(
    label_paths: Sequence[Sequence[Any]],
    level: Union[int, str],
    level_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Derive a one-dimensional label view from hierarchy paths."""

    if not label_paths:
        raise ValueError("At least one label path is required.")
    max_depth = hierarchy_depth(label_paths)
    resolved_level = resolve_hierarchy_level(level, max_depth=max_depth, level_names=level_names)
    missing = [index for index, path in enumerate(label_paths) if len(path) <= resolved_level]
    if missing:
        raise ValueError(
            "Hierarchy level " f"{display_label(level)} is missing for {len(missing)} samples."
        )
    labels = np.asarray(
        [_format_label_prefix(path[: resolved_level + 1]) for path in label_paths],
        dtype=object,
    )
    level_name = _hierarchy_level_name(resolved_level, level_names)
    return labels, {
        "kind": "hierarchy",
        "level": int(resolved_level),
        "name": level_name,
        "requested_level": level,
        "key": f"hierarchy:{resolved_level}:{level_name}",
        "max_depth": int(max_depth),
        "path_delimiter": LABEL_PATH_DELIMITER,
    }


def resolve_hierarchy_level(
    level: Union[int, str],
    max_depth: int,
    level_names: Optional[Sequence[str]] = None,
) -> int:
    """Resolve an integer or named hierarchy level to a concrete index."""

    if isinstance(level, str):
        if level_names is None:
            raise ValueError(
                f"Hierarchy level {level!r} requires level_names metadata on the dataset."
            )
        try:
            return int(level_names.index(level))
        except ValueError as exc:
            raise ValueError(f"Unknown hierarchy level name {level!r}.") from exc
    resolved = int(level)
    if resolved < 0:
        resolved += max_depth
    if resolved < 0 or resolved >= max_depth:
        raise ValueError(f"Hierarchy level {level!r} is out of range for depth {max_depth}.")
    return resolved


def default_label_view_metadata() -> Dict[str, Any]:
    """Return metadata for the default dataset label view."""

    return {"kind": "primary", "name": "primary", "key": "primary"}


def default_target_view_metadata() -> Dict[str, Any]:
    """Return metadata for the default dataset target view."""

    return {"kind": "primary", "name": "primary", "key": "primary"}


def label_view_suffix(label_view: Optional[Dict[str, Any]]) -> str:
    """Format a label-view suffix for result names."""

    if not label_view or label_view.get("kind") == "primary":
        return ""
    return f"[level={label_view.get('name', 'view')}]"


def target_view_suffix(target_view: Optional[Dict[str, Any]]) -> str:
    """Format a target-view suffix for result names."""

    if not target_view or target_view.get("kind") == "primary":
        return ""
    return f"[target={target_view.get('name', 'view')}]"


def _target_metadata(
    target_type_value: str,
    labels: np.ndarray,
    label_names: Optional[Sequence[Any]] = None,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if target_type_value == MULTI_LABEL_TARGET:
        if label_names is None:
            raise ValueError("Multi-label metadata requires label_names.")
        counts = _multi_label_counts(labels, tuple(label_names))
        cardinalities = [len(tuple(labelset)) for labelset in labels]
        return {
            "target_type": MULTI_LABEL_TARGET,
            "label_names": tuple(label_names),
            "n_classes": int(len(label_names)),
            "class_counts": counts,
            "labelset_counts": _labelset_counts_normalized(labels),
            "mean_label_cardinality": float(np.mean(cardinalities)) if cardinalities else 0.0,
            "label_density": (
                float(np.mean(cardinalities) / len(label_names)) if label_names else 0.0
            ),
        }
    if target_type_value == REGRESSION_TARGET:
        regression = np.asarray(labels, dtype=float)
        if regression.ndim == 1:
            regression = regression.reshape(-1, 1)
        resolved_names = (
            tuple(target_names)
            if target_names is not None
            else tuple(f"target_{index}" for index in range(regression.shape[1]))
        )
        variances = np.var(regression, axis=0).astype(float, copy=False)
        means = np.mean(regression, axis=0).astype(float, copy=False)
        constant_targets = tuple(
            resolved_names[index]
            for index, variance in enumerate(variances.tolist())
            if float(variance) <= 0.0
        )
        nonconstant_targets = tuple(name for name in resolved_names if name not in constant_targets)
        return {
            "target_type": REGRESSION_TARGET,
            "n_classes": 0,
            "class_counts": {},
            "n_targets": int(regression.shape[1]),
            "target_names": resolved_names,
            "target_means": {
                name: float(value) for name, value in zip(resolved_names, means.tolist())
            },
            "target_variances": {
                name: float(value) for name, value in zip(resolved_names, variances.tolist())
            },
            "constant_targets": constant_targets,
            "nonconstant_targets": nonconstant_targets,
        }
    counts = _single_label_counts(labels)
    return {
        "target_type": SINGLE_LABEL_TARGET,
        "n_classes": int(len(counts)),
        "class_counts": counts,
    }


def _single_label_counts(labels: np.ndarray) -> Dict[Any, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {
        label.item() if hasattr(label, "item") else label: int(count)
        for label, count in zip(unique, counts)
    }


def _multi_label_counts(labels: np.ndarray, label_names: Sequence[Any]) -> Dict[Any, int]:
    counts = {label: 0 for label in label_names}
    for labelset in labels:
        for label in tuple(labelset):
            counts[label] = counts.get(label, 0) + 1
    return counts


def _normalize_regression_targets(
    labels: np.ndarray,
    target_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        regression = np.asarray(labels, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Regression targets must be numeric.") from exc
    if regression.ndim == 1:
        normalized = regression.astype(float, copy=False)
        n_targets = 1
    elif regression.ndim == 2:
        normalized = regression.astype(float, copy=False)
        n_targets = normalized.shape[1]
    else:
        raise ValueError("Regression targets must be one- or two-dimensional.")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Regression targets must be finite.")
    if target_names is not None and len(target_names) != n_targets:
        raise ValueError(
            "target_names must match the regression target columns; "
            f"got {len(target_names)} names for {n_targets} targets."
        )
    return normalized, _target_metadata(
        REGRESSION_TARGET,
        normalized,
        target_names=target_names,
    )


def _normalize_target_names(target_names: Optional[Iterable[Any]]) -> Optional[Tuple[str, ...]]:
    if target_names is None:
        return None
    normalized = tuple(str(name) for name in target_names)
    if not normalized:
        raise ValueError("target_names must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("target_names must be unique.")
    return normalized


def _validate_target_mode(
    target_type: str,
    label_names: Optional[Sequence[Any]],
    target_names: Optional[Sequence[str]],
) -> None:
    allowed = {"auto", SINGLE_LABEL_TARGET, MULTI_LABEL_TARGET, REGRESSION_TARGET}
    if target_type not in allowed:
        raise ValueError(f"target_type must be one of {sorted(allowed)}.")
    if target_type == REGRESSION_TARGET and label_names is not None:
        raise ValueError("label_names cannot be provided for regression targets.")
    if target_type != REGRESSION_TARGET and target_names is not None:
        raise ValueError("target_names can only be provided for regression targets.")


def _labelset_counts_normalized(labels: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for labelset in labels:
        key = display_labelset(labelset)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _labels_from_indicator(
    labels: np.ndarray,
    label_names: Optional[Sequence[Any]],
) -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError("Indicator labels must be two-dimensional.")
    resolved_names = label_names if label_names is not None else tuple(range(labels.shape[1]))
    if len(resolved_names) != labels.shape[1]:
        raise ValueError(
            "label_names length must match the indicator label columns; "
            f"got {len(resolved_names)} names for {labels.shape[1]} columns."
        )
    indicator = _validate_indicator_matrix(labels)
    normalized = np.empty(indicator.shape[0], dtype=object)
    rows: List[Tuple[Any, ...]] = []
    for row_index, row in enumerate(indicator):
        active = tuple(resolved_names[index] for index in np.flatnonzero(row))
        if not active:
            raise ValueError(f"Multi-label sample {row_index} must contain at least one label.")
        rows.append(active)
    normalized[:] = rows
    return normalized


def _normalize_label_sequences(
    rows: Iterable[Any],
    label_names: Optional[Sequence[Any]],
) -> Tuple[np.ndarray, Tuple[Any, ...]]:
    raw_labelsets: List[Tuple[Any, ...]] = []
    observed: List[Any] = []
    observed_set: set[Any] = set()
    allowed = set(label_names) if label_names is not None else None
    for row_index, row in enumerate(rows):
        if isinstance(row, np.ndarray):
            values = row.tolist()
        elif isinstance(row, (set, frozenset)):
            values = sorted(row, key=display_label)
        elif isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(
                "Multi-label targets must contain non-string label sequences; "
                f"sample {row_index} has {type(row).__name__}."
            )
        else:
            values = list(row)
        if not values:
            raise ValueError(f"Multi-label sample {row_index} must contain at least one label.")
        normalized_values = tuple(_normalize_scalar(value) for value in values)
        if any(_is_missing_label(value) for value in normalized_values):
            raise ValueError(f"Multi-label sample {row_index} contains a missing label value.")
        if len(set(normalized_values)) != len(normalized_values):
            raise ValueError(f"Multi-label sample {row_index} contains duplicate labels.")
        if allowed is not None:
            unknown = [value for value in normalized_values if value not in allowed]
            if unknown:
                raise ValueError(
                    f"Multi-label sample {row_index} contains labels not present "
                    f"in label_names: {unknown}."
                )
        for value in normalized_values:
            if value not in observed_set:
                observed.append(value)
                observed_set.add(value)
        raw_labelsets.append(normalized_values)
    resolved_names = tuple(label_names) if label_names is not None else tuple(observed)
    positions = {label: index for index, label in enumerate(resolved_names)}
    normalized = np.empty(len(raw_labelsets), dtype=object)
    normalized[:] = [
        tuple(sorted(labelset, key=lambda label: positions[label])) for labelset in raw_labelsets
    ]
    return normalized, resolved_names


def _is_indicator_matrix(labels: np.ndarray) -> bool:
    if labels.ndim != 2:
        return False
    if labels.dtype.kind not in {"b", "i", "u", "f"}:
        return False
    values = np.asarray(labels)
    if values.size == 0:
        return True
    return bool(np.all((values == 0) | (values == 1)))


def _validate_indicator_matrix(labels: np.ndarray) -> np.ndarray:
    if not _is_indicator_matrix(labels):
        raise ValueError("Indicator labels must contain only 0/1 or boolean values.")
    return np.asarray(labels, dtype=int)


def _is_sequence_label_array(labels: np.ndarray) -> bool:
    return any(_is_label_sequence(label) for label in labels)


def _is_label_sequence(value: Any) -> bool:
    return not isinstance(value, (str, bytes)) and isinstance(
        value, (Sequence, set, frozenset, np.ndarray)
    )


def _has_missing_single_labels(labels: np.ndarray) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(labels).any())
    except ImportError:
        if labels.dtype.kind in {"f", "c"}:
            return bool(np.isnan(labels).any())
        return any(_is_missing_label(label) for label in labels)


def _single_label_subsample_indices(
    labels: np.ndarray,
    rate: float,
    rng: np.random.Generator,
    min_samples_per_class: int,
) -> np.ndarray:
    selected = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        target = int(np.floor(len(class_indices) * rate))
        if len(class_indices) >= min_samples_per_class:
            target = max(min_samples_per_class, target)
        target = max(1, min(len(class_indices), target))
        selected.extend(rng.choice(class_indices, size=target, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def _format_label_prefix(prefix: Sequence[Any]) -> str:
    return LABEL_PATH_DELIMITER.join(display_label(value) for value in prefix)


def _hierarchy_level_name(level: int, level_names: Optional[Sequence[str]]) -> str:
    if level_names is not None:
        return str(level_names[level])
    return f"level_{level}"


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _is_missing_label(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return bool(np.isnan(value))
    return False
