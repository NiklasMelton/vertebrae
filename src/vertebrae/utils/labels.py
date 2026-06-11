"""Label helpers."""

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

LABEL_PATH_DELIMITER = " > "


def class_counts(y: np.ndarray) -> Dict[Any, int]:
    """Count labels while preserving scalar label values.

    Args:
        y: One-dimensional label array.

    Returns:
        Mapping from label values to sample counts.
    """

    labels, counts = np.unique(y, return_counts=True)
    return {
        label.item() if hasattr(label, "item") else label: int(count)
        for label, count in zip(labels, counts)
    }


def display_label(label: Any) -> str:
    """Convert a label value to display text.

    Args:
        label: Label value.

    Returns:
        Human-readable label string.
    """

    if hasattr(label, "item"):
        label = label.item()
    return str(label)


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


def label_view_suffix(label_view: Optional[Dict[str, Any]]) -> str:
    """Format a label-view suffix for result names."""

    if not label_view or label_view.get("kind") == "primary":
        return ""
    return f"[level={label_view.get('name', 'view')}]"


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
