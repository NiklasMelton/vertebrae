"""Label helpers."""

from typing import Any, Dict

import numpy as np


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
