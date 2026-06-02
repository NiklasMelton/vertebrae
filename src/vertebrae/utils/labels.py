"""Label helpers."""

from typing import Any, Dict

import numpy as np


def class_counts(y: np.ndarray) -> Dict[Any, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {
        label.item() if hasattr(label, "item") else label: int(count)
        for label, count in zip(labels, counts)
    }


def display_label(label: Any) -> str:
    if hasattr(label, "item"):
        label = label.item()
    return str(label)
