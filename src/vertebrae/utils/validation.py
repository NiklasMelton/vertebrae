"""Validation helpers for embeddings."""

from typing import Any

import numpy as np


def ensure_2d_numeric_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values.")
    arr = arr.astype(float, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return arr


def l2_normalize_rows(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return value / norms
