"""Conservative cache fingerprint helpers."""

import hashlib
import json
from typing import Any

import numpy as np


def hash_json(value: Any) -> str:
    payload = json.dumps(_fingerprintable(value), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_array_like(value: Any) -> str:
    return hash_json(value)


def fingerprint_extractor_recipe(recipe: dict) -> str:
    return hash_json(recipe)


def _fingerprintable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _fingerprintable(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprintable(item) for item in value[:100]]
    if isinstance(value, np.ndarray):
        sample = _array_sample(value)
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sample": sample,
        }
    if hasattr(value, "to_numpy"):
        return _fingerprintable(value.to_numpy())
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _fingerprintable(np.asarray(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _array_sample(arr: np.ndarray) -> Any:
    flat = arr.reshape(-1) if arr.size else arr
    if flat.size <= 50:
        return flat.tolist()
    positions = np.linspace(0, flat.size - 1, num=50, dtype=int)
    return flat[positions].tolist()
