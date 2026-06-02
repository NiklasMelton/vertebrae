"""Serialization helpers for reportable result objects."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def make_json_safe(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(make_json_safe(key)): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tocoo"):
        coo = value.tocoo()
        return {
            "format": "coo",
            "shape": list(coo.shape),
            "row": coo.row.tolist(),
            "col": coo.col.tolist(),
            "data": coo.data.tolist(),
        }
    return value
