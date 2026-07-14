"""Serialization helpers for reportable result objects and persisted metadata."""

import base64
import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableSet

import numpy as np

from vertebrae.cache.fingerprint import exact_json_value

_MAPPING_KEY_PREFIX = "@vertebrae-key-v1:"


def make_json_safe(value: Any) -> Any:
    """Convert supported scientific Python values into deterministic JSON data.

    Unsupported values, cycles, and non-finite numbers fail with a path identifying
    the offending value. This function intentionally does not fall back to ``str``.
    """

    return _make_json_safe(value, path="$", active=set())


def json_dumps_strict(
    value: Any,
    *,
    indent: Any = None,
    sort_keys: bool = True,
) -> str:
    """Normalize and serialize a value using standards-compliant JSON."""

    return json.dumps(
        make_json_safe(value),
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def _make_json_safe(value: Any, *, path: str, active: MutableSet[int]) -> Any:
    if isinstance(value, Enum):
        return _make_json_safe(value.value, path=path, active=active)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"Cannot serialize non-finite float at {path}.")
        return value
    if isinstance(value, np.generic):
        return _make_json_safe(value.item(), path=path, active=active)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _with_cycle_guard(
            value,
            path=path,
            active=active,
            normalize=lambda: _make_json_safe(value.tolist(), path=path, active=active),
        )
    if hasattr(value, "tocoo"):
        return _with_cycle_guard(
            value,
            path=path,
            active=active,
            normalize=lambda: _sparse_json(value, path=path, active=active),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return _with_cycle_guard(
            value,
            path=path,
            active=active,
            normalize=lambda: {
                field.name: _make_json_safe(
                    getattr(value, field.name),
                    path=_mapping_value_path(path, field.name),
                    active=active,
                )
                for field in fields(value)
            },
        )
    if isinstance(value, Mapping):
        return _with_cycle_guard(
            value,
            path=path,
            active=active,
            normalize=lambda: _mapping_json(value, path=path, active=active),
        )
    if isinstance(value, (list, tuple)):
        return _with_cycle_guard(
            value,
            path=path,
            active=active,
            normalize=lambda: [
                _make_json_safe(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            ],
        )
    if isinstance(value, (set, frozenset)):
        return _with_cycle_guard(
            value,
            path=path,
            active=active,
            normalize=lambda: _set_json(value, path=path, active=active),
        )
    raise TypeError(f"Cannot serialize unsupported {type(value).__name__} at {path}.")


def _with_cycle_guard(
    value: Any,
    *,
    path: str,
    active: MutableSet[int],
    normalize: Any,
) -> Any:
    identity = id(value)
    if identity in active:
        raise TypeError(f"Cannot serialize recursive value at {path}.")
    active.add(identity)
    try:
        return normalize()
    finally:
        active.remove(identity)


def _mapping_json(value: Mapping[Any, Any], *, path: str, active: MutableSet[int]) -> dict:
    result: dict[str, Any] = {}
    source_keys: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = _mapping_key(key, path=path)
        if normalized_key in result:
            raise TypeError(
                f"Cannot serialize mapping at {path}: keys {source_keys[normalized_key]!r} "
                f"and {key!r} normalize to the same JSON key."
            )
        result[normalized_key] = _make_json_safe(
            item,
            path=_mapping_value_path(path, normalized_key),
            active=active,
        )
        source_keys[normalized_key] = key
    return result


def _mapping_key(value: Any, *, path: str) -> str:
    if isinstance(value, str) and not value.startswith(_MAPPING_KEY_PREFIX):
        return value
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, float) and not math.isfinite(scalar):
        raise TypeError(f"Cannot serialize non-finite float mapping key at {path}.")
    try:
        canonical = json.dumps(
            exact_json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Cannot serialize unsupported {type(value).__name__} mapping key at {path}."
        ) from exc
    encoded = base64.urlsafe_b64encode(canonical.encode("utf-8")).decode("ascii")
    return f"{_MAPPING_KEY_PREFIX}{encoded}"


def _mapping_value_path(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key)}]"


def _set_json(value: Any, *, path: str, active: MutableSet[int]) -> list:
    normalized = [
        _make_json_safe(item, path=f"{path}<set-item>", active=active) for item in value
    ]
    return sorted(
        normalized,
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


def _sparse_json(value: Any, *, path: str, active: MutableSet[int]) -> dict:
    coo = value.tocoo()
    return {
        "format": "coo",
        "shape": [int(size) for size in coo.shape],
        "row": _make_json_safe(coo.row, path=f"{path}.row", active=active),
        "col": _make_json_safe(coo.col, path=f"{path}.col", active=active),
        "data": _make_json_safe(coo.data, path=f"{path}.data", active=active),
    }
