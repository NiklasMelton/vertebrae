"""Portable identities for semantic labels and typed report metadata."""

import base64
import json
import math
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from vertebrae.cache.fingerprint import exact_json_value, hash_exact_json_value, hash_json_exact

LABEL_ENCODING = "vertebrae.semantic-label/v1"
LABEL_KEY_PREFIX = "@vertebrae-label-v1:"
STRING_KEY_PREFIX = "@vertebrae-string-v1:"
MAPPING_KEY_PREFIX = "@vertebrae-key-v1:"


class SemanticLabelKey(str):
    """Marker for a value already converted to a canonical semantic label key."""


def semantic_label_key(value: Any) -> str:
    """Return a collision-resistant JSON object key for one semantic label."""

    if isinstance(value, SemanticLabelKey):
        return str(value)
    if type(value) is str and not value.startswith(
        (LABEL_KEY_PREFIX, STRING_KEY_PREFIX, MAPPING_KEY_PREFIX)
    ):
        return value
    if type(value) is str:
        encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
        return f"{STRING_KEY_PREFIX}{encoded}"
    return f"{LABEL_KEY_PREFIX}{hash_json_exact(value)}"


def semantic_label_catalog(values: Iterable[Any]) -> List[Dict[str, Any]]:
    """Build an ordered, JSON-safe catalog without importing user label classes."""

    catalog: List[Dict[str, Any]] = []
    observed: Dict[str, Any] = {}
    for value in values:
        key = semantic_label_key(value)
        encoded = (
            {"type": "semantic_label_key", "value": str(value)}
            if isinstance(value, SemanticLabelKey)
            else exact_json_value(value)
        )
        if key in observed:
            if observed[key] != encoded:
                raise ValueError("Semantic label key collision detected; rename the labels.")
            continue
        observed[key] = encoded
        catalog.append(
            {
                "key": key,
                "value": encoded,
                "type": _semantic_type(encoded),
                "display": str(value),
            }
        )
    display_counts: Dict[str, int] = {}
    for item in catalog:
        display_counts[item["display"]] = display_counts.get(item["display"], 0) + 1
    for item in catalog:
        item["report_display"] = (
            f"{item['display']} [{item['type']}]"
            if display_counts[item["display"]] > 1
            else item["display"]
        )
    return catalog


def semantic_label_keys(values: Iterable[Any]) -> List[str]:
    """Serialize a sequence of labels into stable scoring keys."""

    return [semantic_label_key(value) for value in values]


def canonical_semantic_array(values: Iterable[Any]) -> np.ndarray:
    """Return a one-dimensional object array of marked semantic keys."""

    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError("Semantic value arrays must be one-dimensional.")
        items = values.tolist()
    else:
        items = list(values)
    result = np.empty(len(items), dtype=object)
    result[:] = [SemanticLabelKey(semantic_label_key(value)) for value in items]
    return result


def label_display(value: Any, catalog: Sequence[Mapping[str, Any]]) -> str:
    """Resolve either an original label or serialized key to reportable text."""

    key = (
        value
        if isinstance(value, str) and any(item.get("key") == value for item in catalog)
        else None
    )
    if key is None:
        try:
            key = semantic_label_key(value)
        except TypeError:
            return str(value)
    for item in catalog:
        if item.get("key") == key:
            return str(item.get("report_display", item.get("display", key)))
    return str(value)


def portable_json(value: Any) -> Any:
    """Convert typed values to deterministic JSON without lossy ``str`` fallbacks."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else exact_json_value(value)
    if isinstance(value, np.generic):
        return portable_json(value.item())
    if isinstance(value, np.ndarray):
        return portable_json(value.tolist())
    if isinstance(value, list):
        return [portable_json(item) for item in value]
    if isinstance(value, tuple):
        return exact_json_value(value)
    if isinstance(value, (set, frozenset)):
        return exact_json_value(value)
    if isinstance(value, Mapping):
        return {_portable_mapping_key(key): portable_json(item) for key, item in value.items()}
    if hasattr(value, "tocoo"):
        coo = value.tocoo()
        return {
            "format": "coo",
            "shape": list(coo.shape),
            "row": coo.row.tolist(),
            "col": coo.col.tolist(),
            "data": portable_json(coo.data.tolist()),
        }
    if isinstance(value, Path) or is_dataclass(value):
        return exact_json_value(value)
    try:
        return exact_json_value(value)
    except TypeError as exc:
        raise TypeError(
            f"Cannot serialize {type(value).__name__} deterministically; convert it to a "
            "supported semantic value."
        ) from exc


def validate_label_catalog(catalog: Any) -> List[Dict[str, Any]]:
    """Validate an artifact label catalog and return normalized entries."""

    if not isinstance(catalog, list):
        raise ValueError("label_catalog must be an ordered list.")
    entries: List[Dict[str, Any]] = []
    keys = set()
    for item in catalog:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            raise ValueError("label_catalog entries must declare string keys.")
        key = item["key"]
        if key in keys:
            raise ValueError("label_catalog contains duplicate keys.")
        if not all(field in item for field in ("value", "type", "display")):
            raise ValueError("label_catalog entries are incomplete.")
        encoded = item["value"]
        if key != _semantic_key_from_encoded(encoded) or item["type"] != _semantic_type(encoded):
            raise ValueError("label_catalog key or type does not match its value.")
        keys.add(key)
        entries.append(dict(item))
    return entries


def strict_json_dumps(value: Any) -> str:
    """Serialize a value under the strict JSON rules used by protocol tests."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _portable_mapping_key(value: Any) -> str:
    if isinstance(value, str) and not value.startswith(MAPPING_KEY_PREFIX):
        return value
    canonical = json.dumps(exact_json_value(value), sort_keys=True, separators=(",", ":"))
    encoded = base64.urlsafe_b64encode(canonical.encode("utf-8")).decode("ascii")
    return f"{MAPPING_KEY_PREFIX}{encoded}"


def _semantic_type(encoded: Any) -> str:
    if isinstance(encoded, dict):
        class_name = encoded.get("class")
        kind = str(encoded.get("type", "unknown"))
        return f"{kind}:{class_name}" if class_name else kind
    return type(encoded).__name__


def _semantic_key_from_encoded(encoded: Any) -> str:
    if isinstance(encoded, dict) and encoded.get("type") == "semantic_label_key":
        value = encoded.get("value")
        if not isinstance(value, str):
            raise ValueError("Semantic label key catalog entries require string values.")
        return value
    if isinstance(encoded, dict) and encoded.get("type") == "str":
        return semantic_label_key(encoded.get("value"))
    return f"{LABEL_KEY_PREFIX}{hash_exact_json_value(encoded)}"
