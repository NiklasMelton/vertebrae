"""Conservative cache fingerprint helpers."""

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np


def hash_json(value: Any) -> str:
    """Hash a value after conservative JSON normalization.

    Args:
        value: Value to fingerprint.

    Returns:
        SHA-256 hash string.
    """

    payload = json.dumps(_fingerprintable(value), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_json_exact(value: Any) -> str:
    """Hash complete canonical content without sampling large sequences or arrays.

    This is intended for protocol and evaluation identities where every declared
    prompt or configuration entry must affect the result key.
    """

    payload = canonical_json_exact(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_json_value(value: Any) -> Any:
    """Return the complete typed JSON representation used by exact hashes."""

    return _exact_fingerprintable(value)


def canonical_json_exact(value: Any) -> str:
    """Return canonical JSON for complete typed content."""

    return json.dumps(
        exact_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def hash_exact_json_value(value: Any) -> str:
    """Hash a value that is already in :func:`exact_json_value` form."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_array_like(value: Any) -> str:
    """Fingerprint an array-like or sparse-matrix value.

    Args:
        value: Value to fingerprint.

    Returns:
        SHA-256 hash string.
    """

    return hash_json(value)


def fingerprint_extractor_recipe(recipe: dict) -> str:
    """Fingerprint a serializable extractor recipe.

    Args:
        recipe: Extractor recipe dictionary.

    Returns:
        SHA-256 hash string.
    """

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
    if _is_sparse_matrix(value):
        return {
            "type": "sparse_matrix",
            "format": value.getformat(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nnz": int(value.nnz),
            "data_sample": _array_sample(value.data),
            "indices_sample": _array_sample(value.indices) if hasattr(value, "indices") else None,
        }
    if hasattr(value, "to_numpy"):
        return _fingerprintable(value.to_numpy())
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _fingerprintable(np.asarray(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _exact_fingerprintable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": "dataclass",
            "class": _type_identity(type(value)),
            "fields": {
                field.name: _exact_fingerprintable(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, dict):
        entries = [
            [_exact_fingerprintable(key), _exact_fingerprintable(item)]
            for key, item in value.items()
        ]
        return {
            "type": "mapping",
            "items": sorted(entries, key=lambda item: _exact_json(item[0])),
        }
    if isinstance(value, list):
        return {"type": "list", "items": [_exact_fingerprintable(item) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_exact_fingerprintable(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        normalized = [_exact_fingerprintable(item) for item in value]
        return {
            "type": "set" if isinstance(value, set) else "frozenset",
            "items": sorted(normalized, key=_exact_json),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {
                "type": "object_ndarray",
                "shape": list(value.shape),
                "values": _exact_fingerprintable(value.tolist()),
            }
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if _is_sparse_matrix(value):
        matrix = value.tocsr()
        return {
            "type": "sparse_matrix",
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "data_sha256": hashlib.sha256(matrix.data.tobytes()).hexdigest(),
            "indices_sha256": hashlib.sha256(matrix.indices.tobytes()).hexdigest(),
            "indptr_sha256": hashlib.sha256(matrix.indptr.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _exact_fingerprintable(value.item())
    if isinstance(value, UUID):
        return {"type": "uuid", "value": value.hex}
    if isinstance(value, Enum):
        return {
            "type": "enum",
            "class": _type_identity(type(value)),
            "member": value.name,
        }
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat(), "fold": value.fold}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, time):
        return {"type": "time", "value": value.isoformat(), "fold": value.fold}
    if isinstance(value, Decimal):
        return {"type": "decimal", "value": _normalized_decimal(value)}
    if isinstance(value, Fraction):
        return {
            "type": "fraction",
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
        }
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if hasattr(value, "to_numpy"):
        return _exact_fingerprintable(value.to_numpy())
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _exact_fingerprintable(np.asarray(value))
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": repr(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(
        "hash_json_exact does not support "
        f"{type(value).__name__}; use JSON-compatible values, dataclasses, NumPy arrays, "
        "sparse matrices, paths, bytes, UUIDs, enums, datetimes, Decimals, or Fractions."
    )


def _exact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _type_identity(value_type: type[Any]) -> str:
    """Return a stable qualified identity for a declared semantic type."""

    return f"{value_type.__module__}:{value_type.__qualname__}"


def _normalized_decimal(value: Decimal) -> str:
    """Represent decimal labels without insignificant finite trailing zeroes."""

    if value.is_finite():
        return str(value.normalize())
    return str(value)


def _array_sample(arr: np.ndarray) -> Any:
    flat = arr.reshape(-1) if arr.size else arr
    if flat.size <= 50:
        return flat.tolist()
    positions = np.linspace(0, flat.size - 1, num=50, dtype=int)
    return flat[positions].tolist()


def _is_sparse_matrix(value: Any) -> bool:
    try:
        from scipy import sparse
    except ImportError:
        return False
    return bool(sparse.issparse(value))
