"""Conservative cache fingerprint helpers."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import numpy as np


def hash_json(value: Any) -> str:
    """Hash the complete typed value using the canonical identity representation.

    ``hash_json`` is the public identity hash.  It intentionally has the same
    exact semantics as :func:`hash_json_exact`; the latter name remains useful
    at call sites that want to emphasize that every value participates in the
    identity.  Unsupported opaque objects fail instead of silently falling back
    to unstable ``repr`` output.
    """

    return hash_json_exact(value)


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


def fingerprint_extractor_recipe(recipe: dict) -> str:
    """Fingerprint a serializable extractor recipe.

    Args:
        recipe: Extractor recipe dictionary.

    Returns:
        SHA-256 hash string.
    """

    if not isinstance(recipe, dict):
        raise TypeError("Extractor recipes must be dictionaries.")
    return hash_json_exact({"identity_schema": 2, "recipe": recipe})


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
    if isinstance(value, Mapping):
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
                "values_sha256": _hash_object_array(value),
            }
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _hash_array_bytes(value),
        }
    if _is_sparse_matrix(value):
        matrix = value
        result = {
            "type": "sparse_matrix",
            "format": matrix.getformat(),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
        }
        for name in ("data", "indices", "indptr", "row", "col", "offsets"):
            component = getattr(matrix, name, None)
            if component is not None:
                result[f"{name}_sha256"] = _hash_array_bytes(np.asarray(component))
        if matrix.getformat() == "dok":
            result["values"] = _exact_fingerprintable(dict(matrix.items()))
        elif matrix.getformat() == "lil":
            result["rows"] = _exact_fingerprintable(matrix.rows)
            result["values"] = _exact_fingerprintable(matrix.data)
        return result
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


def _hash_array_bytes(value: np.ndarray, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash an array in C order without materializing a full contiguous copy."""

    array = np.asarray(value)
    digest = hashlib.sha256()
    if array.size == 0:
        return digest.hexdigest()
    if array.flags.c_contiguous:
        view = memoryview(cast(Any, array)).cast("B")
        for start in range(0, len(view), chunk_bytes):
            digest.update(view[start : start + chunk_bytes])
        return digest.hexdigest()
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],  # type: ignore[list-item]
        op_flags=[["readonly"]],
        order="C",
        buffersize=max(1, chunk_bytes // max(1, array.dtype.itemsize)),
    )
    for chunk in iterator:
        digest.update(np.asarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def _hash_object_array(value: np.ndarray) -> str:
    """Hash object-array entries incrementally with unambiguous framing."""

    array = np.asarray(value, dtype=object)
    digest = hashlib.sha256()
    for index in np.ndindex(array.shape):
        payload = _exact_json(_exact_fingerprintable(array[index])).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def _is_sparse_matrix(value: Any) -> bool:
    try:
        from scipy import sparse
    except ImportError:
        return False
    return bool(sparse.issparse(value))
