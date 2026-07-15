"""Defensive snapshots for protocol datasets whose identity must not drift."""

from copy import deepcopy
from typing import Any, Iterator, Mapping

import numpy as np

from vertebrae.utils.validation import is_sparse_matrix


class ReadOnlyMapping(Mapping[Any, Any]):
    """Small pickle-friendly immutable mapping used by protocol objects."""

    def __init__(self, values: Mapping[Any, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: Any) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"ReadOnlyMapping({self._values!r})"

    def __reduce__(self) -> Any:
        return type(self), (self._values,)


def immutable_value(value: Any) -> Any:
    """Return a recursively immutable detached protocol value."""

    snapshot = snapshot_value(value)
    if isinstance(snapshot, Mapping):
        return ReadOnlyMapping(
            {deepcopy(key): immutable_value(item) for key, item in snapshot.items()}
        )
    if isinstance(snapshot, tuple):
        return tuple(immutable_value(item) for item in snapshot)
    if isinstance(snapshot, frozenset):
        return frozenset(immutable_value(item) for item in snapshot)
    return snapshot


def snapshot_value(value: Any) -> Any:
    """Return a detached snapshot, making numeric storage read-only where possible."""

    if isinstance(value, np.ndarray):
        copied = deepcopy(value) if value.dtype.hasobject else value.copy()
        copied.setflags(write=False)
        return copied
    if is_sparse_matrix(value):
        copied = value.copy()
        for attribute in ("data", "indices", "indptr", "row", "col"):
            array = getattr(copied, attribute, None)
            if isinstance(array, np.ndarray):
                array.setflags(write=False)
        return copied
    if isinstance(value, Mapping):
        return {deepcopy(key): snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(snapshot_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(snapshot_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(snapshot_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(snapshot_value(item) for item in value)
    return deepcopy(value)


def outward_copy(value: Any) -> Any:
    """Return a detached caller-owned copy of a private protocol snapshot."""

    if isinstance(value, np.ndarray):
        copied = deepcopy(value) if value.dtype.hasobject else value.copy()
        copied.setflags(write=False)
        return copied
    if is_sparse_matrix(value):
        copied = value.copy()
        for attribute in ("data", "indices", "indptr", "row", "col"):
            array = getattr(copied, attribute, None)
            if isinstance(array, np.ndarray):
                array.setflags(write=False)
        return copied
    if isinstance(value, Mapping):
        return {deepcopy(key): outward_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(outward_copy(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(outward_copy(item) for item in value)
    return deepcopy(value)
