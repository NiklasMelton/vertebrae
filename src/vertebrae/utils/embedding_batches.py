"""Deterministic batching helpers shared by endpoint embedding workflows."""

from collections.abc import Mapping
from typing import Any, Callable, Iterator, Tuple

import numpy as np

from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


def endpoint_n_rows(values: Any) -> int:
    """Return aligned endpoint row count for common input containers."""

    if isinstance(values, Mapping):
        lengths = {endpoint_n_rows(value) for value in values.values()}
        if len(lengths) != 1:
            raise ValueError("Endpoint mapping values must have aligned row counts.")
        return lengths.pop() if lengths else 0
    return int(len(values))


def take_endpoint_rows(values: Any, indices: np.ndarray) -> Any:
    """Select endpoint rows while preserving common container types."""

    if isinstance(values, Mapping):
        return {key: take_endpoint_rows(value, indices) for key, value in values.items()}
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    if is_sparse_matrix(values) or isinstance(values, np.ndarray):
        return values[indices]
    sequence = list(values)
    return [sequence[int(index)] for index in indices]


def iter_endpoint_batches(values: Any, batch_size: int) -> Iterator[Tuple[np.ndarray, Any]]:
    """Yield deterministic contiguous endpoint batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    n_rows = endpoint_n_rows(values)
    for start in range(0, n_rows, batch_size):
        indices = np.arange(start, min(n_rows, start + batch_size), dtype=int)
        yield indices, take_endpoint_rows(values, indices)


def encode_endpoint_batches(
    values: Any,
    *,
    batch_size: int,
    encode: Callable[[Any], Any],
    owner: str,
    profiler: Any = None,
    call_type: str = "encode_endpoint",
) -> Any:
    """Encode and combine deterministic endpoint batches without sparse densification."""

    batches = []
    expected_dim = None
    expected_sparse = None
    expected_dtype = None
    for indices, batch in iter_endpoint_batches(values, batch_size):

        def call(batch: Any = batch) -> Any:
            return encode(batch)

        encoded = (
            profiler.measure_call(
                call,
                samples=len(indices),
                call_type=call_type,
            )
            if profiler is not None
            else call()
        )
        matrix = ensure_numeric_matrix(encoded, owner, allow_sparse=True)
        if matrix.shape[0] != len(indices):
            raise ValueError(
                f"{owner} returned {matrix.shape[0]} embeddings for {len(indices)} rows."
            )
        is_sparse = is_sparse_matrix(matrix)
        if expected_dim is None:
            expected_dim = int(matrix.shape[1])
            expected_sparse = is_sparse
            expected_dtype = matrix.dtype
        elif (
            int(matrix.shape[1]) != expected_dim
            or is_sparse != expected_sparse
            or matrix.dtype != expected_dtype
        ):
            raise ValueError(f"{owner} changed embedding format between batches.")
        batches.append(matrix)
    if not batches:
        raise ValueError(f"{owner} requires at least one input row.")
    if expected_sparse:
        from scipy import sparse as scipy_sparse

        return scipy_sparse.vstack(batches, format="csr")
    return np.vstack([np.asarray(batch) for batch in batches])
