"""Deterministic batching helpers shared by endpoint embedding workflows."""

from collections.abc import Mapping
from typing import Any, Callable, Iterator, Optional, Tuple

import numpy as np

from vertebrae.config import MemoryConfig
from vertebrae.utils.memory import (
    IncrementalMatrixReferenceStager,
    IncrementalMatrixStager,
)
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
    memory_config: Optional[MemoryConfig] = None,
) -> Any:
    """Encode and combine deterministic endpoint batches without sparse densification.

    Encoded rows and their final ordering references are admitted progressively
    through shared matrix and metadata stagers. No-spill runs fail as soon as
    retaining the next row or reference would cross the configured budget;
    spill-enabled runs write both to temporary storage immediately. Omitting
    ``memory_config`` uses the default memory policy.
    """

    effective_memory_config = memory_config or MemoryConfig()
    n_rows = endpoint_n_rows(values)
    expected_dim: Optional[int] = None
    expected_sparse: Optional[bool] = None
    expected_dtype: Any = None
    next_position = 0
    with (
        IncrementalMatrixStager(
            effective_memory_config,
            purpose=owner,
        ) as stager,
        IncrementalMatrixReferenceStager(
            effective_memory_config,
            purpose=owner,
            matrix_stager=stager,
        ) as reference_stager,
    ):
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
            for row_index in range(int(matrix.shape[0])):
                reference_stager.append(
                    "endpoint",
                    next_position,
                    stager.append("endpoint", matrix[row_index : row_index + 1]),
                )
                next_position += 1
        if expected_dim is None:
            raise ValueError(f"{owner} requires at least one input row.")
        return reference_stager.assemble(
            "endpoint",
            expected_rows=n_rows,
            purpose=owner,
        ).matrix
