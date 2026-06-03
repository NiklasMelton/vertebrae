"""Validation helpers for dense and sparse embeddings."""

from typing import Any

import numpy as np


def ensure_2d_numeric_array(value: Any, name: str) -> np.ndarray:
    """Validate a dense 2D numeric array.

    Args:
        value: Array-like value to validate.
        name: Human-readable name used in error messages.

    Returns:
        A NumPy array with numeric, finite values.

    Raises:
        ValueError: If `value` is sparse, non-2D, non-numeric, or non-finite.
    """

    if is_sparse_matrix(value):
        raise ValueError(f"{name} must be dense; got sparse matrix with shape {value.shape}.")
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values.")
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(float, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return arr


def ensure_dense_numeric_2d(value: Any, name: str) -> np.ndarray:
    """Validate dense 2D numeric output from an extractor.

    Args:
        value: Array-like value to validate.
        name: Human-readable name used in error messages.

    Returns:
        A dense numeric NumPy array.

    Raises:
        ValueError: If `value` is sparse or invalid.
    """

    return ensure_2d_numeric_array(value, name)


def ensure_numeric_matrix(value: Any, name: str, allow_sparse: bool = True) -> Any:
    """Validate dense or sparse numeric 2D embeddings.

    Args:
        value: Dense array-like object or scipy sparse matrix.
        name: Human-readable name used in error messages.
        allow_sparse: Whether sparse matrices are accepted.

    Returns:
        A NumPy array for dense input, or the original sparse matrix for sparse input.

    Raises:
        ValueError: If the matrix is not 2D, numeric, finite, or sparse is disallowed.
    """

    if is_sparse_matrix(value):
        if not allow_sparse:
            raise ValueError(f"{name} must be dense; got sparse matrix with shape {value.shape}.")
        return ensure_sparse_numeric_2d(value, name)
    return ensure_2d_numeric_array(value, name)


def ensure_sparse_numeric_2d(value: Any, name: str) -> Any:
    """Validate a scipy sparse numeric matrix.

    Args:
        value: Sparse matrix to validate.
        name: Human-readable name used in error messages.

    Returns:
        The original sparse matrix.

    Raises:
        ValueError: If the sparse matrix is non-2D, non-numeric, or non-finite.
    """

    if not is_sparse_matrix(value):
        raise ValueError(f"{name} must be a scipy sparse matrix.")
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix; got shape {value.shape}.")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values.")
    if not np.all(np.isfinite(value.data)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return value


def sparse_to_dense(value: Any, name: str, max_dense_bytes: int) -> np.ndarray:
    """Convert a sparse matrix to dense after checking memory size.

    Args:
        value: Sparse matrix to densify.
        name: Human-readable name used in error messages.
        max_dense_bytes: Maximum allowed dense allocation size.

    Returns:
        A dense numeric NumPy array.

    Raises:
        ValueError: If the dense representation would exceed `max_dense_bytes`.
    """

    sparse = ensure_sparse_numeric_2d(value, name)
    dense_bytes = estimate_dense_nbytes(sparse)
    if dense_bytes > max_dense_bytes:
        raise ValueError(
            f"{name} would require {dense_bytes} bytes when densified, exceeding "
            f"max_dense_bytes={max_dense_bytes}."
        )
    return ensure_2d_numeric_array(sparse.toarray(), name)


def estimate_dense_nbytes(value: Any) -> int:
    """Estimate bytes required to represent a matrix densely."""

    dtype = getattr(value, "dtype", np.dtype(float))
    rows, cols = value.shape
    return int(rows) * int(cols) * int(np.dtype(dtype).itemsize)


def is_sparse_matrix(value: Any) -> bool:
    """Return whether `value` is a scipy sparse matrix."""

    try:
        from scipy import sparse
    except ImportError:
        return False
    return bool(sparse.issparse(value))


def l2_normalize_rows(value: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a dense numeric array."""

    norms = np.linalg.norm(value, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return value / norms
