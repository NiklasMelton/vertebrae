"""Memory estimation and admission helpers."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import psutil

from vertebrae.config import MemoryConfig
from vertebrae.utils.validation import estimate_dense_nbytes, is_sparse_matrix


@dataclass(frozen=True)
class MemoryBudget:
    """Resolved memory budget from a `MemoryConfig`.

    Attributes:
        total_bytes: Total system memory.
        available_bytes: Currently available system memory.
        reserve_system_bytes: Bytes reserved for system use.
        max_memory_bytes: Maximum bytes vertebrae should use for planned work.
    """

    total_bytes: int
    available_bytes: int
    reserve_system_bytes: int
    max_memory_bytes: int


@dataclass(frozen=True)
class EmbeddingMemoryEstimate:
    """Estimated memory footprint for an embedding artifact.

    Attributes:
        n_samples: Number of embedding rows.
        embedding_dim: Number of embedding columns.
        dtype: Embedding dtype string.
        resident_bytes: Estimated bytes to hold the embedding artifact in memory.
        dense_scoring_bytes: Estimated dense bytes required by scoring.
        batch_embedding_bytes: Estimated bytes for one embedding batch.
        strategy: Planned strategy: `"in_memory"` or `"stream_to_disk"`.
    """

    n_samples: int
    embedding_dim: int
    dtype: str
    resident_bytes: int
    dense_scoring_bytes: int
    batch_embedding_bytes: int
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the estimate for result metadata.

        Returns:
            JSON-compatible memory estimate.
        """

        return {
            "n_samples": self.n_samples,
            "embedding_dim": self.embedding_dim,
            "dtype": self.dtype,
            "resident_bytes": self.resident_bytes,
            "dense_scoring_bytes": self.dense_scoring_bytes,
            "batch_embedding_bytes": self.batch_embedding_bytes,
            "strategy": self.strategy,
        }


def resolve_memory_budget(config: MemoryConfig) -> MemoryBudget:
    """Resolve an effective memory budget using psutil.

    Args:
        config: Memory configuration.

    Returns:
        Resolved memory budget.
    """

    memory = psutil.virtual_memory()
    total = int(memory.total)
    available = int(memory.available)
    reserve = (
        int(config.reserve_system_bytes)
        if config.reserve_system_bytes is not None
        else _default_reserve_bytes(total)
    )
    if config.max_memory_bytes is not None:
        limit = int(config.max_memory_bytes)
    else:
        limit = int(min(available * config.max_fraction, max(1, available - reserve)))
    return MemoryBudget(
        total_bytes=total,
        available_bytes=available,
        reserve_system_bytes=reserve,
        max_memory_bytes=max(1, limit),
    )


def estimate_embedding_from_probe(
    probe_embeddings: Any,
    n_samples: int,
    batch_size: int,
    memory_config: MemoryConfig,
) -> EmbeddingMemoryEstimate:
    """Estimate full embedding memory from a probe batch.

    Args:
        probe_embeddings: Dense or sparse probe embedding batch.
        n_samples: Full dataset sample count.
        batch_size: Planned embedding batch size.
        memory_config: Memory configuration.

    Returns:
        Estimated embedding footprint and strategy.
    """

    if is_sparse_matrix(probe_embeddings):
        dim = int(probe_embeddings.shape[1])
        dtype = str(probe_embeddings.dtype)
        density = _safe_density(probe_embeddings)
        resident = estimate_sparse_bytes(
            n_samples=n_samples,
            n_features=dim,
            dtype=np.dtype(probe_embeddings.dtype),
            density=density,
        )
        dense_scoring = n_samples * dim * np.dtype(probe_embeddings.dtype).itemsize
        batch_bytes = estimate_sparse_bytes(
            n_samples=batch_size,
            n_features=dim,
            dtype=np.dtype(probe_embeddings.dtype),
            density=density,
        )
    else:
        arr = np.asarray(probe_embeddings)
        dim = int(arr.shape[1])
        dtype = str(arr.dtype)
        resident = n_samples * dim * np.dtype(arr.dtype).itemsize
        dense_scoring = resident
        batch_bytes = batch_size * dim * np.dtype(arr.dtype).itemsize
    budget = resolve_memory_budget(memory_config)
    strategy = (
        "stream_to_disk"
        if memory_config.allow_disk_spill and resident > budget.max_memory_bytes
        else "in_memory"
    )
    return EmbeddingMemoryEstimate(
        n_samples=n_samples,
        embedding_dim=dim,
        dtype=dtype,
        resident_bytes=int(resident),
        dense_scoring_bytes=int(dense_scoring),
        batch_embedding_bytes=int(batch_bytes),
        strategy=strategy,
    )


def estimate_matrix_resident_bytes(matrix: Any) -> int:
    """Estimate memory required to hold a dense or sparse matrix.

    Args:
        matrix: Dense or sparse matrix.

    Returns:
        Estimated resident bytes.
    """

    if is_sparse_matrix(matrix):
        return sparse_matrix_nbytes(matrix)
    return estimate_dense_nbytes(np.asarray(matrix))


def estimate_metadata_resident_bytes(metadata: dict[str, Any]) -> Optional[int]:
    """Estimate resident bytes from embedding metadata.

    Args:
        metadata: Embedding metadata dictionary.

    Returns:
        Estimated bytes, or `None` if metadata is incomplete.
    """

    shape = metadata.get("shape")
    dtype = metadata.get("dtype")
    if not shape or len(shape) != 2 or dtype is None:
        return None
    if metadata.get("sparse"):
        nnz = metadata.get("nnz")
        if nnz is None:
            return None
        return sparse_nbytes_from_nnz(
            nnz=int(nnz),
            n_rows=int(shape[0]),
            dtype=np.dtype(dtype),
        )
    return int(shape[0]) * int(shape[1]) * np.dtype(dtype).itemsize


def estimate_metadata_dense_scoring_bytes(metadata: dict[str, Any]) -> Optional[int]:
    """Estimate dense bytes needed for scoring from embedding metadata.

    Args:
        metadata: Embedding metadata dictionary.

    Returns:
        Estimated dense scoring bytes, or `None` if metadata is incomplete.
    """

    shape = metadata.get("shape")
    dtype = metadata.get("dtype")
    if not shape or len(shape) != 2 or dtype is None:
        return None
    return int(shape[0]) * int(shape[1]) * np.dtype(dtype).itemsize


def assert_within_memory(
    required_bytes: int,
    memory_config: MemoryConfig,
    purpose: str,
) -> MemoryBudget:
    """Fail fast when a planned allocation exceeds the memory budget.

    Args:
        required_bytes: Estimated required bytes.
        memory_config: Memory configuration.
        purpose: Human-readable description of the planned work.

    Returns:
        Resolved memory budget.

    Raises:
        ValueError: If `required_bytes` exceeds the configured budget.
    """

    budget = resolve_memory_budget(memory_config)
    if memory_config.fail_fast and required_bytes > budget.max_memory_bytes:
        raise ValueError(
            f"{purpose} is estimated to require {required_bytes} bytes, exceeding "
            f"the memory budget of {budget.max_memory_bytes} bytes. Increase "
            "MemoryConfig.max_memory_bytes, enable disk spill where applicable, "
            "reduce batch size, or run fewer concurrent jobs."
        )
    return budget


def largest_fitting_subsample_rate(
    required_bytes: int,
    memory_config: MemoryConfig,
) -> float:
    """Estimate the largest sample fraction that fits the memory budget.

    Args:
        required_bytes: Estimated bytes for the full sample set.
        memory_config: Memory configuration.

    Returns:
        Fraction in `(0, 1]` that should fit the configured budget.
    """

    if required_bytes < 1:
        return 1.0
    budget = resolve_memory_budget(memory_config)
    return min(1.0, max(0.0, budget.max_memory_bytes / float(required_bytes)))


def sparse_matrix_nbytes(matrix: Any) -> int:
    """Estimate resident bytes for an existing scipy sparse matrix."""

    total = int(matrix.data.nbytes)
    for attr in ("indices", "indptr", "row", "col"):
        values = getattr(matrix, attr, None)
        if values is not None:
            total += int(values.nbytes)
    return total


def sparse_nbytes_from_nnz(nnz: int, n_rows: int, dtype: np.dtype) -> int:
    """Estimate CSR sparse bytes from non-zero count and row count."""

    index_bytes = np.dtype(np.int32).itemsize
    return int(nnz) * (np.dtype(dtype).itemsize + index_bytes) + (int(n_rows) + 1) * index_bytes


def estimate_sparse_bytes(
    n_samples: int,
    n_features: int,
    dtype: np.dtype,
    density: float,
) -> int:
    """Estimate sparse matrix bytes from shape, dtype, and density."""

    nnz = int(round(n_samples * n_features * density))
    return sparse_nbytes_from_nnz(nnz=nnz, n_rows=n_samples, dtype=dtype)


def _default_reserve_bytes(total_bytes: int) -> int:
    gib = 1024**3
    return int(min(8 * gib, max(1 * gib, total_bytes * 0.2)))


def _safe_density(matrix: Any) -> float:
    total = int(matrix.shape[0]) * int(matrix.shape[1])
    if total == 0:
        return 0.0
    return min(1.0, max(0.0, float(matrix.nnz) / float(total)))
