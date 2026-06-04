"""Distributed-ready job and shard helpers."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ShardSpec:
    """Deterministic non-overlapping sample shard specification.

    Args:
        total_shards: Total number of shards in the embedding job.
        shard_index: Zero-based index for this shard.
    """

    total_shards: int = 1
    shard_index: int = 0

    def __post_init__(self) -> None:
        """Validate shard bounds.

        Raises:
            ValueError: If shard counts or index bounds are invalid.
        """

        if self.total_shards < 1:
            raise ValueError("total_shards must be >= 1.")
        if not 0 <= self.shard_index < self.total_shards:
            raise ValueError("shard_index must be in [0, total_shards).")

    def owns(self, sample_index: int) -> bool:
        """Return whether this shard owns a sample index.

        Args:
            sample_index: Zero-based sample index.

        Returns:
            Whether the sample belongs to this shard.
        """

        return sample_index % self.total_shards == self.shard_index

    def indices(self, n_samples: int) -> np.ndarray:
        """Return all sample indices owned by this shard.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            One-dimensional array of non-overlapping sample indices.
        """

        return np.arange(self.shard_index, n_samples, self.total_shards, dtype=int)

    @property
    def is_complete(self) -> bool:
        """Return whether this spec covers the complete dataset."""

        return self.total_shards == 1 and self.shard_index == 0


@dataclass(frozen=True)
class SampleBatch:
    """A deterministic batch of sample references.

    Attributes:
        indices: Original dataset row indices for this batch.
        X: Batch inputs sliced from the dataset.
    """

    indices: np.ndarray
    X: Any


@dataclass(frozen=True)
class EmbeddingJob:
    """Description of a future distributed embedding job.

    Attributes:
        dataset_id: Dataset identifier or fingerprint.
        extractor_id: Extractor identifier or fingerprint.
        recipe_hash: Hash of the extractor recipe.
        shard: Deterministic shard assignment.
        output_uri: Destination embedding artifact URI or path.
    """

    dataset_id: str
    extractor_id: str
    recipe_hash: str
    shard: ShardSpec
    output_uri: str


@dataclass(frozen=True)
class ResourceSpec:
    """Resource request for local or distributed work.

    Attributes:
        cpus: Number of CPU cores requested.
        memory_bytes: Optional memory budget for the job.
        gpus: Number of GPUs requested.
        gpu_memory_bytes: Optional GPU memory budget.
        walltime_seconds: Optional walltime limit for schedulers such as SLURM.
        queue: Optional queue, partition, or scheduling lane.
    """

    cpus: int = 1
    memory_bytes: Optional[int] = None
    gpus: int = 0
    gpu_memory_bytes: Optional[int] = None
    walltime_seconds: Optional[int] = None
    queue: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate resource bounds.

        Raises:
            ValueError: If resource values are negative or zero where invalid.
        """

        if self.cpus < 1:
            raise ValueError("ResourceSpec.cpus must be >= 1.")
        if self.memory_bytes is not None and self.memory_bytes < 1:
            raise ValueError("ResourceSpec.memory_bytes must be >= 1.")
        if self.gpus < 0:
            raise ValueError("ResourceSpec.gpus must be >= 0.")
        if self.gpu_memory_bytes is not None and self.gpu_memory_bytes < 1:
            raise ValueError("ResourceSpec.gpu_memory_bytes must be >= 1.")
        if self.walltime_seconds is not None and self.walltime_seconds < 1:
            raise ValueError("ResourceSpec.walltime_seconds must be >= 1.")


@dataclass(frozen=True)
class EmbeddingShardJob:
    """Executable embedding shard job.

    Attributes:
        dataset: Dataset object available to the worker.
        extractor: Feature extractor available to the worker.
        shard: Deterministic shard assignment.
        output_key: Artifact-store key for the shard output.
        batch_size: Number of samples per transform batch.
        resources: Resource request for the shard.
    """

    dataset: Any
    extractor: Any
    shard: ShardSpec
    output_key: str
    batch_size: int = 128
    resources: ResourceSpec = ResourceSpec()

    def __post_init__(self) -> None:
        """Validate job settings.

        Raises:
            ValueError: If `batch_size` is invalid.
        """

        if self.batch_size < 1:
            raise ValueError("EmbeddingShardJob.batch_size must be >= 1.")


@dataclass(frozen=True)
class EmbeddingMergeJob:
    """Executable embedding merge job.

    Attributes:
        shard_keys: Artifact-store keys for shard outputs.
        output_key: Artifact-store key for the merged embedding artifact.
        n_samples: Expected full dataset sample count.
        resources: Resource request for the merge.
    """

    shard_keys: tuple[str, ...]
    output_key: str
    n_samples: int
    resources: ResourceSpec = ResourceSpec()

    def __post_init__(self) -> None:
        """Validate merge job settings.

        Raises:
            ValueError: If no shards are provided or sample count is invalid.
        """

        if not self.shard_keys:
            raise ValueError("EmbeddingMergeJob.shard_keys must not be empty.")
        if self.n_samples < 1:
            raise ValueError("EmbeddingMergeJob.n_samples must be >= 1.")


@dataclass(frozen=True)
class ScoringJob:
    """Description of a scoring job over persisted embeddings.

    Attributes:
        embedding_key: Artifact-store key for embeddings.
        labels_key: Artifact-store key or URI for labels.
        output_key: Artifact-store key for scoring results.
        scoring_config: Optional OverlapIndex scoring configuration.
        seed: Optional scoring seed, commonly used for stability repeats.
        resources: Resource request for scoring.
    """

    embedding_key: str
    labels_key: str
    output_key: str
    scoring_config: Any = None
    seed: Optional[int] = None
    resources: ResourceSpec = ResourceSpec()
