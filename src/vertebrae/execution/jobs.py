"""Distributed-ready job and shard helpers."""

from dataclasses import dataclass
from typing import Any

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
