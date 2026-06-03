"""Execution backends."""

from vertebrae.execution.jobs import EmbeddingJob, SampleBatch, ShardSpec
from vertebrae.execution.local import LocalBackend

__all__ = ["EmbeddingJob", "LocalBackend", "SampleBatch", "ShardSpec"]
