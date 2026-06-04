"""Execution backends."""

from vertebrae.execution.distributed import (
    embedding_artifact_key,
    embedding_shard_key,
    labels_artifact_key,
    materialize_and_merge_embeddings,
    materialize_embedding_shard,
    materialize_embedding_shards,
    materialize_label_artifact,
    merge_embedding_shards,
    plan_embedding_shard_jobs,
    score_embedding_artifact,
    score_embedding_artifacts,
    scoring_artifact_key,
)
from vertebrae.execution.jobs import (
    EmbeddingJob,
    EmbeddingMergeJob,
    EmbeddingShardJob,
    ResourceSpec,
    SampleBatch,
    ScoringJob,
    ShardSpec,
)
from vertebrae.execution.local import LocalBackend, LocalJobHandle

__all__ = [
    "EmbeddingJob",
    "EmbeddingMergeJob",
    "EmbeddingShardJob",
    "LocalBackend",
    "LocalJobHandle",
    "ResourceSpec",
    "SampleBatch",
    "ScoringJob",
    "ShardSpec",
    "embedding_artifact_key",
    "embedding_shard_key",
    "labels_artifact_key",
    "materialize_and_merge_embeddings",
    "materialize_embedding_shard",
    "materialize_embedding_shards",
    "materialize_label_artifact",
    "merge_embedding_shards",
    "plan_embedding_shard_jobs",
    "score_embedding_artifact",
    "score_embedding_artifacts",
    "scoring_artifact_key",
]
