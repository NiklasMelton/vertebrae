"""Feature extractor benchmarking with MiniBatchKMeans-backed OverlapIndex."""

__version__ = "0.1.0"

from vertebrae.benchmark import Benchmark
from vertebrae.config import (
    CacheConfig,
    EmbeddingConfig,
    MemoryConfig,
    OverlapScoringConfig,
    ProbeConfig,
    StabilityConfig,
)
from vertebrae.datasets import BenchmarkDataset
from vertebrae.evaluator import Evaluator
from vertebrae.execution import (
    EmbeddingJob,
    EmbeddingMergeJob,
    EmbeddingShardJob,
    LocalBackend,
    ResourceSpec,
    SampleBatch,
    ScoringJob,
    ShardSpec,
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
from vertebrae.results import BenchmarkResult, ExtractorResult

__all__ = [
    "Benchmark",
    "BenchmarkDataset",
    "BenchmarkResult",
    "CacheConfig",
    "EmbeddingConfig",
    "EmbeddingJob",
    "EmbeddingMergeJob",
    "EmbeddingShardJob",
    "Evaluator",
    "ExtractorResult",
    "LocalBackend",
    "MemoryConfig",
    "OverlapScoringConfig",
    "ProbeConfig",
    "ResourceSpec",
    "SampleBatch",
    "ScoringJob",
    "ShardSpec",
    "StabilityConfig",
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
