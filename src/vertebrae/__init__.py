"""Feature extractor benchmarking with MiniBatchKMeans-backed OverlapIndex."""

__version__ = "0.1.0"

from vertebrae.benchmark import Benchmark
from vertebrae.config import (
    CacheConfig,
    EmbeddingConfig,
    OverlapScoringConfig,
    ProbeConfig,
    StabilityConfig,
)
from vertebrae.datasets import BenchmarkDataset
from vertebrae.evaluator import Evaluator
from vertebrae.execution import EmbeddingJob, SampleBatch, ShardSpec
from vertebrae.results import BenchmarkResult, ExtractorResult

__all__ = [
    "Benchmark",
    "BenchmarkDataset",
    "BenchmarkResult",
    "CacheConfig",
    "EmbeddingConfig",
    "EmbeddingJob",
    "Evaluator",
    "ExtractorResult",
    "OverlapScoringConfig",
    "ProbeConfig",
    "SampleBatch",
    "ShardSpec",
    "StabilityConfig",
]
