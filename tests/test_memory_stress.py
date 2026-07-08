import os

import numpy as np
import pytest

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import (
    CacheConfig,
    EmbeddingConfig,
    MemoryConfig,
    OverlapScoringConfig,
    ProbeConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.extractors import CallableExtractor

pytestmark = [
    pytest.mark.memorystress,
    pytest.mark.skipif(
        os.environ.get("VERTABRAE_RUN_MEMORY_STRESS") != "1",
        reason="set VERTABRAE_RUN_MEMORY_STRESS=1 to run memory stress tests",
    ),
]


def test_large_streaming_embedding_auto_subsamples_under_memory_budget(tmp_path):
    n_per_class = int(os.environ.get("VERTABRAE_MEMORY_STRESS_SAMPLES_PER_CLASS", "1000"))
    feature_dim = int(os.environ.get("VERTABRAE_MEMORY_STRESS_INPUT_DIM", "24"))
    embedding_dim = int(os.environ.get("VERTABRAE_MEMORY_STRESS_EMBEDDING_DIM", "512"))
    dataset = _large_classification_dataset(n_per_class=n_per_class, feature_dim=feature_dim)
    extractor = CallableExtractor(
        "large_streaming_projection",
        transform_fn=_wide_projection,
        streaming_safe=True,
        recipe_data={"embedding_dim": embedding_dim},
    )

    result = Benchmark(
        dataset,
        extractors=[extractor],
        scoring_config=OverlapScoringConfig(
            k=1,
            min_samples_per_cluster=1,
            kmeans_kwargs={"random_state": 29, "batch_size": 128, "n_init": 2},
        ),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path / "memory-cache")),
        embedding_config=EmbeddingConfig(batch_size=128),
        memory_config=MemoryConfig(
            max_memory_bytes=2_500_000,
            subsample_random_state=17,
            min_subsample_samples_per_class=10,
        ),
    ).run()

    item = result.extractor_results[0]
    metadata = item.embedding_metadata
    estimate = metadata["memory_estimate"]

    assert metadata["subsampled"] is True
    assert metadata["subsample_reason"] == "memory_limit"
    assert metadata["parent_n_samples"] == len(dataset.y)
    assert metadata["n_samples"] < len(dataset.y)
    assert metadata["n_samples"] >= 20
    assert metadata["sample_indices"] == sorted(metadata["sample_indices"])
    assert estimate["embedding_dim"] == embedding_dim
    assert estimate["dense_scoring_bytes"] <= 2_500_000
    assert any("memory estimate exceeded" in warning for warning in item.warnings)
    assert 0.0 <= item.overlap.score <= 1.0


def test_user_requested_large_subsample_preserves_class_balance(tmp_path):
    dataset = _large_classification_dataset(n_per_class=750, feature_dim=16)
    extractor = CallableExtractor(
        "large_user_subsample",
        transform_fn=lambda batch: np.asarray(batch, dtype=np.float32)[:, :8],
        streaming_safe=True,
    )

    result = Benchmark(
        dataset,
        extractors=[extractor],
        scoring_config=OverlapScoringConfig(
            k=1,
            min_samples_per_cluster=1,
            kmeans_kwargs={"random_state": 31, "batch_size": 128, "n_init": 2},
        ),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path / "subsample-cache")),
        embedding_config=EmbeddingConfig(batch_size=128),
        memory_config=MemoryConfig(
            max_memory_bytes=100_000_000,
            subsample_rate=0.2,
            subsample_random_state=23,
        ),
    ).run()

    item = result.extractor_results[0]
    metadata = item.embedding_metadata
    selected_labels = dataset.y[np.asarray(metadata["sample_indices"], dtype=int)]
    labels, counts = np.unique(selected_labels, return_counts=True)

    assert metadata["subsampled"] is True
    assert metadata["subsample_reason"] == "user_requested"
    assert metadata["n_samples"] == 300
    assert set(labels.tolist()) == {"left", "right"}
    assert counts.tolist() == [150, 150]
    assert item.embedding_metadata["streamed"] is True
    assert 0.0 <= item.overlap.score <= 1.0


def _large_classification_dataset(n_per_class, feature_dim):
    rng = np.random.default_rng(123)
    left = rng.normal(loc=0.0, scale=0.2, size=(n_per_class, feature_dim))
    right = rng.normal(loc=1.5, scale=0.2, size=(n_per_class, feature_dim))
    X = np.vstack([left, right]).astype(np.float32)
    y = np.array(["left"] * n_per_class + ["right"] * n_per_class)
    return BenchmarkDataset.from_arrays(X, y, modality="tabular")


def _wide_projection(batch):
    embedding_dim = int(os.environ.get("VERTABRAE_MEMORY_STRESS_EMBEDDING_DIM", "512"))
    values = np.asarray(batch, dtype=np.float32)
    base = values.mean(axis=1, keepdims=True)
    offsets = np.linspace(0.0, 1.0, embedding_dim, dtype=np.float32).reshape(1, -1)
    return base + offsets
