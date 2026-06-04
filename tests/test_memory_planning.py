import numpy as np
import pytest

from vertebrae import BenchmarkDataset, EmbeddingConfig, Evaluator, MemoryConfig
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import CallableExtractor
from vertebrae.utils.memory import resolve_memory_budget


def test_memory_config_resolves_explicit_budget():
    budget = resolve_memory_budget(MemoryConfig(max_memory_bytes=123_456))

    assert budget.max_memory_bytes == 123_456
    assert budget.available_bytes > 0


def test_streaming_embedding_fails_after_probe_when_auto_subsampling_disabled(
    fake_overlapindex,
):
    seen = []

    def embed(batch):
        values = np.asarray(batch)
        seen.extend(values[:, 0].astype(int).tolist())
        return np.ones((len(values), 100), dtype=np.float64)

    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        ["a"] * 5 + ["b"] * 5,
        modality="tabular",
    )
    extractor = CallableExtractor("large_embeddings", embed, streaming_safe=True)

    with pytest.raises(ValueError, match="Dense scoring input"):
        Evaluator(
            dataset=dataset,
            extractor=extractor,
            scoring_config=OverlapScoringConfig(k=1),
            stability_config=StabilityConfig(enabled=False),
            probe_config=ProbeConfig(enabled=False),
            cache_config=CacheConfig(enabled=False),
            embedding_config=EmbeddingConfig(batch_size=2),
            memory_config=MemoryConfig(
                max_memory_bytes=2_000,
                auto_subsample_on_memory_exceeded=False,
            ),
        ).run()

    assert seen == [0, 3]


def test_streaming_embedding_auto_subsamples_when_scoring_would_exceed_memory(
    tmp_path,
    fake_overlapindex,
):
    seen = []

    def embed(batch):
        values = np.asarray(batch)
        seen.extend(values[:, 0].astype(int).tolist())
        return np.ones((len(values), 100), dtype=np.float64)

    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        ["a"] * 5 + ["b"] * 5,
        modality="tabular",
    )
    extractor = CallableExtractor("large_embeddings", embed, streaming_safe=True)

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=2),
        memory_config=MemoryConfig(max_memory_bytes=4_000),
    ).run()

    extractor_result = result.extractor_results[0]
    metadata = extractor_result.embedding_metadata
    assert metadata["subsampled"] is True
    assert metadata["subsample_reason"] == "memory_limit"
    assert metadata["n_samples"] == 4
    assert metadata["effective_subsample_rate"] == 0.4
    assert len(metadata["sample_indices"]) == 4
    assert any("memory estimate exceeded" in warning for warning in extractor_result.warnings)
    assert seen[:2] == [0, 3]
    assert len(seen[2:]) == 4


def test_user_requested_subsample_rate_limits_benchmark_samples(tmp_path, fake_overlapindex):
    seen = []

    def embed(batch):
        values = np.asarray(batch)
        seen.extend(values[:, 0].astype(int).tolist())
        return np.asarray(batch)[:, :2].astype(np.float32)

    dataset = BenchmarkDataset.from_arrays(
        np.arange(36).reshape(12, 3),
        ["a"] * 6 + ["b"] * 6,
        modality="tabular",
    )
    extractor = CallableExtractor("subsampled_embeddings", embed, streaming_safe=True)

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=3),
        memory_config=MemoryConfig(max_memory_bytes=10_000_000, subsample_rate=0.5),
    ).run()

    extractor_result = result.extractor_results[0]
    metadata = extractor_result.embedding_metadata
    assert metadata["subsampled"] is True
    assert metadata["subsample_reason"] == "user_requested"
    assert metadata["n_samples"] == 6
    assert metadata["effective_subsample_rate"] == 0.5
    assert set(metadata["sample_indices"]).issubset(set(range(12)))
    assert len(seen) == 6
    assert any("user-requested" in warning for warning in extractor_result.warnings)


def test_streaming_embedding_records_memory_estimate(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
    )
    extractor = CallableExtractor(
        "small_embeddings",
        lambda batch: np.asarray(batch)[:, :2].astype(np.float32),
        streaming_safe=True,
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=3),
        memory_config=MemoryConfig(max_memory_bytes=10_000_000),
    ).run()

    estimate = result.extractor_results[0].embedding_metadata["memory_estimate"]
    assert estimate["embedding_dim"] == 2
    assert estimate["resident_bytes"] == 64
    assert estimate["strategy"] == "in_memory"


def test_streaming_probe_is_reused_when_no_subsampling_is_needed(tmp_path, fake_overlapindex):
    seen = []

    def embed(batch):
        values = np.asarray(batch)
        seen.extend(values[:, 0].astype(int).tolist())
        return np.asarray(batch)[:, :2].astype(np.float32)

    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
    )
    extractor = CallableExtractor("small_embeddings", embed, streaming_safe=True)

    Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=3),
        memory_config=MemoryConfig(max_memory_bytes=10_000_000),
    ).run()

    assert seen == list(range(0, 24, 3))
