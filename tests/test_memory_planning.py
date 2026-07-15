import numpy as np
import pytest

from vertebrae import BenchmarkDataset, DatasetIdentity, EmbeddingConfig, Evaluator, MemoryConfig
from vertebrae.config import (
    CacheConfig,
    ContinuousOverlapScoringConfig,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.extractors import CallableExtractor
from vertebrae.utils.memory import resolve_memory_budget


class _FitRowTrackingExtractor:
    name = "fit_row_tracking"
    modality = "tabular"
    extractor_type = "test"
    streaming_safe = True

    def __init__(self):
        self.fit_calls = 0
        self.fit_rows = None

    def fit(self, X, y=None):
        del y
        self.fit_calls += 1
        self.fit_rows = np.asarray(X).copy()
        return self

    def transform(self, X):
        return np.ones((len(X), 100), dtype=np.float64)

    def recipe(self):
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "cache_safe": True,
        }


def test_memory_config_resolves_explicit_budget():
    budget = resolve_memory_budget(MemoryConfig(max_memory_bytes=123_456))

    assert budget.max_memory_bytes == 123_456
    assert budget.available_bytes > 0


def test_memory_config_falls_back_to_available_fraction_when_reserve_exceeds_available(
    monkeypatch,
):
    class _Memory:
        total = 16_000
        available = 400

    monkeypatch.setattr("vertebrae.utils.memory.psutil.virtual_memory", lambda: _Memory())

    budget = resolve_memory_budget(MemoryConfig(max_fraction=0.5))

    assert budget.reserve_system_bytes > budget.available_bytes
    assert budget.max_memory_bytes == 200


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
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor("large_embeddings", embed, streaming_safe=True)

    with pytest.raises(ValueError, match="Dense scoring input"):
        Evaluator(
            dataset=dataset,
            extractor=extractor,
            scoring_config=OverlapScoringConfig(k=1),
            stability_config=StabilityConfig(enabled=False),
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
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor("large_embeddings", embed, streaming_safe=True)

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
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


def test_auto_memory_probe_fits_live_extractor_once_on_final_rows(
    tmp_path,
    fake_overlapindex,
):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        ["a"] * 5 + ["b"] * 5,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = _FitRowTrackingExtractor()

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=2),
        memory_config=MemoryConfig(max_memory_bytes=4_000),
    ).run()

    assert result.extractor_results[0].embedding_metadata["n_samples"] == 4
    assert extractor.fit_calls == 1
    assert extractor.fit_rows is not None
    assert len(extractor.fit_rows) == 4


def test_auto_memory_probe_uses_derived_budget_without_fitting_live_extractor_twice(
    tmp_path,
    fake_overlapindex,
    monkeypatch,
):
    class _Memory:
        total = 8_000
        available = 5_334

    monkeypatch.setattr("vertebrae.utils.memory.psutil.virtual_memory", lambda: _Memory())
    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        ["a"] * 5 + ["b"] * 5,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = _FitRowTrackingExtractor()

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=2),
        memory_config=MemoryConfig(reserve_system_bytes=0),
    ).run()

    metadata = result.extractor_results[0].embedding_metadata
    assert metadata["subsampled"] is True
    assert metadata["n_samples"] == 4
    assert extractor.fit_calls == 1
    assert extractor.fit_rows is not None
    assert len(extractor.fit_rows) == 4


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
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor("subsampled_embeddings", embed, streaming_safe=True)

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
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
    # The disposable sizing clone probes one batch; the live extractor still
    # materializes exactly the six final rows once.
    assert len(seen) == 9
    assert len(seen[:3]) == 3
    assert len(seen[3:]) == 6
    assert any("user-requested" in warning for warning in extractor_result.warnings)


def test_user_requested_regression_subsample_expands_to_valid_minimum(
    tmp_path,
    fake_overlapindex,
):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        np.asarray([0.0] * 9 + [1.0]),
        modality="tabular",
        target_type="regression",
        target_names=["score"],
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "regression_subsample",
        lambda batch: np.asarray(batch)[:, :2].astype(np.float32),
        streaming_safe=True,
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=ContinuousOverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        memory_config=MemoryConfig(max_memory_bytes=10_000_000, subsample_rate=0.01),
    ).run()

    item = result.extractor_results[0]
    assert item.embedding_metadata["requested_subsample_rate"] == 0.01
    assert item.embedding_metadata["effective_subsample_rate"] == 0.3
    assert item.embedding_metadata["n_samples"] == 3
    assert np.var(fake_overlapindex.continuous_calls[-1]["fit_y"]) > 0.0
    assert any(
        "user-requested target-preserving regression subsample" in warning
        for warning in item.warnings
    )


def test_memory_triggered_regression_subsample_preserves_variation(
    tmp_path,
    fake_overlapindex,
):
    def embed(batch):
        return np.ones((len(batch), 100), dtype=np.float64)

    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        np.asarray([0.0] * 9 + [1.0]),
        modality="tabular",
        target_type="regression",
        target_names=["score"],
        identity=DatasetIdentity.ephemeral(),
    )

    result = Evaluator(
        dataset=dataset,
        extractor=CallableExtractor("large_regression_embeddings", embed, streaming_safe=True),
        scoring_config=ContinuousOverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=2),
        memory_config=MemoryConfig(max_memory_bytes=4_000),
    ).run()

    item = result.extractor_results[0]
    assert item.embedding_metadata["subsample_reason"] == "memory_limit"
    assert item.embedding_metadata["requested_subsample_rate"] == 0.5
    assert item.embedding_metadata["effective_subsample_rate"] == 0.5
    assert np.var(fake_overlapindex.continuous_calls[-1]["fit_y"]) > 0.0
    assert any("target-preserving regression subsample" in warning for warning in item.warnings)


def test_streaming_embedding_records_memory_estimate(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
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
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=3),
        memory_config=MemoryConfig(max_memory_bytes=10_000_000),
    ).run()

    estimate = result.extractor_results[0].embedding_metadata["memory_estimate"]
    assert estimate["embedding_dim"] == 2
    assert estimate["resident_bytes"] == 64
    assert estimate["strategy"] == "in_memory"


def test_disposable_streaming_probe_is_not_reused_by_live_extractor(
    tmp_path,
    fake_overlapindex,
):
    seen = []

    def embed(batch):
        values = np.asarray(batch)
        seen.extend(values[:, 0].astype(int).tolist())
        return np.asarray(batch)[:, :2].astype(np.float32)

    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor("small_embeddings", embed, streaming_safe=True)

    Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=3),
        memory_config=MemoryConfig(max_memory_bytes=10_000_000),
    ).run()

    assert seen[:3] == [0, 3, 6]
    assert seen[3:] == list(range(0, 24, 3))
