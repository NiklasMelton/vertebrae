from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

import vertebrae
import vertebrae._version as version_module
from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    BenchmarkExecutionError,
    CacheConfig,
    DatasetIdentity,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    EmbeddingOutputSpec,
    ExecutionConfig,
    LabelViewConfig,
    LocalBackend,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.extractors import CallableExtractor, MultiOutputExtractor


class RecordingBackend(LocalBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def map(self, fn, jobs):
        values = list(jobs)
        self.calls.append((fn, len(values)))
        return super().map(fn, values)


class FailingBackend(LocalBackend):
    def map(self, fn, jobs):
        raise RuntimeError("worker unavailable")


class FitOnceExtractor:
    name = "fit_once"
    extractor_type = "test"
    streaming_safe = True

    def __init__(self) -> None:
        self.fit_calls = 0
        self.offset = None

    def fit(self, X, y=None):
        self.fit_calls += 1
        self.offset = float(np.asarray(X)[0, 0])
        return self

    def transform(self, X):
        if self.offset is None:
            raise RuntimeError("extractor was not fitted")
        return np.asarray(X, dtype=float)[:, :2] + self.offset

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def recipe(self):
        return {"name": self.name, "extractor_type": self.extractor_type}


def _dataset():
    return BenchmarkDataset.from_arrays(
        np.arange(24, dtype=float).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )


def _kwargs(tmp_path):
    return {
        "scoring_config": OverlapScoringConfig(k=1),
        "stability_config": StabilityConfig(enabled=False),
        "separatix_config": SeparatixConfig(enabled=False),
        "cache_config": CacheConfig(enabled=False, cache_dir=str(tmp_path)),
        "embedding_config": EmbeddingConfig(batch_size=2),
    }


def test_runtime_version_comes_from_distribution_metadata():
    assert vertebrae.__version__ == version("vertebrae")


def test_runtime_version_has_unknown_fallback(monkeypatch):
    def missing(_distribution):
        raise version_module.PackageNotFoundError

    monkeypatch.setattr(version_module, "version", missing)

    assert version_module.resolve_version() == "0+unknown"


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"total_shards": 0}, ValueError, "total_shards"),
        ({"total_shards": 1.5}, TypeError, "integer"),
        ({"dispatch_stages": ("scoring", "scoring")}, ValueError, "duplicates"),
        ({"dispatch_stages": ("bogus",)}, ValueError, "unknown stages"),
    ],
)
def test_execution_config_validates_inputs(kwargs, error, match):
    with pytest.raises(error, match=match):
        ExecutionConfig(**kwargs)


def test_execution_config_requires_explicit_backend(tmp_path):
    with pytest.raises(ValueError, match="requires an explicit"):
        Benchmark(
            _dataset(),
            [CallableExtractor("identity", np.asarray)],
            execution_config=ExecutionConfig(total_shards=2),
            **_kwargs(tmp_path),
        )


def test_explicit_backend_runs_artifact_pipeline_and_matches_direct(tmp_path, fake_overlapindex):
    dataset = _dataset()
    direct = Benchmark(
        dataset,
        [CallableExtractor("identity", np.asarray, streaming_safe=True)],
        **_kwargs(tmp_path / "direct"),
    ).run()
    backend = RecordingBackend()
    dispatched = Benchmark(
        dataset,
        [CallableExtractor("identity", np.asarray, streaming_safe=True)],
        execution=backend,
        execution_config=ExecutionConfig(total_shards=3),
        **_kwargs(tmp_path / "dispatched"),
    ).run()

    assert dispatched.extractor_results[0].primary_score == (
        direct.extractor_results[0].primary_score
    )
    assert dispatched.metadata["execution"]["artifact_backed"] is True
    assert dispatched.metadata["vertebrae_version"] == version("vertebrae")
    assert dispatched.metadata["execution"]["effective_total_shards"] == [3]
    assert len(backend.calls) == 2  # embedding and scoring; diagnostics are disabled
    runs_dir = tmp_path / "dispatched" / "runs"
    assert not runs_dir.exists() or not any(runs_dir.iterdir())


def test_local_parallel_backend_runs_end_to_end(tmp_path, fake_overlapindex):
    result = Benchmark(
        _dataset(),
        [CallableExtractor("identity", np.asarray, streaming_safe=True)],
        execution=LocalBackend(n_jobs=2, joblib_backend="threading"),
        execution_config=ExecutionConfig(total_shards=3),
        **_kwargs(tmp_path),
    ).run()

    assert len(result.extractor_results) == 1
    assert result.metadata["execution"]["backend"] == "LocalBackend"


def test_dispatched_embedding_fits_once_before_sharding(tmp_path, fake_overlapindex):
    extractor = FitOnceExtractor()
    result = Benchmark(
        _dataset(),
        [extractor],
        execution=LocalBackend(),
        execution_config=ExecutionConfig(total_shards=4),
        **_kwargs(tmp_path),
    ).run()

    assert result.extractor_results
    assert extractor.fit_calls == 1


def test_dispatch_stages_can_keep_embedding_on_driver(tmp_path, fake_overlapindex):
    backend = RecordingBackend()
    Benchmark(
        _dataset(),
        [CallableExtractor("identity", np.asarray, streaming_safe=True)],
        execution=backend,
        execution_config=ExecutionConfig(total_shards=2, dispatch_stages=("scoring",)),
        **_kwargs(tmp_path),
    ).run()

    assert len(backend.calls) == 1


def test_cache_hit_prunes_embedding_jobs(tmp_path, fake_overlapindex):
    dataset = _dataset()
    config = CacheConfig(enabled=True, cache_dir=str(tmp_path))
    first_backend = RecordingBackend()
    common = {
        **_kwargs(tmp_path),
        "cache_config": config,
    }
    Benchmark(
        dataset,
        [CallableExtractor("identity", np.asarray, streaming_safe=True)],
        execution=first_backend,
        execution_config=ExecutionConfig(total_shards=2),
        **common,
    ).run()
    second_backend = RecordingBackend()
    Benchmark(
        dataset,
        [CallableExtractor("identity", np.asarray, streaming_safe=True)],
        execution=second_backend,
        execution_config=ExecutionConfig(total_shards=2),
        **common,
    ).run()

    assert len(first_backend.calls) == 2
    assert len(second_backend.calls) == 1


def test_uncached_artifacts_can_be_retained_for_inspection(tmp_path, fake_overlapindex):
    result = Benchmark(
        _dataset(),
        [CallableExtractor("identity", np.asarray, streaming_safe=True)],
        execution=LocalBackend(),
        execution_config=ExecutionConfig(
            total_shards=2,
            retain_intermediate_artifacts=True,
        ),
        **_kwargs(tmp_path),
    ).run()

    run_id = result.metadata["execution"]["run_id"]
    assert Path(tmp_path, "runs", run_id).exists()


def test_dispatched_multi_output_compression_and_stability(tmp_path, fake_overlapindex):
    extractor = MultiOutputExtractor(
        "multi",
        output_specs=[EmbeddingOutputSpec("left"), EmbeddingOutputSpec("right")],
        transform_many_fn=lambda values: {
            "left": np.asarray(values)[:, :2],
            "right": np.asarray(values)[:, 1:3],
        },
        streaming_safe=True,
    )
    result = Benchmark(
        _dataset(),
        [extractor],
        execution=LocalBackend(),
        execution_config=ExecutionConfig(total_shards=3),
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(repeats=2),
        separatix_config=SeparatixConfig(enabled=False),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="prefix_truncate",
            n_components=1,
            assume_matryoshka=True,
        ),
        cache_config=CacheConfig(enabled=False, cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=2),
    ).run()

    assert [item.name for item in result.extractor_results] == [
        "multi:left[prefix_truncate_1]",
        "multi:right[prefix_truncate_1]",
    ]
    assert all(item.stability["repeats"] == 2 for item in result.extractor_results)
    assert all(
        item.compression_metadata["compressed_dim"] == 1 for item in result.extractor_results
    )


def test_dispatched_output_views_preserve_declared_row_alignment(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24, dtype=float).reshape(8, 3),
        ["husky", "husky", "pug", "pug", "sedan", "sedan", "suv", "suv"],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    ).with_label_hierarchy(
        [
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
        ],
        level_names=("domain", "family", "leaf"),
    )
    extractor = MultiOutputExtractor(
        "views",
        output_specs=[EmbeddingOutputSpec("coarse"), EmbeddingOutputSpec("fine")],
        transform_many_fn=lambda values: {
            "coarse": np.asarray(values)[:, :2],
            "fine": np.asarray(values)[:, 1:3],
        },
        streaming_safe=True,
    )
    result = Benchmark(
        dataset,
        [extractor],
        execution=LocalBackend(),
        execution_config=ExecutionConfig(total_shards=2),
        label_view_config=LabelViewConfig(output_levels={"coarse": "domain", "fine": "leaf"}),
        **_kwargs(tmp_path),
    ).run()

    assert [item.name for item in result.extractor_results] == [
        "views:coarse[level=domain]",
        "views:fine[level=leaf]",
    ]


def test_backend_failure_is_wrapped_without_local_fallback(tmp_path):
    with pytest.raises(BenchmarkExecutionError, match="embedding") as error:
        Benchmark(
            _dataset(),
            [CallableExtractor("identity", np.asarray, streaming_safe=True)],
            execution=FailingBackend(),
            execution_config=ExecutionConfig(total_shards=2),
            **_kwargs(tmp_path),
        ).run()

    assert isinstance(error.value.__cause__, RuntimeError)
    runs_dir = Path(tmp_path, "runs")
    assert not runs_dir.exists() or not any(runs_dir.iterdir())
