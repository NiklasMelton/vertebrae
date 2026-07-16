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
from vertebrae.cache import LocalArtifactStore
from vertebrae.execution import embedding_artifact_key
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


class CachePolicyExtractor:
    name = "cache_policy"
    extractor_type = "test"
    streaming_safe = True

    def __init__(self, *, cache_embeddings=True, cache_safe=True):
        self.cache_embeddings = cache_embeddings
        self.cache_safe = cache_safe

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, :2]

    def recipe(self):
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "cache_safe": self.cache_safe,
        }


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
        [
            CallableExtractor(
                "identity",
                np.asarray,
                streaming_safe=True,
                cache_identity="identity-v1",
            )
        ],
        **_kwargs(tmp_path / "direct"),
    ).run()
    backend = RecordingBackend()
    dispatched = Benchmark(
        dataset,
        [
            CallableExtractor(
                "identity",
                np.asarray,
                streaming_safe=True,
                cache_identity="identity-v1",
            )
        ],
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
        [
            CallableExtractor(
                "identity",
                np.asarray,
                streaming_safe=True,
                cache_identity="identity-v1",
            )
        ],
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
        [
            CallableExtractor(
                "identity",
                np.asarray,
                streaming_safe=True,
                cache_identity="identity-v1",
            )
        ],
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
    first = Benchmark(
        dataset,
        [
            CallableExtractor(
                "identity",
                np.asarray,
                streaming_safe=True,
                cache_identity="identity-v1",
            )
        ],
        execution=first_backend,
        execution_config=ExecutionConfig(total_shards=2),
        **common,
    ).run()
    first_metadata = first.extractor_results[0].embedding_metadata
    assert first_metadata["cache_hit"] is False
    assert first_metadata["cache_eligible"] is True
    assert first_metadata["cache_status"] == "miss"
    assert (
        LocalArtifactStore(str(tmp_path)).get_json(first_metadata["cache_key"])["cache_status"]
        == "miss"
    )
    second_backend = RecordingBackend()
    second = Benchmark(
        dataset,
        [
            CallableExtractor(
                "identity",
                np.asarray,
                streaming_safe=True,
                cache_identity="identity-v1",
            )
        ],
        execution=second_backend,
        execution_config=ExecutionConfig(total_shards=2),
        **common,
    ).run()

    assert len(first_backend.calls) == 2
    assert len(second_backend.calls) == 0
    second_metadata = second.extractor_results[0].embedding_metadata
    assert second_metadata["cache_hit"] is True
    assert second_metadata["cache_eligible"] is True
    assert second_metadata["cache_status"] == "hit"
    assert (
        LocalArtifactStore(str(tmp_path)).get_json(second_metadata["cache_key"])["cache_status"]
        == "miss"
    )


def test_cache_hit_reuses_compression_scoring_and_diagnostics(
    tmp_path,
    fake_overlapindex,
    fake_separatix,
):
    dataset = _dataset()
    common = {
        "scoring_config": OverlapScoringConfig(k=1),
        "stability_config": StabilityConfig(repeats=2),
        "separatix_config": SeparatixConfig(enabled=True, overlap_threshold=0.0),
        "cache_config": CacheConfig(enabled=True, cache_dir=str(tmp_path)),
        "compression_config": EmbeddingCompressionConfig(
            enabled=True,
            method="prefix_truncate",
            n_components=2,
            assume_matryoshka=True,
        ),
        "embedding_config": EmbeddingConfig(batch_size=2),
    }
    first_backend = RecordingBackend()
    Benchmark(
        dataset,
        [
            CallableExtractor(
                "identity",
                np.asarray,
                streaming_safe=True,
                cache_identity="identity-v1",
            )
        ],
        execution=first_backend,
        execution_config=ExecutionConfig(total_shards=2),
        **common,
    ).run()
    second_backend = RecordingBackend()
    Benchmark(
        dataset,
        [
            CallableExtractor(
                "identity",
                np.asarray,
                streaming_safe=True,
                cache_identity="identity-v1",
            )
        ],
        execution=second_backend,
        execution_config=ExecutionConfig(total_shards=2),
        **common,
    ).run()

    assert len(first_backend.calls) == 5
    assert len(second_backend.calls) == 0


def test_multi_output_cache_status_reaches_parent_outputs_and_results(tmp_path, fake_overlapindex):
    dataset = _dataset()
    extractor = MultiOutputExtractor(
        "multi-cache",
        output_specs=[EmbeddingOutputSpec("left"), EmbeddingOutputSpec("right")],
        transform_many_fn=lambda values: {
            "left": np.asarray(values)[:, :2],
            "right": np.asarray(values)[:, 1:3],
        },
        streaming_safe=True,
        cache_identity="multi-cache-v1",
    )
    common = {
        **_kwargs(tmp_path),
        "cache_config": CacheConfig(enabled=True, cache_dir=str(tmp_path)),
    }
    first = Benchmark(
        dataset,
        [extractor],
        execution=LocalBackend(),
        execution_config=ExecutionConfig(total_shards=2),
        **common,
    ).run()
    store = LocalArtifactStore(str(tmp_path))
    parent_key = embedding_artifact_key(dataset, extractor)
    first_parent = store.get_json(parent_key)

    assert first_parent["cache_status"] == "miss"
    assert {item.embedding_metadata["cache_status"] for item in first.extractor_results} == {"miss"}
    assert all(
        store.get_json(output["output_key"])["cache_status"] == "miss"
        for output in first_parent["outputs"]
    )

    second = Benchmark(
        dataset,
        [extractor],
        execution=LocalBackend(),
        execution_config=ExecutionConfig(total_shards=2),
        **common,
    ).run()
    second_parent = store.get_json(parent_key)
    # Cache-access status belongs to the current result, not the immutable reusable artifact.
    assert second_parent["cache_status"] == "miss"
    assert {item.embedding_metadata["cache_status"] for item in second.extractor_results} == {"hit"}
    assert all(
        store.get_json(output["output_key"])["cache_status"] == "miss"
        for output in second_parent["outputs"]
    )


@pytest.mark.parametrize(
    ("extractor_kwargs", "expected_cache_status"),
    [
        ({"cache_embeddings": False, "cache_safe": True}, "disabled"),
        (
            {"cache_embeddings": True, "cache_safe": False},
            "bypassed_unsafe_identity",
        ),
    ],
)
def test_artifact_pipeline_honors_per_extractor_cache_policy(
    tmp_path,
    fake_overlapindex,
    extractor_kwargs,
    expected_cache_status,
):
    common = {
        **_kwargs(tmp_path),
        "cache_config": CacheConfig(enabled=True, cache_dir=str(tmp_path)),
    }
    dataset = _dataset()
    for _ in range(2):
        backend = RecordingBackend()
        result = Benchmark(
            dataset,
            [CachePolicyExtractor(**extractor_kwargs)],
            execution=backend,
            execution_config=ExecutionConfig(total_shards=2),
            **common,
        ).run()
        assert len(backend.calls) == 2
        metadata = result.extractor_results[0].embedding_metadata
        assert metadata["cache_hit"] is False
        assert metadata["cache_eligible"] is False
        assert metadata["cache_status"] == expected_cache_status

    assert not Path(tmp_path, "embeddings").exists()
    runs_dir = Path(tmp_path, "runs")
    assert not runs_dir.exists() or not any(runs_dir.iterdir())


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
