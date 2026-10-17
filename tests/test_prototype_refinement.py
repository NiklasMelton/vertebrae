"""Adapter contracts for opt-in upstream prototype refinement."""

import json
import pickle
from dataclasses import asdict

import numpy as np
import pytest

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CacheConfig,
    DatasetIdentity,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.cache import LocalArtifactStore
from vertebrae.cli import main
from vertebrae.config import overlap_scoring_config_recipe
from vertebrae.execution import (
    LocalBackend,
    materialize_and_merge_embeddings,
    materialize_label_artifact,
    plan_scoring_jobs,
    score_embedding_artifact,
    stability_artifact_key,
)
from vertebrae.extractors import PrecomputedExtractor
from vertebrae.scoring import overlap as overlap_module
from vertebrae.scoring.metrics import OverlapMetric
from vertebrae.scoring.overlap import OverlapIndexScorer
from vertebrae.scoring.stability import run_stability_analysis
from vertebrae.utils.serialization import make_json_safe


def _data():
    return np.arange(72, dtype=float).reshape(18, 4), np.asarray(["a"] * 9 + ["b"] * 9)


@pytest.mark.parametrize("value", [0, 1, "true", "false", None, [], np.bool_(True)])
def test_refinement_requires_a_strict_boolean(value):
    with pytest.raises(TypeError, match="prototype_refinement must be a boolean"):
        OverlapScoringConfig(prototype_refinement=value)


@pytest.mark.parametrize("cross_fit", [False, True])
def test_default_and_explicit_false_match(fake_overlapindex, cross_fit):
    Z, y = _data()
    results = []
    for config in (OverlapScoringConfig(), OverlapScoringConfig(prototype_refinement=False)):
        scorer = OverlapIndexScorer(config)
        method = scorer.score_cross_fitted if cross_fit else scorer.score
        results.append(method(Z, y, seed=3).to_dict())
    assert results[0] == results[1]
    assert all(call["prototype_refinement"] is False for call in fake_overlapindex.calls)


@pytest.mark.parametrize("cross_fit", [False, True])
def test_refinement_preserves_adaptive_k_and_upstream_diagnostics(
    monkeypatch, fake_overlapindex, cross_fit
):
    initial_k = 3 if cross_fit else 4
    diagnostics = {
        "initial_counts": {"a": initial_k, "b": initial_k},
        "refined_counts": {"a": 5, "b": 4},
    }

    class RefiningIndex(fake_overlapindex.OverlapIndex):
        def fit_offline(self, *args, **kwargs):
            result = super().fit_offline(*args, **kwargs)
            self.prototype_refinement_ = diagnostics
            return result

    monkeypatch.setattr(overlap_module, "_load_overlap_index", lambda: RefiningIndex)
    Z, y = _data()
    scorer = OverlapIndexScorer(
        OverlapScoringConfig(k=8, min_samples_per_cluster=2, prototype_refinement=True)
    )
    result = (
        scorer.score_cross_fitted(Z, y, n_splits=3, seed=11)
        if cross_fit
        else scorer.score(Z, y, seed=11)
    )
    expected_k = {"a": 3, "b": 3} if cross_fit else {"a": 4, "b": 4}
    assert result.k_per_class == expected_k
    assert result.metadata["requested_k"] == 8
    assert any("Reduced k" in warning for warning in result.warnings)
    assert result.metadata["prototype_refinement"] is True
    for call in fake_overlapindex.calls:
        assert call["model_type"] == "MiniBatchKMeans"
        assert call["prototype_refinement"] is True
        assert call["kmeans_k"] == expected_k
    if cross_fit:
        for fold in result.metadata["cross_fit"]["folds"]:
            assert fold["k_per_class"] == expected_k
            assert fold["prototype_refinement_diagnostics"] == diagnostics
        assert [call["kmeans_kwargs"]["random_state"] for call in fake_overlapindex.calls] == [
            11,
            12,
            13,
        ]
    else:
        assert result.metadata["prototype_refinement_diagnostics"] == diagnostics
    assert json.loads(json.dumps(result.to_dict()))["metadata"] == result.to_dict()["metadata"]


@pytest.mark.parametrize("target_type", ["multi_label", "regression"])
def test_refinement_rejects_unsupported_targets_before_coercion(fake_overlapindex, target_type):
    Z, _ = _data()
    y = np.tile([[1, 0], [0, 1]], (9, 1)) if target_type == "multi_label" else np.arange(18)
    scorer = OverlapIndexScorer(OverlapScoringConfig(prototype_refinement=True))
    with pytest.raises(ValueError, match="prototype_refinement=True.*single-label"):
        scorer.score(Z, y, target_type=target_type)
    assert fake_overlapindex.calls == []


def test_regression_benchmark_explains_unsupported_refinement():
    Z, _ = _data()
    dataset = BenchmarkDataset.from_embeddings(
        Z,
        np.arange(len(Z), dtype=float),
        target_type="regression",
        identity=DatasetIdentity.ephemeral(),
    )
    benchmark = Benchmark(
        dataset,
        [PrecomputedExtractor()],
        scoring_config=OverlapScoringConfig(prototype_refinement=True),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
    )
    with pytest.raises(ValueError, match="prototype_refinement=True.*single-label"):
        benchmark.run()


@pytest.mark.parametrize("cross_fit", [False, True])
def test_enabled_refinement_cannot_be_filtered_for_an_old_dependency(monkeypatch, cross_fit):
    class OldIndex:
        def __init__(self, model_type, kmeans_k, kmeans_kwargs, offline_chunk_size):
            pytest.fail("An incompatible index must not be instantiated with refinement disabled")

    monkeypatch.setattr(overlap_module, "_load_overlap_index", lambda: OldIndex)
    Z, y = _data()
    scorer = OverlapIndexScorer(OverlapScoringConfig(prototype_refinement=True))
    method = scorer.score_cross_fitted if cross_fit else scorer.score
    with pytest.raises(RuntimeError, match="Install overlapindex>=0.1.5,<0.2.0"):
        method(Z, y)


def test_false_retains_existing_constructor_compatibility(monkeypatch, fake_overlapindex):
    class OldIndex(fake_overlapindex.OverlapIndex):
        def __init__(self, model_type, kmeans_k, kmeans_kwargs, offline_chunk_size):
            super().__init__(
                model_type=model_type,
                kmeans_k=kmeans_k,
                kmeans_kwargs=kmeans_kwargs,
                offline_chunk_size=offline_chunk_size,
            )

    monkeypatch.setattr(overlap_module, "_load_overlap_index", lambda: OldIndex)
    Z, y = _data()
    assert np.isfinite(OverlapIndexScorer().score(Z, y).score)


@pytest.mark.parametrize("mode", ["prototype", "subsample"])
def test_stability_repeats_retain_refinement(fake_overlapindex, mode):
    Z, y = _data()
    config = OverlapScoringConfig(prototype_refinement=True)
    result = run_stability_analysis(Z, y, config, StabilityConfig(repeats=3, mode=mode))
    assert len(result["scores"]) == 3
    assert len(fake_overlapindex.calls) == 3
    assert all(call["prototype_refinement"] is True for call in fake_overlapindex.calls)
    assert [call["kmeans_kwargs"]["random_state"] for call in fake_overlapindex.calls] == result[
        "seeds"
    ]


def test_config_recipes_and_pickle_round_trip():
    config = OverlapScoringConfig(prototype_refinement=True)
    recipe = json.loads(json.dumps(overlap_scoring_config_recipe(config)))
    assert OverlapScoringConfig(**recipe) == config
    assert OverlapScoringConfig(**asdict(config)) == config
    assert pickle.loads(pickle.dumps(config)) == config
    assert OverlapMetric(config).recipe()["config"]["prototype_refinement"] is True


def test_artifact_and_cli_scoring_share_raw_embeddings_but_not_score_identity(
    tmp_path, capsys, fake_overlapindex
):
    Z, y = _data()
    dataset = BenchmarkDataset.from_embeddings(Z, y, identity=DatasetIdentity.ephemeral())
    store = LocalArtifactStore(str(tmp_path / "cache"))
    manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=PrecomputedExtractor(),
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=3,
    )
    labels = materialize_label_artifact(dataset, store)
    embedding_key, labels_key = manifest["output_key"], labels["output_key"]
    original_embeddings = store.get_array(embedding_key).copy()
    configs = [OverlapScoringConfig(prototype_refinement=value) for value in (False, True)]
    jobs = [plan_scoring_jobs(embedding_key, labels_key, [7], config)[0] for config in configs]
    assert jobs[0].output_key != jobs[1].output_key
    assert jobs[0].embedding_key == jobs[1].embedding_key
    assert stability_artifact_key(
        embedding_key,
        labels_key=labels_key,
        scoring_config=configs[0],
        stability_config=StabilityConfig(),
    ) != stability_artifact_key(
        embedding_key,
        labels_key=labels_key,
        scoring_config=configs[1],
        stability_config=StabilityConfig(),
    )
    for job, enabled in zip(jobs, (False, True)):
        score = score_embedding_artifact(pickle.loads(pickle.dumps(job)), store)
        assert score["scoring_config"]["prototype_refinement"] is enabled
        assert score["metrics"]["overlap"]["metadata"]["prototype_refinement"] is enabled
        assert store.get_json(job.output_key) == score
    config_path = tmp_path / "scoring.pkl"
    config_path.write_bytes(pickle.dumps(configs[1]))
    assert (
        main(
            [
                "score",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--embedding-key",
                embedding_key,
                "--labels-key",
                labels_key,
                "--scoring-config-pickle",
                str(config_path),
                "--seed",
                "7",
            ]
        )
        == 0
    )
    cli_score = json.loads(capsys.readouterr().out)
    assert cli_score["output_key"] == jobs[1].output_key
    assert cli_score["scoring_config"]["prototype_refinement"] is True
    assert fake_overlapindex.calls[-1]["prototype_refinement"] is True
    np.testing.assert_array_equal(store.get_array(embedding_key), original_embeddings)


@pytest.mark.parametrize("distributed", [False, True])
def test_benchmark_reuses_raw_cache_when_refinement_changes(
    tmp_path, fake_overlapindex, distributed
):
    Z, y = _data()
    dataset = BenchmarkDataset.from_embeddings(Z, y, identity=DatasetIdentity.ephemeral())
    results = []
    for enabled in (False, True):
        results.append(
            Benchmark(
                dataset,
                [PrecomputedExtractor()],
                scoring_config=OverlapScoringConfig(prototype_refinement=enabled),
                stability_config=StabilityConfig(enabled=False),
                separatix_config=SeparatixConfig(enabled=False),
                cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path)),
                execution=LocalBackend() if distributed else None,
            ).run()
        )
    first, second = (result.extractor_results[0] for result in results)
    assert first.embedding_metadata["cache_key"] == second.embedding_metadata["cache_key"]
    assert first.embedding_metadata["cache_hit"] is False
    assert second.embedding_metadata["cache_hit"] is True
    assert [call["prototype_refinement"] for call in fake_overlapindex.calls] == [False, True]
    serialized = json.loads(json.dumps(results[1].to_dict()))
    assert serialized["metadata"]["scoring_config"]["prototype_refinement"] is True
    assert (
        serialized["extractor_results"][0]["metrics"]["overlap"]["metadata"]["prototype_refinement"]
        is True
    )


@pytest.fixture
def real_indices(monkeypatch):
    """Record the real dependency calls without replacing its numerical implementation."""
    real_index = overlap_module._load_overlap_index()
    calls = []

    def build_index(**kwargs):
        index = real_index(**kwargs)
        calls.append((kwargs, index))
        return index

    monkeypatch.setattr(overlap_module, "_load_overlap_index", lambda: build_index)
    return calls


def _real_data():
    rng = np.random.default_rng(1729)
    Z = rng.normal(size=(60, 5))
    Z[30:, 0] += 1.5
    return Z, np.asarray(["a"] * 30 + ["b"] * 30)


@pytest.mark.parametrize("cross_fit", [False, True])
def test_real_minibatch_refinement_and_fold_diagnostics(real_indices, cross_fit):
    Z, y = _real_data()
    scorer = OverlapIndexScorer(
        OverlapScoringConfig(k=50, min_samples_per_cluster=5, prototype_refinement=True)
    )
    result = (
        scorer.score_cross_fitted(Z, y, n_splits=3, seed=11)
        if cross_fit
        else scorer.score(Z, y, seed=11)
    )
    expected_k = {"a": 4, "b": 4} if cross_fit else {"a": 6, "b": 6}
    assert result.k_per_class == expected_k
    assert result.metadata["requested_k"] == 50
    assert result.metadata["normalize_embeddings"] is True
    assert any("Reduced k" in warning for warning in result.warnings)
    assert np.isfinite(result.score)
    assert len(real_indices) == (3 if cross_fit else 1)
    diagnostics = (
        [fold["prototype_refinement_diagnostics"] for fold in result.metadata["cross_fit"]["folds"]]
        if cross_fit
        else [result.metadata["prototype_refinement_diagnostics"]]
    )
    for fold, ((kwargs, index), preserved) in enumerate(zip(real_indices, diagnostics)):
        assert kwargs["model_type"] == "MiniBatchKMeans"
        assert kwargs["prototype_refinement"] is True
        assert kwargs["kmeans_k"] == expected_k
        assert kwargs["kmeans_kwargs"]["random_state"] == 11 + fold
        assert preserved == make_json_safe(index.prototype_refinement_)
        assert preserved["prototype_count_before"] == sum(expected_k.values())
        assert preserved["prototype_count_after"] >= preserved["prototype_count_before"]
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("cross_fit", [False, True])
def test_real_default_false_parity(real_indices, cross_fit):
    Z, y = _real_data()
    results = []
    for config in (OverlapScoringConfig(), OverlapScoringConfig(prototype_refinement=False)):
        scorer = OverlapIndexScorer(config)
        results.append(
            scorer.score_cross_fitted(Z, y, seed=17) if cross_fit else scorer.score(Z, y, seed=17)
        )
    assert results[0].score == results[1].score
    assert results[0].per_class_scores == results[1].per_class_scores
    assert results[0].k_per_class == results[1].k_per_class
    assert all(kwargs["prototype_refinement"] is False for kwargs, _ in real_indices)


def test_real_stability_refines_every_repeat(real_indices):
    Z, y = _real_data()
    result = run_stability_analysis(
        Z, y, OverlapScoringConfig(prototype_refinement=True), StabilityConfig(repeats=2)
    )
    assert len(real_indices) == 2
    assert np.all(np.isfinite(result["scores"]))
    for kwargs, index in real_indices:
        assert kwargs["model_type"] == "MiniBatchKMeans"
        assert kwargs["prototype_refinement"] is True
        assert index.prototype_refinement_
