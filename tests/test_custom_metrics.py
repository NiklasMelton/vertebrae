import numpy as np
import pytest

from vertebrae import Benchmark, BenchmarkDataset, CallableMetric, DatasetIdentity, OverlapMetric
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.config import CacheConfig, OverlapScoringConfig, SeparatixConfig, StabilityConfig
from vertebrae.execution import (
    ScoringJob,
    benchmark_result_from_artifacts,
    score_embedding_artifact,
)
from vertebrae.extractors import CallableExtractor
from vertebrae.reports.markdown_report import render_markdown_report
from vertebrae.scoring.metrics import MetricResult, load_metric_callable


def mean_embedding_metric(embeddings, labels, *, target_metadata=None, groups=None, seed=None):
    return {
        "score": float(np.asarray(embeddings).mean()),
        "diagnostics": {"n_labels": len(labels)},
        "metadata": {
            "received_target_type": (target_metadata or {}).get("target_type"),
            "received_groups": None if groups is None else np.asarray(groups).tolist(),
        },
    }


def test_callable_metric_normalizes_results_and_import_path():
    metric = CallableMetric("mean_embedding", mean_embedding_metric)
    result = metric.score(np.asarray([[1.0], [3.0]]), ["a", "b"])

    assert isinstance(result, MetricResult)
    assert result.score == 2.0
    assert metric.recipe()["portable"] is True
    loaded = load_metric_callable("test_custom_metrics:mean_embedding_metric")
    assert loaded is mean_embedding_metric


def test_benchmark_ranks_by_custom_primary_metric(fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(16, dtype=float).reshape(8, 2),
        np.array(["a"] * 4 + ["b"] * 4),
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    result = Benchmark(
        dataset,
        extractors=[
            CallableExtractor("identity", lambda values: values, modality="tabular"),
            CallableExtractor("scaled", lambda values: values * 2, modality="tabular"),
        ],
        metrics=[CallableMetric("mean_embedding", mean_embedding_metric)],
        primary_metric="mean_embedding",
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    assert result.ranked_results()[0].name == "scaled"
    assert result.ranked_results()[0].primary_metric_name == "mean_embedding"
    assert result.ranked_results()[0].metrics["mean_embedding"].diagnostics["n_labels"] == 8
    assert result.to_dataframe().loc[0, "primary_metric"] == "mean_embedding"
    assert all(item.overlap is not None for item in result.extractor_results)


def test_custom_metrics_always_include_overlap_without_duplicate_serialization():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(16, dtype=float).reshape(8, 2),
        np.array(["a"] * 4 + ["b"] * 4),
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    result = Benchmark(
        dataset,
        extractors=[CallableExtractor("identity", lambda values: values, modality="tabular")],
        metrics=[CallableMetric("mean_embedding", mean_embedding_metric)],
        primary_metric="mean_embedding",
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    item = result.extractor_results[0]
    assert item.overlap is item.metrics["overlap"]
    assert item.primary_score == pytest.approx(7.5)
    assert item.overlap.macro_score == item.overlap.score
    assert "overlap" not in item.to_dict()
    assert "primary_metric" in result.to_dataframe().columns
    assert "Primary metric: mean_embedding" in render_markdown_report(result)


def test_configured_overlap_metric_replaces_the_default(fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(16, dtype=float).reshape(8, 2),
        np.array(["a"] * 4 + ["b"] * 4),
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    result = Benchmark(
        dataset,
        extractors=[CallableExtractor("identity", lambda values: values, modality="tabular")],
        metrics=[OverlapMetric(config=OverlapScoringConfig(k=1))],
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    assert list(result.extractor_results[0].metrics) == ["overlap"]
    assert fake_overlapindex.calls[-1]["kmeans_k"] == {"a": 1, "b": 1}


def test_custom_metric_artifact_round_trip(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    embedding_key = "embeddings/example"
    labels_key = "labels/example"
    store.put_array(embedding_key, np.asarray([[1.0], [3.0], [5.0], [7.0]]))
    store.put_json(
        embedding_key,
        {"n_samples": 4, "embedding_dim": 1, "extractor_name": "example"},
    )
    store.put_labels(labels_key, ["a", "a", "b", "b"])
    store.put_json(labels_key, {"n_samples": 4, "target_type": "single_label"})

    artifact = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            output_key="scores/custom",
            metrics=[CallableMetric("mean_embedding", mean_embedding_metric)],
            primary_metric="mean_embedding",
        ),
        store,
    )
    result = benchmark_result_from_artifacts(artifact["output_key"], store)

    assert artifact["artifact_type"] == "metric_evaluation"
    assert set(artifact["metrics"]) == {"overlap", "mean_embedding"}
    assert artifact["primary_metric"] == "mean_embedding"
    assert artifact["metrics"]["mean_embedding"]["metadata"]["received_groups"] is None
    assert result["extractor_results"][0]["metrics"]["mean_embedding"]["score"] == 4.0
    assert result["extractor_results"][0]["primary_metric_name"] == "mean_embedding"
    assert "overlap" not in result["extractor_results"][0]


def test_custom_metric_artifact_receives_validated_groups(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    embedding_key = "embeddings/grouped"
    labels_key = "labels/grouped"
    groups_key = "groups/grouped"
    identity_key = "dataset/example"
    store.put_array(embedding_key, np.asarray([[1.0], [3.0], [5.0], [7.0]]))
    store.put_json(
        embedding_key,
        {"n_samples": 4, "embedding_dim": 1, "dataset_identity_key": identity_key},
    )
    store.put_labels(labels_key, ["a", "a", "b", "b"])
    store.put_json(
        labels_key,
        {
            "artifact_type": "labels",
            "n_samples": 4,
            "target_type": "single_label",
            "dataset_identity_key": identity_key,
        },
    )
    store.put_labels(groups_key, [10, 10, 20, 20])
    store.put_json(
        groups_key,
        {
            "artifact_type": "groups",
            "n_samples": 4,
            "n_groups": 2,
            "group_name": "patient",
            "dataset_identity_key": identity_key,
        },
    )

    artifact = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            groups_key=groups_key,
            output_key="scores/grouped",
            metrics=[CallableMetric("mean_embedding", mean_embedding_metric)],
            primary_metric="mean_embedding",
        ),
        store,
    )

    assert artifact["groups_key"] == groups_key
    assert artifact["group_metadata"]["group_name"] == "patient"
    assert artifact["metrics"]["mean_embedding"]["metadata"]["received_groups"] == [
        10,
        10,
        20,
        20,
    ]


@pytest.mark.parametrize(
    "case,match",
    [
        ("artifact_type", "artifact_type='groups'"),
        ("row_count", "different row counts"),
        ("identity", "different dataset identities"),
        ("loaded_length", "expected 4 rows, loaded 3"),
    ],
)
def test_group_artifact_validation_fails_before_metric_execution(tmp_path, case, match):
    store = LocalArtifactStore(str(tmp_path))
    embedding_key = "embeddings/invalid-groups"
    labels_key = "labels/invalid-groups"
    groups_key = "groups/invalid"
    store.put_array(embedding_key, np.ones((4, 1)))
    store.put_json(
        embedding_key,
        {"n_samples": 4, "dataset_identity_key": "dataset/a"},
    )
    store.put_labels(labels_key, ["a", "a", "b", "b"])
    store.put_json(
        labels_key,
        {
            "artifact_type": "labels",
            "n_samples": 4,
            "target_type": "single_label",
            "dataset_identity_key": "dataset/a",
        },
    )
    store.put_labels(groups_key, [0, 0, 1] if case == "loaded_length" else [0, 0, 1, 1])
    store.put_json(
        groups_key,
        {
            "artifact_type": "labels" if case == "artifact_type" else "groups",
            "n_samples": 3 if case == "row_count" else 4,
            "dataset_identity_key": "dataset/b" if case == "identity" else "dataset/a",
        },
    )
    called = []

    def metric(*_args, **_kwargs):
        called.append(True)
        return 1.0

    with pytest.raises(ValueError, match=match):
        score_embedding_artifact(
            ScoringJob(
                embedding_key=embedding_key,
                labels_key=labels_key,
                groups_key=groups_key,
                output_key="scores/invalid-groups",
                metrics=[CallableMetric("never_called", metric)],
            ),
            store,
        )

    assert called == []
