from datetime import date
from decimal import Decimal
from uuid import UUID

import numpy as np
import pytest
from scipy import sparse

from vertebrae import (
    BenchmarkDataset,
    DatasetIdentity,
    EmbeddingCompressionConfig,
    OverlapScoringConfig,
    RetrievalConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.datasets import TargetView
from vertebrae.execution import (
    EmbeddingMergeJob,
    EmbeddingShardJob,
    LocalBackend,
    ResourceSpec,
    ScoringJob,
    SeparatixJob,
    ShardSpec,
    benchmark_result_from_artifacts,
    collect_score_artifacts,
    compress_embedding_artifact,
    diagnose_embedding_artifact,
    embedding_artifact_key,
    groups_artifact_key,
    labels_artifact_key,
    materialize_and_merge_embeddings,
    materialize_embedding_shard,
    materialize_group_artifact,
    materialize_label_artifact,
    merge_embedding_shards,
    plan_compression_job,
    plan_embedding_shard_jobs,
    plan_scoring_jobs,
    retrieval_scoring_artifact_key,
    score_embedding_artifact,
    score_embedding_artifacts,
    scoring_artifact_key,
    separatix_artifact_key,
    stability_artifact_key,
)
from vertebrae.extractors import CallableExtractor, MultiOutputExtractor, PrecomputedExtractor
from vertebrae.extractors.base import EmbeddingOutputSpec
from vertebrae.scoring.metrics import CallableMetric, OverlapMetric
from vertebrae.utils.semantic_labels import semantic_label_key

COARSE_TARGETS = [
    "mammal",
    "mammal",
    "mammal",
    "mammal",
    "avian",
    "avian",
    "mammal",
    "mammal",
]


def _default_scoring_key(embedding_key, labels_key, groups_key=None, *, seed=None):
    return scoring_artifact_key(
        embedding_key,
        seed=seed,
        labels_key=labels_key,
        groups_key=groups_key,
        scoring_config=OverlapScoringConfig(),
        metrics=(),
        primary_metric="overlap",
    )


def _default_separatix_key(embedding_key, labels_key, score_key, groups_key=None):
    return separatix_artifact_key(
        embedding_key,
        labels_key=labels_key,
        groups_key=groups_key,
        score_key=score_key,
        separatix_config=SeparatixConfig(),
    )


class _FitOnceExtractor:
    name = "fit_once"
    modality = "tabular"
    extractor_type = "test"
    streaming_safe = True

    def __init__(self):
        self.fit_calls = 0
        self.offset = None

    def fit(self, X, y=None):
        self.fit_calls += 1
        self.offset = np.asarray(X).mean(axis=0)
        return self

    def transform(self, X):
        if self.offset is None:
            raise RuntimeError("extractor must be fitted")
        return np.asarray(X) - self.offset

    def recipe(self):
        return {"name": self.name, "extractor_type": self.extractor_type}


def test_resource_spec_validates_bounds():
    with pytest.raises(ValueError, match="cpus"):
        ResourceSpec(cpus=0)
    with pytest.raises(ValueError, match="gpus"):
        ResourceSpec(gpus=-1)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ShardSpec(total_shards=True),
        lambda: ShardSpec(total_shards=1.5),
        lambda: ShardSpec(shard_index=False),
        lambda: ResourceSpec(cpus=True),
        lambda: ResourceSpec(gpus=0.5),
        lambda: ResourceSpec(memory_bytes=True),
        lambda: ResourceSpec(gpu_memory_bytes=1.5),
        lambda: ResourceSpec(walltime_seconds=False),
        lambda: EmbeddingMergeJob(("shard",), "merged", n_samples=True),
        lambda: EmbeddingShardJob(object(), object(), ShardSpec(), "shard", batch_size=1.5),
    ],
)
def test_execution_integer_fields_reject_booleans_and_fractionals(factory):
    with pytest.raises(ValueError, match="integer"):
        factory()


def test_shard_runtime_indices_require_exact_nonnegative_integers():
    shard = ShardSpec()
    with pytest.raises(ValueError, match="n_samples.*integer"):
        shard.indices(True)
    with pytest.raises(ValueError, match="sample_index.*>= 0"):
        shard.owns(-1)


def test_group_artifact_and_separatix_job_preserve_group_safety(
    tmp_path,
    fake_overlapindex,
    fake_separatix,
):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24, dtype=float).reshape(8, 3),
        np.array(["a"] * 4 + ["b"] * 4),
        identity=DatasetIdentity.ephemeral(),
    ).with_groups(np.repeat(np.arange(4), 2), name="image_id")
    store = LocalArtifactStore(tmp_path)
    extractor = CallableExtractor("identity", transform_fn=lambda value: value)
    embedding_manifest = materialize_and_merge_embeddings(
        dataset,
        extractor,
        store,
        LocalBackend(),
        total_shards=1,
    )
    labels = materialize_label_artifact(dataset, store)
    groups = materialize_group_artifact(dataset, store)
    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=labels["output_key"],
            groups_key=groups["output_key"],
            output_key=_default_scoring_key(
                embedding_manifest["output_key"],
                labels["output_key"],
                groups["output_key"],
            ),
        ),
        store,
    )

    diagnostic = diagnose_embedding_artifact(
        SeparatixJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=labels["output_key"],
            groups_key=groups["output_key"],
            score_key=score["output_key"],
            output_key=_default_separatix_key(
                embedding_manifest["output_key"],
                labels["output_key"],
                score["output_key"],
                groups["output_key"],
            ),
        ),
        store,
    )
    result = benchmark_result_from_artifacts(score["output_key"], store)

    assert groups["output_key"] == groups_artifact_key(dataset)
    assert groups["n_groups"] == 4
    assert fake_separatix.ComplexityProfiler.calls[-1]["groups"].tolist() == [
        semantic_label_key(value) for value in dataset.groups().tolist()
    ]
    assert diagnostic["diagnostic"]["metadata"]["grouped"] is True
    assert score["groups_key"] == groups["output_key"]
    assert result["dataset_summary"]["grouped"] is True
    assert result["dataset_summary"]["group_name"] == "image_id"
    assert result["dataset_summary"]["n_groups"] == 4
    assert result["metadata"]["source_groups_key"] == groups["output_key"]


def test_group_artifact_counts_semantically_distinct_heterogeneous_ids(tmp_path):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24, dtype=float).reshape(8, 3),
        np.array(["a"] * 4 + ["b"] * 4),
        identity=DatasetIdentity.ephemeral(),
    ).with_groups(
        np.asarray([1, True, "1", (1,), 1, True, "1", (1,)], dtype=object),
        name="source",
    )

    manifest = materialize_group_artifact(dataset, LocalArtifactStore(tmp_path))

    assert manifest["n_groups"] == 4


def test_local_backend_submit_gather_status():
    backend = LocalBackend()
    handle = backend.submit(lambda value: value + 1, 2)

    assert backend.status(handle) == "finished"
    assert backend.gather([handle]) == [3]


def test_plan_materialize_and_merge_dense_embedding_shards(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "dense_sharded",
        lambda batch: np.asarray(batch)[:, :2] * 2,
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))

    jobs = plan_embedding_shard_jobs(dataset, extractor, total_shards=2, batch_size=2)
    manifests = [materialize_embedding_shard(job, store) for job in jobs]
    output_key = embedding_artifact_key(dataset, extractor)
    merged = merge_embedding_shards(
        EmbeddingMergeJob(
            shard_keys=tuple(manifest["output_key"] for manifest in manifests),
            output_key=output_key,
            n_samples=len(dataset.y),
        ),
        store,
    )

    embeddings = store.get_array(output_key)
    assert merged["shape"] == [8, 2]
    assert np.array_equal(embeddings, dataset.X[:, :2] * 2)
    assert sorted(manifests[0]["sample_indices"] + manifests[1]["sample_indices"]) == list(range(8))


def test_materialize_and_merge_embeddings_with_local_parallel_backend(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(36).reshape(12, 3),
        ["a"] * 6 + ["b"] * 6,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "parallel_dense_sharded",
        lambda batch: np.asarray(batch)[:, :2] + 1,
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))

    manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(n_jobs=2, joblib_backend="threading"),
        total_shards=3,
        batch_size=2,
    )

    embeddings = store.get_array(manifest["output_key"])
    assert manifest["n_shards"] == 3
    assert np.array_equal(embeddings, dataset.X[:, :2] + 1)


def test_materialize_and_merge_fits_stateful_extractor_exactly_once(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24, dtype=float).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = _FitOnceExtractor()
    store = LocalArtifactStore(str(tmp_path))

    manifest = materialize_and_merge_embeddings(
        dataset,
        extractor,
        store,
        LocalBackend(),
        total_shards=4,
        batch_size=1,
    )

    assert extractor.fit_calls == 1
    assert np.allclose(store.get_array(manifest["output_key"]), dataset.X - dataset.X.mean(0))


def test_embedding_shard_planner_caps_oversharding_and_rejects_invalid_counts():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(12).reshape(4, 3),
        ["a", "a", "b", "b"],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "identity",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )

    jobs = plan_embedding_shard_jobs(dataset, extractor, total_shards=100)

    assert len(jobs) == 4
    assert all(job.shard.total_shards == 4 for job in jobs)
    assert all(not hasattr(job, "fit_extractor") for job in jobs)
    non_streaming = CallableExtractor("full_only", lambda batch: np.asarray(batch))
    full_jobs = plan_embedding_shard_jobs(dataset, non_streaming, total_shards=100)
    assert len(full_jobs) == 1
    assert full_jobs[0].streaming_enabled is False
    assert full_jobs[0].shard == ShardSpec(total_shards=1, shard_index=0)
    with pytest.raises(ValueError, match=">= 1"):
        plan_embedding_shard_jobs(dataset, extractor, total_shards=0)
    with pytest.raises(ValueError, match="integer"):
        plan_embedding_shard_jobs(dataset, extractor, total_shards=True)


def test_unsafe_embedding_identity_is_run_scoped_instead_of_reused():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(12).reshape(4, 3),
        ["a", "a", "b", "b"],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "unsafe",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )

    jobs = plan_embedding_shard_jobs(dataset, extractor, total_shards=2)

    assert all(job.output_key.startswith("runs/") for job in jobs)
    assert all(not job.cache_eligible for job in jobs)
    assert all(job.cache_status == "bypassed_unsafe_identity" for job in jobs)
    assert (
        jobs[0].output_key.split("/embeddings/", 1)[0]
        == jobs[1].output_key.split("/embeddings/", 1)[0]
    )


def test_disabled_embedding_cache_is_run_scoped_in_distributed_plans():
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(12).reshape(4, 3),
        ["a", "a", "b", "b"],
        identity=DatasetIdentity.from_content(),
    )

    jobs = plan_embedding_shard_jobs(
        dataset,
        PrecomputedExtractor(cache_embeddings=False),
        total_shards=2,
    )

    assert all(job.output_key.startswith("runs/") for job in jobs)
    assert all(not job.cache_eligible for job in jobs)
    assert all(job.cache_status == "disabled" for job in jobs)


def test_disabled_embedding_cache_propagates_to_derived_artifacts(
    tmp_path,
    fake_overlapindex,
):
    store = LocalArtifactStore(tmp_path)
    embedding_key = "runs/test/embedding"
    labels_key = "runs/test/labels"
    store.put_artifact(
        embedding_key,
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        {
            "artifact_type": "embedding",
            "dataset_identity_key": "dataset",
            "cache_eligible": False,
            "cache_status": "disabled",
        },
    )
    store.put_labels_artifact(
        labels_key,
        ["a", "a", "b", "b"],
        {
            "artifact_type": "labels",
            "dataset_identity_key": "dataset",
            "content_digest": "labels",
        },
    )
    compression = compress_embedding_artifact(
        plan_compression_job(
            embedding_key,
            EmbeddingCompressionConfig(
                enabled=True,
                method="prefix_truncate",
                n_components=1,
                assume_matryoshka=True,
            ),
        ),
        store,
    )
    compressed_metadata = compression["embedding_metadata"]
    assert compressed_metadata["cache_eligible"] is False
    assert compressed_metadata["cache_status"] == "disabled"

    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            output_key="runs/test/score",
            scoring_config=OverlapScoringConfig(),
            metrics=(),
            primary_metric="overlap",
        ),
        store,
    )
    assert score["cache_eligible"] is False
    assert score["cache_status"] == "disabled"


def test_materialize_and_merge_sparse_embedding_shards(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        ["a"] * 5 + ["b"] * 5,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "sparse_sharded",
        lambda batch: sparse.csr_matrix(np.asarray(batch)[:, :2]),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))

    jobs = plan_embedding_shard_jobs(dataset, extractor, total_shards=3, batch_size=2)
    manifests = [materialize_embedding_shard(job, store) for job in jobs]
    output_key = embedding_artifact_key(dataset, extractor)
    merged = merge_embedding_shards(
        EmbeddingMergeJob(
            shard_keys=tuple(manifest["output_key"] for manifest in manifests),
            output_key=output_key,
            n_samples=len(dataset.y),
        ),
        store,
    )

    embeddings = store.get_array(output_key)
    assert merged["sparse"] is True
    assert embeddings.shape == (10, 2)
    assert np.array_equal(embeddings.toarray(), dataset.X[:, :2])


def test_merge_embedding_shards_rejects_duplicate_sample_indices(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(12).reshape(4, 3),
        ["a", "a", "b", "b"],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "duplicate_shard",
        lambda batch: np.asarray(batch)[:, :2],
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    job = plan_embedding_shard_jobs(dataset, extractor, total_shards=2, batch_size=2)[0]
    manifest = materialize_embedding_shard(job, store)

    with pytest.raises(ValueError, match="Duplicate embedding rows"):
        merge_embedding_shards(
            EmbeddingMergeJob(
                shard_keys=(manifest["output_key"], manifest["output_key"]),
                output_key=embedding_artifact_key(dataset, extractor),
                n_samples=len(dataset.y),
            ),
            store,
        )


def test_plan_materialize_and_merge_multi_output_embedding_shards(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = MultiOutputExtractor(
        name="multi",
        output_specs=[EmbeddingOutputSpec("a/b"), EmbeddingOutputSpec("a_b")],
        transform_many_fn=lambda batch: {
            "a/b": np.asarray(batch)[:, :2],
            "a_b": np.asarray(batch)[:, 1:3],
        },
        modality="tabular",
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))

    jobs = plan_embedding_shard_jobs(dataset, extractor, total_shards=2, batch_size=2)
    manifests = [materialize_embedding_shard(job, store) for job in jobs]
    merged = merge_embedding_shards(
        EmbeddingMergeJob(
            shard_keys=tuple(manifest["output_key"] for manifest in manifests),
            output_key=embedding_artifact_key(dataset, extractor),
            n_samples=len(dataset.y),
        ),
        store,
    )

    assert merged["artifact_type"] == "multi_output_embedding"
    assert [output["output_name"] for output in merged["outputs"]] == ["a/b", "a_b"]
    assert len({output["output_key"] for output in merged["outputs"]}) == 2
    assert all("/outputs/output-v1-a-b--" in output["output_key"] for output in merged["outputs"])
    assert [output["output_recipe"] for output in merged["outputs"]] == [
        {"output_name": "a/b"},
        {"output_name": "a_b"},
    ]
    assert all(output["output_metadata"] == {} for output in merged["outputs"])
    left = store.get_array(merged["outputs"][0]["output_key"])
    right = store.get_array(merged["outputs"][1]["output_key"])
    assert np.array_equal(left, dataset.X[:, :2])
    assert np.array_equal(right, dataset.X[:, 1:3])


def test_materialize_and_merge_embeddings_supports_multi_output(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = MultiOutputExtractor(
        name="multi",
        output_specs=[EmbeddingOutputSpec("left"), EmbeddingOutputSpec("right")],
        transform_many_fn=lambda batch: {
            "left": np.asarray(batch)[:, :2] * 2,
            "right": np.asarray(batch)[:, 1:3] + 1,
        },
        modality="tabular",
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))

    manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )

    assert manifest["artifact_type"] == "multi_output_embedding"
    left_key = manifest["outputs"][0]["output_key"]
    right_key = manifest["outputs"][1]["output_key"]
    assert np.array_equal(store.get_array(left_key), dataset.X[:, :2] * 2)
    assert np.array_equal(store.get_array(right_key), dataset.X[:, 1:3] + 1)


def test_multimodal_multi_output_shards_preserve_row_order(tmp_path):
    dataset = BenchmarkDataset.from_multimodal(
        inputs={
            "image": ["a.png", "b.png", "c.png", "d.png", "e.png", "f.png"],
            "caption": ["one", "two", "three", "four", "five", "six"],
        },
        labels=["a", "a", "a", "b", "b", "b"],
        modalities={"image": "image", "caption": "text"},
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = MultiOutputExtractor(
        name="fusion",
        output_specs=[EmbeddingOutputSpec("image_branch"), EmbeddingOutputSpec("fused")],
        transform_many_fn=lambda batch: {
            "image_branch": np.asarray([[len(item)] for item in batch["image"]], dtype=float),
            "fused": np.asarray(
                [[len(image), len(text)] for image, text in zip(batch["image"], batch["caption"])],
                dtype=float,
            ),
        },
        modality="multimodal",
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))

    manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=3,
        batch_size=1,
    )

    image_branch = store.get_array(manifest["outputs"][0]["output_key"])
    fused = store.get_array(manifest["outputs"][1]["output_key"])
    assert image_branch.tolist() == [[5.0], [5.0], [5.0], [5.0], [5.0], [5.0]]
    assert fused.tolist() == [
        [5.0, 3.0],
        [5.0, 3.0],
        [5.0, 5.0],
        [5.0, 4.0],
        [5.0, 4.0],
        [5.0, 3.0],
    ]


def test_score_embedding_artifact_consumes_persisted_embeddings_and_labels(
    tmp_path,
    fake_overlapindex,
):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "score_artifact",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )
    label_manifest = materialize_label_artifact(dataset, store)

    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            output_key=f'{embedding_manifest["output_key"]}/scores/default',
        ),
        store,
    )

    assert score["artifact_type"] == "metric_evaluation"
    assert score["metrics"]["overlap"]["diagnostics"]["macro_score"] == 0.8
    assert score["embedding_key"] == embedding_manifest["output_key"]
    assert score["labels_key"] == labels_artifact_key(dataset)
    assert (
        store.get_json(score["output_key"])["metrics"]["overlap"]["metadata"]["backend"]
        == "MiniBatchKMeans"
    )


def test_score_embedding_artifact_preserves_typed_semantic_labels(
    tmp_path,
    fake_overlapindex,
):
    distinct = [
        Decimal("1.25"),
        date(2026, 7, 15),
        UUID("12345678-1234-5678-1234-567812345678"),
        1,
        True,
        "1",
    ]
    labels = np.empty(12, dtype=object)
    labels[:] = distinct + distinct
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(36, dtype=float).reshape(12, 3),
        labels,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "typed_labels",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )
    label_manifest = materialize_label_artifact(dataset, store)

    score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            output_key=_default_scoring_key(
                embedding_manifest["output_key"], label_manifest["output_key"]
            ),
        ),
        store,
    )

    expected = {semantic_label_key(value) for value in distinct}
    loaded, metadata = store.get_labels_artifact(label_manifest["output_key"])
    assert set(loaded.tolist()) == expected
    assert {item["key"] for item in metadata["label_catalog"]} == expected
    assert set(fake_overlapindex.calls[-1]["fit_y"].tolist()) == expected


def test_score_embedding_artifact_supports_multilabel_labels(tmp_path, fake_overlapindex):
    labels = [
        ("red", "round"),
        ("red",),
        ("round",),
        ("red", "sweet"),
        ("round", "sweet"),
        ("sweet",),
        ("red", "round"),
        ("red",),
        ("round",),
        ("red", "sweet"),
        ("round", "sweet"),
        ("sweet",),
    ]
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(36).reshape(12, 3), labels, identity=DatasetIdentity.ephemeral()
    )
    extractor = CallableExtractor(
        "score_multilabel_artifact",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )
    label_manifest = materialize_label_artifact(dataset, store)

    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            output_key=_default_scoring_key(
                embedding_manifest["output_key"], label_manifest["output_key"]
            ),
        ),
        store,
    )
    result = benchmark_result_from_artifacts(score_key=score["output_key"], store=store)

    assert label_manifest["target_type"] == "multi_label"
    assert label_manifest["label_names"] == ["red", "round", "sweet"]
    assert store.get_labels(label_manifest["output_key"]).tolist()[0] == ("red", "round")
    assert fake_overlapindex.calls[-1]["fit_y_shape"] == [12, 3]
    assert score["metrics"]["overlap"]["metadata"]["target_type"] == "multi_label"
    assert result["dataset_summary"]["target_type"] == "multi_label"
    assert result["dataset_summary"]["labelset_counts"]["red + round"] == 2


def test_score_embedding_artifact_supports_regression_labels(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(18, dtype=float).reshape(6, 3),
        np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]),
        target_type="regression",
        target_names=["score"],
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "score_regression_artifact",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )
    label_manifest = materialize_label_artifact(dataset, store)

    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            output_key=_default_scoring_key(
                embedding_manifest["output_key"], label_manifest["output_key"]
            ),
        ),
        store,
    )
    result = benchmark_result_from_artifacts(score_key=score["output_key"], store=store)

    assert label_manifest["target_type"] == "regression"
    assert label_manifest["target_names"] == ["score"]
    assert store.get_labels(label_manifest["output_key"]).shape == (6,)
    assert fake_overlapindex.continuous_calls[-1]["fit_y_shape"] == [6]
    assert score["metrics"]["overlap"]["score"] == 0.62
    assert score["metrics"]["overlap"]["metadata"]["target_type"] == "regression"
    assert result["dataset_summary"]["target_type"] == "regression"
    assert result["extractor_results"][0]["metrics"]["overlap"]["score"] == 0.62


def test_label_artifact_preserves_active_label_view_metadata(tmp_path):
    dataset = (
        BenchmarkDataset.from_embeddings(
            np.arange(24).reshape(8, 3),
            ["husky", "husky", "pug", "pug", "sedan", "sedan", "suv", "suv"],
            identity=DatasetIdentity.ephemeral(),
        )
        .with_label_hierarchy(
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
        .label_view("family")
    )
    store = LocalArtifactStore(str(tmp_path))

    manifest = materialize_label_artifact(dataset, store)

    assert manifest["label_view"]["name"] == "family"
    assert store.get_json(manifest["output_key"])["label_view"]["level"] == 1


def test_label_artifact_preserves_active_target_view_metadata(tmp_path):
    dataset = (
        BenchmarkDataset.from_embeddings(
            np.arange(24).reshape(8, 3),
            ["cat", "cat", "dog", "dog", "bird", "bird", "fox", "fox"],
            identity=DatasetIdentity.ephemeral(),
        )
        .with_target_views(
            [
                TargetView(
                    name="coarse",
                    targets=COARSE_TARGETS,
                )
            ]
        )
        .target_view("coarse")
    )
    store = LocalArtifactStore(str(tmp_path))

    manifest = materialize_label_artifact(dataset, store)

    assert manifest["target_view"]["name"] == "coarse"
    assert store.get_json(manifest["output_key"])["target_view"]["kind"] == "named_target"


def test_score_repeats_collect_and_benchmark_from_artifacts(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        identity=DatasetIdentity.ephemeral(),
    ).with_groups(np.repeat(np.arange(4), 2), name="source")
    extractor = CallableExtractor(
        "repeat_score_artifact",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )
    label_manifest = materialize_label_artifact(dataset, store)
    group_manifest = materialize_group_artifact(dataset, store)
    jobs = plan_scoring_jobs(
        embedding_key=embedding_manifest["output_key"],
        labels_key=label_manifest["output_key"],
        groups_key=group_manifest["output_key"],
        seeds=[3, 5, 7],
    )

    scores = score_embedding_artifacts(jobs, store, LocalBackend())
    collection = collect_score_artifacts(
        [score["output_key"] for score in scores],
        store,
        output_key=f'{embedding_manifest["output_key"]}/scores/stability',
    )
    result = benchmark_result_from_artifacts(
        score_key=scores[0]["output_key"],
        store=store,
        stability_key=collection["output_key"],
    )

    assert collection["artifact_type"] == "score_collection"
    assert collection["seeds"] == [3, 5, 7]
    assert collection["groups_key"] == group_manifest["output_key"]
    assert len({score["protocol_fingerprint"] for score in scores}) == 3
    assert len({score["collection_protocol_fingerprint"] for score in scores}) == 1
    assert collection["protocol_fingerprints"] == [
        score["protocol_fingerprint"] for score in scores
    ]
    assert all(job.groups_key == group_manifest["output_key"] for job in jobs)
    assert result["metadata"]["distributed_artifacts"] is True
    assert result["extractor_results"][0]["stability"]["summary"]["mean"] > 0.0


def test_scoring_job_binds_configless_overlap_and_rejects_conflicts(
    tmp_path,
    fake_overlapindex,
):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        identity=DatasetIdentity.ephemeral(),
    )
    store = LocalArtifactStore(str(tmp_path))
    extractor = CallableExtractor(
        "identity",
        np.asarray,
        streaming_safe=True,
        cache_identity="score-binding-v1",
    )
    embedding = materialize_and_merge_embeddings(
        dataset,
        extractor,
        store,
        LocalBackend(),
        total_shards=2,
    )
    labels = materialize_label_artifact(dataset, store)
    config = OverlapScoringConfig(k=1, normalize_embeddings=False)
    jobs = plan_scoring_jobs(
        embedding["output_key"],
        labels["output_key"],
        seeds=[13],
        scoring_config=config,
        metrics=[OverlapMetric(config=None)],
    )

    artifact = score_embedding_artifact(jobs[0], store)

    assert artifact["scoring_config"] == artifact["protocol"]["scoring_config"]
    assert artifact["metric_recipes"][0]["config"]["k"] == 1
    assert artifact["protocol"]["seed"] == 13
    with pytest.raises(ValueError, match="conflicts with ScoringJob.scoring_config"):
        plan_scoring_jobs(
            embedding["output_key"],
            labels["output_key"],
            seeds=[None],
            scoring_config=config,
            metrics=[OverlapMetric(config=OverlapScoringConfig(k=2))],
        )


def test_unsafe_metric_score_plans_are_run_scoped():
    metric = CallableMetric("unsafe", lambda embeddings, labels: 0.5)

    jobs = plan_scoring_jobs(
        "embeddings/safe",
        "labels/safe",
        seeds=[1, 2],
        scoring_config=OverlapScoringConfig(k=1),
        metrics=[metric],
    )

    assert all(job.output_key.startswith("runs/") for job in jobs)
    assert jobs[0].output_key.split("/scores/", 1)[0] == jobs[1].output_key.split("/scores/", 1)[0]


def test_collect_score_artifacts_rejects_mixed_group_protocols(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    for index, groups_key in enumerate(("groups/a", "groups/b")):
        store.put_json(
            f"scores/{index}",
            {
                "groups_key": groups_key,
                "primary_metric": "overlap",
                "metrics": {"overlap": {"score": 0.5, "warnings": []}},
                "seed": index,
            },
        )

    with pytest.raises(ValueError, match="share one groups protocol"):
        collect_score_artifacts(
            ["scores/0", "scores/1"],
            store,
            output_key="scores/collection",
        )


def test_scoring_keys_include_labels_groups_metrics_and_configuration():
    base = scoring_artifact_key(
        "embeddings/a",
        labels_key="labels/a",
        groups_key="groups/a",
        scoring_config={"k": 10},
        metrics=(),
        primary_metric="overlap",
    )

    assert base != scoring_artifact_key(
        "embeddings/a",
        labels_key="labels/b",
        groups_key="groups/a",
        scoring_config={"k": 10},
        metrics=(),
        primary_metric="overlap",
    )
    assert base != scoring_artifact_key(
        "embeddings/a",
        labels_key="labels/a",
        groups_key="groups/b",
        scoring_config={"k": 10},
        metrics=(),
        primary_metric="overlap",
    )
    assert base != scoring_artifact_key(
        "embeddings/a",
        labels_key="labels/a",
        groups_key="groups/a",
        scoring_config={"k": 11},
        metrics=(),
        primary_metric="overlap",
    )


def test_protocol_key_helpers_reject_incomplete_identities():
    with pytest.raises(ValueError, match="labels_key"):
        scoring_artifact_key(
            "embeddings/a",
            labels_key="",
            groups_key=None,
            scoring_config=OverlapScoringConfig(),
            metrics=(),
            primary_metric="overlap",
        )
    with pytest.raises(ValueError, match="scoring_config"):
        scoring_artifact_key(
            "embeddings/a",
            labels_key="labels/a",
            groups_key=None,
            scoring_config=None,
            metrics=(),
            primary_metric="overlap",
        )
    with pytest.raises(TypeError):
        scoring_artifact_key("embeddings/a")
    with pytest.raises(ValueError, match="stability_config"):
        stability_artifact_key(
            "embeddings/a",
            labels_key="labels/a",
            scoring_config=OverlapScoringConfig(),
            stability_config=None,
        )
    with pytest.raises(ValueError, match="relevance_key"):
        retrieval_scoring_artifact_key(
            "retrieval/query",
            "retrieval/gallery",
            relevance_key="",
            exclusions_key=None,
            retrieval_config=RetrievalConfig(),
        )
    with pytest.raises(ValueError, match="score_key"):
        separatix_artifact_key(
            "embeddings/a",
            labels_key="labels/a",
            groups_key=None,
            score_key="",
            separatix_config=SeparatixConfig(),
        )

    assert stability_artifact_key(
        "embeddings/a",
        labels_key="labels/a",
        scoring_config=OverlapScoringConfig(),
        stability_config=StabilityConfig(),
    )


def test_label_and_group_keys_digest_exact_aligned_content():
    class ProtocolDataset:
        metadata = {"target_type": "single_label", "group_name": "source"}

        def __init__(self, labels, groups):
            self.y = np.asarray(labels, dtype=object)
            self._groups = np.asarray(groups)

        def identity_key(self):
            return "declared-shared-identity"

        def groups(self):
            return self._groups

        def active_target_view(self):
            return None

        def active_label_view(self):
            return None

    first = ProtocolDataset(
        ["a", "a", "a", "b", "b", "b"],
        [0, 0, 1, 1, 2, 2],
    )
    second = ProtocolDataset(
        ["a", "a", "b", "a", "b", "b"],
        [0, 1, 0, 1, 2, 2],
    )

    assert first.identity_key() == second.identity_key()
    assert labels_artifact_key(first) != labels_artifact_key(second)
    assert groups_artifact_key(first) != groups_artifact_key(second)


def test_score_collection_rejects_different_evaluation_fingerprints(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    for index, fingerprint in enumerate(("protocol-a", "protocol-b")):
        store.put_json(
            f"scores/{index}",
            {
                "artifact_type": "metric_evaluation",
                "groups_key": None,
                "evaluation_fingerprint": fingerprint,
                "primary_metric": "overlap",
                "metrics": {
                    "overlap": {
                        "score": 0.5,
                        "warnings": [],
                        "higher_is_better": True,
                    }
                },
                "seed": index,
            },
        )

    with pytest.raises(ValueError, match="complete embedding.*metric protocol"):
        collect_score_artifacts(
            ["scores/0", "scores/1"],
            store,
            output_key="scores/collection",
        )


def test_compressed_embedding_manifest_rebuilds_physical_provenance(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    source = np.arange(24, dtype=np.float64).reshape(8, 3)
    store.put_artifact(
        "raw",
        source,
        {
            "artifact_type": "embedding",
            "output_key": "raw",
            "cache_key": "raw",
            "artifact_path": "stale/raw.npy",
            "n_samples": 8,
            "embedding_dim": 3,
            "dtype": "float64",
            "sparse": False,
        },
    )
    job = plan_compression_job(
        "raw",
        EmbeddingCompressionConfig(
            enabled=True,
            method="prefix_truncate",
            n_components=2,
            assume_matryoshka=True,
            dtype="float32",
        ),
    )

    compress_embedding_artifact(job, store)
    metadata = store.get_json(job.output_key)

    assert metadata["artifact_type"] == "compressed_embedding"
    assert metadata["output_key"] == job.output_key
    assert metadata["cache_key"] == job.output_key
    assert metadata["source_embedding_key"] == "raw"
    assert metadata["artifact_path"] != "stale/raw.npy"
    assert metadata["shape"] == [8, 2]
    assert metadata["dtype"] == "float32"


def test_benchmark_from_artifacts_carries_label_view_metadata(tmp_path, fake_overlapindex):
    dataset = (
        BenchmarkDataset.from_embeddings(
            np.arange(24).reshape(8, 3),
            ["husky", "husky", "pug", "pug", "sedan", "sedan", "suv", "suv"],
            identity=DatasetIdentity.ephemeral(),
        )
        .with_label_hierarchy(
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
        .label_view("family")
    )
    extractor = CallableExtractor(
        "artifact_family",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )
    label_manifest = materialize_label_artifact(dataset, store)
    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            output_key=_default_scoring_key(
                embedding_manifest["output_key"], label_manifest["output_key"]
            ),
        ),
        store,
    )

    result = benchmark_result_from_artifacts(score_key=score["output_key"], store=store)

    assert result["dataset_summary"]["label_view"]["name"] == "family"
    assert result["extractor_results"][0]["label_view"]["name"] == "family"


def test_benchmark_from_artifacts_carries_target_view_metadata(tmp_path, fake_overlapindex):
    dataset = (
        BenchmarkDataset.from_embeddings(
            np.arange(24).reshape(8, 3),
            ["cat", "cat", "dog", "dog", "bird", "bird", "fox", "fox"],
            identity=DatasetIdentity.ephemeral(),
        )
        .with_target_views(
            [
                TargetView(
                    name="coarse",
                    targets=COARSE_TARGETS,
                )
            ]
        )
        .target_view("coarse")
    )
    extractor = CallableExtractor(
        "artifact_coarse",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )
    label_manifest = materialize_label_artifact(dataset, store)
    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            output_key=_default_scoring_key(
                embedding_manifest["output_key"], label_manifest["output_key"]
            ),
        ),
        store,
    )

    result = benchmark_result_from_artifacts(score_key=score["output_key"], store=store)

    assert result["dataset_summary"]["target_view"]["name"] == "coarse"
    assert result["extractor_results"][0]["target_view"]["name"] == "coarse"


def test_diagnose_embedding_artifact_and_attach_to_benchmark_result(
    tmp_path,
    fake_overlapindex,
    fake_separatix,
):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24).reshape(8, 3),
        [1] * 4 + [2] * 4,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "diagnose_artifact",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )
    label_manifest = materialize_label_artifact(dataset, store)
    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            output_key=_default_scoring_key(
                embedding_manifest["output_key"], label_manifest["output_key"]
            ),
            scoring_config=OverlapScoringConfig(k=1, exclude_classes=[1]),
        ),
        store,
    )

    diagnostic = diagnose_embedding_artifact(
        job=SeparatixJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            score_key=score["output_key"],
            output_key=_default_separatix_key(
                embedding_manifest["output_key"],
                label_manifest["output_key"],
                score["output_key"],
            ),
        ),
        store=store,
    )
    result = benchmark_result_from_artifacts(
        score_key=score["output_key"],
        store=store,
        separatix_key=diagnostic["output_key"],
    )

    assert diagnostic["artifact_type"] == "separatix_diagnostic"
    assert diagnostic["diagnostic"]["ran"] is True
    assert fake_separatix.ComplexityProfiler.calls[-1]["n_labels"] == 4
    assert diagnostic["diagnostic"]["probe_summary"]["status"] == "executed"
    assert result["extractor_results"][0]["separatix"]["recommendation"] == (
        "smooth_nonlinear_recommended"
    )
    assert (
        result["extractor_results"][0]["separatix"]["probe_summary"]["best_probe"] == "smooth_poly"
    )


@pytest.mark.parametrize(
    ("excluded", "expected_rows", "expected_columns", "expected_ran"),
    [
        (None, 8, 3, True),
        (["round"], 7, 2, True),
        (["red", "round", "sweet"], 0, 0, False),
    ],
)
def test_distributed_multilabel_separatix_preserves_sparse_targets_and_exclusions(
    tmp_path,
    fake_overlapindex,
    fake_separatix,
    excluded,
    expected_rows,
    expected_columns,
    expected_ran,
):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24).reshape(8, 3),
        [
            ("red", "round"),
            ("red",),
            ("round",),
            ("sweet",),
            ("red", "sweet"),
            ("round", "sweet"),
            ("red", "round"),
            ("red", "sweet"),
        ],
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "diagnose_multilabel_artifact",
        lambda batch: np.asarray(batch),
        streaming_safe=True,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_manifest = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=LocalBackend(),
        total_shards=2,
        batch_size=2,
    )
    label_manifest = materialize_label_artifact(dataset, store)
    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            output_key=_default_scoring_key(
                embedding_manifest["output_key"],
                label_manifest["output_key"],
            ),
            scoring_config=OverlapScoringConfig(
                k=1,
                min_samples_per_cluster=1,
                exclude_classes=excluded,
            ),
        ),
        store,
    )

    diagnostic = diagnose_embedding_artifact(
        SeparatixJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            score_key=score["output_key"],
            output_key=_default_separatix_key(
                embedding_manifest["output_key"],
                label_manifest["output_key"],
                score["output_key"],
            ),
        ),
        store,
    )

    assert diagnostic["diagnostic"]["ran"] is expected_ran
    if expected_ran:
        call = fake_separatix.ComplexityProfiler.calls[-1]
        assert call["target_mode"] == "multilabel"
        assert call["y_sparse"] is True
        assert call["y_shape"] == [expected_rows, expected_columns]
    else:
        assert "all classes were excluded" in diagnostic["diagnostic"]["skipped_reason"]
