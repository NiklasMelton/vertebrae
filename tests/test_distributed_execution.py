import numpy as np
import pytest
from scipy import sparse

from vertebrae import BenchmarkDataset
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.datasets import TargetView
from vertebrae.execution import (
    EmbeddingMergeJob,
    LocalBackend,
    ResourceSpec,
    ScoringJob,
    SeparatixJob,
    benchmark_result_from_artifacts,
    collect_score_artifacts,
    diagnose_embedding_artifact,
    embedding_artifact_key,
    groups_artifact_key,
    labels_artifact_key,
    materialize_and_merge_embeddings,
    materialize_embedding_shard,
    materialize_group_artifact,
    materialize_label_artifact,
    merge_embedding_shards,
    plan_embedding_shard_jobs,
    plan_scoring_jobs,
    score_embedding_artifact,
    score_embedding_artifacts,
    scoring_artifact_key,
    separatix_artifact_key,
)
from vertebrae.extractors import CallableExtractor, MultiOutputExtractor
from vertebrae.extractors.base import EmbeddingOutputSpec

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


def test_resource_spec_validates_bounds():
    with pytest.raises(ValueError, match="cpus"):
        ResourceSpec(cpus=0)
    with pytest.raises(ValueError, match="gpus"):
        ResourceSpec(gpus=-1)


def test_group_artifact_and_separatix_job_preserve_group_safety(
    tmp_path,
    fake_overlapindex,
    fake_separatix,
):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24, dtype=float).reshape(8, 3),
        np.array(["a"] * 4 + ["b"] * 4),
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
            output_key=scoring_artifact_key(embedding_manifest["output_key"]),
        ),
        store,
    )

    diagnostic = diagnose_embedding_artifact(
        SeparatixJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=labels["output_key"],
            groups_key=groups["output_key"],
            score_key=score["output_key"],
            output_key=separatix_artifact_key(embedding_manifest["output_key"]),
        ),
        store,
    )

    assert groups["output_key"] == groups_artifact_key(dataset)
    assert groups["n_groups"] == 4
    assert fake_separatix.ComplexityProfiler.calls[-1]["groups"].tolist() == (
        dataset.groups().tolist()
    )
    assert diagnostic["diagnostic"]["metadata"]["grouped"] is True


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

    embeddings = store.get_array(embedding_artifact_key(dataset, extractor))
    assert manifest["n_shards"] == 3
    assert np.array_equal(embeddings, dataset.X[:, :2] + 1)


def test_materialize_and_merge_sparse_embedding_shards(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        ["a"] * 5 + ["b"] * 5,
        modality="tabular",
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
    )
    extractor = MultiOutputExtractor(
        name="multi",
        output_specs=[EmbeddingOutputSpec("left"), EmbeddingOutputSpec("right")],
        transform_many_fn=lambda batch: {
            "left": np.asarray(batch)[:, :2],
            "right": np.asarray(batch)[:, 1:3],
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
    assert [output["output_name"] for output in merged["outputs"]] == ["left", "right"]
    left = store.get_array(merged["outputs"][0]["output_key"])
    right = store.get_array(merged["outputs"][1]["output_key"])
    assert np.array_equal(left, dataset.X[:, :2])
    assert np.array_equal(right, dataset.X[:, 1:3])


def test_materialize_and_merge_embeddings_supports_multi_output(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
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
    dataset = BenchmarkDataset.from_embeddings(np.arange(36).reshape(12, 3), labels)
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
            output_key=scoring_artifact_key(embedding_manifest["output_key"]),
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
            output_key=scoring_artifact_key(embedding_manifest["output_key"]),
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
    )
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
    jobs = plan_scoring_jobs(
        embedding_key=embedding_manifest["output_key"],
        labels_key=label_manifest["output_key"],
        seeds=[3, 5, 7],
    )

    scores = score_embedding_artifacts(jobs, store, LocalBackend())
    collection = collect_score_artifacts(
        [score["output_key"] for score in scores],
        store,
        output_key=f'{embedding_manifest["output_key"]}/scores/stability',
    )
    result = benchmark_result_from_artifacts(
        score_key=scoring_artifact_key(embedding_manifest["output_key"], seed=3),
        store=store,
        stability_key=collection["output_key"],
    )

    assert collection["artifact_type"] == "score_collection"
    assert collection["seeds"] == [3, 5, 7]
    assert result["metadata"]["distributed_artifacts"] is True
    assert result["extractor_results"][0]["stability"]["summary"]["mean"] > 0.0


def test_benchmark_from_artifacts_carries_label_view_metadata(tmp_path, fake_overlapindex):
    dataset = (
        BenchmarkDataset.from_embeddings(
            np.arange(24).reshape(8, 3),
            ["husky", "husky", "pug", "pug", "sedan", "sedan", "suv", "suv"],
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
            output_key=scoring_artifact_key(embedding_manifest["output_key"]),
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
            output_key=scoring_artifact_key(embedding_manifest["output_key"]),
        ),
        store,
    )

    result = benchmark_result_from_artifacts(score_key=score["output_key"], store=store)

    assert result["dataset_summary"]["target_view"]["name"] == "coarse"
    assert result["extractor_results"][0]["target_view"]["name"] == "coarse"


def test_diagnose_embedding_artifact_and_attach_to_benchmark_result(
    tmp_path,
    fake_overlapindex,
):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
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
            output_key=scoring_artifact_key(embedding_manifest["output_key"]),
        ),
        store,
    )

    diagnostic = diagnose_embedding_artifact(
        job=SeparatixJob(
            embedding_key=embedding_manifest["output_key"],
            labels_key=label_manifest["output_key"],
            score_key=score["output_key"],
            output_key=separatix_artifact_key(embedding_manifest["output_key"]),
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
    assert diagnostic["diagnostic"]["probe_summary"]["status"] == "executed"
    assert result["extractor_results"][0]["separatix"]["recommendation"] == (
        "smooth_nonlinear_recommended"
    )
    assert (
        result["extractor_results"][0]["separatix"]["probe_summary"]["best_probe"] == "smooth_poly"
    )
