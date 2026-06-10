import numpy as np
import pytest
from scipy import sparse

from vertebrae import BenchmarkDataset
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.execution import (
    EmbeddingMergeJob,
    LocalBackend,
    ResourceSpec,
    ScoringJob,
    benchmark_result_from_artifacts,
    collect_score_artifacts,
    embedding_artifact_key,
    labels_artifact_key,
    materialize_and_merge_embeddings,
    materialize_embedding_shard,
    materialize_label_artifact,
    merge_embedding_shards,
    plan_embedding_shard_jobs,
    plan_scoring_jobs,
    score_embedding_artifact,
    score_embedding_artifacts,
    scoring_artifact_key,
)
from vertebrae.extractors import CallableExtractor, MultiOutputExtractor
from vertebrae.extractors.base import EmbeddingOutputSpec


def test_resource_spec_validates_bounds():
    with pytest.raises(ValueError, match="cpus"):
        ResourceSpec(cpus=0)
    with pytest.raises(ValueError, match="gpus"):
        ResourceSpec(gpus=-1)


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


def test_plan_embedding_shards_rejects_multi_output_extractors():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(12).reshape(4, 3),
        ["a", "a", "b", "b"],
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

    with pytest.raises(ValueError, match="single-output extractors only"):
        plan_embedding_shard_jobs(dataset, extractor, total_shards=2, batch_size=2)


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

    assert score["artifact_type"] == "overlap_score"
    assert score["score"]["macro_score"] == 0.8
    assert score["embedding_key"] == embedding_manifest["output_key"]
    assert score["labels_key"] == labels_artifact_key(dataset)
    assert store.get_json(score["output_key"])["score"]["metadata"]["backend"] == "MiniBatchKMeans"


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
