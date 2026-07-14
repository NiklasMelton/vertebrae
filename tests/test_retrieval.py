import numpy as np
import pytest
from scipy import sparse

from vertebrae import (
    DatasetIdentity,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    LabelRetrievalMetric,
    ResourceProfilingConfig,
    RetrievalBenchmark,
    RetrievalConfig,
    RetrievalDataset,
)
from vertebrae.cache import LocalArtifactStore
from vertebrae.execution import (
    EmbeddingMergeJob,
    RetrievalCompressionJob,
    RetrievalEmbeddingShardJob,
    RetrievalScoringJob,
    ShardSpec,
    compress_retrieval_embedding_artifacts,
    materialize_retrieval_embedding_shard,
    merge_retrieval_embedding_shards,
    retrieval_benchmark_result_from_artifacts,
    retrieval_embedding_artifact_key,
    score_retrieval_artifact,
)
from vertebrae.extractors import CallableRetrievalExtractor, PrecomputedExtractor
from vertebrae.retrieval import render_retrieval_markdown_report
from vertebrae.scoring import RetrievalScorer
from vertebrae.utils.embedding_batches import encode_endpoint_batches


def test_retrieval_dataset_accepts_sparse_grades_and_preserves_equal_ids():
    dataset = RetrievalDataset.from_arrays(
        ["q0", "q1"],
        ["g0", "g1"],
        [("same", "same", 2), ("q1", "g1", 1)],
        query_ids=["same", "q1"],
        gallery_ids=["same", "g1"],
        query_modality="text",
        gallery_modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    assert dataset.relevance[0] == {0: 2.0}
    assert not dataset.exclusions


def test_retrieval_scorer_reports_graded_and_binary_metrics():
    scorer = RetrievalScorer(RetrievalConfig(ks=(1, 2), primary_metric="ndcg@1"))
    result = scorer.score(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]]),
        {0: {0: 3.0, 1: 1.0}, 1: {2: 2.0}},
    )
    assert result.score == pytest.approx(1.0)
    assert result.metrics["hit_rate@1"] == pytest.approx(1.0)
    assert result.metrics["recall@2"] == pytest.approx(1.0)
    assert result.metrics["map"] == pytest.approx(1.0)


def test_blockwise_scoring_matches_large_gallery_blocks():
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    gallery = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    relevance = {0: {0: 2.0, 1: 1.0}, 1: {2: 2.0, 3: 1.0}}
    full = RetrievalScorer(
        RetrievalConfig(ks=(1, 2), primary_metric="ndcg@1", gallery_batch_size=100)
    ).score(queries, gallery, relevance)
    blocked = RetrievalScorer(
        RetrievalConfig(ks=(1, 2), primary_metric="ndcg@1", gallery_batch_size=1)
    ).score(queries, gallery, relevance)
    assert blocked.metrics == pytest.approx(full.metrics)


def test_query_batch_scoring_matches_single_query_batches():
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    gallery = np.asarray([[1.0, 0.0], [0.2, 0.8], [0.0, 1.0], [0.8, 0.2]])
    relevance = {0: {0: 2.0, 3: 1.0}, 1: {2: 2.0}, 2: {3: 1.0}}
    single = RetrievalScorer(
        RetrievalConfig(
            ks=(1, 2), primary_metric="ndcg@1", query_batch_size=1, gallery_batch_size=2
        )
    ).score(queries, gallery, relevance)
    batched = RetrievalScorer(
        RetrievalConfig(
            ks=(1, 2), primary_metric="ndcg@1", query_batch_size=3, gallery_batch_size=2
        )
    ).score(queries, gallery, relevance)
    assert batched.metrics == pytest.approx(single.metrics)


def test_dense_relevance_matrix_and_gallery_compression_benchmark():
    dataset = RetrievalDataset.from_relevance_matrix(
        np.eye(2),
        np.eye(2),
        [[2.0, 0.0], [0.0, 1.0]],
        query_modality="embeddings",
        gallery_modality="embeddings",
        identity=DatasetIdentity.ephemeral(),
    )
    result = RetrievalBenchmark(
        dataset,
        [PrecomputedExtractor()],
        retrieval_config=RetrievalConfig(ks=(1,), primary_metric="ndcg@1"),
    ).run()
    assert result.ranked_results()[0].primary_score == pytest.approx(1.0)
    assert result.ranked_results()[0].compression_metadata["fit_side"] == "gallery"


def test_retrieval_local_compression_skips_nonreducing_pca_and_preserves_dtype():
    query = np.eye(3, dtype=np.float64)
    gallery = np.eye(3, dtype=np.float64)
    dataset = RetrievalDataset.from_embeddings(
        query,
        gallery,
        [(index, index, 1.0) for index in range(3)],
        identity=DatasetIdentity.ephemeral(),
    )
    result = RetrievalBenchmark(
        dataset,
        [PrecomputedExtractor()],
        retrieval_config=RetrievalConfig(ks=(1,), primary_metric="ndcg@1"),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="pca",
            n_components=3,
            dtype="float32",
        ),
    ).run()

    metadata = result.extractor_results[0].compression_metadata
    assert metadata["applied"] is False
    assert metadata["dtype"] == "float64"
    assert metadata["fit_side"] == "gallery"
    assert any("skipping compression" in warning for warning in metadata["warnings"])
    assert any(
        "skipping compression" in warning for warning in result.extractor_results[0].warnings
    )


def test_raw_cross_modal_retrieval_uses_explicit_callable_branches():
    dataset = RetrievalDataset.from_arrays(
        ["zero", "one"],
        ["image-zero", "image-one"],
        [(0, 0, 1.0), (1, 1, 1.0)],
        query_modality="text",
        gallery_modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableRetrievalExtractor(
        "paired",
        query_fn=lambda values: np.asarray(
            [[1.0, 0.0] if value == "zero" else [0.0, 1.0] for value in values]
        ),
        gallery_fn=lambda values: np.asarray(
            [[1.0, 0.0] if value == "image-zero" else [0.0, 1.0] for value in values]
        ),
        query_modality="text",
        gallery_modality="image",
    )
    result = RetrievalBenchmark(
        dataset,
        [extractor],
        retrieval_config=RetrievalConfig(ks=(1,), primary_metric="ndcg@1"),
        query_branch="query",
        gallery_branch="gallery",
    ).run()
    assert result.ranked_results()[0].primary_score == pytest.approx(1.0)


def test_retrieval_profiles_endpoints_with_deterministic_batches():
    calls = {"query": [], "gallery": []}

    def encode_query(values):
        calls["query"].append(list(values))
        return np.asarray([[float(value), 1.0] for value in values])

    def encode_gallery(values):
        calls["gallery"].append(list(values))
        return np.asarray([[float(value), 1.0] for value in values])

    dataset = RetrievalDataset.from_arrays(
        [0, 1, 2],
        [0, 1, 2],
        [(index, index, 1.0) for index in range(3)],
        query_modality="text",
        gallery_modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    result = RetrievalBenchmark(
        dataset,
        [
            CallableRetrievalExtractor(
                "batched",
                encode_query,
                encode_gallery,
                query_modality="text",
                gallery_modality="image",
            )
        ],
        retrieval_config=RetrievalConfig(ks=(1,), primary_metric="ndcg@1"),
        query_branch="query",
        gallery_branch="gallery",
        embedding_config=EmbeddingConfig(batch_size=2),
        resource_profiling_config=ResourceProfilingConfig(enabled=True),
    ).run()

    item = result.extractor_results[0]
    assert calls == {"query": [[0, 1], [2]], "gallery": [[0, 1], [2]]}
    assert item.resource_profiles["query"].inference.batch_sizes == [2, 1]
    assert item.resource_profiles["gallery"].context["modality"] == "image"
    assert item.resource_profiles["query"].context["process_first_inference"] is True
    assert item.resource_profiles["gallery"].context["process_first_inference"] is False
    assert "query_warm_median_seconds" in result.to_dataframe().columns
    assert "Resources for quality-similar candidates" in render_retrieval_markdown_report(result)


def test_endpoint_batch_combination_preserves_sparse_output_and_order():
    combined = encode_endpoint_batches(
        [3, 1, 2],
        batch_size=2,
        encode=lambda values: sparse.csr_matrix(
            np.asarray([[value, value + 1] for value in values])
        ),
        owner="sparse endpoint",
    )

    assert sparse.issparse(combined)
    assert combined.toarray().tolist() == [[3, 4], [1, 2], [2, 3]]


def test_dataset_validates_direct_construction_and_identifies_endpoints():
    with pytest.raises(ValueError, match="Every query"):
        RetrievalDataset(
            queries=["q"],
            gallery=["g"],
            query_ids=["q"],
            gallery_ids=["g"],
            relevance=[],
            identity=DatasetIdentity.ephemeral(),
            query_modality="text",
            gallery_modality="text",
        )
    first = RetrievalDataset.from_embeddings(
        np.asarray([[1.0]]),
        np.asarray([[1.0]]),
        [(0, 0, 1)],
        identity=DatasetIdentity.from_content(),
    )
    second = RetrievalDataset.from_embeddings(
        np.asarray([[2.0]]),
        np.asarray([[1.0]]),
        [(0, 0, 1)],
        identity=DatasetIdentity.from_content(),
    )
    assert first.identity_key() != second.identity_key()


def test_bidirectional_requires_every_gallery_to_have_relevance():
    dataset = RetrievalDataset.from_embeddings(
        np.eye(2),
        np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        [(0, 0, 1), (1, 1, 1)],
        identity=DatasetIdentity.ephemeral(),
    )
    with pytest.raises(ValueError, match="bidirectional"):
        RetrievalBenchmark(
            dataset,
            [PrecomputedExtractor()],
            retrieval_config=RetrievalConfig(ks=(1,), primary_metric="ndcg@1", bidirectional=True),
        ).run()


def test_label_retrieval_is_leave_one_out_and_rejects_multilabel():
    metric = LabelRetrievalMetric(RetrievalConfig(ks=(1,), primary_metric="ndcg@1"))
    result = metric.score(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        ["a", "a", "b", "b"],
    )
    assert result.score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="single-label"):
        metric.score(np.eye(2), [("a",), ("b",)], target_metadata={"target_type": "multi_label"})


def test_retrieval_artifact_scoring_round_trip(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_array("q", np.eye(2))
    store.put_json(
        "q", {"n_samples": 2, "side": "query", "dataset_identity_key": "d", "recipe_hash": "e"}
    )
    store.put_array("g", np.eye(2))
    store.put_json(
        "g", {"n_samples": 2, "side": "gallery", "dataset_identity_key": "d", "recipe_hash": "e"}
    )
    store.put_json(
        "r", {"relevance": {"0": {"0": 1.0}, "1": {"1": 2.0}}, "dataset_identity_key": "d"}
    )
    artifact = score_retrieval_artifact(
        RetrievalScoringJob(
            "q", "g", "r", "out", RetrievalConfig(ks=(1,), primary_metric="ndcg@1")
        ),
        store,
    )
    assert artifact["artifact_type"] == "retrieval_evaluation"
    assert artifact["result"]["score"] == pytest.approx(1.0)
    result = retrieval_benchmark_result_from_artifacts(["out"], store)
    assert result.ranked_results()[0].primary_score == pytest.approx(1.0)
    assert result.dataset_summary["n_queries"] == 2


def test_retrieval_artifact_rejects_misaligned_ids(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_array("q", np.eye(2))
    store.put_json(
        "q", {"n_samples": 2, "side": "query", "dataset_identity_key": "d", "recipe_hash": "e"}
    )
    store.put_array("g", np.eye(2))
    store.put_json(
        "g", {"n_samples": 2, "side": "gallery", "dataset_identity_key": "d", "recipe_hash": "e"}
    )
    store.put_json(
        "r",
        {
            "relevance": {"0": {"0": 1.0}, "1": {"1": 1.0}},
            "query_ids": ["one"],
            "dataset_identity_key": "d",
        },
    )
    with pytest.raises(ValueError, match="query IDs"):
        score_retrieval_artifact(
            RetrievalScoringJob("q", "g", "r", "out"),
            store,
        )


def test_retrieval_endpoint_shards_merge_and_paired_compression(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    dataset = RetrievalDataset.from_embeddings(
        np.eye(4),
        np.eye(4),
        [(index, index, 1.0) for index in range(4)],
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = PrecomputedExtractor()
    query_shards = []
    gallery_shards = []
    for side, keys in (("query", query_shards), ("gallery", gallery_shards)):
        for index in range(2):
            key = f"{side}/shard/{index}"
            materialize_retrieval_embedding_shard(
                RetrievalEmbeddingShardJob(
                    dataset=dataset,
                    extractor=extractor,
                    side=side,
                    shard=ShardSpec(total_shards=2, shard_index=index),
                    output_key=key,
                    batch_size=1,
                    resource_profiling_config=ResourceProfilingConfig(enabled=True),
                ),
                store,
            )
            keys.append(key)
    query_manifest = merge_retrieval_embedding_shards(
        EmbeddingMergeJob(tuple(query_shards), "query", n_samples=4), store
    )
    merge_retrieval_embedding_shards(
        EmbeddingMergeJob(tuple(gallery_shards), "gallery", n_samples=4), store
    )
    artifact = compress_retrieval_embedding_artifacts(
        RetrievalCompressionJob(
            "query",
            "gallery",
            "compressed/query",
            "compressed/gallery",
            EmbeddingCompressionConfig(
                enabled=True,
                method="prefix_truncate",
                n_components=2,
                assume_matryoshka=True,
                dtype="float32",
            ),
        ),
        store,
    )
    assert store.get_array("compressed/query").shape == (4, 2)
    assert store.get_array("compressed/gallery").shape == (4, 2)
    assert store.get_array("compressed/query").dtype == np.float32
    assert store.get_array("compressed/gallery").dtype == np.float32
    assert store.get_json("compressed/query")["dtype"] == "float32"
    assert store.get_json("compressed/gallery")["dtype"] == "float32"
    assert artifact["compression_metadata"]["fit_side"] == "gallery"
    distributed = query_manifest["distributed_resource_profile"]
    assert distributed["scope"] == "distributed_shards"
    assert distributed["shard_count"] == 2
    assert distributed["worker_first_calls"]["count"] == 2
    assert distributed["embedding"]["evaluated_persisted"]["status"] == "measured"


def test_retrieval_artifact_compression_skips_nonreducing_pca(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    query = np.eye(3, dtype=np.float64)
    gallery = np.eye(3, dtype=np.float64)
    for key, values, side in (("query", query, "query"), ("gallery", gallery, "gallery")):
        store.put_array(key, values)
        store.put_json(
            key,
            {
                "side": side,
                "dataset_identity_key": "dataset-v1",
                "recipe_hash": "extractor-v1",
            },
        )

    artifact = compress_retrieval_embedding_artifacts(
        RetrievalCompressionJob(
            "query",
            "gallery",
            "compressed/query",
            "compressed/gallery",
            EmbeddingCompressionConfig(
                enabled=True,
                method="pca",
                n_components=4,
                dtype="float32",
            ),
        ),
        store,
    )

    assert artifact["compression_metadata"]["applied"] is False
    assert artifact["compression_metadata"]["dtype"] == "float64"
    assert np.array_equal(store.get_array("compressed/query"), query)
    assert np.array_equal(store.get_array("compressed/gallery"), gallery)
    assert store.get_array("compressed/query").dtype == np.float64
    assert store.get_array("compressed/gallery").dtype == np.float64


def test_retrieval_branch_keys_are_distinct_and_wrong_provenance_is_rejected(tmp_path):
    dataset = RetrievalDataset.from_embeddings(
        np.eye(2), np.eye(2), [(0, 0, 1), (1, 1, 1)], identity=DatasetIdentity.ephemeral()
    )
    extractor = PrecomputedExtractor()
    assert retrieval_embedding_artifact_key(
        dataset, extractor, "query", "first"
    ) != retrieval_embedding_artifact_key(dataset, extractor, "query", "second")
    store = LocalArtifactStore(str(tmp_path))
    store.put_array("q", np.eye(2))
    store.put_array("g", np.eye(2))
    store.put_json(
        "q", {"n_samples": 2, "side": "query", "dataset_identity_key": "one", "recipe_hash": "e"}
    )
    store.put_json(
        "g", {"n_samples": 2, "side": "gallery", "dataset_identity_key": "two", "recipe_hash": "e"}
    )
    store.put_json(
        "r", {"relevance": {"0": {"0": 1}, "1": {"1": 1}}, "dataset_identity_key": "one"}
    )
    with pytest.raises(ValueError, match="dataset identities"):
        score_retrieval_artifact(RetrievalScoringJob("q", "g", "r", "out"), store)
