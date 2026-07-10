import numpy as np
import pytest

from vertebrae import (
    EmbeddingCompressionConfig,
    LabelRetrievalMetric,
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
    score_retrieval_artifact,
)
from vertebrae.extractors import CallableRetrievalExtractor, PrecomputedExtractor
from vertebrae.scoring import RetrievalScorer


def test_retrieval_dataset_accepts_sparse_grades_and_preserves_equal_ids():
    dataset = RetrievalDataset.from_arrays(
        ["q0", "q1"],
        ["g0", "g1"],
        [("same", "same", 2), ("q1", "g1", 1)],
        query_ids=["same", "q1"],
        gallery_ids=["same", "g1"],
        query_modality="text",
        gallery_modality="image",
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


def test_dense_relevance_matrix_and_gallery_compression_benchmark():
    dataset = RetrievalDataset.from_relevance_matrix(
        np.eye(2),
        np.eye(2),
        [[2.0, 0.0], [0.0, 1.0]],
        query_modality="embeddings",
        gallery_modality="embeddings",
    )
    result = RetrievalBenchmark(
        dataset,
        [PrecomputedExtractor()],
        retrieval_config=RetrievalConfig(ks=(1,), primary_metric="ndcg@1"),
    ).run()
    assert result.ranked_results()[0].primary_score == pytest.approx(1.0)
    assert result.ranked_results()[0].compression_metadata["fit_side"] == "gallery"


def test_raw_cross_modal_retrieval_uses_explicit_callable_branches():
    dataset = RetrievalDataset.from_arrays(
        ["zero", "one"],
        ["image-zero", "image-one"],
        [(0, 0, 1.0), (1, 1, 1.0)],
        query_modality="text",
        gallery_modality="image",
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


def test_dataset_validates_direct_construction_and_fingerprints_endpoints():
    with pytest.raises(ValueError, match="Every query"):
        RetrievalDataset(
            queries=["q"],
            gallery=["g"],
            query_ids=["q"],
            gallery_ids=["g"],
            relevance=[],
            query_modality="text",
            gallery_modality="text",
        )
    first = RetrievalDataset.from_embeddings(np.asarray([[1.0]]), np.asarray([[1.0]]), [(0, 0, 1)])
    second = RetrievalDataset.from_embeddings(np.asarray([[2.0]]), np.asarray([[1.0]]), [(0, 0, 1)])
    assert first.fingerprint() != second.fingerprint()


def test_bidirectional_requires_every_gallery_to_have_relevance():
    dataset = RetrievalDataset.from_embeddings(
        np.eye(2),
        np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        [(0, 0, 1), (1, 1, 1)],
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
    store.put_json("q", {"n_samples": 2})
    store.put_array("g", np.eye(2))
    store.put_json("g", {"n_samples": 2})
    store.put_json("r", {"relevance": {"0": {"0": 1.0}, "1": {"1": 2.0}}})
    artifact = score_retrieval_artifact(
        RetrievalScoringJob(
            "q", "g", "r", "out", RetrievalConfig(ks=(1,), primary_metric="ndcg@1")
        ),
        store,
    )
    assert artifact["artifact_type"] == "retrieval_evaluation"
    assert artifact["result"]["score"] == pytest.approx(1.0)


def test_retrieval_artifact_rejects_misaligned_ids(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_array("q", np.eye(2))
    store.put_json("q", {"n_samples": 2})
    store.put_array("g", np.eye(2))
    store.put_json("g", {"n_samples": 2})
    store.put_json(
        "r",
        {"relevance": {"0": {"0": 1.0}, "1": {"1": 1.0}}, "query_ids": ["one"]},
    )
    with pytest.raises(ValueError, match="query IDs"):
        score_retrieval_artifact(
            RetrievalScoringJob("q", "g", "r", "out"),
            store,
        )


def test_retrieval_endpoint_shards_merge_and_paired_compression(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    dataset = RetrievalDataset.from_embeddings(
        np.eye(4), np.eye(4), [(index, index, 1.0) for index in range(4)]
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
                ),
                store,
            )
            keys.append(key)
    merge_retrieval_embedding_shards(
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
            ),
        ),
        store,
    )
    assert store.get_array("compressed/query").shape == (4, 2)
    assert store.get_array("compressed/gallery").shape == (4, 2)
    assert artifact["compression_metadata"]["fit_side"] == "gallery"
