import numpy as np
import pytest
from scipy import sparse

from vertebrae import (
    DatasetIdentity,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    LabelRetrievalMetric,
    MemoryConfig,
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
    plan_retrieval_embedding_shard_jobs,
    retrieval_benchmark_result_from_artifacts,
    retrieval_embedding_artifact_key,
    score_retrieval_artifact,
)
from vertebrae.extractors import CallableRetrievalExtractor, PrecomputedExtractor
from vertebrae.retrieval import render_retrieval_markdown_report
from vertebrae.scoring import RetrievalScorer
from vertebrae.utils.embedding_batches import encode_endpoint_batches


def test_precomputed_extractor_serializes_cache_eligibility():
    extractor = PrecomputedExtractor(cache_embeddings=False)

    assert extractor.cache_embeddings is False
    assert extractor.recipe()["cache_embeddings"] is False


def test_retrieval_planning_honors_disabled_embedding_cache():
    dataset = RetrievalDataset.from_embeddings(
        np.eye(4),
        np.eye(4),
        [(index, index, 1.0) for index in range(4)],
        identity=DatasetIdentity.from_content(),
    )

    jobs = plan_retrieval_embedding_shard_jobs(
        dataset,
        PrecomputedExtractor(cache_embeddings=False),
        2,
        side="query",
    )

    assert all(job.output_key.startswith("runs/") for job in jobs)
    assert all(not job.cache_eligible for job in jobs)
    assert all(job.cache_status == "disabled" for job in jobs)


def test_unsafe_retrieval_artifact_plans_are_run_scoped():
    dataset = RetrievalDataset.from_embeddings(
        np.eye(4),
        np.eye(4),
        [(index, index, 1.0) for index in range(4)],
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableRetrievalExtractor(
        "unsafe",
        lambda values: values,
        lambda values: values,
    )

    jobs = plan_retrieval_embedding_shard_jobs(
        dataset,
        extractor,
        2,
        side="query",
        branch="query",
    )

    assert all(job.output_key.startswith("runs/") for job in jobs)
    assert all(job.cache_status == "bypassed_unsafe_identity" for job in jobs)


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


def test_retrieval_dataset_uses_immutable_protocol_snapshots():
    metadata = {"nested": {"version": 1}}
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    gallery = [["g0"], ["g1"]]
    dataset = RetrievalDataset.from_arrays(
        queries,
        gallery,
        [("q0", "g0", 1.0), ("q1", "g1", 1.0)],
        query_ids=["q0", "q1"],
        gallery_ids=["g0", "g1"],
        query_modality="text",
        gallery_modality="image",
        metadata=metadata,
        identity=DatasetIdentity.from_content(),
    )
    identity_key = dataset.identity_key()
    summary = dataset.summary()

    metadata["nested"]["version"] = 2
    with pytest.raises(TypeError):
        dataset.metadata["nested"]["version"] = 3
    with pytest.raises(TypeError):
        dataset.query_ids[0] = "changed"
    with pytest.raises(TypeError):
        dataset.gallery_ids[0] = "changed"
    with pytest.raises(AttributeError):
        dataset.relevance[0].clear()
    with pytest.raises(AttributeError):
        dataset.exclusions = {(1, 1)}
    with pytest.raises(AttributeError):
        dataset.query_modality = "changed"
    with pytest.raises(AttributeError):
        dataset.gallery_modality = "changed"
    with pytest.raises(ValueError):
        dataset.queries[0, 0] = 42.0
    queries[0, 0] = 99.0
    gallery[0][0] = "changed"
    with pytest.raises(AttributeError):
        dataset.queries = np.zeros((2, 2))
    with pytest.raises(AttributeError):
        dataset.gallery = [["changed"], ["changed"]]
    summary["metadata"]["nested"]["version"] = 99

    assert dataset.identity_key() == identity_key
    assert dataset.summary()["metadata"]["nested"]["version"] == 1
    assert dataset.query_id_values() == ("q0", "q1")
    assert dataset.gallery_id_values() == ("g0", "g1")
    assert dataset.eligible_relevance() == {0: {0: 1.0}, 1: {1: 1.0}}
    assert dataset.query_values().tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert dataset.gallery_values() == (("g0",), ("g1",))
    assert dataset.query_values().flags.writeable is False


def test_retrieval_record_dispatch_does_not_treat_three_column_lists_as_matrix():
    dataset = RetrievalDataset.from_arrays(
        ["q0", "q1"],
        ["g0", "g1", "g2"],
        [["q0", "g2", 2.0], ["q1", "g1", 1.0]],
        query_ids=["q0", "q1"],
        gallery_ids=["g0", "g1", "g2"],
        query_modality="text",
        gallery_modality="image",
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.relevance == {0: {2: 2.0}, 1: {1: 1.0}}


def test_sparse_relevance_is_normalized_without_dense_materialization():
    class NoDenseCsr(sparse.csr_matrix):
        def toarray(self, *args, **kwargs):
            raise AssertionError("sparse relevance must not be densified")

    relevance = NoDenseCsr(
        (np.asarray([2.0, 1.0]), (np.asarray([0, 1]), np.asarray([2, 1]))),
        shape=(2, 100_000),
    )
    dataset = RetrievalDataset.from_arrays(
        ["q0", "q1"],
        list(range(100_000)),
        relevance,
        query_modality="text",
        gallery_modality="id",
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.relevance == {0: {2: 2.0}, 1: {1: 1.0}}


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


def test_retrieval_scorer_rejects_queries_without_eligible_positives():
    scorer = RetrievalScorer(RetrievalConfig(ks=(1,), primary_metric="ndcg@1"))
    with pytest.raises(ValueError, match="Every query.*eligible positive"):
        scorer.score(
            np.eye(2),
            np.eye(2),
            {0: {0: 1.0}, 1: {1: 1.0}},
            query_ids=["left", "right"],
            exclusions={(1, 1)},
        )


def test_retrieval_ndcg_is_finite_for_large_finite_grades():
    result = RetrievalScorer(RetrievalConfig(ks=(1, 2), primary_metric="ndcg@2")).score(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[1.0, 0.0], [0.9, 0.1]]),
        {0: {0: 1024.0, 1: 1023.0}},
    )

    assert result.metrics["ndcg@2"] == pytest.approx(1.0)
    assert all(np.isfinite(value) for value in result.metrics.values())


@pytest.mark.parametrize("grades", [(1e-300, 2e-300), (5e307, 1e308)])
def test_retrieval_ndcg_is_stable_for_tiny_and_extreme_finite_grades(grades):
    result = RetrievalScorer(
        RetrievalConfig(
            similarity="dot",
            ks=(2,),
            primary_metric="ndcg@2",
        )
    ).score(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[1.0, 0.0], [0.5, 0.0]]),
        {0: {0: grades[0], 1: grades[1]}},
    )

    assert np.isfinite(result.score)
    assert 0.0 < result.score < 1.0


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


@pytest.mark.parametrize(
    ("endpoint", "use_sparse"),
    [
        ("query", False),
        ("query", True),
        ("gallery", False),
        ("gallery", True),
    ],
)
def test_cosine_retrieval_rejects_zero_norm_rows_with_endpoint_identity(endpoint, use_sparse):
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    gallery = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    query_ids = ["query-ok", "query-zero"]
    gallery_ids = ["gallery-ok", "gallery-zero"]
    if endpoint == "query":
        queries[1] = 0.0
        if use_sparse:
            queries = sparse.csr_matrix(queries)
        expected_endpoint = "Query embeddings"
        expected_id = "query-zero"
    else:
        gallery[1] = 0.0
        if use_sparse:
            gallery = sparse.csr_matrix(gallery)
        expected_endpoint = "Gallery embeddings"
        expected_id = "gallery-zero"

    with pytest.raises(ValueError) as error:
        RetrievalScorer(RetrievalConfig(ks=(1,), primary_metric="ndcg@1")).score(
            queries,
            gallery,
            {0: {0: 1.0}, 1: {1: 1.0}},
            query_ids=query_ids,
            gallery_ids=gallery_ids,
        )

    message = str(error.value)
    assert expected_endpoint in message
    assert "1 zero-norm row(s)" in message
    assert "index 1" in message
    assert expected_id in message
    assert "Cosine similarity is undefined" in message


def test_cosine_retrieval_zero_norm_error_bounds_row_preview():
    queries = np.zeros((12, 2))
    gallery = np.asarray([[1.0, 0.0]])
    relevance = {index: {0: 1.0} for index in range(len(queries))}

    with pytest.raises(ValueError) as error:
        RetrievalScorer(RetrievalConfig(ks=(1,), primary_metric="ndcg@1")).score(
            queries,
            gallery,
            relevance,
        )

    message = str(error.value)
    assert "12 zero-norm row(s)" in message
    assert "index 9" in message
    assert "index 10" not in message
    assert ", ..." in message


@pytest.mark.parametrize("similarity", ["dot", "squared_l2"])
def test_non_cosine_retrieval_accepts_zero_norm_rows(similarity):
    result = RetrievalScorer(
        RetrievalConfig(similarity=similarity, ks=(1,), primary_metric="ndcg@1")
    ).score(
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        {0: {0: 1.0}, 1: {1: 1.0}},
    )

    assert result.score == pytest.approx(1.0)


def test_retrieval_benchmark_rejects_zero_norm_cosine_queries():
    dataset = RetrievalDataset.from_embeddings(
        np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        [
            ("query-ok", "gallery-left", 1.0),
            ("query-zero", "gallery-right", 1.0),
        ],
        query_ids=["query-ok", "query-zero"],
        gallery_ids=["gallery-left", "gallery-right"],
        identity=DatasetIdentity.ephemeral(),
    )

    with pytest.raises(ValueError, match="Query embeddings.*query-zero"):
        RetrievalBenchmark(
            dataset,
            [PrecomputedExtractor()],
            retrieval_config=RetrievalConfig(ks=(1,), primary_metric="ndcg@1"),
        ).run()


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


def test_retrieval_markdown_escapes_all_dynamic_values():
    dataset = RetrievalDataset.from_arrays(
        [0, 1],
        [0, 1],
        [(0, 0, 1.0), (1, 1, 1.0)],
        query_modality="text",
        gallery_modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    result = RetrievalBenchmark(
        dataset,
        [
            CallableRetrievalExtractor(
                "safe",
                lambda values: np.eye(2)[list(values)],
                lambda values: np.eye(2)[list(values)],
                query_modality="text",
                gallery_modality="image",
            )
        ],
        retrieval_config=RetrievalConfig(ks=(1,), primary_metric="ndcg@1"),
        query_branch="query",
        gallery_branch="gallery",
    ).run()
    item = result.extractor_results[0]
    injected = "name|C:\\model\n## injected"
    metric = "ndcg|unsafe\\metric\n## metric-injected"
    item.name = injected
    item.forward.primary_metric = metric
    item.forward.metrics[metric] = 1.0
    item.warnings = ["warning|unsafe\\text\n- injected-list-item"]
    result.dataset_summary = {
        "field|unsafe\\key\n## injected-key": "<script>|unsafe\\value\n## injected-value"
    }

    report = render_retrieval_markdown_report(result)

    assert r"name\|C:\\model<br>## injected" in report
    assert r"ndcg\|unsafe\\metric<br>## metric-injected" in report
    assert r"warning\|unsafe\\text<br>- injected-list-item" in report
    assert "&lt;script&gt;\\|unsafe" in report
    assert "\n## injected" not in report
    assert "<script>" not in report


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


def test_endpoint_batch_admission_fails_before_encoding_all_batches():
    calls = []

    def encode(values):
        calls.append(list(values))
        return np.asarray([[float(value), 1.0] for value in values], dtype=np.float64)

    with pytest.raises(ValueError, match="memory budget"):
        encode_endpoint_batches(
            list(range(8)),
            batch_size=2,
            encode=encode,
            owner="bounded endpoint",
            memory_config=MemoryConfig(
                max_memory_bytes=16,
                allow_disk_spill=False,
            ),
        )

    assert calls == [[0, 1]]


def test_endpoint_batch_admission_spills_without_vstack_accumulation(monkeypatch):
    calls = []

    def encode(values):
        calls.append(list(values))
        return np.asarray([[float(value), 1.0] for value in values], dtype=np.float64)

    def reject_vstack(*_args, **_kwargs):
        raise AssertionError("memory-controlled endpoint batching must not use np.vstack")

    monkeypatch.setattr(np, "vstack", reject_vstack)
    combined = encode_endpoint_batches(
        list(range(5)),
        batch_size=2,
        encode=encode,
        owner="spilled endpoint",
        memory_config=MemoryConfig(max_memory_bytes=1, allow_disk_spill=True),
    )

    assert isinstance(combined, np.memmap)
    assert combined.tolist() == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]
    assert calls == [[0, 1], [2, 3], [4]]


def test_local_retrieval_propagates_memory_admission_before_gallery_encoding():
    query_calls = []
    gallery_calls = []

    def encode_query(values):
        query_calls.append(list(values))
        return np.asarray([[float(value + 1), 1.0] for value in values])

    def encode_gallery(values):
        gallery_calls.append(list(values))
        return np.asarray([[float(value + 1), 1.0] for value in values])

    dataset = RetrievalDataset.from_arrays(
        list(range(6)),
        list(range(6)),
        [(index, index, 1.0) for index in range(6)],
        query_modality="text",
        gallery_modality="image",
        identity=DatasetIdentity.ephemeral(),
    )

    with pytest.raises(ValueError, match="memory budget"):
        RetrievalBenchmark(
            dataset,
            [
                CallableRetrievalExtractor(
                    "bounded",
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
            memory_config=MemoryConfig(
                max_memory_bytes=16,
                allow_disk_spill=False,
            ),
        ).run()

    assert query_calls == [[0, 1]]
    assert gallery_calls == []


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


def test_label_retrieval_uses_exact_semantic_label_equality():
    metric = LabelRetrievalMetric(RetrievalConfig(ks=(1,), primary_metric="ndcg@1"))
    result = metric.score(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        [1, 1, True, True],
    )

    assert result.score == pytest.approx(1.0)
    assert result.diagnostics["recall@1"] == pytest.approx(1.0)


def test_retrieval_artifact_scoring_round_trip(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_artifact(
        "q",
        np.eye(2),
        {
            "n_samples": 2,
            "side": "query",
            "dataset_identity_key": "d",
            "recipe_hash": "e",
            "cache_eligible": False,
            "cache_status": "disabled",
        },
    )
    store.put_artifact(
        "g",
        np.eye(2),
        {
            "n_samples": 2,
            "side": "gallery",
            "dataset_identity_key": "d",
            "recipe_hash": "e",
            "cache_eligible": False,
            "cache_status": "disabled",
        },
    )
    store.put_json(
        "r",
        {
            "relevance": {"0": {"0": 1.0}, "1": {"1": 2.0}},
            "dataset_identity_key": "d",
            "protocol_fingerprint": "d",
        },
    )
    artifact = score_retrieval_artifact(
        RetrievalScoringJob(
            "q", "g", "r", "out", RetrievalConfig(ks=(1,), primary_metric="ndcg@1")
        ),
        store,
    )
    assert artifact["artifact_type"] == "retrieval_evaluation"
    assert artifact["forward"]["score"] == pytest.approx(1.0)
    assert artifact["reverse"] is None
    assert artifact["primary_score"] == pytest.approx(1.0)
    assert artifact["cache_eligible"] is False
    assert artifact["cache_status"] == "disabled"
    compressed = compress_retrieval_embedding_artifacts(
        RetrievalCompressionJob(
            query_embedding_key="q",
            gallery_embedding_key="g",
            query_output_key="compressed/query",
            gallery_output_key="compressed/gallery",
            compression_config=EmbeddingCompressionConfig(
                enabled=True,
                method="prefix_truncate",
                n_components=1,
                assume_matryoshka=True,
            ),
        ),
        store,
    )
    assert compressed["cache_eligible"] is False
    assert compressed["cache_status"] == "disabled"
    assert store.get_json("compressed/query")["cache_status"] == "disabled"
    assert store.get_json("compressed/gallery")["cache_status"] == "disabled"
    result = retrieval_benchmark_result_from_artifacts(["out"], store)
    assert result.ranked_results()[0].primary_score == pytest.approx(1.0)
    assert result.dataset_summary["n_queries"] == 2


def test_retrieval_artifact_scoring_preserves_bidirectional_results(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    for key, side in (("q", "query"), ("g", "gallery")):
        store.put_artifact(
            key,
            np.eye(2),
            {
                "n_samples": 2,
                "side": side,
                "dataset_identity_key": "protocol",
                "recipe_hash": "extractor",
            },
        )
    store.put_json(
        "r",
        {
            "relevance": {"0": {"0": 1.0}, "1": {"1": 1.0}},
            "query_ids": ["q0", "q1"],
            "gallery_ids": ["g0", "g1"],
            "n_queries": 2,
            "n_gallery": 2,
            "dataset_identity_key": "protocol",
            "protocol_fingerprint": "protocol",
        },
    )

    artifact = score_retrieval_artifact(
        RetrievalScoringJob(
            "q",
            "g",
            "r",
            "out",
            RetrievalConfig(ks=(1,), primary_metric="ndcg@1", bidirectional=True),
        ),
        store,
    )
    reconstructed = retrieval_benchmark_result_from_artifacts(["out"], store)

    assert artifact["forward"]["score"] == pytest.approx(1.0)
    assert artifact["reverse"]["score"] == pytest.approx(1.0)
    assert artifact["primary_score"] == pytest.approx(1.0)
    assert reconstructed.extractor_results[0].reverse is not None


def test_retrieval_artifact_bidirectional_requires_reverse_coverage(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_artifact(
        "q",
        np.eye(2, 3),
        {"n_samples": 2, "side": "query", "dataset_identity_key": "d", "recipe_hash": "e"},
    )
    store.put_artifact(
        "g",
        np.eye(3),
        {"n_samples": 3, "side": "gallery", "dataset_identity_key": "d", "recipe_hash": "e"},
    )
    store.put_json(
        "r",
        {
            "relevance": {"0": {"0": 1.0}, "1": {"1": 1.0}},
            "n_queries": 2,
            "n_gallery": 3,
            "dataset_identity_key": "d",
            "protocol_fingerprint": "d",
        },
    )

    with pytest.raises(ValueError, match="eligible reverse relevance"):
        score_retrieval_artifact(
            RetrievalScoringJob(
                "q",
                "g",
                "r",
                "out",
                RetrievalConfig(ks=(1,), primary_metric="ndcg@1", bidirectional=True),
            ),
            store,
        )


def test_retrieval_artifact_comparison_rejects_different_protocol_fingerprints(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    for key, side in (("q", "query"), ("g", "gallery")):
        store.put_artifact(
            key,
            np.eye(2),
            {
                "n_samples": 2,
                "side": side,
                "dataset_identity_key": "d",
                "recipe_hash": "e",
            },
        )
    store.put_json(
        "r",
        {
            "relevance": {"0": {"0": 1.0}, "1": {"1": 1.0}},
            "dataset_identity_key": "d",
            "protocol_fingerprint": "d",
        },
    )
    score_retrieval_artifact(
        RetrievalScoringJob(
            "q", "g", "r", "cosine", RetrievalConfig(ks=(1,), primary_metric="ndcg@1")
        ),
        store,
    )
    score_retrieval_artifact(
        RetrievalScoringJob(
            "q",
            "g",
            "r",
            "dot",
            RetrievalConfig(similarity="dot", ks=(1,), primary_metric="ndcg@1"),
        ),
        store,
    )

    with pytest.raises(ValueError, match="protocol fingerprint"):
        retrieval_benchmark_result_from_artifacts(["cosine", "dot"], store)


def test_retrieval_artifact_reconstruction_recomputes_protocol_fingerprint(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    for key, side in (("q", "query"), ("g", "gallery")):
        store.put_artifact(
            key,
            np.eye(2),
            {
                "n_samples": 2,
                "side": side,
                "dataset_identity_key": "d",
                "recipe_hash": "e",
            },
        )
    store.put_json(
        "r",
        {
            "relevance": {"0": {"0": 1.0}, "1": {"1": 1.0}},
            "dataset_identity_key": "d",
            "protocol_fingerprint": "d",
        },
    )
    artifact = score_retrieval_artifact(
        RetrievalScoringJob(
            "q", "g", "r", "valid", RetrievalConfig(ks=(1,), primary_metric="ndcg@1")
        ),
        store,
    )
    tampered = dict(artifact)
    tampered["retrieval_config"] = {
        **artifact["retrieval_config"],
        "query_batch_size": artifact["retrieval_config"]["query_batch_size"] + 1,
    }
    store.put_json("tampered", tampered)

    with pytest.raises(ValueError, match="protocol fingerprint is inconsistent"):
        retrieval_benchmark_result_from_artifacts(["valid", "tampered"], store)


def test_retrieval_artifact_rejects_zero_norm_cosine_gallery(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_artifact(
        "q",
        np.eye(2),
        {"n_samples": 2, "side": "query", "dataset_identity_key": "d", "recipe_hash": "e"},
    )
    store.put_artifact(
        "g",
        np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        {"n_samples": 2, "side": "gallery", "dataset_identity_key": "d", "recipe_hash": "e"},
    )
    store.put_json(
        "r",
        {
            "relevance": {"0": {"0": 1.0}, "1": {"1": 1.0}},
            "query_ids": ["query-left", "query-right"],
            "gallery_ids": ["gallery-ok", "gallery-zero"],
            "dataset_identity_key": "d",
            "protocol_fingerprint": "d",
        },
    )

    with pytest.raises(ValueError, match="Gallery embeddings.*gallery-zero"):
        score_retrieval_artifact(
            RetrievalScoringJob(
                "q",
                "g",
                "r",
                "out",
                RetrievalConfig(ks=(1,), primary_metric="ndcg@1"),
            ),
            store,
        )


def test_retrieval_artifact_rejects_misaligned_ids(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_artifact(
        "q",
        np.eye(2),
        {"n_samples": 2, "side": "query", "dataset_identity_key": "d", "recipe_hash": "e"},
    )
    store.put_artifact(
        "g",
        np.eye(2),
        {"n_samples": 2, "side": "gallery", "dataset_identity_key": "d", "recipe_hash": "e"},
    )
    store.put_json(
        "r",
        {
            "relevance": {"0": {"0": 1.0}, "1": {"1": 1.0}},
            "query_ids": ["one"],
            "dataset_identity_key": "d",
            "protocol_fingerprint": "d",
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
        store.put_artifact(
            key,
            values,
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
    store.put_artifact(
        "q",
        np.eye(2),
        {"n_samples": 2, "side": "query", "dataset_identity_key": "one", "recipe_hash": "e"},
    )
    store.put_artifact(
        "g",
        np.eye(2),
        {"n_samples": 2, "side": "gallery", "dataset_identity_key": "two", "recipe_hash": "e"},
    )
    store.put_json(
        "r",
        {
            "relevance": {"0": {"0": 1}, "1": {"1": 1}},
            "dataset_identity_key": "one",
            "protocol_fingerprint": "one",
        },
    )
    with pytest.raises(ValueError, match="dataset identities"):
        score_retrieval_artifact(RetrievalScoringJob("q", "g", "r", "out"), store)
