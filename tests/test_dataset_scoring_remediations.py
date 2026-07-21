import numpy as np
import pytest

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CacheConfig,
    DatasetIdentity,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    LabelViewConfig,
    MemoryConfig,
    OverlapMetric,
    OverlapScoringConfig,
    RetrievalConfig,
    RetrievalDataset,
    StabilityConfig,
    TargetView,
    TargetViewConfig,
    UnitAnnotation,
)
from vertebrae.benchmark import _weakest_class
from vertebrae.config import (
    ContinuousOverlapScoringConfig,
    ExecutionConfig,
    SegmentationConfig,
    ZeroShotConfig,
)
from vertebrae.reports.markdown_report import render_markdown_report
from vertebrae.reports.recommendations import (
    recommendation_for_extractor,
    recommendations_for_benchmark,
)
from vertebrae.results import BenchmarkResult, ExtractorResult
from vertebrae.scoring.separatix import summarize_probe_diagnostics
from vertebrae.utils.labels import normalize_level_names, resolve_hierarchy_level
from vertebrae.utils.semantic_labels import semantic_label_catalog, semantic_label_key


def _identity():
    return DatasetIdentity.ephemeral()


def _balanced_labels():
    return ["a", "a", "b", "b"]


def test_embedding_factory_rejects_invalid_matrix_contracts():
    for embeddings, message in (
        ([1.0, 2.0, 3.0, 4.0], "2D"),
        (np.empty((4, 0)), "feature column"),
        (np.asarray([[0.0], [1.0], [np.nan], [2.0]]), "finite"),
        (np.asarray([["x"], ["y"], ["z"], ["w"]]), "numeric"),
    ):
        with pytest.raises(ValueError, match=message):
            BenchmarkDataset.from_embeddings(
                embeddings,
                _balanced_labels(),
                identity=_identity(),
            )


def test_structural_metadata_cannot_override_constructor_invariants():
    with pytest.raises(ValueError, match="structural keys"):
        BenchmarkDataset.from_node_embeddings(
            np.eye(4),
            _balanced_labels(),
            identity=_identity(),
            node_ids=["n0", "n1", "n2", "n3"],
            metadata={"node_ids": ["wrong"] * 4},
        )
    with pytest.raises(ValueError, match="structural keys"):
        BenchmarkDataset.from_embeddings(
            np.eye(4),
            _balanced_labels(),
            identity=_identity(),
            metadata={"precomputed_embeddings": False},
        )


def test_unit_row_metadata_registry_survives_reordered_nested_subsets():
    dataset = BenchmarkDataset.from_embedding_units(
        np.arange(16, dtype=float).reshape(8, 2),
        ["a", "a", "a", "a", "b", "b", "b", "b"],
        identity=_identity(),
        unit_ids=[f"u{index}" for index in range(8)],
        parent_ids=["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"],
        positions=list(range(8)),
        provenance=[{"row": index} for index in range(8)],
    )

    subset = dataset.subset([7, 1, 5, 3, 4, 2]).subset([2, 1, 3, 4])

    assert subset.metadata["sample_indices"] == [5, 1, 3, 4]
    assert subset.metadata["unit_ids"] == ["u5", "u1", "u3", "u4"]
    assert subset.metadata["parent_ids"] == ["p2", "p0", "p1", "p2"]
    assert subset.metadata["unit_positions"] == [5, 1, 3, 4]
    assert subset.metadata["unit_provenance"] == [
        {"row": 5},
        {"row": 1},
        {"row": 3},
        {"row": 4},
    ]
    assert len(subset.groups()) == len(subset.y) == 4


def test_all_registered_row_metadata_survives_direct_reordered_nested_subsets():
    embeddings = np.arange(16, dtype=float).reshape(8, 2)
    labels = ["a"] * 4 + ["b"] * 4
    ids = list(range(8))
    pair_rows = [[index, index + 10] for index in ids]
    triplet_rows = [[index, index + 10, index + 20] for index in ids]
    datasets = [
        BenchmarkDataset(
            X=embeddings,
            y=np.asarray(labels, dtype=object),
            modality="tabular",
            identity=_identity(),
            metadata={
                "custom_rows": [{"row": index} for index in ids],
                "_row_aligned_metadata_keys": ["custom_rows"],
            },
        ),
        BenchmarkDataset.from_node_embeddings(
            embeddings,
            labels,
            identity=_identity(),
            node_ids=[f"node-{index}" for index in ids],
        ),
        BenchmarkDataset.from_entity_embeddings(
            embeddings,
            labels,
            identity=_identity(),
            entity_ids=[f"entity-{index}" for index in ids],
        ),
        BenchmarkDataset.from_edge_embeddings(
            edge_embeddings=embeddings,
            edge_index=pair_rows,
            labels=labels,
            identity=_identity(),
        ),
        BenchmarkDataset.from_pair_embeddings(
            pair_embeddings=embeddings,
            pairs=pair_rows,
            labels=labels,
            identity=_identity(),
        ),
        BenchmarkDataset.from_triplet_embeddings(
            triplet_embeddings=embeddings,
            triplets=triplet_rows,
            labels=labels,
            identity=_identity(),
        ),
        BenchmarkDataset.from_embedding_units(
            embeddings,
            labels,
            identity=_identity(),
            unit_ids=[f"unit-{index}" for index in ids],
            parent_ids=[f"parent-{index}" for index in ids],
            positions=ids,
            spans=[[index, index + 1] for index in ids],
            coordinates=[[index, 0, index + 1, 1] for index in ids],
            provenance=[{"row": index} for index in ids],
        ),
        BenchmarkDataset.from_arrays(
            embeddings,
            labels,
            modality="tabular",
            identity=_identity(),
        ).with_groups([f"group-{index}" for index in ids]),
        BenchmarkDataset.from_arrays(
            embeddings,
            labels,
            modality="text",
            identity=_identity(),
        ).with_unit_annotations(
            [
                UnitAnnotation(
                    labels=["x", "y"],
                    metadata={"parent_row": index},
                )
                for index in ids
            ],
            unit_type="token",
        ),
    ]
    observed_keys = set()
    expected_original_rows = [5, 1, 3, 4]

    for dataset in datasets:
        registered = list(dataset.metadata["_row_aligned_metadata_keys"])
        observed_keys.update(registered)
        expected = {
            key: np.asarray(dataset.metadata[key], dtype=object)[expected_original_rows].tolist()
            for key in registered
        }

        subset = dataset.subset([7, 1, 5, 3, 4, 2]).subset([2, 1, 3, 4])

        for key, values in expected.items():
            assert subset.metadata[key] == values
        assert subset.metadata["sample_indices"] == expected_original_rows
        assert subset.metadata["parent_row_positions"] == [2, 1, 3, 4]

    assert observed_keys == {
        "custom_rows",
        "node_ids",
        "entity_ids",
        "edge_index",
        "pair_ids",
        "triplet_ids",
        "unit_ids",
        "parent_ids",
        "unit_positions",
        "unit_spans",
        "unit_coordinates",
        "unit_provenance",
        "groups",
        "unit_annotations",
    }


def test_group_accessor_preserves_exact_mixed_semantic_types():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(16).reshape(8, 2),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=_identity(),
    ).with_groups([1, 1, "1", "1", True, True, b"1", b"1"])

    groups = dataset.groups()

    assert groups.dtype == object
    assert [type(value) for value in groups[::2].tolist()] == [int, str, bool, bytes]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CacheConfig(storage_options={"client": {"timeout": np.inf}}),
        lambda: CacheConfig(metadata={"array": np.asarray([1.0, np.nan])}),
        lambda: CacheConfig(storage_options={1: "not-a-string-key"}),
        lambda: CacheConfig(metadata={"opaque": object()}),
        lambda: EmbeddingCompressionConfig(
            enabled=True,
            method="pca",
            n_components=2,
            algorithm_kwargs={"nested": {"tolerance": np.nan}},
        ),
        lambda: EmbeddingCompressionConfig(
            enabled=True,
            method="pca",
            n_components=2,
            algorithm_kwargs={"opaque": object()},
        ),
    ],
)
def test_nested_cache_and_compression_options_require_finite_typed_values(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize("indices", [[0.5, 1], [True, 1], [-1, 1], [0, 0]])
def test_subset_indices_are_not_coerced(indices):
    dataset = BenchmarkDataset.from_arrays(
        np.eye(4),
        _balanced_labels(),
        modality="tabular",
        identity=_identity(),
    )
    with pytest.raises((TypeError, ValueError, IndexError)):
        dataset.subset(indices)


def test_relational_and_retrieval_ids_use_exact_typed_identity():
    relational = BenchmarkDataset.from_pair_embeddings(
        entity_embeddings=np.eye(4),
        entity_ids=[1, True, "x", "y"],
        pairs=[(1, "x"), (True, "y"), (1, "y"), (True, "x")],
        labels=_balanced_labels(),
        identity=_identity(),
    )
    assert not np.array_equal(relational.X[0], relational.X[1])

    retrieval = RetrievalDataset.from_arrays(
        ["q-int", "q-bool"],
        ["g-int", "g-bool"],
        [(1, 1, 1.0), (True, True, 1.0)],
        query_ids=[1, True],
        gallery_ids=[1, True],
        query_modality="text",
        gallery_modality="text",
        identity=_identity(),
    )
    assert retrieval.relevance == {0: {0: 1.0}, 1: {1: 1.0}}


def test_hierarchy_prefixes_have_delimiter_free_identity():
    dataset = BenchmarkDataset.from_arrays(
        np.eye(4),
        ["left", "left", "right", "right"],
        modality="tabular",
        identity=_identity(),
    ).with_label_hierarchy(
        [
            ("a > b", "c"),
            ("a > b", "c"),
            ("a", "b > c"),
            ("a", "b > c"),
        ]
    )

    view = dataset.label_view(1)
    assert len(set(view.y.tolist())) == 2
    assert len(view.class_counts()) == 2
    assert all("path=" in label for label in view.class_counts())


def test_mixed_labels_remain_distinct_through_overlap_and_reports(fake_overlapindex):
    labels = np.asarray([1, 1, "1", "1"], dtype=object)
    dataset = BenchmarkDataset.from_arrays(
        np.eye(4),
        labels,
        modality="tabular",
        identity=_identity(),
    )
    metric = OverlapMetric(OverlapScoringConfig(k=1)).score(
        np.eye(4),
        dataset.y,
        target_metadata=dataset.metadata,
    )
    fit_labels = fake_overlapindex.calls[-1]["fit_y"].tolist()
    assert set(fit_labels) == {semantic_label_key(1), semantic_label_key("1")}
    assert len(metric.metadata["label_catalog"]) == 2
    assert set(metric.per_class_scores) == {
        semantic_label_key(1),
        semantic_label_key("1"),
    }
    assert set(metric.metadata["label_display_by_key"].values()) == {
        "1 [int]",
        "1 [str]",
    }

    item = ExtractorResult(
        name="mixed",
        extractor_type="precomputed",
        stability=None,
        separatix=None,
        embedding_metadata={"embedding_dim": 4},
        compression_metadata={"method": "none"},
        runtime={},
        warnings=[],
        recommendation="",
        metrics={"overlap": metric},
    )
    report = render_markdown_report(
        BenchmarkResult(
            dataset_summary=dataset.summary(),
            extractor_results=[item],
            recommendations=[],
        )
    )
    assert "1 [int]" in report
    assert "1 [str]" in report


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OverlapScoringConfig(k=True),
        lambda: OverlapScoringConfig(offline_chunk_size=1.5),
        lambda: RetrievalConfig(ks=(True,), primary_metric="ndcg@True"),
        lambda: StabilityConfig(enabled=1),
        lambda: EmbeddingConfig(batch_size=True),
        lambda: MemoryConfig(max_memory_bytes=True),
        lambda: EmbeddingCompressionConfig(n_components=True),
    ],
)
def test_config_integer_and_boolean_fields_are_strict(factory):
    with pytest.raises(TypeError):
        factory()


def test_config_integral_values_are_normalized_and_range_checked():
    retrieval = RetrievalConfig(ks=(np.int64(1),), primary_metric="ndcg@1")
    zero_shot = ZeroShotConfig(top_k=(np.int64(1),))
    execution = ExecutionConfig(total_shards=np.int64(2))

    assert retrieval.ks == (1,)
    assert zero_shot.top_k == (1,)
    assert execution.total_shards == 2
    with pytest.raises(ValueError, match="n_target_cells"):
        ContinuousOverlapScoringConfig(n_target_cells=np.int64(-1))


@pytest.mark.parametrize(
    "factory,error",
    [
        (lambda: OverlapScoringConfig(kmeans_kwargs=[]), TypeError),
        (lambda: OverlapScoringConfig(kmeans_kwargs={"tol": np.nan}), ValueError),
        (
            lambda: ContinuousOverlapScoringConfig(target_cover_kwargs={"bad": object()}),
            ValueError,
        ),
        (lambda: LabelViewConfig(output_levels=[]), TypeError),
        (lambda: TargetViewConfig(output_views=[]), TypeError),
        (lambda: SegmentationConfig(background_label=[]), TypeError),
        (lambda: SegmentationConfig(ignore_instance_ids=(np.nan,)), ValueError),
    ],
)
def test_mapping_and_semantic_config_values_are_strict(factory, error):
    with pytest.raises(error):
        factory()


def test_target_and_hierarchy_names_are_trimmed_and_collisions_rejected():
    dataset = BenchmarkDataset.from_arrays(
        np.eye(4),
        _balanced_labels(),
        modality="tabular",
        identity=_identity(),
    ).with_target_views([TargetView(name=" coarse ", targets=["x", "x", "y", "y"])])
    assert dataset.target_view_names() == ["coarse"]
    config = TargetViewConfig(
        views=(" coarse ",),
        output_views={" output ": " coarse "},
    )
    assert config.views == ("coarse",)
    assert config.output_views == {"output": "coarse"}
    with pytest.raises(ValueError, match="Duplicate target view"):
        dataset.with_target_views(
            [
                TargetView(name="role", targets=["x", "x", "y", "y"]),
                TargetView(name=" role ", targets=["x", "x", "y", "y"]),
            ]
        )
    assert normalize_level_names([" family ", "leaf"], 2) == ("family", "leaf")
    with pytest.raises(ValueError, match="unique"):
        normalize_level_names(["family", " family "], 2)
    with pytest.raises(TypeError, match="non-boolean"):
        resolve_hierarchy_level(True, max_depth=2)


def test_overlap_metric_can_bind_a_benchmark_config_without_mutation():
    metric = OverlapMetric()
    config = OverlapScoringConfig(k=1)
    bound = metric.with_config(config)

    assert metric.config is None
    assert bound.config is config
    explicit = OverlapMetric(OverlapScoringConfig(k=2))
    assert explicit.with_config(config) is explicit


def test_benchmark_binds_reused_configless_overlap_metric_without_mutation():
    metric = OverlapMetric()
    classification = BenchmarkDataset.from_embeddings(
        np.eye(4),
        ["a", "a", "b", "b"],
        identity=_identity(),
    )
    regression = BenchmarkDataset.from_embeddings(
        np.eye(4),
        [0.0, 0.25, 0.75, 1.0],
        target_type="regression",
        identity=_identity(),
    )

    classification_benchmark = Benchmark(classification, metrics=[metric])
    regression_benchmark = Benchmark(regression, metrics=[metric])

    assert metric.config is None
    assert isinstance(classification_benchmark.overlap_metric.config, OverlapScoringConfig)
    assert isinstance(
        regression_benchmark.overlap_metric.config,
        ContinuousOverlapScoringConfig,
    )


def test_semantic_exclusions_distinguish_typed_labels_and_multilabel_columns():
    labels = [1, 1, True, True, "1", "1"]
    dataset = BenchmarkDataset.from_embeddings(
        np.eye(6),
        labels,
        identity=_identity(),
    )
    benchmark = Benchmark(
        dataset,
        scoring_config=OverlapScoringConfig(k=1, exclude_classes=[1]),
    )

    _, filtered, _, _ = benchmark._diagnostic_inputs(
        dataset.X,
        dataset.y,
        None,
        target_type="single_label",
        scoring_config=benchmark.scoring_config,
        label_names=dataset.metadata.get("label_names"),
    )
    assert filtered.tolist() == [True, True, "1", "1"]

    catalog = semantic_label_catalog(labels)
    weakest, score = _weakest_class(
        {
            semantic_label_key(1): 0.01,
            semantic_label_key(True): 0.20,
            semantic_label_key("1"): 0.30,
        },
        excluded_classes=[1],
        label_catalog=catalog,
    )
    assert weakest == "True"
    assert score == pytest.approx(0.20)

    multilabel = BenchmarkDataset.from_embeddings(
        np.eye(6),
        [
            ("red", "round"),
            ("red",),
            ("round",),
            ("red", "sweet"),
            ("round", "sweet"),
            ("sweet",),
        ],
        identity=_identity(),
    )
    multilabel_benchmark = Benchmark(
        multilabel,
        scoring_config=OverlapScoringConfig(k=1, exclude_classes=["round"]),
    )
    _, diagnostic_labels, _, diagnostic_names = multilabel_benchmark._diagnostic_inputs(
        multilabel.X,
        multilabel.y,
        None,
        target_type="multi_label",
        scoring_config=multilabel_benchmark.scoring_config,
        label_names=multilabel.metadata["label_names"],
    )
    assert diagnostic_labels.shape == (5, 2)
    assert diagnostic_names == ("red", "sweet")


def test_invalid_aggregates_are_not_ranked_and_markdown_escapes_dynamic_values():
    from vertebrae.scoring.metrics import MetricResult

    valid = ExtractorResult(
        name="safe|name\n## injected",
        extractor_type="custom|type",
        stability=None,
        separatix=None,
        embedding_metadata={"embedding_dim": 2},
        compression_metadata={"method": "none"},
        runtime={},
        warnings=["warning\n- injected"],
        recommendation="use|care",
        metrics={"custom": MetricResult(name="custom", score=0.8)},
        primary_metric_name="custom",
    )
    invalid = ExtractorResult(
        name="invalid",
        extractor_type="custom",
        stability=None,
        separatix=None,
        embedding_metadata={"embedding_dim": 2},
        compression_metadata={"method": "none"},
        runtime={},
        warnings=[],
        recommendation="",
        metrics={
            "custom": MetricResult(
                name="custom",
                score=1.0,
                metadata={"aggregate_valid": False},
            )
        },
        primary_metric_name="custom",
    )
    result = BenchmarkResult(
        dataset_summary={
            "n_samples": 4,
            "n_classes": 2,
            "target_type": "single_label",
            "modality": "tabular",
        },
        extractor_results=[invalid, valid],
        recommendations=[],
    )

    assert result.ranked_results() == [valid]
    report = render_markdown_report(result)
    assert "safe\\|name<br>## injected" in report
    assert "warning<br>- injected" in report
    assert "| invalid |" not in report

    no_valid = BenchmarkResult(
        dataset_summary=result.dataset_summary,
        extractor_results=[invalid],
        recommendations=recommendations_for_benchmark([invalid]),
    )
    unavailable_report = render_markdown_report(no_valid)
    assert "Ranking unavailable because no valid aggregate remains" in unavailable_report
    assert no_valid.recommendations == [
        "Ranking unavailable because no valid aggregate remains under this protocol."
    ]


@pytest.mark.parametrize(
    ("score", "stability", "expected"),
    [
        (0.0, None, "continuous_overlap_null_like"),
        (0.4, None, "continuous_structure_above_null"),
        (
            0.4,
            {"summary": {"lower": 0.0, "upper": 0.6}},
            "continuous_overlap_null_indeterminate",
        ),
        (
            0.4,
            {"summary": {"lower": 0.1, "upper": 0.6}},
            "continuous_structure_above_null",
        ),
    ],
)
def test_regression_recommendations_use_zero_as_the_continuous_null_endpoint(
    score,
    stability,
    expected,
):
    assert (
        recommendation_for_extractor(
            score,
            stability,
            weakest_class_score=None,
            target_type="regression",
        )
        == expected
    )


def test_lower_is_better_probe_metrics_select_minimum_and_positive_improvement():
    report = {
        "metrics": {
            "baseline": {"best_probe": "smooth_poly", "primary_metric": "mae"},
            "probes": {
                "linear": {"mae": 0.5, "evaluation_mode": "holdout"},
                "smooth_poly": {"mae": 0.4, "evaluation_mode": "holdout"},
                "kernel_approx": {"mae": 0.3, "evaluation_mode": "holdout"},
            },
        }
    }

    comparison = summarize_probe_diagnostics(
        report,
        target_type="regression",
        grouped=False,
        n_groups=None,
    )["comparison"]
    assert comparison["nonlinear_probe"] == "kernel_approx"
    assert comparison["delta"] == pytest.approx(0.2)
    assert comparison["raw_delta"] == pytest.approx(-0.2)
    assert comparison["favored_family"] == "nonlinear"
