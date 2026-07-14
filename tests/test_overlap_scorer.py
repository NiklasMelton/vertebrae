import numpy as np
import pytest

from vertebrae.config import ContinuousOverlapScoringConfig, OverlapScoringConfig
from vertebrae.reports.markdown_report import render_markdown_report
from vertebrae.results import BenchmarkResult, ExtractorResult
from vertebrae.scoring.metrics import OverlapMetric
from vertebrae.scoring.overlap import OverlapIndexScorer


def test_overlap_scorer_uses_minibatch_kmeans_backend(fake_overlapindex):
    Z = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    y = np.array(["a", "a", "b", "b"])
    scorer = OverlapIndexScorer(OverlapScoringConfig(k=2, min_samples_per_cluster=1))

    result = scorer.score(Z, y, seed=123)

    assert result.macro_score > 0
    assert fake_overlapindex.calls[-1]["model_type"] == "MiniBatchKMeans"
    assert fake_overlapindex.calls[-1]["kmeans_kwargs"]["random_state"] == 123
    assert fake_overlapindex.calls[-1]["kmeans_k"] == {"a": 2, "b": 2}


def test_overlap_scorer_passes_multilabel_indicator_targets(fake_overlapindex):
    Z = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.4, 0.6],
        ]
    )
    y = [
        ("red", "round"),
        ("red",),
        ("round",),
        ("red", "sweet"),
        ("round", "sweet"),
        ("sweet",),
    ]
    scorer = OverlapIndexScorer(OverlapScoringConfig(k=2, min_samples_per_cluster=1))

    result = scorer.score(Z, y)

    assert result.metadata["target_type"] == "multi_label"
    assert result.metadata["label_names"] == ("red", "round", "sweet")
    assert result.class_counts == {"red": 3, "round": 3, "sweet": 3}
    assert set(result.per_class_scores) == {"red", "round", "sweet"}
    assert fake_overlapindex.calls[-1]["kmeans_k"] == {0: 2, 1: 2, 2: 2}
    assert fake_overlapindex.calls[-1]["fit_y_shape"] == [6, 3]
    assert fake_overlapindex.calls[-1]["fit_y"].tolist() == [
        [1, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
    ]


def test_overlap_scorer_passes_reporting_exclusions(fake_overlapindex):
    Z = np.eye(6)
    y = np.array(["background", "background", "cat", "cat", "dog", "dog"])

    result = OverlapIndexScorer(
        OverlapScoringConfig(
            k=1,
            min_samples_per_cluster=1,
            exclude_classes="background",
        )
    ).score(Z, y)

    assert fake_overlapindex.calls[-1]["exclude_classes"] == "background"
    assert result.metadata["exclude_classes"] == ["background"]
    assert result.metadata["aggregation_classes"] == ["cat", "dog"]
    assert result.metadata["aggregate_valid"] is True


def test_overlap_scorer_supports_explicit_regression_targets(fake_overlapindex):
    Z = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.0, 1.0],
        ]
    )
    y = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

    result = OverlapIndexScorer(ContinuousOverlapScoringConfig(k=3, n_null_permutations=5)).score(
        Z,
        y,
        seed=11,
        target_type="regression",
        target_names=["score"],
    )

    assert result.metadata["target_type"] == "regression"
    assert result.metadata["target_names"] == ("score",)
    assert result.score == 0.631
    assert result.macro_score == 0.611
    assert result.weighted_score == 0.631
    assert result.actual_loss == 0.12
    assert result.null_loss == 0.24
    assert result.loss_ratio == 0.50
    assert fake_overlapindex.continuous_calls[-1]["model_type"] == "MiniBatchKMeans"
    assert fake_overlapindex.continuous_calls[-1]["kmeans_k"] == 3
    assert fake_overlapindex.continuous_calls[-1]["random_state"] == 11


@pytest.mark.parametrize(
    ("macro_value", "weighted_value", "expected_macro", "expected_weighted"),
    [
        (0.0, 0.0, 0.0, 0.0),
        (None, None, 0.75, 0.75),
        (0.0, None, 0.0, 0.75),
        (None, 0.0, 0.75, 0.0),
    ],
)
def test_continuous_overlap_uses_only_none_for_aggregate_fallback(
    monkeypatch,
    macro_value,
    weighted_value,
    expected_macro,
    expected_weighted,
):
    from vertebrae.scoring import overlap as overlap_module

    class AggregateContinuousOverlapIndex:
        def __init__(self, **kwargs):
            self.index = 0.75
            self.macro_index_ = macro_value
            self._weighted_value = weighted_value

        @property
        def weighted_index(self):
            return self._weighted_value

        def fit_offline(self, embeddings, targets, reset_state=True):
            return self.index

    monkeypatch.setattr(
        overlap_module,
        "_load_continuous_overlap_index",
        lambda: AggregateContinuousOverlapIndex,
    )
    result = OverlapIndexScorer(ContinuousOverlapScoringConfig(k=1)).score(
        np.eye(4),
        np.asarray([0.0, 0.25, 0.75, 1.0]),
        target_type="regression",
    )

    assert result.macro_score == expected_macro
    assert result.weighted_score == expected_weighted


def test_zero_continuous_aggregates_survive_metric_serialization_and_markdown(monkeypatch):
    from vertebrae.scoring import overlap as overlap_module

    class ZeroAggregateContinuousOverlapIndex:
        def __init__(self, **kwargs):
            self.index = 0.75
            self.macro_index_ = 0.0

        @property
        def weighted_index(self):
            return 0.0

        def fit_offline(self, embeddings, targets, reset_state=True):
            return self.index

    monkeypatch.setattr(
        overlap_module,
        "_load_continuous_overlap_index",
        lambda: ZeroAggregateContinuousOverlapIndex,
    )
    metric = OverlapMetric(ContinuousOverlapScoringConfig(k=1)).score(
        np.eye(4),
        np.asarray([0.0, 0.25, 0.75, 1.0]),
        target_metadata={"target_type": "regression", "target_names": ["score"]},
    )
    extractor_result = ExtractorResult(
        name="zero-aggregate",
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
    benchmark_result = BenchmarkResult(
        dataset_summary={
            "n_samples": 4,
            "n_targets": 1,
            "target_type": "regression",
            "modality": "embedding",
        },
        extractor_results=[extractor_result],
        recommendations=[],
    )

    serialized = benchmark_result.to_dict()
    serialized_overlap = serialized["extractor_results"][0]["metrics"]["overlap"]
    assert serialized_overlap["diagnostics"]["macro_score"] == 0.0
    assert serialized_overlap["diagnostics"]["weighted_score"] == 0.0
    assert "| 0.7500 | 0.0000 | 0.0000 |" in render_markdown_report(benchmark_result)
