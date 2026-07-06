import numpy as np

from vertebrae.config import OverlapScoringConfig
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
    assert fake_overlapindex.calls[-1]["kmeans_k"] == {
        "red": 2,
        "round": 2,
        "sweet": 2,
    }
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
