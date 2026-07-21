import numpy as np
import pytest
from scipy import sparse

from vertebrae.config import (
    ContinuousOverlapScoringConfig,
    OverlapScoringConfig,
    StabilityConfig,
)
from vertebrae.scoring.stability import run_stability_analysis


def test_prototype_stability_repeat_count(fake_overlapindex):
    Z = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    y = np.array(["a", "a", "b", "b"])

    result = run_stability_analysis(
        Z,
        y,
        OverlapScoringConfig(k=1),
        StabilityConfig(repeats=5),
    )

    assert result is not None
    assert result["repeats"] == 5
    assert len(result["scores"]) == 5
    assert len(fake_overlapindex.calls) == 5


def test_regression_stability_uses_primary_continuous_score(fake_overlapindex):
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

    result = run_stability_analysis(
        Z,
        y,
        OverlapScoringConfig(k=1),
        StabilityConfig(repeats=3),
        target_type="regression",
        target_names=["score"],
    )

    assert result is not None
    assert result["repeats"] == 3
    assert len(result["scores"]) == 3
    assert all(score >= 0.62 for score in result["scores"])
    assert len(fake_overlapindex.continuous_calls) == 3


@pytest.mark.parametrize("sparse_type", [sparse.csr_matrix, sparse.csc_matrix])
@pytest.mark.parametrize("mode", ["prototype", "subsample"])
def test_stability_preserves_sparse_embeddings_until_scoring(
    fake_overlapindex, monkeypatch, sparse_type, mode
):
    from vertebrae.scoring import stability as stability_module

    embeddings = sparse_type(
        np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.2, 0.8],
            ]
        )
    )
    labels = np.asarray(["a", "a", "a", "b", "b", "b"])
    observed = []
    original_score = stability_module.OverlapIndexScorer.score

    def record_sparse(self, values, target, **kwargs):
        observed.append(sparse.issparse(values))
        return original_score(self, values, target, **kwargs)

    monkeypatch.setattr(stability_module.OverlapIndexScorer, "score", record_sparse)
    result = run_stability_analysis(
        embeddings,
        labels,
        OverlapScoringConfig(k=1, max_dense_bytes=10_000),
        StabilityConfig(
            repeats=2,
            mode=mode,
            subsample_fraction=1.0,
            random_state=7,
        ),
    )

    assert result is not None
    assert observed == [True, True]


def test_sparse_stability_does_not_apply_dense_memory_guard(fake_overlapindex):
    embeddings = sparse.csr_matrix(np.eye(4))
    labels = np.asarray(["a", "a", "b", "b"])

    result = run_stability_analysis(
        embeddings,
        labels,
        OverlapScoringConfig(k=1, max_dense_bytes=1),
        StabilityConfig(repeats=1),
    )

    assert len(result["scores"]) == 1
    assert fake_overlapindex.calls[-1]["fit_X_sparse"] is True


def test_single_label_subsample_stability_is_target_aware(fake_overlapindex):
    embeddings = np.eye(8)
    labels = np.asarray(["a"] * 4 + ["b"] * 4)

    result = run_stability_analysis(
        embeddings,
        labels,
        OverlapScoringConfig(k=1),
        StabilityConfig(
            mode="subsample",
            repeats=3,
            subsample_fraction=0.5,
            random_state=17,
        ),
    )

    assert result is not None
    assert result["effective_sample_counts"] == [4, 4, 4]
    assert result["effective_subsample_fractions"] == [0.5, 0.5, 0.5]
    assert len(result["sampling_seeds"]) == 3
    for call in fake_overlapindex.calls:
        values, counts = np.unique(call["fit_y"], return_counts=True)
        assert dict(zip(values.tolist(), counts.tolist())) == {"a": 2, "b": 2}


def test_multilabel_subsample_stability_preserves_active_labels(fake_overlapindex):
    embeddings = np.eye(8)
    labels = [
        ("a", "b"),
        ("a",),
        ("a", "c"),
        ("a", "b"),
        ("b", "c"),
        ("b",),
        ("c",),
        ("c",),
    ]

    result = run_stability_analysis(
        embeddings,
        labels,
        OverlapScoringConfig(k=1),
        StabilityConfig(
            mode="subsample",
            repeats=3,
            subsample_fraction=0.5,
            random_state=19,
        ),
        label_names=["a", "b", "c"],
        target_type="multi_label",
    )

    assert result is not None
    assert len(result["sampling_seeds"]) == 3
    for call in fake_overlapindex.calls:
        assert np.all(np.sum(call["fit_y"], axis=0) >= 2)


def test_categorical_subsample_stability_fails_before_scoring(fake_overlapindex):
    embeddings = np.eye(4)
    labels = np.asarray(["a", "a", "b", "b"])

    with pytest.raises(
        ValueError,
        match=r"Increase subsample_fraction to at least 1.*mode='prototype'",
    ):
        run_stability_analysis(
            embeddings,
            labels,
            OverlapScoringConfig(k=1),
            StabilityConfig(mode="subsample", repeats=3, subsample_fraction=0.5),
        )

    assert fake_overlapindex.calls == []


def test_regression_subsample_stability_fails_before_scoring(fake_overlapindex):
    embeddings = np.eye(4)
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])

    with pytest.raises(
        ValueError,
        match=r"regression stability requires at least 3.*mode='prototype'",
    ):
        run_stability_analysis(
            embeddings,
            targets,
            ContinuousOverlapScoringConfig(k=1),
            StabilityConfig(mode="subsample", repeats=3, subsample_fraction=0.5),
            target_type="regression",
        )

    assert fake_overlapindex.continuous_calls == []


def test_regression_subsample_stability_preserves_nonconstant_target(fake_overlapindex):
    embeddings = np.eye(10)
    targets = np.column_stack(
        [
            np.ones(10),
            np.asarray([0.0] * 8 + [1.0] * 2),
        ]
    )

    result = run_stability_analysis(
        embeddings,
        targets,
        ContinuousOverlapScoringConfig(k=1),
        StabilityConfig(
            mode="subsample",
            repeats=5,
            subsample_fraction=0.5,
            random_state=23,
        ),
        target_type="regression",
        target_names=["constant", "signal"],
    )

    assert result is not None
    assert result["effective_sample_counts"] == [5] * 5
    for call in fake_overlapindex.continuous_calls:
        assert np.any(np.var(call["fit_y"], axis=0) > 0.0)


def test_subsample_stability_is_deterministic(fake_overlapindex):
    embeddings = np.eye(12)
    labels = np.asarray(["a"] * 6 + ["b"] * 6)
    config = StabilityConfig(
        mode="subsample",
        repeats=3,
        subsample_fraction=0.5,
        random_state=29,
    )

    first = run_stability_analysis(embeddings, labels, OverlapScoringConfig(k=1), config)
    first_targets = [call["fit_y"].tolist() for call in fake_overlapindex.calls]
    fake_overlapindex.calls.clear()
    second = run_stability_analysis(embeddings, labels, OverlapScoringConfig(k=1), config)
    second_targets = [call["fit_y"].tolist() for call in fake_overlapindex.calls]
    fake_overlapindex.calls.clear()
    third = run_stability_analysis(
        embeddings,
        labels,
        OverlapScoringConfig(k=1),
        StabilityConfig(
            mode="subsample",
            repeats=3,
            subsample_fraction=0.5,
            random_state=30,
        ),
    )

    assert first == second
    assert first_targets == second_targets
    assert first["sampling_seeds"] != third["sampling_seeds"]


def test_dense_and_sparse_subsample_stability_select_same_rows(fake_overlapindex, monkeypatch):
    from vertebrae.scoring import stability as stability_module

    dense = np.column_stack([np.arange(12, dtype=float), np.ones(12)])
    labels = np.asarray(["a"] * 6 + ["b"] * 6)
    selected_rows = []
    original_score = stability_module.OverlapIndexScorer.score

    def record_rows(self, values, target, **kwargs):
        matrix = values.toarray() if sparse.issparse(values) else np.asarray(values)
        selected_rows.append(matrix[:, 0].tolist())
        return original_score(self, values, target, **kwargs)

    monkeypatch.setattr(stability_module.OverlapIndexScorer, "score", record_rows)
    config = StabilityConfig(
        mode="subsample",
        repeats=2,
        subsample_fraction=0.5,
        random_state=31,
    )
    for values in (dense, sparse.csr_matrix(dense), sparse.csc_matrix(dense)):
        run_stability_analysis(values, labels, OverlapScoringConfig(k=1), config)

    assert selected_rows[0:2] == selected_rows[2:4] == selected_rows[4:6]


def test_prototype_stability_does_not_apply_subsample_feasibility(fake_overlapindex):
    result = run_stability_analysis(
        np.eye(4),
        np.asarray(["a", "a", "b", "b"]),
        OverlapScoringConfig(k=1),
        StabilityConfig(mode="prototype", repeats=1, subsample_fraction=0.1),
    )

    assert result is not None
    assert "sampling_seeds" not in result
