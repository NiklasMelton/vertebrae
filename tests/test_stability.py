import numpy as np
import pytest
from scipy import sparse

from vertebrae.config import OverlapScoringConfig, StabilityConfig
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


def test_sparse_stability_retains_scoring_dense_memory_guard(fake_overlapindex):
    embeddings = sparse.csr_matrix(np.eye(4))
    labels = np.asarray(["a", "a", "b", "b"])

    with pytest.raises(ValueError, match="max_dense_bytes"):
        run_stability_analysis(
            embeddings,
            labels,
            OverlapScoringConfig(k=1, max_dense_bytes=1),
            StabilityConfig(repeats=1),
        )
