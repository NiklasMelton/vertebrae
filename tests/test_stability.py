import numpy as np

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
