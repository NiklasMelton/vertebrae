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
