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
