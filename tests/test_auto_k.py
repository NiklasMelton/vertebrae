from dataclasses import fields

import numpy as np

from vertebrae.config import OverlapScoringConfig
from vertebrae.scoring.overlap import auto_k_for_class, resolve_kmeans_k


def test_auto_k_respects_class_size_ceiling():
    assert auto_k_for_class(100, min_k=10, max_k=50, min_samples_per_cluster=5) == 10
    assert auto_k_for_class(20, min_k=10, max_k=50, min_samples_per_cluster=5) == 4


def test_resolve_kmeans_k_reduces_requested_k_and_warns():
    y = np.array(["a"] * 8 + ["b"] * 30)
    config = OverlapScoringConfig(k=10, min_samples_per_cluster=5)

    k_per_class, warnings = resolve_kmeans_k(y, config, return_warnings=True)

    assert k_per_class["a"] == 1
    assert k_per_class["b"] == 6
    assert warnings


def test_public_scoring_config_does_not_expose_backend_selection():
    field_names = {field.name for field in fields(OverlapScoringConfig)}

    assert "model_type" not in field_names
    assert "rho" not in field_names
    assert "match_tracking" not in field_names
