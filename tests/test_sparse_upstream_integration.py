import os

import numpy as np
import pytest
from scipy import sparse

from vertebrae.config import (
    ContinuousOverlapScoringConfig,
    OverlapScoringConfig,
    SeparatixConfig,
)
from vertebrae.scoring.overlap import OverlapIndexScorer
from vertebrae.scoring.separatix import SeparatixScorer

pytestmark = [
    pytest.mark.realworld,
    pytest.mark.skipif(
        os.environ.get("VERTABRAE_RUN_REALWORLD") != "1",
        reason="set VERTABRAE_RUN_REALWORLD=1 to run real dependency integration tests",
    ),
]


def test_real_overlapindex_dense_and_sparse_classification_are_equivalent():
    embeddings, labels = _classification_fixture()
    config = OverlapScoringConfig(
        k=2,
        min_samples_per_cluster=1,
        kmeans_kwargs={"random_state": 7, "n_init": 1, "batch_size": 8},
    )

    dense = OverlapIndexScorer(config).score(embeddings, labels, seed=7)
    sparse_result = OverlapIndexScorer(config).score(
        sparse.csc_array(embeddings),
        labels,
        seed=7,
    )

    assert sparse_result.metadata["scoring_input_format"] == "csr"
    assert sparse_result.score == pytest.approx(dense.score, abs=1e-10)


def test_real_overlapindex_accepts_sparse_features_and_sparse_multilabel_targets():
    embeddings, _ = _classification_fixture()
    targets = sparse.csr_array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )

    result = OverlapIndexScorer(
        OverlapScoringConfig(
            k=2,
            min_samples_per_cluster=1,
            kmeans_kwargs={"random_state": 11, "n_init": 1, "batch_size": 8},
        )
    ).score(
        sparse.csr_array(embeddings),
        targets,
        label_names=["red", "round", "sweet"],
        target_type="multi_label",
        seed=11,
    )

    assert np.isfinite(result.score)
    assert result.metadata["target_type"] == "multi_label"
    assert result.metadata["scoring_input_format"] == "csr"


def test_real_continuous_overlap_accepts_sparse_features():
    embeddings, _ = _classification_fixture()
    targets = np.linspace(0.0, 1.0, len(embeddings))
    config = ContinuousOverlapScoringConfig(
        k=2,
        kmeans_kwargs={"random_state": 13, "n_init": 1, "batch_size": 8},
        n_projections=4,
        n_null_permutations=2,
    )

    dense = OverlapIndexScorer(config).score(
        embeddings,
        targets,
        target_type="regression",
        seed=13,
    )
    sparse_result = OverlapIndexScorer(config).score(
        sparse.csc_array(embeddings),
        targets,
        target_type="regression",
        seed=13,
    )

    assert sparse_result.metadata["scoring_input_format"] == "csr"
    assert sparse_result.score == pytest.approx(dense.score, abs=0.02)


def test_real_separatix_preserves_sparse_input_and_policy_audit():
    embeddings, labels = _classification_fixture()

    result = SeparatixScorer(
        config=SeparatixConfig(
            budget="fast",
            densify_policy="skip",
            max_samples=len(labels),
            max_dense_bytes=1_048_576,
        ),
        overlap_config=OverlapScoringConfig(normalize_embeddings=True),
    ).score(sparse.csc_array(embeddings), labels)

    assert result.ran is True
    assert result.metadata["sparse_input"] is True
    assert result.metadata["densify_policy"] == "skip"
    assert result.preprocessing["is_sparse"] is True
    assert result.report["config"]["densify_policy"] == "skip"


def _classification_fixture():
    first = np.asarray(
        [
            [1.0, 0.0, 0.1, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.1, 0.0],
            [0.85, 0.15, 0.0, 0.1],
            [0.95, 0.05, 0.1, 0.0],
            [0.75, 0.25, 0.0, 0.1],
        ]
    )
    second = first[:, [1, 0, 3, 2]]
    return np.vstack([first, second]), np.asarray(["a"] * 6 + ["b"] * 6)
