import numpy as np
import pytest
from scipy import sparse

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.config import (
    CacheConfig,
    ContinuousOverlapScoringConfig,
    OverlapScoringConfig,
    StabilityConfig,
)
from vertebrae.extractors import PrecomputedExtractor, SklearnExtractor
from vertebrae.scoring.overlap import OverlapIndexScorer
from vertebrae.utils.labels import multilabel_indicator


def test_dataset_from_sparse_embeddings_preserves_sparse_matrix():
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])

    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    assert sparse.issparse(dataset.X)
    assert dataset.X.shape == (6, 6)


@pytest.mark.parametrize("sparse_type", [sparse.csr_matrix, sparse.csc_array])
def test_local_store_round_trips_sparse_embeddings(tmp_path, sparse_type):
    store = LocalArtifactStore(str(tmp_path))
    embeddings = sparse_type(np.eye(4, dtype=np.float32))

    path = store.put_array("embeddings/test", embeddings)
    loaded = store.get_array("embeddings/test")

    assert path.endswith(".npz")
    assert sparse.isspmatrix_csr(loaded)
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded.toarray(), embeddings.toarray())


def test_overlap_scorer_accepts_sparse_embeddings(fake_overlapindex):
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    scorer = OverlapIndexScorer(OverlapScoringConfig(k=1, max_dense_bytes=1_000))

    result = scorer.score(embeddings, labels)

    assert result.metadata["sparse_input"] is True
    assert result.metadata["scoring_input_format"] == "csr"
    assert fake_overlapindex.calls[-1]["fit_X_sparse"] is True
    assert fake_overlapindex.calls[-1]["fit_X_format"] == "csr"
    assert not any("densified" in warning for warning in result.warnings)


def test_overlap_scorer_sparse_input_ignores_dense_diagnostic_limit(fake_overlapindex):
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    scorer = OverlapIndexScorer(OverlapScoringConfig(k=1, max_dense_bytes=1))

    result = scorer.score(embeddings, labels)

    assert result.metadata["scoring_input_format"] == "csr"


def test_overlap_scorer_normalizes_sparse_arrays_to_csr(fake_overlapindex):
    embeddings = sparse.csc_array(np.eye(6, dtype=np.int16))
    labels = np.array(["a", "a", "a", "b", "b", "b"])

    result = OverlapIndexScorer(OverlapScoringConfig(k=1)).score(embeddings, labels)

    assert result.metadata["scoring_input_format"] == "csr"
    assert fake_overlapindex.calls[-1]["fit_X_sparse"] is True
    assert fake_overlapindex.calls[-1]["fit_X_format"] == "csr"


def test_continuous_overlap_scorer_preserves_sparse_features(fake_overlapindex):
    embeddings = sparse.csc_array(np.eye(6))
    targets = np.linspace(0.0, 1.0, 6)

    result = OverlapIndexScorer(
        ContinuousOverlapScoringConfig(k=1, n_null_permutations=1)
    ).score(
        embeddings,
        targets,
        target_type="regression",
    )

    assert result.metadata["sparse_input"] is True
    assert result.metadata["scoring_input_format"] == "csr"
    assert fake_overlapindex.continuous_calls[-1]["fit_X_sparse"] is True


def test_sparse_multilabel_targets_are_validated_and_passed_as_csr(fake_overlapindex):
    embeddings = sparse.csr_array(np.eye(6))
    targets = sparse.csc_array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ]
    )
    dataset = BenchmarkDataset.from_embeddings(
        embeddings,
        targets,
        label_names=["red", "round", "sweet"],
        identity=DatasetIdentity.ephemeral(),
    )

    result = OverlapIndexScorer(
        OverlapScoringConfig(k=1, min_samples_per_cluster=1)
    ).score(
        dataset.X,
        dataset.y,
        label_names=dataset.metadata["label_names"],
        target_type="multi_label",
    )

    assert dataset.y.tolist() == [
        ("red", "round"),
        ("red",),
        ("round",),
        ("sweet",),
        ("red", "sweet"),
        ("round", "sweet"),
    ]
    assert result.metadata["target_type"] == "multi_label"
    assert fake_overlapindex.calls[-1]["fit_y_sparse"] is True
    indicator = multilabel_indicator(
        dataset.y,
        dataset.metadata["label_names"],
        sparse_output=True,
    )
    assert sparse.isspmatrix_csr(indicator)
    assert indicator.nnz == 9


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (sparse.csr_array([[1, 2], [1, 0]]), "only 0/1"),
        (sparse.csr_array([[1, np.nan], [1, 0]]), "only 0/1"),
        (sparse.csr_array([[1, 0], [0, 0]]), "at least one label"),
    ],
)
def test_sparse_multilabel_targets_reject_invalid_indicators(targets, message):
    with pytest.raises(ValueError, match=message):
        BenchmarkDataset.from_arrays(
            np.eye(2),
            targets,
            modality="embeddings",
            identity=DatasetIdentity.ephemeral(),
        )


def test_sparse_multilabel_targets_validate_label_names():
    with pytest.raises(ValueError, match="1 names for 2 columns"):
        BenchmarkDataset.from_arrays(
            np.eye(2),
            sparse.csr_array([[1, 0], [0, 1]]),
            label_names=["only-one-name"],
            modality="embeddings",
            identity=DatasetIdentity.ephemeral(),
        )


def test_sparse_regression_targets_are_rejected():
    with pytest.raises(ValueError, match="Regression targets must be dense"):
        BenchmarkDataset.from_arrays(
            np.eye(3),
            sparse.csr_array([[0.0], [0.5], [1.0]]),
            target_type="regression",
            modality="embeddings",
            identity=DatasetIdentity.ephemeral(),
        )


def test_benchmark_sparse_precomputed_workflow(tmp_path, fake_overlapindex):
    embeddings = sparse.csr_matrix(np.eye(8))
    labels = np.array(["a"] * 4 + ["b"] * 4)
    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor("sparse_embeddings"),
        scoring_config=OverlapScoringConfig(k=1, max_dense_bytes=10_000),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
    ).run()

    metadata = result.extractor_results[0].embedding_metadata
    assert metadata["sparse"] is True
    assert metadata["storage_format"] == "csr"
    assert metadata["nnz"] == 8


def test_sklearn_sparse_extractor_can_flow_to_benchmark(tmp_path, fake_overlapindex):
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = ["alpha beta", "alpha gamma", "delta epsilon", "delta zeta"]
    labels = np.array(["left", "left", "right", "right"])
    dataset = BenchmarkDataset.from_arrays(
        texts, labels, modality="text", identity=DatasetIdentity.ephemeral()
    )
    extractor = SklearnExtractor(
        "tfidf_sparse",
        TfidfVectorizer(),
        allow_sparse=True,
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1, max_dense_bytes=10_000),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    assert result.extractor_results[0].embedding_metadata["sparse"] is True
