import numpy as np
import pytest
from scipy import sparse

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import PrecomputedExtractor, SklearnExtractor
from vertebrae.scoring.overlap import OverlapIndexScorer


def test_dataset_from_sparse_embeddings_preserves_sparse_matrix():
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])

    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    assert sparse.issparse(dataset.X)
    assert dataset.X.shape == (6, 6)


def test_local_store_round_trips_sparse_embeddings(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    embeddings = sparse.csr_matrix(np.eye(4))

    path = store.put_array("embeddings/test", embeddings)
    loaded = store.get_array("embeddings/test")

    assert path.endswith(".npz")
    assert sparse.issparse(loaded)
    assert np.array_equal(loaded.toarray(), embeddings.toarray())


def test_overlap_scorer_accepts_sparse_embeddings(fake_overlapindex):
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    scorer = OverlapIndexScorer(OverlapScoringConfig(k=1, max_dense_bytes=1_000))

    result = scorer.score(embeddings, labels)

    assert result.metadata["sparse_input"] is True
    assert result.metadata["scoring_input_format"] == "dense"
    assert any("Sparse embeddings were densified" in warning for warning in result.warnings)


def test_overlap_scorer_sparse_densification_memory_limit():
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    scorer = OverlapIndexScorer(OverlapScoringConfig(k=1, max_dense_bytes=1))

    with pytest.raises(ValueError, match="max_dense_bytes"):
        scorer.score(embeddings, labels)


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
