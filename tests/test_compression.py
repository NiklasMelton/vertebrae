import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from vertebrae import Benchmark, BenchmarkDataset, EmbeddingCompressionConfig, Evaluator
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.execution import (
    ScoringJob,
    benchmark_result_from_artifacts,
    compress_embedding_artifact,
    plan_compression_job,
    score_embedding_artifact,
)
from vertebrae.extractors import CallableExtractor, PrecomputedExtractor, SklearnExtractor


def test_pca_compression_reduces_dense_embeddings(fake_overlapindex):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(12, 8))
    labels = np.array(["a"] * 6 + ["b"] * 6)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="dense"),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="pca",
            n_components=3,
            dtype="float32",
        ),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    item = result.extractor_results[0]
    assert item.name == "dense[pca_3]"
    assert item.compression_metadata["compressed_dim"] == 3
    assert item.compression_metadata["method"] == "pca"
    assert "explained_variance_total" in item.compression_metadata


def test_truncated_svd_accepts_sparse_embeddings(tmp_path, fake_overlapindex):
    texts = ["alpha beta", "alpha gamma", "delta epsilon", "delta zeta"]
    labels = np.array(["left", "left", "right", "right"])
    dataset = BenchmarkDataset.from_arrays(texts, labels, modality="text")
    extractor = SklearnExtractor("tfidf_sparse", TfidfVectorizer(), allow_sparse=True)

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="truncated_svd",
            n_components=2,
            random_state=7,
        ),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
    ).run()

    item = result.extractor_results[0]
    assert item.compression_metadata["method"] == "truncated_svd"
    assert item.compression_metadata["input_sparse"] is True
    assert item.compression_metadata["compressed_dim"] == 2


def test_prefix_truncate_dense_embeddings_preserves_prefix(fake_overlapindex):
    embeddings = np.arange(48, dtype=float).reshape(8, 6)
    labels = np.array(["a"] * 4 + ["b"] * 4)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="dense_prefix"),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="prefix_truncate",
            n_components=3,
            assume_matryoshka=True,
        ),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    item = result.extractor_results[0]
    assert item.name == "dense_prefix[prefix_truncate_3]"
    assert item.compression_metadata["compressed_dim"] == 3
    assert item.compression_metadata["assume_matryoshka"] is True


def test_prefix_truncate_sparse_embeddings_warns_without_matryoshka(fake_overlapindex):
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor("sparse_prefix"),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="prefix_truncate",
            n_components=4,
        ),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    item = result.extractor_results[0]
    assert item.compression_metadata["input_sparse"] is True
    assert item.compression_metadata["output_sparse"] is True
    assert any("Prefix truncation" in warning for warning in item.compression_metadata["warnings"])


def test_pca_rejects_sparse_embeddings(fake_overlapindex):
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    evaluator = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor("sparse"),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="pca",
            n_components=2,
        ),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    )

    try:
        evaluator.run()
    except ValueError as exc:
        assert "requires dense embeddings" in str(exc)
    else:
        raise AssertionError("Expected PCA compression on sparse input to fail.")


def test_multi_compression_configs_expand_results(fake_overlapindex):
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.array(["a"] * 10 + ["b"] * 10)
    dataset = BenchmarkDataset.from_arrays(X, y, modality="tabular")

    benchmark = Benchmark(
        dataset,
        compression_configs=[
            EmbeddingCompressionConfig(),
            EmbeddingCompressionConfig(enabled=True, method="pca", n_components=2),
        ],
        stability_config=StabilityConfig(repeats=2),
        cache_config=CacheConfig(enabled=False),
    )
    benchmark.add_extractor(CallableExtractor("identity", lambda value: value, modality="tabular"))

    result = benchmark.run()

    assert len(result.extractor_results) == 2
    assert set(result.to_dataframe()["compression_method"]) == {"none", "pca"}


def test_quantize_float16_changes_precision(fake_overlapindex):
    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(12, 6)).astype(np.float32)
    labels = np.array(["a"] * 6 + ["b"] * 6)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor("float16_quant"),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="quantize",
            precision="float16",
        ),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    item = result.extractor_results[0]
    assert item.name == "float16_quant[quantize_float16]"
    assert item.compression_metadata["precision"] == "float16"
    assert item.compression_metadata["dtype"] == "float16"


def test_quantize_int8_round_trip_records_calibration(fake_overlapindex):
    rng = np.random.default_rng(5)
    embeddings = rng.normal(size=(12, 6)).astype(np.float32)
    labels = np.array(["a"] * 6 + ["b"] * 6)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor("int8_quant"),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="quantize",
            precision="int8",
        ),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    item = result.extractor_results[0]
    assert item.compression_metadata["precision"] == "int8"
    assert item.compression_metadata["quantization_mode"] == "symmetric_absmax"
    assert item.compression_metadata["encoded_dtype"] == "int8"
    assert item.compression_metadata["estimated_encoded_bytes"] > 0
    assert "calibration" in item.compression_metadata


def test_compressed_embeddings_are_reused_from_cache(tmp_path, fake_overlapindex):
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(12, 6))
    labels = np.array(["a"] * 6 + ["b"] * 6)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)
    config = EmbeddingCompressionConfig(enabled=True, method="pca", n_components=3)

    kwargs = dict(
        dataset=dataset,
        extractor=PrecomputedExtractor("cached"),
        compression_config=config,
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
    )
    first = Evaluator(**kwargs).run()
    second = Evaluator(**kwargs).run()

    assert first.extractor_results[0].compression_metadata["cache_hit"] is False
    assert second.extractor_results[0].compression_metadata["cache_hit"] is True


def test_compress_embedding_artifact_round_trip(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24, dtype=float).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
    )
    store = LocalArtifactStore(str(tmp_path))
    embedding_key = "embeddings/raw"
    labels_key = "labels/raw"
    embeddings = np.asarray(dataset.X)
    store.put_array(embedding_key, embeddings)
    store.put_json(
        embedding_key,
        {
            "cache_key": embedding_key,
            "extractor_name": "artifact",
            "extractor_type": "precomputed",
            "extractor_recipe": {"name": "artifact", "extractor_type": "precomputed"},
            "dataset_fingerprint": dataset.fingerprint(),
            "n_samples": int(len(dataset.y)),
            "embedding_dim": int(embeddings.shape[1]),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "sparse": False,
            "modality": "embeddings",
        },
    )
    store.put_labels(labels_key, dataset.y)
    store.put_json(
        labels_key,
        {
            "dataset_fingerprint": dataset.fingerprint(),
            "n_samples": int(len(dataset.y)),
            "class_counts": dataset.class_counts(),
        },
    )

    job = plan_compression_job(
        embedding_key,
        EmbeddingCompressionConfig(enabled=True, method="pca", n_components=2),
    )
    compressed = compress_embedding_artifact(job, store)
    score = score_embedding_artifact(
        ScoringJob(
            embedding_key=compressed["output_key"],
            labels_key=labels_key,
            output_key=f'{compressed["output_key"]}/scores/default',
        ),
        store,
    )
    result = benchmark_result_from_artifacts(score["output_key"], store=store)

    assert compressed["artifact_type"] == "compressed_embedding"
    assert compressed["compression_metadata"]["compressed_dim"] == 2
    assert result["extractor_results"][0]["compression_metadata"]["method"] == "pca"


def test_markdown_report_includes_compression_columns(tmp_path, fake_overlapindex):
    rng = np.random.default_rng(2)
    embeddings = rng.normal(size=(12, 6))
    labels = np.array(["a"] * 6 + ["b"] * 6)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="report"),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="pca",
            n_components=3,
        ),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    path = tmp_path / "report.md"
    result.save_markdown(str(path))
    text = path.read_text(encoding="utf-8")

    assert "compression" in text
    assert "pca" in text


def test_quantize_sparse_int8_rejects_sparse_input(fake_overlapindex):
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    evaluator = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor("sparse_quant"),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="quantize",
            precision="int8",
        ),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    )

    try:
        evaluator.run()
    except ValueError as exc:
        assert "requires dense embeddings" in str(exc)
    else:
        raise AssertionError("Expected int8 quantization on sparse input to fail.")
