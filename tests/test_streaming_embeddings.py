import json

import numpy as np
import pytest

import vertebrae.benchmark as benchmark_module
from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    DatasetIdentity,
    EmbeddingConfig,
    Evaluator,
    ShardSpec,
)
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.config import CacheConfig, MemoryConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import CallableExtractor, MultiOutputExtractor
from vertebrae.extractors.base import EmbeddingOutputSpec
from vertebrae.utils.memory import (
    IncrementalMatrixReferenceStager,
    IncrementalMatrixStager,
    MatrixRowReference,
)


def test_matrix_reference_stager_spills_bounded_state_and_restores_sample_order():
    config = MemoryConfig(max_memory_bytes=1, allow_disk_spill=True)
    n_rows = 500
    with (
        IncrementalMatrixStager(
            config,
            purpose="ordered reference stress",
        ) as matrix_stager,
        IncrementalMatrixReferenceStager(
            config,
            purpose="ordered reference stress",
            matrix_stager=matrix_stager,
        ) as reference_stager,
    ):
        for position in reversed(range(n_rows)):
            reference_stager.append(
                "embedding",
                position,
                matrix_stager.append(
                    "embedding",
                    np.asarray([[float(position)]], dtype=np.float64),
                ),
            )

        assert reference_stager.strategy == "disk"
        assert reference_stager.resident_bytes == 0
        assert reference_stager.count_rows("embedding") == n_rows
        assembly = reference_stager.assemble(
            "embedding",
            expected_rows=n_rows,
            purpose="ordered reference stress",
        )

    assert isinstance(assembly.matrix, np.memmap)
    assert assembly.matrix[:, 0].tolist() == [float(index) for index in range(n_rows)]


def test_matrix_reference_stager_charges_no_spill_order_metadata():
    config = MemoryConfig(max_memory_bytes=4_096, allow_disk_spill=False)
    with (
        IncrementalMatrixStager(
            config,
            purpose="bounded reference metadata",
        ) as matrix_stager,
        IncrementalMatrixReferenceStager(
            config,
            purpose="bounded reference metadata",
            matrix_stager=matrix_stager,
        ) as reference_stager,
    ):
        with pytest.raises(ValueError, match="memory budget"):
            for position in range(1_000):
                reference_stager.append(
                    "embedding",
                    position,
                    MatrixRowReference(
                        token=position,
                        output_name="embedding",
                        width=1,
                        dtype="float64",
                        resident_bytes=264,
                    ),
                )

        assert 0 < reference_stager.count_rows("embedding") < 1_000
        assert 0 < reference_stager.resident_bytes <= config.max_memory_bytes


def test_matrix_reference_stager_rejects_duplicate_and_missing_positions():
    config = MemoryConfig(max_memory_bytes=1, allow_disk_spill=True)
    with (
        IncrementalMatrixStager(
            config,
            purpose="reference coverage",
        ) as matrix_stager,
        IncrementalMatrixReferenceStager(
            config,
            purpose="reference coverage",
            matrix_stager=matrix_stager,
        ) as reference_stager,
    ):
        first = matrix_stager.append("embedding", np.asarray([[0.0]]))
        last = matrix_stager.append("embedding", np.asarray([[2.0]]))
        reference_stager.append("embedding", 0, first)
        with pytest.raises(ValueError, match="Duplicate embedding rows"):
            reference_stager.append("embedding", 0, first)
        reference_stager.append("embedding", 2, last)

        with pytest.raises(ValueError, match=r"missing \[1\]"):
            reference_stager.assemble(
                "embedding",
                expected_rows=3,
                purpose="reference coverage",
            )


def test_dataset_iter_batches_shards_are_disjoint_and_complete():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        ["a"] * 5 + ["b"] * 5,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )

    shard_0 = list(
        dataset.iter_batches(batch_size=2, shard=ShardSpec(total_shards=2, shard_index=0))
    )
    shard_1 = list(
        dataset.iter_batches(batch_size=2, shard=ShardSpec(total_shards=2, shard_index=1))
    )
    indices_0 = np.concatenate([batch.indices for batch in shard_0])
    indices_1 = np.concatenate([batch.indices for batch in shard_1])

    assert set(indices_0).isdisjoint(set(indices_1))
    assert sorted([*indices_0, *indices_1]) == list(range(10))


def test_local_store_put_array_batches_rejects_duplicate_indices(tmp_path):
    store = LocalArtifactStore(str(tmp_path))

    with pytest.raises(ValueError, match="Duplicate embedding rows"):
        store.put_array_batches(
            "embeddings/duplicate",
            [
                (np.array([0, 1]), np.ones((2, 3))),
                (np.array([1, 2]), np.ones((2, 3))),
            ],
            n_samples=3,
        )


def test_streaming_benchmark_materializes_embeddings_once_per_sample(tmp_path, fake_overlapindex):
    seen = []

    def transform_batch(batch):
        values = np.asarray(batch)
        seen.extend(values[:, 0].astype(int).tolist())
        return values[:, :2] * 2

    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        "streaming_callable",
        transform_batch,
        modality="tabular",
        streaming_safe=True,
        cache_identity="streaming-materialization-test-v1",
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=3),
        memory_config=MemoryConfig(auto_subsample_on_memory_exceeded=False),
    ).run()

    metadata = result.extractor_results[0].embedding_metadata
    assert sorted(seen) == [0, 3, 6, 9, 12, 15, 18, 21]
    assert len(seen) == len(set(seen))
    assert metadata["streamed"] is True
    assert metadata["stream_batch_size"] == 3
    artifact_dir = tmp_path / metadata["cache_key"]
    artifact_manifest = json.loads((artifact_dir / "artifact-manifest.json").read_text())
    assert (artifact_dir / artifact_manifest["array"]["filename"]).exists()


def test_already_fitted_non_streaming_extractor_receives_one_full_transform(
    tmp_path,
    fake_overlapindex,
):
    class _NonStreamingExtractor:
        name = "non_streaming_fitted"
        modality = "tabular"
        extractor_type = "test"
        streaming_safe = False
        already_fitted = True

        def __init__(self):
            self.transform_sizes = []

        def fit(self, X, y=None):
            del X, y
            return self

        def transform(self, X):
            self.transform_sizes.append(len(X))
            return np.asarray(X)[:, :2].astype(float)

        def fit_transform(self, X, y=None):
            del y
            return self.transform(X)

        def recipe(self):
            return {
                "name": self.name,
                "extractor_type": self.extractor_type,
                "cache_safe": True,
            }

    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = _NonStreamingExtractor()

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=2, streaming_enabled=True),
    ).run()

    assert extractor.transform_sizes == [8]
    assert result.extractor_results[0].embedding_metadata["streamed"] is False


def test_streaming_multi_output_uses_incremental_disk_staging(
    monkeypatch,
    tmp_path,
    fake_overlapindex,
):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(36, dtype=np.float32).reshape(12, 3),
        ["a"] * 6 + ["b"] * 6,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = MultiOutputExtractor(
        "staged_multi",
        [EmbeddingOutputSpec("base"), EmbeddingOutputSpec("scaled")],
        transform_many_fn=lambda value: {
            "base": np.asarray(value)[:, :2],
            "scaled": np.asarray(value)[:, :2] * 2,
        },
        modality="tabular",
        streaming_safe=True,
        cache_identity="staged-multi-v1",
    )

    def reject_eager_batch_collection(*_args, **_kwargs):
        raise AssertionError("multi-output streaming must not collect complete batch lists")

    monkeypatch.setattr(
        benchmark_module,
        "_combine_embedding_batches",
        reject_eager_batch_collection,
    )
    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False, cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=3),
        memory_config=MemoryConfig(
            # Each 12x2 output needs 96 bytes and can be scored independently,
            # while retaining both dense outputs would exceed this budget.
            max_memory_bytes=128,
            auto_subsample_on_memory_exceeded=False,
        ),
    ).run()

    assert len(result.extractor_results) == 2
    assert all(
        item.embedding_metadata["materialization"]["staging_strategy"] == "disk"
        for item in result.extractor_results
    )
    assert all(
        item.embedding_metadata["materialization"]["strategy"] == "disk_spill"
        for item in result.extractor_results
    )


def test_cached_multi_output_load_spills_when_aggregate_exceeds_budget(
    tmp_path,
    fake_overlapindex,
):
    transform_calls = []
    dataset = BenchmarkDataset.from_arrays(
        np.arange(36, dtype=np.float32).reshape(12, 3),
        ["a"] * 6 + ["b"] * 6,
        modality="tabular",
        identity=DatasetIdentity.declared("cached-multi-output-spill", "v1"),
    )

    def transform_many(value):
        transform_calls.append(len(value))
        array = np.asarray(value)
        return {"base": array[:, :2], "scaled": array[:, :2] * 2}

    def make_extractor():
        return MultiOutputExtractor(
            "cached_staged_multi",
            [EmbeddingOutputSpec("base"), EmbeddingOutputSpec("scaled")],
            transform_many_fn=transform_many,
            modality="tabular",
            streaming_safe=True,
            cache_identity="cached-staged-multi-v1",
        )

    kwargs = {
        "dataset": dataset,
        "stability_config": StabilityConfig(enabled=False),
        "cache_config": CacheConfig(cache_dir=str(tmp_path)),
        "embedding_config": EmbeddingConfig(batch_size=3),
        "memory_config": MemoryConfig(
            max_memory_bytes=128,
            auto_subsample_on_memory_exceeded=False,
        ),
    }
    Benchmark(extractors=[make_extractor()], **kwargs).run()
    assert transform_calls == [3, 3, 3, 3]
    transform_calls.clear()

    cached = Benchmark(extractors=[make_extractor()], **kwargs).run()

    assert transform_calls == []
    assert all(item.embedding_metadata["cache_hit"] for item in cached.extractor_results)
    assert all(
        item.embedding_metadata["materialization"]["strategy"] == "disk_spill"
        for item in cached.extractor_results
    )


def test_embedding_config_no_longer_accepts_partial_shards():
    with pytest.raises(TypeError, match="unexpected keyword argument 'shard'"):
        EmbeddingConfig(
            batch_size=2,
            shard=ShardSpec(total_shards=2, shard_index=0),
        )


@pytest.mark.parametrize("change", ["width", "dtype", "sparse_format"])
def test_streaming_benchmark_rejects_batch_contract_changes(
    tmp_path,
    fake_overlapindex,
    change,
):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )

    def transform_batch(batch):
        first = int(np.asarray(batch)[0, 0]) == 0
        if change == "width":
            return np.ones((len(batch), 2 if first else 1), dtype=np.float32)
        if change == "dtype":
            dtype = np.float32 if first else np.float64
            return np.ones((len(batch), 2), dtype=dtype)
        from scipy import sparse

        matrix = np.ones((len(batch), 2), dtype=np.float32)
        return sparse.csr_matrix(matrix) if first else sparse.csc_matrix(matrix)

    extractor = CallableExtractor(
        "changing_contract",
        transform_batch,
        streaming_safe=True,
    )
    with pytest.raises(ValueError, match="changed matrix contract"):
        Evaluator(
            dataset=dataset,
            extractor=extractor,
            stability_config=StabilityConfig(enabled=False),
            cache_config=CacheConfig(enabled=False, cache_dir=str(tmp_path)),
            embedding_config=EmbeddingConfig(batch_size=3),
        ).run()
