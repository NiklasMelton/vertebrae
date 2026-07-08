import numpy as np
import pytest

from vertebrae import BenchmarkDataset, EmbeddingConfig, Evaluator, ShardSpec
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import CallableExtractor


def test_dataset_iter_batches_shards_are_disjoint_and_complete():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(30).reshape(10, 3),
        ["a"] * 5 + ["b"] * 5,
        modality="tabular",
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
    )
    extractor = CallableExtractor(
        "streaming_callable",
        transform_batch,
        modality="tabular",
        streaming_safe=True,
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        embedding_config=EmbeddingConfig(batch_size=3),
    ).run()

    metadata = result.extractor_results[0].embedding_metadata
    assert sorted(seen) == [0, 3, 6, 9, 12, 15, 18, 21]
    assert len(seen) == len(set(seen))
    assert metadata["streamed"] is True
    assert metadata["stream_batch_size"] == 3
    assert (tmp_path / metadata["cache_key"] / "embeddings.npy").exists()


def test_local_benchmark_rejects_partial_embedding_shards(fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(12).reshape(4, 3),
        ["a", "a", "b", "b"],
        modality="tabular",
    )
    extractor = CallableExtractor(
        "streaming_callable",
        lambda batch: np.asarray(batch)[:, :2],
        streaming_safe=True,
    )

    with pytest.raises(ValueError, match="complete embedding artifact"):
        Evaluator(
            dataset=dataset,
            extractor=extractor,
            scoring_config=OverlapScoringConfig(k=1),
            stability_config=StabilityConfig(enabled=False),
            cache_config=CacheConfig(enabled=False),
            embedding_config=EmbeddingConfig(
                batch_size=2,
                shard=ShardSpec(total_shards=2, shard_index=0),
            ),
        ).run()
