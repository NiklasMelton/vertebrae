"""Run a network-free fixed-prompt zero-shot comparison."""

import numpy as np

from vertebrae import (
    BenchmarkDataset,
    CacheConfig,
    DatasetIdentity,
    ZeroShotBenchmark,
    ZeroShotDataset,
)
from vertebrae.extractors import CallableRetrievalExtractor


def _sample_embeddings(values):
    return np.asarray(
        [[1.0, 0.0] if str(value).startswith("cat") else [0.0, 1.0] for value in values]
    )


def _text_embeddings(values):
    return np.asarray([[1.0, 0.0] if value == "cat" else [0.0, 1.0] for value in values])


dataset = BenchmarkDataset.from_arrays(
    ["cat-0", "cat-1", "dog-0", "dog-1"],
    ["cat", "cat", "dog", "dog"],
    modality="image",
    identity=DatasetIdentity.ephemeral(),
)
protocol = ZeroShotDataset.from_templates(dataset, templates=("{label}",))
extractor = CallableRetrievalExtractor(
    "synthetic_aligned_encoder",
    query_fn=_sample_embeddings,
    gallery_fn=_text_embeddings,
    query_modality="image",
    gallery_modality="text",
)
result = ZeroShotBenchmark(
    protocol,
    [extractor],
    sample_branch="query",
    text_branch="gallery",
    cache_config=CacheConfig(enabled=False),
).run()
print(result.to_dataframe())
