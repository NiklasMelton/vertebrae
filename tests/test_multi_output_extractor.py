import numpy as np

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.extractors import MultiOutputExtractor
from vertebrae.extractors.base import EmbeddingOutputSpec


def test_multi_output_extractor_expands_to_multiple_results(fake_overlapindex):
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.array(["a"] * 10 + ["b"] * 10)
    dataset = BenchmarkDataset.from_arrays(X, y, modality="tabular")
    extractor = MultiOutputExtractor(
        name="callable_multi",
        output_specs=[
            EmbeddingOutputSpec(name="identity"),
            EmbeddingOutputSpec(name="scaled"),
        ],
        transform_many_fn=lambda value: {
            "identity": np.asarray(value),
            "scaled": np.asarray(value) * 2,
        },
        modality="tabular",
        streaming_safe=True,
    )

    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        stability_config=StabilityConfig(repeats=2),
        cache_config=CacheConfig(enabled=False),
    ).run()

    assert len(result.extractor_results) == 2
    assert {item.name for item in result.extractor_results} == {
        "callable_multi:identity",
        "callable_multi:scaled",
    }
    assert all(
        item.embedding_metadata["parent_extractor_name"] == "callable_multi"
        for item in result.extractor_results
    )
    assert {item.embedding_metadata["output_name"] for item in result.extractor_results} == {
        "identity",
        "scaled",
    }
    assert set(result.to_dataframe()["extractor"]) == {
        "callable_multi:identity",
        "callable_multi:scaled",
    }
    assert len(fake_overlapindex.calls) == 6
