import numpy as np
import pytest

from vertebrae import Benchmark, BenchmarkDataset, DatasetIdentity
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.extractors import MultiOutputExtractor
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec


def test_multi_output_extractor_expands_to_multiple_results(fake_overlapindex):
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.array(["a"] * 10 + ["b"] * 10)
    dataset = BenchmarkDataset.from_arrays(
        X, y, modality="tabular", identity=DatasetIdentity.ephemeral()
    )
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


def test_multi_output_cache_identity_rejects_local_lambda_without_opt_in():
    specs = [EmbeddingOutputSpec(name="identity")]
    unsafe = MultiOutputExtractor(
        "unsafe",
        specs,
        transform_many_fn=lambda value: {"identity": np.asarray(value)},
    )
    explicit = MultiOutputExtractor(
        "explicit",
        specs,
        transform_many_fn=lambda value: {"identity": np.asarray(value)},
        cache_identity="multi-v1",
    )

    assert unsafe.recipe()["cache_safe"] is False
    assert explicit.recipe()["cache_safe"] is True
    with pytest.raises(ValueError, match="cache_identity"):
        MultiOutputExtractor("bad", specs, transform_many_fn=np.asarray, cache_identity="")


def test_multi_output_requires_exact_raw_named_outputs():
    specs = [EmbeddingOutputSpec("left"), EmbeddingOutputSpec("right")]
    extra = MultiOutputExtractor(
        "extra",
        specs,
        transform_many_fn=lambda value: {
            "left": value,
            "right": value,
            "undeclared": value,
        },
    )
    duplicate = MultiOutputExtractor(
        "duplicate",
        specs,
        transform_many_fn=lambda value: [
            EmbeddingOutput("left", value, {}),
            EmbeddingOutput(" left ", value, {}),
            EmbeddingOutput("right", value, {}),
        ],
    )

    with pytest.raises(ValueError, match="extra=.*undeclared"):
        extra.transform_many(np.ones((2, 2)))
    with pytest.raises(ValueError, match="duplicate output"):
        duplicate.transform_many(np.ones((2, 2)))
