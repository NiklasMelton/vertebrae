import numpy as np

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import CacheConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import CallableExtractor


def test_multi_extractor_benchmark(fake_overlapindex):
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.array(["a"] * 10 + ["b"] * 10)
    dataset = BenchmarkDataset.from_arrays(X, y, modality="tabular")

    benchmark = Benchmark(
        dataset,
        stability_config=StabilityConfig(repeats=2),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    )
    benchmark.add_extractor(CallableExtractor("identity", lambda value: value, modality="tabular"))
    benchmark.add_extractor(
        CallableExtractor("scaled", lambda value: value * 2, modality="tabular")
    )

    result = benchmark.run()

    assert len(result.extractor_results) == 2
    assert list(result.to_dataframe()["rank"]) == [1, 2]
    assert len(fake_overlapindex.calls) == 6
