import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import CacheConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import CallableExtractor, PrecomputedExtractor, SklearnExtractor


def test_full_benchmark_multiple_extractor_families_generates_reports(
    tmp_path,
    fake_overlapindex,
):
    X = np.vstack(
        [
            np.random.default_rng(1).normal(0, 0.2, size=(10, 5)),
            np.random.default_rng(2).normal(3, 0.2, size=(10, 5)),
        ]
    )
    y = np.array(["left"] * 10 + ["right"] * 10)
    dataset = BenchmarkDataset.from_embeddings(X, y)
    benchmark = Benchmark(
        dataset,
        stability_config=StabilityConfig(repeats=2),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    )
    benchmark.add_extractor(PrecomputedExtractor("precomputed"))
    benchmark.add_extractor(
        SklearnExtractor(
            "scaled_pca",
            Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=3))]),
        )
    )
    benchmark.add_extractor(
        CallableExtractor(
            "stat_features",
            lambda values: np.column_stack([values.mean(axis=1), values.std(axis=1)]),
            recipe_data={"features": ["row_mean", "row_std"]},
        )
    )

    result = benchmark.run()
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "report.md"
    result.save_json(str(json_path))
    result.save_markdown(str(markdown_path))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(result.to_dataframe()) == 3
    assert len(payload["extractor_results"]) == 3
    assert np.isfinite(result.extractor_results[0].overlap.macro_score)
    assert "Recipe summary" in markdown_path.read_text(encoding="utf-8")
    for item in payload["extractor_results"]:
        assert "recipe" in item["embedding_metadata"]
