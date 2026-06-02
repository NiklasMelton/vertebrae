"""Compare several feature extractors on the same labeled numeric dataset."""

import numpy as np
from _common import CACHE_DIR, ensure_output_dir, make_separated_blobs, print_ranking
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import CallableExtractor, SklearnExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    X, labels = make_separated_blobs(samples_per_class=38, n_features=12, random_state=23)
    dataset = BenchmarkDataset.from_arrays(
        X,
        labels,
        modality="tabular",
        metadata={"example": "multi_extractor_comparison"},
    )

    benchmark = Benchmark(
        dataset=dataset,
        scoring_config=OverlapScoringConfig(k=4, min_samples_per_cluster=5),
        stability_config=StabilityConfig(repeats=4, random_state=29),
        probe_config=ProbeConfig(methods=("nearest_centroid", "knn")),
        cache_config=CacheConfig(cache_dir=str(CACHE_DIR)),
    )
    benchmark.add_extractor(
        CallableExtractor(
            name="raw_numeric_features",
            transform_fn=lambda values: np.asarray(values),
            modality="tabular",
        )
    )
    benchmark.add_extractor(
        SklearnExtractor(
            name="scaled_pca_4d",
            pipeline=Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=4, random_state=31)),
                ]
            ),
        )
    )
    benchmark.add_extractor(
        CallableExtractor(
            name="first_half_features",
            transform_fn=lambda values: np.asarray(values)[:, :6],
            modality="tabular",
        )
    )

    result = benchmark.run()
    result.save_json(str(output_dir / "multi_extractor_comparison.json"))
    result.save_markdown(str(output_dir / "multi_extractor_comparison.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
