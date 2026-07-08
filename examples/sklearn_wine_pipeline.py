"""Compare real scikit-learn pipelines on the bundled Wine dataset.

This example stays network-free while using real UCI data distributed with
scikit-learn. The pipelines are intentionally modest so they run quickly on a
laptop, but they still exercise the same extractor comparison, scoring,
stability, probe, cache, JSON, and Markdown reporting path used for larger
projects.
"""

from _common import CACHE_DIR, ensure_output_dir, print_ranking
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import SklearnExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    wine = load_wine(as_frame=True)
    df = wine.frame.copy()
    df["cultivar"] = df["target"].map(dict(enumerate(wine.target_names)))

    dataset = BenchmarkDataset.from_dataframe(
        df,
        input_col=list(wine.feature_names),
        label_col="cultivar",
        modality="tabular",
        metadata={
            "example": "sklearn_wine_pipeline",
            "source": "sklearn.datasets.load_wine",
        },
    )

    benchmark = Benchmark(
        dataset=dataset,
        scoring_config=OverlapScoringConfig(k=4, min_samples_per_cluster=8),
        stability_config=StabilityConfig(repeats=5, random_state=23),
        cache_config=CacheConfig(cache_dir=str(CACHE_DIR)),
    )
    benchmark.add_extractor(
        SklearnExtractor(
            name="wine_standard_scaler_pca_6",
            pipeline=Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=6, random_state=42)),
                ]
            ),
        )
    )
    benchmark.add_extractor(
        SklearnExtractor(
            name="wine_standard_scaler_all_features",
            pipeline=Pipeline(
                [
                    ("scale", StandardScaler()),
                ]
            ),
        )
    )
    benchmark.add_extractor(
        SklearnExtractor(
            name="wine_minmax_pca_2",
            pipeline=Pipeline(
                [
                    ("scale", MinMaxScaler()),
                    ("pca", PCA(n_components=2, random_state=42)),
                ]
            ),
        )
    )
    benchmark.add_extractor(
        SklearnExtractor(
            name="wine_quantile_pca_1",
            pipeline=Pipeline(
                [
                    (
                        "quantile",
                        QuantileTransformer(
                            n_quantiles=64,
                            output_distribution="normal",
                            random_state=42,
                        ),
                    ),
                    ("pca", PCA(n_components=1, random_state=42)),
                ]
            ),
        )
    )

    result = benchmark.run()

    result.save_json(str(output_dir / "sklearn_wine_pipeline.json"))
    result.save_markdown(str(output_dir / "sklearn_wine_pipeline.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
