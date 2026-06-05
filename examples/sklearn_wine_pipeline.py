"""Evaluate a real scikit-learn pipeline on the bundled Wine dataset.

This example stays network-free while using real UCI data distributed with
scikit-learn. The pipeline is intentionally modest so it runs quickly on a
laptop, but it still exercises the same extractor, scoring, stability, probe,
cache, JSON, and Markdown reporting path used for larger projects.
"""

from _common import CACHE_DIR, ensure_output_dir, print_ranking
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
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

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=6, random_state=42)),
        ]
    )

    result = Evaluator(
        dataset=dataset,
        extractor=SklearnExtractor(name="wine_standard_scaler_pca", pipeline=pipeline),
        scoring_config=OverlapScoringConfig(k=4, min_samples_per_cluster=8),
        stability_config=StabilityConfig(repeats=5, random_state=23),
        probe_config=ProbeConfig(methods=("nearest_centroid", "logistic_regression")),
        cache_config=CacheConfig(cache_dir=str(CACHE_DIR)),
    ).run()

    result.save_json(str(output_dir / "sklearn_wine_pipeline.json"))
    result.save_markdown(str(output_dir / "sklearn_wine_pipeline.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
