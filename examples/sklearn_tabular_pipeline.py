"""Evaluate a mixed tabular scikit-learn pipeline on a pandas dataframe."""

import pandas as pd
from _common import CACHE_DIR, ensure_output_dir, print_ranking
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import SklearnExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    df = _make_customer_dataframe()
    dataset = BenchmarkDataset.from_dataframe(
        df,
        input_col=["age", "income", "region", "plan"],
        label_col="segment",
        modality="tabular",
        metadata={"example": "sklearn_tabular_pipeline"},
    )

    pipeline = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), ["age", "income"]),
                        ("categorical", OneHotEncoder(sparse_output=False), ["region", "plan"]),
                    ]
                ),
            ),
            ("pca", PCA(n_components=4, random_state=42)),
        ]
    )

    result = Evaluator(
        dataset=dataset,
        extractor=SklearnExtractor(name="mixed_tabular_pca", pipeline=pipeline),
        scoring_config=OverlapScoringConfig(k=3, min_samples_per_cluster=4),
        stability_config=StabilityConfig(repeats=4, random_state=47),
        probe_config=ProbeConfig(methods=("nearest_centroid", "knn")),
        cache_config=CacheConfig(cache_dir=str(CACHE_DIR)),
    ).run()

    result.save_json(str(output_dir / "sklearn_tabular_pipeline.json"))
    result.save_markdown(str(output_dir / "sklearn_tabular_pipeline.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


def _make_customer_dataframe() -> pd.DataFrame:
    rows = []
    segments = {
        "starter": {"age": 24, "income": 45_000, "region": "west", "plan": "basic"},
        "growth": {"age": 37, "income": 82_000, "region": "east", "plan": "pro"},
        "enterprise": {"age": 51, "income": 135_000, "region": "central", "plan": "enterprise"},
    }
    for segment, base in segments.items():
        for idx in range(20):
            rows.append(
                {
                    "age": base["age"] + (idx % 5),
                    "income": base["income"] + 1_500 * (idx % 7),
                    "region": base["region"] if idx % 4 else "south",
                    "plan": base["plan"],
                    "segment": segment,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
