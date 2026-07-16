"""Evaluate a scikit-learn text feature pipeline on a pandas dataframe."""

import pandas as pd
from _common import ensure_output_dir, print_ranking
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import SklearnExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    df = _make_support_ticket_dataframe()

    dataset = BenchmarkDataset.from_dataframe(
        df,
        input_col="ticket_text",
        label_col="team",
        modality="text",
        metadata={"example": "sklearn_text_pipeline"},
        identity=DatasetIdentity.ephemeral(),
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("svd", TruncatedSVD(n_components=8, random_state=13)),
            ("normalize", Normalizer()),
        ]
    )

    result = Evaluator(
        dataset=dataset,
        extractor=SklearnExtractor(name="tfidf_svd", pipeline=pipeline),
        scoring_config=OverlapScoringConfig(k=3, min_samples_per_cluster=4),
        stability_config=StabilityConfig(repeats=4, random_state=19),
        cache_config=CacheConfig(enabled=False),
    ).run()

    result.save_json(str(output_dir / "sklearn_text_pipeline.json"))
    result.save_markdown(str(output_dir / "sklearn_text_pipeline.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


def _make_support_ticket_dataframe() -> pd.DataFrame:
    templates = {
        "billing": [
            "invoice payment renewal subscription charge receipt",
            "billing statement refund tax invoice account",
            "credit card failed payment invoice support",
        ],
        "platform": [
            "api latency deployment logs outage service",
            "server timeout endpoint error monitoring release",
            "database queue worker incident performance",
        ],
        "product": [
            "feature request workflow dashboard export",
            "user onboarding settings navigation feedback",
            "report filter chart workspace permissions",
        ],
    }
    rows = []
    for team, texts in templates.items():
        for idx in range(18):
            base = texts[idx % len(texts)]
            rows.append(
                {
                    "ticket_text": f"{base} customer {idx % 5} priority {idx % 3}",
                    "team": team,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
