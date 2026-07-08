"""Evaluate a set of precomputed embeddings and write JSON/Markdown reports."""

from _common import ensure_output_dir, make_separated_blobs, print_ranking

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import PrecomputedExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    embeddings, labels = make_separated_blobs(samples_per_class=40, n_features=10)

    dataset = BenchmarkDataset.from_embeddings(
        embeddings=embeddings,
        labels=labels,
        metadata={"example": "precomputed_embeddings"},
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="frozen_backbone_embeddings"),
        scoring_config=OverlapScoringConfig(k=4, min_samples_per_cluster=5),
        stability_config=StabilityConfig(repeats=5, random_state=11),
        cache_config=CacheConfig(enabled=False),
    ).run()

    result.save_json(str(output_dir / "precomputed_embeddings.json"))
    result.save_markdown(str(output_dir / "precomputed_embeddings.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
