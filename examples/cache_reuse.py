"""Show how the local embedding cache is reused across benchmark runs."""

from _common import CACHE_DIR, ensure_output_dir, make_separated_blobs

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import CallableExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    X, labels = make_separated_blobs(samples_per_class=28, n_features=8, random_state=41)
    dataset = BenchmarkDataset.from_arrays(
        X,
        labels,
        modality="tabular",
        metadata={"example": "cache_reuse"},
        identity=DatasetIdentity.ephemeral(),
    )

    cache_config = CacheConfig(cache_dir=str(CACHE_DIR))
    extractor = CallableExtractor(
        name="expensive_domain_features",
        transform_fn=lambda values: values[:, :4] * 1.25,
        modality="tabular",
        # This lambda is intentionally local and therefore has no portable code
        # identity. Declare the transformation revision explicitly to opt into reuse.
        cache_identity="expensive-domain-features-v1",
    )
    kwargs = {
        "scoring_config": OverlapScoringConfig(k=3, min_samples_per_cluster=4),
        "stability_config": StabilityConfig(enabled=False),
        "cache_config": cache_config,
    }

    first = Evaluator(dataset=dataset, extractor=extractor, **kwargs).run()
    second = Evaluator(dataset=dataset, extractor=extractor, **kwargs).run()

    first_hit = first.extractor_results[0].embedding_metadata["cache_hit"]
    second_hit = second.extractor_results[0].embedding_metadata["cache_hit"]
    print(f"First run cache hit: {first_hit}")
    print(f"Second run cache hit: {second_hit}")

    second.save_markdown(str(output_dir / "cache_reuse.md"))
    print(f"\nReport written to {output_dir / 'cache_reuse.md'}")


if __name__ == "__main__":
    main()
