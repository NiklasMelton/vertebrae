"""Hugging Face text extractor API example.

Requires optional dependencies and a model available locally or from Hugging Face:

    poetry install -E hf
"""

from _common import CACHE_DIR, ensure_output_dir, print_ranking

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import HFTextExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    texts = [
        "The invoice is overdue and needs payment.",
        "Please send the receipt for the renewal.",
        "The service API timed out during deployment.",
        "The database worker queue is delayed.",
        "The dashboard export needs a better filter.",
        "The onboarding workflow needs clearer settings.",
    ]
    labels = ["finance", "finance", "platform", "platform", "product", "product"]
    dataset = BenchmarkDataset.from_arrays(texts, labels, modality="text")

    extractor = HFTextExtractor(
        name="distilbert_mean_pool",
        model_id="distilbert-base-uncased",
        pooling="mean",
        batch_size=2,
    )

    try:
        result = Evaluator(
            dataset=dataset,
            extractor=extractor,
            scoring_config=OverlapScoringConfig(k=1, min_samples_per_cluster=2),
            stability_config=StabilityConfig(repeats=3),
            probe_config=ProbeConfig(enabled=False),
            cache_config=CacheConfig(cache_dir=str(CACHE_DIR)),
        ).run()
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E hf")
        return

    result.save_markdown(str(output_dir / "hf_text_extractor.md"))
    print_ranking(result)
    print(f"\nReport written to {output_dir / 'hf_text_extractor.md'}")


if __name__ == "__main__":
    main()
