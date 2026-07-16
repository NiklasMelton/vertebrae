"""Sentence-transformers extractor API example.

Requires optional dependencies and a model available locally or from Hugging Face:

    poetry install -E hf
"""

from _common import ensure_output_dir, print_ranking

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import SentenceTransformerExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    texts = [
        "contract signature clause agreement",
        "legal filing agreement review",
        "invoice payment receipt balance",
        "billing renewal invoice refund",
        "kitten sleeps on the sofa",
        "puppy chased a ball outside",
    ]
    labels = ["legal", "legal", "finance", "finance", "animal", "animal"]
    dataset = BenchmarkDataset.from_arrays(
        texts, labels, modality="text", identity=DatasetIdentity.ephemeral()
    )

    extractor = SentenceTransformerExtractor(
        name="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=4,
        normalize_embeddings=True,
    )

    try:
        result = Evaluator(
            dataset=dataset,
            extractor=extractor,
            scoring_config=OverlapScoringConfig(k=1, min_samples_per_cluster=2),
            stability_config=StabilityConfig(repeats=3),
            # This example intentionally uses an unpinned remote model name.
            cache_config=CacheConfig(enabled=False),
        ).run()
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E hf")
        return

    result.save_markdown(str(output_dir / "sentence_transformer_extractor.md"))
    print_ranking(result)
    print(f"\nReport written to {output_dir / 'sentence_transformer_extractor.md'}")


if __name__ == "__main__":
    main()
