"""Evaluate precomputed embeddings for a multi-label classification dataset."""

import numpy as np

from vertebrae import CacheConfig, Evaluator, ProbeConfig, StabilityConfig
from vertebrae.datasets import BenchmarkDataset
from vertebrae.extractors import PrecomputedExtractor


def main() -> None:
    rng = np.random.default_rng(7)
    embeddings = np.vstack(
        [
            rng.normal(loc=(-1.0, 0.0, 0.0), scale=0.15, size=(6, 3)),
            rng.normal(loc=(0.0, 1.0, 0.0), scale=0.15, size=(6, 3)),
            rng.normal(loc=(0.0, 0.0, 1.0), scale=0.15, size=(6, 3)),
        ]
    )
    labels = [
        ("outdoor", "animal"),
        ("outdoor", "animal"),
        ("outdoor",),
        ("animal",),
        ("outdoor", "vehicle"),
        ("animal", "vehicle"),
        ("indoor",),
        ("indoor",),
        ("indoor", "animal"),
        ("indoor", "vehicle"),
        ("animal",),
        ("vehicle",),
        ("outdoor", "vehicle"),
        ("vehicle",),
        ("outdoor",),
        ("indoor", "vehicle"),
        ("animal", "vehicle"),
        ("indoor", "animal"),
    ]
    dataset = BenchmarkDataset.from_embeddings(embeddings=embeddings, labels=labels)
    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="toy_multilabel_embeddings"),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=True),
        cache_config=CacheConfig(enabled=False),
    ).run()

    print(result.to_dataframe())


if __name__ == "__main__":
    main()
