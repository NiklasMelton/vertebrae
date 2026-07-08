"""Evaluate graph node and edge embeddings through the standard benchmark path."""

import numpy as np
from _common import ensure_output_dir, print_ranking

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import PrecomputedExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    rng = np.random.default_rng(13)
    node_embeddings = np.vstack(
        [
            rng.normal(loc=-1.5, scale=0.25, size=(12, 6)),
            rng.normal(loc=1.5, scale=0.25, size=(12, 6)),
        ]
    )
    node_ids = [f"node-{idx}" for idx in range(node_embeddings.shape[0])]
    node_labels = np.asarray(["source_like"] * 12 + ["sink_like"] * 12)

    node_dataset = BenchmarkDataset.from_node_embeddings(
        node_embeddings,
        node_labels,
        node_ids=node_ids,
        metadata={"example": "relational_node_embeddings"},
    )
    node_result = Evaluator(
        dataset=node_dataset,
        extractor=PrecomputedExtractor(name="node_embeddings"),
        scoring_config=OverlapScoringConfig(k=3, min_samples_per_cluster=4),
        stability_config=StabilityConfig(repeats=4, random_state=17),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    edge_index = [
        ("node-0", "node-1"),
        ("node-2", "node-3"),
        ("node-4", "node-16"),
        ("node-5", "node-17"),
        ("node-12", "node-13"),
        ("node-14", "node-15"),
        ("node-6", "node-18"),
        ("node-7", "node-19"),
    ]
    edge_labels = np.asarray(["within"] * 2 + ["across"] * 2 + ["within"] * 2 + ["across"] * 2)
    edge_dataset = BenchmarkDataset.from_edge_embeddings(
        labels=edge_labels,
        edge_index=edge_index,
        node_embeddings=node_embeddings,
        node_ids=node_ids,
        composition="abs_diff",
        metadata={"example": "relational_edge_embeddings"},
    )
    edge_result = Evaluator(
        dataset=edge_dataset,
        extractor=PrecomputedExtractor(name="edge_embeddings"),
        scoring_config=OverlapScoringConfig(k=2, min_samples_per_cluster=2),
        stability_config=StabilityConfig(repeats=4, random_state=19),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    node_result.save_json(str(output_dir / "relational_node_embeddings.json"))
    node_result.save_markdown(str(output_dir / "relational_node_embeddings.md"))
    edge_result.save_json(str(output_dir / "relational_edge_embeddings.json"))
    edge_result.save_markdown(str(output_dir / "relational_edge_embeddings.md"))

    print("Node embedding diagnostic")
    print_ranking(node_result)
    print("\nEdge embedding diagnostic")
    print_ranking(edge_result)
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
