import json

import numpy as np

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.extractors import PrecomputedExtractor


def test_precomputed_single_extractor_workflow_writes_reports(tmp_path, fake_overlapindex):
    embeddings = np.vstack(
        [
            np.random.default_rng(0).normal(loc=0, scale=0.1, size=(8, 4)),
            np.random.default_rng(1).normal(loc=4, scale=0.1, size=(8, 4)),
        ]
    )
    labels = np.array(["left"] * 8 + ["right"] * 8)
    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="embeddings"),
        stability_config=StabilityConfig(repeats=3),
        cache_config=CacheConfig(enabled=False),
    ).run()

    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "report.md"
    result.save_json(str(json_path))
    result.save_markdown(str(markdown_path))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["extractor_results"][0]["name"] == "embeddings"
    assert "Ranking" in markdown_path.read_text(encoding="utf-8")
    assert len(fake_overlapindex.calls) == 4


def test_benchmark_metadata_excludes_probe_config(fake_overlapindex):
    embeddings = np.arange(48, dtype=float).reshape(16, 3)
    labels = np.array(["left"] * 8 + ["right"] * 8)
    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="embeddings"),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    item = result.extractor_results[0]
    assert item.separatix is not None
    assert "probe_config" not in result.metadata


def test_node_embedding_dataset_runs_through_existing_benchmark(tmp_path, fake_overlapindex):
    embeddings = np.vstack(
        [
            np.random.default_rng(2).normal(loc=0, scale=0.1, size=(6, 3)),
            np.random.default_rng(3).normal(loc=3, scale=0.1, size=(6, 3)),
        ]
    )
    labels = np.array(["low"] * 6 + ["high"] * 6)
    dataset = BenchmarkDataset.from_node_embeddings(
        embeddings,
        labels,
        node_ids=[f"node-{idx}" for idx in range(12)],
        identity=DatasetIdentity.ephemeral(),
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="node-embeddings"),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()
    markdown_path = tmp_path / "node_report.md"
    result.save_markdown(str(markdown_path))

    item = result.extractor_results[0]
    assert item.overlap.metadata["target_type"] == "single_label"
    assert result.dataset_summary["metadata"]["relational_unit"] == "node"
    assert "Relational unit: node" in markdown_path.read_text(encoding="utf-8")
