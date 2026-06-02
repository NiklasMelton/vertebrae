import json

import numpy as np

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import PrecomputedExtractor


def test_precomputed_single_extractor_workflow_writes_reports(tmp_path, fake_overlapindex):
    embeddings = np.vstack(
        [
            np.random.default_rng(0).normal(loc=0, scale=0.1, size=(8, 4)),
            np.random.default_rng(1).normal(loc=4, scale=0.1, size=(8, 4)),
        ]
    )
    labels = np.array(["left"] * 8 + ["right"] * 8)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="embeddings"),
        stability_config=StabilityConfig(repeats=3),
        probe_config=ProbeConfig(enabled=False),
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
