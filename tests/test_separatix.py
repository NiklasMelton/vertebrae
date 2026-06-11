import json

import numpy as np
import pytest
from scipy import sparse

from vertebrae import BenchmarkDataset, Evaluator, SeparatixConfig
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import PrecomputedExtractor
from vertebrae.scoring.separatix import SeparatixScorer


def test_separatix_config_validates_threshold_and_limits():
    with pytest.raises(ValueError, match="overlap_threshold"):
        SeparatixConfig(overlap_threshold=1.1)
    with pytest.raises(ValueError, match="budget"):
        SeparatixConfig(budget="tiny")
    with pytest.raises(ValueError, match="max_samples"):
        SeparatixConfig(max_samples=0)
    with pytest.raises(ValueError, match="max_dense_bytes"):
        SeparatixConfig(max_dense_bytes=0)
    with pytest.raises(ValueError, match="n_jobs"):
        SeparatixConfig(n_jobs=0)


def test_separatix_scorer_preserves_sparse_input_and_records_normalization(fake_separatix):
    embeddings = sparse.csr_matrix(np.eye(6))
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    scorer = SeparatixScorer(
        config=SeparatixConfig(max_dense_bytes=1_048_576),
        overlap_config=OverlapScoringConfig(normalize_embeddings=True),
    )

    result = scorer.score(embeddings, labels)

    assert result.ran is True
    assert result.metadata["normalized_embeddings"] is True
    assert result.metadata["sparse_input"] is True
    assert result.metadata["max_dense_mb"] == 1
    assert fake_separatix.ComplexityProfiler.calls[0]["kind"] == "diagnose"
    assert fake_separatix.ComplexityProfiler.calls[0]["is_sparse"] is True


def test_benchmark_runs_and_skips_separatix_by_threshold(tmp_path, fake_overlapindex):
    embeddings = np.arange(48, dtype=float).reshape(16, 3)
    labels = np.array(["a"] * 8 + ["b"] * 8)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    baseline_kwargs = dict(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="embeddings"),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
    )

    ran_result = Evaluator(
        separatix_config=SeparatixConfig(overlap_threshold=0.80),
        **baseline_kwargs,
    ).run()
    skipped_result = Evaluator(
        separatix_config=SeparatixConfig(overlap_threshold=0.81),
        **baseline_kwargs,
    ).run()

    ran_item = ran_result.extractor_results[0]
    skipped_item = skipped_result.extractor_results[0]
    frame = ran_result.to_dataframe()
    assert ran_item.separatix is not None
    assert ran_item.separatix.ran is True
    assert skipped_item.separatix is not None
    assert skipped_item.separatix.ran is False
    assert "below the configured threshold" in (skipped_item.separatix.skipped_reason or "")
    assert bool(frame.loc[0, "separatix_ran"]) is True
    assert frame.loc[0, "separatix_recommendation"] == "smooth_nonlinear_recommended"
    assert frame.loc[0, "separatix_confidence"] == "high"


def test_reports_include_separatix_content(tmp_path, fake_overlapindex):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(12, 4))
    labels = np.array(["a"] * 6 + ["b"] * 6)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="dense"),
        separatix_config=SeparatixConfig(overlap_threshold=0.80),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "report.md"
    result.save_json(str(json_path))
    result.save_markdown(str(markdown_path))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    extractor_payload = payload["extractor_results"][0]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert extractor_payload["separatix"]["ran"] is True
    assert extractor_payload["separatix"]["report"]["recommendation"] == (
        "smooth_nonlinear_recommended"
    )
    assert "Separatix complexity diagnostic" in markdown
    assert "smooth_nonlinear_recommended" in markdown
    assert "Probe signal is strong." in markdown
    assert "| 0.9100 |" in markdown


def test_native_probes_remain_available_when_explicitly_enabled(fake_overlapindex):
    rng = np.random.default_rng(2)
    embeddings = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(10, 4)),
            rng.normal(loc=1.0, scale=0.2, size=(10, 4)),
        ]
    )
    labels = np.array(["a"] * 10 + ["b"] * 10)
    dataset = BenchmarkDataset.from_embeddings(embeddings, labels)

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="dense"),
        separatix_config=SeparatixConfig(overlap_threshold=0.80),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=True),
        cache_config=CacheConfig(enabled=False),
    ).run()

    item = result.extractor_results[0]
    frame = result.to_dataframe()
    assert item.probes is not None
    assert item.probes["enabled"] is True
    assert set(item.probes["results"]) == {"knn", "logistic_regression"}
    assert frame.loc[0, "probe_accuracy"] not in ("", None)
