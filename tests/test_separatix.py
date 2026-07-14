import json

import numpy as np
import pytest
from scipy import sparse

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator, SeparatixConfig
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import PrecomputedExtractor
from vertebrae.scoring.separatix import (
    SeparatixScorer,
    probe_summary_for_result,
    summarize_probe_diagnostics,
)


def test_separatix_config_validates_threshold_and_limits():
    with pytest.raises(ValueError, match="overlap_threshold"):
        SeparatixConfig(overlap_threshold=1.1)
    with pytest.raises(ValueError, match="regression_overlap_threshold"):
        SeparatixConfig(regression_overlap_threshold=1.1)
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


def test_separatix_scorer_passes_multilabel_target_mode(fake_separatix):
    embeddings = np.arange(18, dtype=float).reshape(6, 3)
    labels = [
        ("red", "round"),
        ("red",),
        ("round",),
        ("red", "sweet"),
        ("round", "sweet"),
        ("sweet",),
    ]
    scorer = SeparatixScorer(overlap_config=OverlapScoringConfig(normalize_embeddings=False))

    result = scorer.score(embeddings, labels)

    call = fake_separatix.ComplexityProfiler.calls[-1]
    assert result.metadata["target_type"] == "multi_label"
    assert result.metadata["label_names"] == ("red", "round", "sweet")
    assert call["kind"] == "diagnose"
    assert call["target_mode"] == "multilabel"
    assert call["y_shape"] == [6, 3]


def test_separatix_scorer_passes_groups_without_serializing_ids(fake_separatix):
    embeddings = np.arange(24, dtype=float).reshape(8, 3)
    labels = np.array(["a"] * 4 + ["b"] * 4)
    groups = np.array(["image-a"] * 2 + ["image-b"] * 2 + ["image-c"] * 2 + ["image-d"] * 2)

    result = SeparatixScorer().score(embeddings, labels, groups=groups)

    call = fake_separatix.ComplexityProfiler.calls[-1]
    assert call["groups"].tolist() == groups.tolist()
    assert result.metadata["grouped"] is True
    assert result.metadata["n_groups"] == 4
    assert result.probe_summary["evaluation"]["grouped"] is True
    assert result.probe_summary["evaluation"]["n_groups"] == 4
    assert "image-a" not in json.dumps(result.to_dict())


def test_separatix_scorer_passes_regression_target_mode_and_mlp_settings(fake_separatix):
    embeddings = np.arange(18, dtype=float).reshape(6, 3)
    labels = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

    result = SeparatixScorer(
        config=SeparatixConfig(
            mlp_probes=True,
            mlp_device="cpu",
            mlp_trigger_skill_threshold=0.7,
            mlp_min_improvement=0.03,
            mlp_max_parameters=1000,
        ),
        overlap_config=OverlapScoringConfig(normalize_embeddings=False),
    ).score(
        embeddings,
        labels,
        target_type="regression",
        target_names=["score"],
    )

    call = fake_separatix.ComplexityProfiler.calls[-1]
    assert result.metadata["target_type"] == "regression"
    assert result.metadata["target_names"] == ("score",)
    assert call["target_mode"] == "regression"
    assert call["mlp_probes"] is True
    assert call["mlp_max_parameters"] == 1000
    assert result.report["metrics"]["mlp_probes"]["status"] == "executed"
    assert result.probe_summary["primary_metric"] == {"name": "r2", "value": 0.84}
    assert result.probe_summary["metrics"] == {"r2": 0.84, "mae": 0.11, "rmse": 0.15}


def test_benchmark_runs_and_skips_separatix_by_threshold(tmp_path, fake_overlapindex):
    embeddings = np.arange(48, dtype=float).reshape(16, 3)
    labels = np.array(["a"] * 8 + ["b"] * 8)
    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    baseline_kwargs = dict(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="embeddings"),
        stability_config=StabilityConfig(enabled=False),
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
    assert skipped_item.separatix.probe_summary["status"] == "skipped"
    assert bool(frame.loc[0, "separatix_ran"]) is True
    assert frame.loc[0, "separatix_recommendation"] == "smooth_nonlinear_recommended"
    assert frame.loc[0, "separatix_confidence"] == "high"


def test_reports_include_separatix_content(tmp_path, fake_overlapindex):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(12, 4))
    labels = np.array(["a"] * 6 + ["b"] * 6)
    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="dense"),
        separatix_config=SeparatixConfig(overlap_threshold=0.80),
        stability_config=StabilityConfig(enabled=False),
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
    assert extractor_payload["separatix"]["probe_summary"]["primary_metric"] == {
        "name": "balanced_accuracy",
        "value": 0.89,
    }
    assert "Separatix complexity diagnostic" in markdown
    assert "smooth_nonlinear_recommended" in markdown
    assert "Probe signal is strong." in markdown
    assert "| 0.9100 |" in markdown


def test_dataframe_uses_target_appropriate_separatix_probe_fields(fake_overlapindex):
    rng = np.random.default_rng(2)
    embeddings = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(10, 4)),
            rng.normal(loc=1.0, scale=0.2, size=(10, 4)),
        ]
    )
    labels = np.array(["a"] * 10 + ["b"] * 10)
    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="dense"),
        separatix_config=SeparatixConfig(overlap_threshold=0.80),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    item = result.extractor_results[0]
    frame = result.to_dataframe()
    assert item.separatix is not None
    assert item.separatix.ran is True
    assert "probe_accuracy" not in frame.columns
    assert frame.loc[0, "probe_status"] == "executed"
    assert frame.loc[0, "best_probe"] == "smooth_poly"
    assert frame.loc[0, "probe_metric"] == "balanced_accuracy"
    assert frame.loc[0, "probe_score"] == 0.89
    assert frame.loc[0, "probe_metrics"]["accuracy"] == 0.91
    assert frame.loc[0, "probe_linear_score"] == 0.82
    assert frame.loc[0, "probe_nonlinear_score"] == 0.89
    assert frame.loc[0, "probe_nonlinear_delta"] == pytest.approx(0.07)
    assert frame.loc[0, "probe_evaluation_mode"] == "cross_validation"
    assert bool(frame.loc[0, "probe_sampled"]) is False


def test_multilabel_benchmark_runs_separatix(
    tmp_path,
    fake_overlapindex,
):
    embeddings = np.arange(36, dtype=float).reshape(12, 3)
    labels = [
        ("red", "round"),
        ("red",),
        ("round",),
        ("red", "sweet"),
        ("round", "sweet"),
        ("sweet",),
        ("red", "round"),
        ("red",),
        ("round",),
        ("red", "sweet"),
        ("round", "sweet"),
        ("sweet",),
    ]
    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="multilabel_dense"),
        separatix_config=SeparatixConfig(overlap_threshold=0.80),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
    ).run()

    item = result.extractor_results[0]
    frame = result.to_dataframe()
    markdown_path = tmp_path / "multilabel_report.md"
    result.save_markdown(str(markdown_path))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert result.dataset_summary["target_type"] == "multi_label"
    assert item.overlap.metadata["target_type"] == "multi_label"
    assert item.separatix is not None
    assert item.separatix.ran is True
    assert item.separatix.metadata["target_type"] == "multi_label"
    assert frame.loc[0, "target_type"] == "multi_label"
    assert frame.loc[0, "probe_metric"] == "macro_f1"
    assert frame.loc[0, "probe_score"] == 0.82
    assert frame.loc[0, "probe_metrics"] == {
        "micro_f1": 0.86,
        "macro_f1": 0.82,
        "sample_jaccard": 0.77,
    }
    assert "accuracy" not in frame.loc[0, "probe_metrics"]
    assert "Target type: multi_label" in markdown
    assert "| macro_f1 | 0.8200 |" in markdown


def test_regression_benchmark_uses_regression_threshold(
    tmp_path,
    fake_overlapindex,
):
    embeddings = np.arange(18, dtype=float).reshape(6, 3)
    targets = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    dataset = BenchmarkDataset.from_embeddings(
        embeddings,
        targets,
        target_type="regression",
        target_names=["score"],
        identity=DatasetIdentity.ephemeral(),
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="regression_dense"),
        separatix_config=SeparatixConfig(regression_overlap_threshold=0.63),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
    ).run()

    item = result.extractor_results[0]
    assert item.overlap.metadata["target_type"] == "regression"
    assert item.overlap.score == 0.62
    assert item.separatix is not None
    assert item.separatix.ran is False
    assert "below the configured threshold" in (item.separatix.skipped_reason or "")


def test_reports_include_separatix_mlp_status(tmp_path, fake_overlapindex):
    embeddings = np.arange(24, dtype=float).reshape(8, 3)
    labels = np.array(["a"] * 4 + ["b"] * 4)
    dataset = BenchmarkDataset.from_embeddings(
        embeddings, labels, identity=DatasetIdentity.ephemeral()
    )

    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name="dense"),
        separatix_config=SeparatixConfig(overlap_threshold=0.80, mlp_probes=True),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    markdown_path = tmp_path / "report.md"
    result.save_markdown(str(markdown_path))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert "MLP status" in markdown
    assert "executed" in markdown
    assert "MLP reason" in markdown


def test_disabled_separatix_has_probe_summary_without_result():
    summary = probe_summary_for_result(None)

    assert summary["status"] == "disabled"
    assert summary["best_probe"] is None
    assert summary["skip_reason"] == "Separatix diagnostics were disabled."


def test_multilabel_summary_does_not_invent_primary_metric():
    summary = summarize_probe_diagnostics(
        {
            "metrics": {
                "baseline": {"best_probe": "linear", "best_probe_score": 0.72},
                "probes": {
                    "linear": {
                        "micro_f1": 0.76,
                        "macro_f1": 0.72,
                        "sample_jaccard": 0.65,
                    }
                },
            }
        },
        target_type="multi_label",
        grouped=False,
        n_groups=None,
    )

    assert summary["primary_metric"] is None
    assert summary["metrics"] == {
        "micro_f1": 0.76,
        "macro_f1": 0.72,
        "sample_jaccard": 0.65,
    }
