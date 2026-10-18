import json

import numpy as np
import pytest
from scipy import sparse

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    DatasetIdentity,
    LinearProbeConfig,
    LinearProbeMetric,
    LinearProbeScorer,
)
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.cli import main
from vertebrae.config import CacheConfig, SeparatixConfig, StabilityConfig
from vertebrae.execution import (
    ScoringJob,
    benchmark_result_from_artifacts,
    score_embedding_artifact,
)
from vertebrae.extractors import CallableExtractor
from vertebrae.reports.markdown_report import render_markdown_report


def _classification_problem(n_per_class=10):
    negative = np.tile(np.asarray([-1.0, 0.25]), (n_per_class, 1))
    positive = np.tile(np.asarray([1.0, -0.25]), (n_per_class, 1))
    return (
        np.vstack([negative, positive]),
        np.asarray(["negative"] * n_per_class + ["positive"] * n_per_class),
    )


def test_linear_probe_config_validates_protocol_options():
    with pytest.raises(ValueError, match="n_splits"):
        LinearProbeConfig(n_splits=1)
    with pytest.raises(ValueError, match="class_weight"):
        LinearProbeConfig(class_weight="unsupported")
    with pytest.raises(ValueError, match="regression_alpha"):
        LinearProbeConfig(regression_alpha=0.0)
    with pytest.raises(ValueError, match="primary_metric"):
        LinearProbeConfig(primary_metric="roc_auc")


def test_single_label_probe_is_oof_deterministic_sparse_and_group_aware():
    embeddings, labels = _classification_problem()
    order = np.asarray([index for pair in zip(range(10), range(10, 20)) for index in pair])
    embeddings = embeddings[order]
    labels = labels[order]
    groups = np.repeat(np.arange(10), 2)
    scorer = LinearProbeScorer(LinearProbeConfig(n_splits=5, random_state=7))

    first = scorer.score(sparse.csr_matrix(embeddings), labels, groups=groups)
    second = scorer.score(sparse.csr_matrix(embeddings), labels, groups=groups)

    assert first.score == pytest.approx(1.0)
    assert first.metrics == second.metrics
    assert first.primary_metric == "balanced_accuracy"
    assert first.metadata["evaluation_mode"] == "out_of_fold"
    assert first.metadata["cv_method"] == "stratified_group_kfold"
    assert first.metadata["group_aware"] is True
    assert first.metadata["n_splits"] == 5
    assert len(first.folds) == 5
    assert sum(fold["holdout_size"] for fold in first.folds) == len(labels)


def test_single_label_probe_preserves_typed_label_identity():
    labels = np.empty(8, dtype=object)
    labels[:] = [1, 1, 1, 1, True, True, True, True]
    embeddings = np.asarray([[-2.0], [-1.5], [-1.0], [-0.5], [0.5], [1.0], [1.5], [2.0]])

    result = LinearProbeScorer(LinearProbeConfig(n_splits=2)).score(embeddings, labels)

    assert len(result.per_class_scores) == 2
    assert len(result.metadata["label_catalog"]) == 2


def test_probe_supports_multilabel_and_regression_targets():
    indicator = np.asarray([[1, index % 2] for index in range(20)], dtype=np.uint8)
    multilabel_embeddings = np.column_stack(
        [indicator[:, 0] * 2.0 - 1.0, indicator[:, 1] * 2.0 - 1.0]
    )
    multilabel = LinearProbeScorer().score(
        multilabel_embeddings,
        indicator,
        target_metadata={"target_type": "multi_label", "label_names": ["base", "even"]},
    )

    values = np.linspace(-2.0, 2.0, 30)
    regression = LinearProbeMetric(
        LinearProbeConfig(primary_metric="rmse", normalize_embeddings=False)
    ).score(
        np.column_stack([values, values**2]),
        values,
        target_metadata={"target_type": "regression", "target_names": ["value"]},
    )

    assert multilabel.primary_metric == "macro_f1"
    assert set(multilabel.metrics) == {
        "macro_f1",
        "micro_f1",
        "sample_jaccard",
        "subset_accuracy",
    }
    assert regression.kind == "linear_probe"
    assert regression.diagnostics["primary_metric"] == "rmse"
    assert regression.higher_is_better is False
    assert regression.score >= 0.0
    assert "value" in regression.diagnostics["per_target_scores"]


def test_benchmark_scores_both_metrics_and_can_rank_by_linear_probe(fake_overlapindex):
    embeddings, labels = _classification_problem()
    dataset = BenchmarkDataset.from_embeddings(
        embeddings,
        labels,
        identity=DatasetIdentity.ephemeral(),
    )
    extractors = [
        CallableExtractor("constant", lambda values: np.ones_like(values)),
        CallableExtractor("separable", np.asarray),
    ]
    common = {
        "dataset": dataset,
        "extractors": extractors,
        "metrics": [LinearProbeMetric()],
        "stability_config": StabilityConfig(enabled=False),
        "separatix_config": SeparatixConfig(enabled=False),
        "cache_config": CacheConfig(enabled=False),
    }

    default_rank = Benchmark(**common).run()
    probe_rank = Benchmark(**common, primary_metric="linear_probe").run()

    assert default_rank.extractor_results[0].primary_metric_name == "overlap"
    assert all(
        set(item.metrics) == {"overlap", "linear_probe"}
        for item in probe_rank.extractor_results
    )
    assert probe_rank.ranked_results()[0].name == "separable"
    assert probe_rank.ranked_results()[0].linear_probe.score == pytest.approx(1.0)
    frame = probe_rank.to_dataframe()
    assert frame.loc[0, "linear_probe_metric"] == "balanced_accuracy"
    assert frame.loc[0, "linear_probe_score"] == pytest.approx(1.0)
    report = render_markdown_report(probe_rank)
    assert "linear_probe_score" in report
    assert "#### Linear probe" in report


def test_linear_probe_artifact_round_trip(tmp_path, fake_overlapindex):
    embeddings, labels = _classification_problem()
    store = LocalArtifactStore(str(tmp_path))
    store.put_artifact(
        "embeddings/probe",
        embeddings,
        {"n_samples": len(labels), "embedding_dim": 2, "extractor_name": "probe"},
    )
    store.put_labels_artifact(
        "labels/probe",
        labels,
        {"n_samples": len(labels), "target_type": "single_label"},
    )

    artifact = score_embedding_artifact(
        ScoringJob(
            embedding_key="embeddings/probe",
            labels_key="labels/probe",
            output_key="scores/probe",
            metrics=[LinearProbeMetric(LinearProbeConfig(n_splits=4))],
            primary_metric="linear_probe",
        ),
        store,
    )
    restored = benchmark_result_from_artifacts(artifact["output_key"], store)

    assert artifact["primary_metric"] == "linear_probe"
    assert artifact["metrics"]["linear_probe"]["score"] == pytest.approx(1.0)
    assert artifact["metrics"]["linear_probe"]["metadata"]["n_splits"] == 4
    assert restored["extractor_results"][0]["primary_metric_name"] == "linear_probe"
    assert set(restored["extractor_results"][0]["metrics"]) == {
        "overlap",
        "linear_probe",
    }


def test_cli_can_score_and_select_the_linear_probe(tmp_path, capsys, fake_overlapindex):
    embeddings, labels = _classification_problem()
    store = LocalArtifactStore(str(tmp_path))
    store.put_artifact(
        "embeddings/cli-probe",
        embeddings,
        {"n_samples": len(labels), "embedding_dim": 2},
    )
    store.put_labels_artifact(
        "labels/cli-probe",
        labels,
        {"n_samples": len(labels), "target_type": "single_label"},
    )

    status = main(
        [
            "score",
            "--cache-dir",
            str(tmp_path),
            "--embedding-key",
            "embeddings/cli-probe",
            "--labels-key",
            "labels/cli-probe",
            "--linear-probe",
            "--linear-probe-config-json",
            '{"n_splits": 4}',
            "--primary-metric",
            "linear_probe",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["primary_metric"] == "linear_probe"
    assert payload["metrics"]["linear_probe"]["metadata"]["n_splits"] == 4
