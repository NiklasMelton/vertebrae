import io
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vertebrae import (
    BenchmarkDataset,
    CacheConfig,
    CallableMetric,
    CallableSpatialExtractor,
    CallableStructuredExtractor,
    ConsoleReporter,
    DatasetIdentity,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    EvaluationContext,
    EvaluationHistory,
    EvaluationHistoryConfig,
    EvaluationRecord,
    ExecutionConfig,
    LabelViewConfig,
    LocalBackend,
    MemoryConfig,
    MetricResult,
    OverlapScoringConfig,
    RepresentationMonitor,
    ResourceProfilingConfig,
    SegmentationConfig,
    SegmentationDataset,
    SeparatixConfig,
    SpatialLayout,
    SpatialOutputSpec,
    StabilityConfig,
    StructuredOutputSpec,
    StructuredUnitAligner,
    TargetView,
    TargetViewConfig,
    UnitAnnotation,
)
from vertebrae.extractors import MultiOutputExtractor
from vertebrae.extractors.base import EmbeddingOutputSpec
from vertebrae.results import (
    RESULT_ROW_STATIC_COLUMNS,
    BenchmarkResult,
    ExtractorResult,
)


def _dataset():
    return BenchmarkDataset.from_arrays(
        np.arange(24, dtype=float).reshape(8, 3),
        ["left"] * 4 + ["right"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )


def _monitor_options():
    return {
        "stability_config": StabilityConfig(enabled=False),
        "separatix_config": SeparatixConfig(enabled=False),
    }


class MutableExtractor:
    name = "mutable"
    modality = "tabular"
    extractor_type = "mutable_test"
    streaming_safe = False

    def __init__(self):
        self.scale = 1.0
        self.calls = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.calls += 1
        return np.asarray(X, dtype=float) * self.scale

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def recipe(self):
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "cache_identity": "mutable-monitor-test-v1",
            "cache_safe": True,
        }


class FailingExtractor(MutableExtractor):
    name = "failing"

    def transform(self, X):
        raise RuntimeError("evaluation exploded")


class ToggleExtractor(MutableExtractor):
    def __init__(self):
        super().__init__()
        self.fail = False

    def transform(self, X):
        if self.fail:
            raise RuntimeError("evaluation exploded")
        return super().transform(X)


def _mean_metric(embeddings, labels, **kwargs):
    return MetricResult(name="mean", score=float(np.asarray(embeddings).mean()))


def test_evaluation_context_normalizes_identifiers_and_metadata():
    context = EvaluationContext(
        epoch=np.int64(2),
        global_step=7,
        timestamp="2025-01-02T03:04:05-05:00",
        checkpoint=" checkpoint.pt ",
        metadata={"loss": np.float32(0.25)},
        recorded_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    assert context.epoch == 2
    assert context.timestamp == "2025-01-02T08:04:05+00:00"
    assert context.checkpoint == " checkpoint.pt "
    assert context.metadata == {"loss": pytest.approx(0.25)}
    assert context.recorded_at == "2025-01-01T00:00:00+00:00"
    assert "recorded_at" not in context.identity_payload()
    assert "metadata" not in context.identity_payload()

    with pytest.raises(ValueError, match="At least one"):
        EvaluationContext()
    with pytest.raises(TypeError, match="epoch"):
        EvaluationContext(epoch=True)
    with pytest.raises(ValueError, match="timezone"):
        EvaluationContext(timestamp="2025-01-01T00:00:00")
    with pytest.raises(TypeError, match="keys"):
        EvaluationContext(epoch=1, metadata={1: "bad"})
    with pytest.raises(TypeError, match="keys"):
        EvaluationContext(epoch=1, metadata={"nested": {1: "bad"}})
    with pytest.raises(TypeError, match="non-finite"):
        EvaluationContext(epoch=1, metadata={"loss": np.nan})


@pytest.mark.parametrize(
    "identifier",
    [
        {"snapshot_id": "snapshot-a"},
        {"epoch": 0},
        {"global_step": 0},
        {"timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc)},
        {"checkpoint": Path("checkpoint.pt")},
        {
            "snapshot_id": "snapshot-a",
            "epoch": 2,
            "global_step": 7,
            "timestamp": "2025-01-01T00:00:00Z",
            "checkpoint": "checkpoint.pt",
        },
    ],
)
def test_evaluation_context_accepts_each_identifier_and_combinations(identifier):
    context = EvaluationContext(**identifier)

    assert context.identity_payload()


@pytest.mark.parametrize(
    "identifier, error",
    [
        ({"snapshot_id": " "}, ValueError),
        ({"epoch": -1}, ValueError),
        ({"global_step": 1.5}, TypeError),
        ({"checkpoint": ""}, ValueError),
        ({"timestamp": "not-a-timestamp"}, ValueError),
    ],
)
def test_evaluation_context_rejects_invalid_identifiers(identifier, error):
    with pytest.raises(error):
        EvaluationContext(**identifier)


def test_history_config_validates_storage_path_detail_and_resume(tmp_path):
    assert EvaluationHistoryConfig() == EvaluationHistoryConfig(
        storage="memory",
        detail="summary",
    )
    with pytest.raises(ValueError, match="storage"):
        EvaluationHistoryConfig(storage="remote")
    with pytest.raises(ValueError, match="detail"):
        EvaluationHistoryConfig(detail="compact")
    with pytest.raises(ValueError, match="only valid"):
        EvaluationHistoryConfig(path=tmp_path / "history.jsonl")
    with pytest.raises(ValueError, match="required"):
        EvaluationHistoryConfig(storage="disk")
    with pytest.raises(ValueError, match="resume"):
        EvaluationHistoryConfig(resume=True)


def test_monitor_collects_fresh_mutable_results_and_duplicate_contexts(
    tmp_path,
    fake_overlapindex,
):
    extractor = MutableExtractor()
    monitor = RepresentationMonitor(
        _dataset(),
        [extractor],
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path / "cache")),
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="prefix_truncate",
            n_components=2,
            assume_matryoshka=True,
        ),
        metrics=[CallableMetric(name="mean", metric_fn=_mean_metric)],
        primary_metric="mean",
        **_monitor_options(),
    )

    first = monitor.evaluate(epoch=1, metadata={"loss": 2.0})
    extractor.scale = 3.0
    with pytest.warns(UserWarning, match="duplicates"):
        second = monitor.evaluate(epoch=1, metadata={"loss": 1.0})

    assert first is not None and second is not None
    assert extractor.calls == 2
    assert first.extractor_results[0].primary_score != second.extractor_results[0].primary_score
    assert first.extractor_results[0].embedding_metadata["cache_hit"] is False
    assert second.extractor_results[0].embedding_metadata["cache_hit"] is False
    assert first.extractor_results[0].compression_metadata["cache_hit"] is False
    assert second.extractor_results[0].compression_metadata["cache_hit"] is False
    assert monitor.cache_config.force_recompute is True
    frame = monitor.history.to_dataframe()
    assert frame["evaluation_index"].tolist() == [0, 1]
    assert frame["context_metadata.loss"].tolist() == [2.0, 1.0]
    assert frame["metric.mean"].nunique() == 2
    assert frame["warning_count"].gt(0).all()
    assert frame["warnings"].map(bool).all()


def test_monitor_forwards_compression_target_and_label_view_options(fake_overlapindex):
    target_dataset = _dataset().with_target_views(
        [
            TargetView(
                name="alternate",
                targets=["near"] * 4 + ["far"] * 4,
            )
        ]
    )
    target_monitor = RepresentationMonitor(
        target_dataset,
        [MutableExtractor()],
        target_view_config=TargetViewConfig(enabled=True, views=("alternate",)),
        compression_configs=[
            EmbeddingCompressionConfig(),
            EmbeddingCompressionConfig(
                enabled=True,
                method="prefix_truncate",
                n_components=2,
                assume_matryoshka=True,
            ),
        ],
        **_monitor_options(),
    )

    target_monitor.evaluate(epoch=0)
    target_frame = target_monitor.history.to_dataframe()
    assert set(target_frame["target_view"]) == {"alternate"}
    assert set(target_frame["compression_method"]) == {"none", "prefix_truncate"}

    labels = np.array(["husky"] * 6 + ["pug"] * 6 + ["sedan"] * 6 + ["suv"] * 6)
    paths = (
        [("animal", "dog", "husky")] * 6
        + [("animal", "dog", "pug")] * 6
        + [("vehicle", "car", "sedan")] * 6
        + [("vehicle", "car", "suv")] * 6
    )
    label_dataset = BenchmarkDataset.from_arrays(
        np.arange(72, dtype=float).reshape(24, 3),
        labels,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    ).with_label_hierarchy(
        paths,
        level_names=("domain", "family", "leaf"),
    )
    label_monitor = RepresentationMonitor(
        label_dataset,
        [MutableExtractor()],
        label_view_config=LabelViewConfig(
            enabled=True,
            hierarchy_levels=("domain", "family"),
        ),
        **_monitor_options(),
    )

    label_monitor.evaluate(epoch=0)
    assert set(label_monitor.history.to_dataframe()["label_view"]) == {
        "domain",
        "family",
    }


def test_monitor_rows_include_stability_and_separatix_run_and_skip_states(
    fake_overlapindex,
):
    ran = RepresentationMonitor(
        _dataset(),
        [MutableExtractor()],
        stability_config=StabilityConfig(repeats=2),
        separatix_config=SeparatixConfig(overlap_threshold=0.80),
    )
    skipped = RepresentationMonitor(
        _dataset(),
        [MutableExtractor()],
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(overlap_threshold=0.81),
    )

    ran.evaluate(epoch=0)
    skipped.evaluate(epoch=0)
    ran_row = ran.history.to_dataframe().iloc[0]
    skipped_row = skipped.history.to_dataframe().iloc[0]

    assert ran_row["stability_mode"] == "prototype"
    assert ran_row["stability_repeats"] == 2
    assert ran_row["stability_interval_level"] == pytest.approx(0.95)
    assert ran_row["stability_min"] <= ran_row["stability_max"]
    assert ran_row["stability_interval_lower"] <= ran_row["stability_interval_upper"]
    assert bool(ran_row["separatix_ran"]) is True
    assert ran_row["separatix_recommendation"] == "smooth_nonlinear_recommended"
    assert bool(skipped_row["separatix_ran"]) is False
    assert "below the configured threshold" in skipped_row["separatix_skip_reason"]


def test_monitor_supports_resource_profiling_and_execution_backend(
    tmp_path,
    fake_overlapindex,
):
    profiled = RepresentationMonitor(
        _dataset(),
        [MutableExtractor()],
        resource_profiling_config=ResourceProfilingConfig(enabled=True),
        **_monitor_options(),
    )
    result = profiled.evaluate(epoch=0)
    row = profiled.history.to_dataframe().iloc[0]
    assert result is not None
    assert row["resource_profile_status"] == "measured"
    assert row["peak_host_rss_bytes"] >= 0

    dispatched = RepresentationMonitor(
        _dataset(),
        [MutableExtractor()],
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path / "artifacts")),
        execution=LocalBackend(),
        execution_config=ExecutionConfig(total_shards=2),
        **_monitor_options(),
    )
    dispatched_result = dispatched.evaluate(epoch=0)
    assert dispatched_result is not None
    assert dispatched_result.metadata["execution"]["backend"] == "LocalBackend"
    assert dispatched_result.extractor_results[0].embedding_metadata["cache_hit"] is False
    assert "runtime.embedding_seconds" in dispatched.history.to_dataframe()


def test_monitor_history_keeps_invalid_aggregates_with_null_rank(fake_overlapindex):
    def invalid_metric(embeddings, labels, **kwargs):
        return MetricResult(
            name="invalid",
            score=0.5,
            metadata={"aggregate_valid": False},
        )

    monitor = RepresentationMonitor(
        _dataset(),
        [MutableExtractor()],
        metrics=[CallableMetric("invalid", invalid_metric)],
        primary_metric="invalid",
        **_monitor_options(),
    )

    monitor.evaluate(epoch=0)
    row = monitor.history.to_dataframe().iloc[0]

    assert pd.isna(row["rank"])
    assert not bool(row["aggregate_valid"])


def test_monitor_multi_output_rows_include_layer_identity(fake_overlapindex):
    extractor = MultiOutputExtractor(
        name="layers",
        output_specs=[
            EmbeddingOutputSpec(name="middle", hidden_layer=2, pooling="mean"),
            EmbeddingOutputSpec(name="final", hidden_layer=4, pooling="cls"),
        ],
        transform_many_fn=lambda value: {
            "middle": np.asarray(value)[:, :2],
            "final": np.asarray(value),
        },
    )
    monitor = RepresentationMonitor(_dataset(), [extractor], **_monitor_options())

    monitor.evaluate(epoch=0)
    frame = monitor.history.latest_dataframe()

    assert frame["output_name"].tolist() == ["middle", "final"]
    assert frame["hidden_layer"].tolist() == [2, 4]
    assert frame["pooling"].tolist() == ["mean", "cls"]
    assert set(frame["parent_extractor"]) == {"layers"}
    assert frame["rank"].notna().all()
    assert "metric.overlap" in frame
    assert "runtime.embedding_seconds" in frame


def test_history_memory_full_and_disk_round_trip(tmp_path):
    context = EvaluationContext(epoch=3, metadata={"z_loss": 0.5, "a_loss": 0.25})
    record = EvaluationRecord(
        context=context,
        status="success",
        evaluation_index=0,
        rows=[
            {
                "z_result": "last alphabetically",
                "extractor": "layer",
                "primary_score": 0.75,
                "a_result": "first alphabetically",
            }
        ],
        benchmark_result={"dataset_summary": {}, "extractor_results": []},
    )
    memory = EvaluationHistory(EvaluationHistoryConfig(detail="full"))
    memory.append(record)

    path = tmp_path / "history.jsonl"
    disk = EvaluationHistory(EvaluationHistoryConfig(storage="disk", path=path, detail="full"))
    disk.append(record)
    loaded = EvaluationHistory.load(path)

    pd.testing.assert_frame_equal(
        memory.to_dataframe(),
        disk.to_dataframe(),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        disk.to_dataframe(),
        loaded.to_dataframe(),
        check_dtype=False,
    )
    assert list(loaded.iter_records())[0].benchmark_result == record.benchmark_result
    assert loaded.latest_dataframe().loc[0, "evaluation_index"] == 0
    with pytest.raises(RuntimeError, match="read-only"):
        loaded.append(
            EvaluationRecord(
                context=EvaluationContext(epoch=4),
                status="success",
                evaluation_index=1,
                rows=[{}],
            )
        )


def test_summary_history_omits_full_payload_traceback_and_nulls_failure_fields():
    history = EvaluationHistory(EvaluationHistoryConfig(detail="summary"))
    history.append(
        EvaluationRecord(
            context=EvaluationContext(epoch=0),
            status="success",
            evaluation_index=0,
            rows=[{"primary_score": 0.75}],
            benchmark_result={"complete": "payload"},
        )
    )
    history.append(
        EvaluationRecord(
            context=EvaluationContext(epoch=1),
            status="failure",
            evaluation_index=1,
            rows=[{}],
            error_type="RuntimeError",
            error_message="bad",
            error_traceback="traceback",
        )
    )

    records = list(history.iter_records())
    assert records[0].benchmark_result is None
    assert records[1].error_traceback is None
    failure = history.latest_dataframe().iloc[0]
    assert pd.isna(failure["primary_score"])
    assert pd.isna(failure["extractor"])
    assert pd.isna(failure["probe_status"])
    assert pd.isna(failure["compression_precision"])
    assert pd.isna(failure["runtime.embedding_seconds"])
    assert pd.isna(failure["peak_host_rss_bytes"])
    assert failure["error_type"] == "RuntimeError"


def test_real_monitor_memory_disk_and_latest_failure_schema_parity(
    tmp_path,
    fake_overlapindex,
):
    dataset = _dataset()
    memory_extractor = ToggleExtractor()
    disk_extractor = ToggleExtractor()
    common = {
        "error_policy": "continue",
        "metrics": [CallableMetric(name="mean", metric_fn=_mean_metric)],
        "primary_metric": "mean",
        **_monitor_options(),
    }
    memory = RepresentationMonitor(dataset, [memory_extractor], **common)
    disk = RepresentationMonitor(
        dataset,
        [disk_extractor],
        history_config=EvaluationHistoryConfig(
            storage="disk",
            path=tmp_path / "parity.jsonl",
        ),
        **common,
    )
    memory.evaluate(epoch=0)
    disk.evaluate(epoch=0)
    memory_extractor.fail = True
    disk_extractor.fail = True
    assert memory.evaluate(epoch=1) is None
    assert disk.evaluate(epoch=1) is None

    memory_frame = memory.history.to_dataframe()
    disk_frame = disk.history.to_dataframe()
    assert memory_frame.columns.tolist() == disk_frame.columns.tolist()
    assert memory.history.latest_dataframe().columns.tolist() == memory_frame.columns.tolist()
    assert disk.history.latest_dataframe().columns.tolist() == disk_frame.columns.tolist()
    for name in (
        "metric.overlap",
        "metric.mean",
        "runtime.embedding_seconds",
        "probe_status",
        "peak_host_rss_bytes",
    ):
        assert name in memory_frame
        assert pd.isna(memory.history.latest_dataframe().loc[0, name])


def test_disk_history_resume_validation_and_index_continuation(tmp_path):
    path = tmp_path / "history.jsonl"
    history = EvaluationHistory(
        EvaluationHistoryConfig(storage="disk", path=path, detail="summary")
    )
    first = EvaluationRecord(
        context=EvaluationContext(epoch=1),
        status="failure",
        evaluation_index=0,
        rows=[{}],
        error_type="RuntimeError",
        error_message="bad",
    )
    history.append(first)

    with pytest.raises(FileExistsError):
        EvaluationHistory(EvaluationHistoryConfig(storage="disk", path=path, detail="summary"))
    with pytest.raises(ValueError, match="detail mismatch"):
        EvaluationHistory(
            EvaluationHistoryConfig(
                storage="disk",
                path=path,
                detail="full",
                resume=True,
            )
        )

    resumed = EvaluationHistory(
        EvaluationHistoryConfig(
            storage="disk",
            path=path,
            detail="summary",
            resume=True,
        )
    )
    assert resumed.next_evaluation_index == 1
    assert resumed.contains_context(EvaluationContext(epoch=1))
    resumed.append(
        EvaluationRecord(
            context=EvaluationContext(global_step=8),
            status="success",
            evaluation_index=1,
            rows=[{"primary_score": 0.8}],
        )
    )
    assert resumed.to_dataframe()["evaluation_index"].tolist() == [0, 1]

    payload = path.read_text(encoding="utf-8")
    assert payload.endswith("\n")
    lines = payload.splitlines()
    assert json.loads(lines[0])["schema_version"] == 2


def test_monitor_resume_requires_matching_protocol_and_preserves_file(
    tmp_path,
    fake_overlapindex,
):
    path = tmp_path / "monitor.jsonl"
    dataset = _dataset()
    first = RepresentationMonitor(
        dataset,
        [MutableExtractor()],
        history_config=EvaluationHistoryConfig(storage="disk", path=path),
        **_monitor_options(),
    )
    first.evaluate(epoch=0)

    resumed = RepresentationMonitor(
        dataset,
        [MutableExtractor()],
        history_config=EvaluationHistoryConfig(
            storage="disk",
            path=path,
            resume=True,
        ),
        **_monitor_options(),
    )
    assert resumed.history.next_evaluation_index == 1
    resumed.evaluate(epoch=1)
    assert resumed.history.to_dataframe()["evaluation_index"].unique().tolist() == [0, 1]

    original = path.read_bytes()
    with pytest.raises(ValueError, match="protocol mismatch.*benchmark"):
        RepresentationMonitor(
            dataset,
            [MutableExtractor()],
            history_config=EvaluationHistoryConfig(
                storage="disk",
                path=path,
                resume=True,
            ),
            stability_config=StabilityConfig(repeats=3),
            separatix_config=SeparatixConfig(enabled=False),
        )
    assert path.read_bytes() == original

    with pytest.raises(ValueError, match="protocol mismatch.*error_policy"):
        RepresentationMonitor(
            dataset,
            [MutableExtractor()],
            history_config=EvaluationHistoryConfig(
                storage="disk",
                path=path,
                resume=True,
            ),
            error_policy="continue",
            **_monitor_options(),
        )
    assert path.read_bytes() == original

    changed_dataset = BenchmarkDataset.from_arrays(
        np.arange(24, dtype=float).reshape(8, 3) + 1.0,
        ["left"] * 4 + ["right"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    with pytest.raises(ValueError, match="protocol mismatch.*dataset"):
        RepresentationMonitor(
            changed_dataset,
            [MutableExtractor()],
            history_config=EvaluationHistoryConfig(
                storage="disk",
                path=path,
                resume=True,
            ),
            **_monitor_options(),
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "change,component",
    [
        ("extractor", "extractors"),
        ("metrics", "benchmark"),
        ("compression", "benchmark"),
        ("execution", "benchmark"),
        ("scoring", "benchmark"),
        ("views", "benchmark"),
        ("separatix", "benchmark"),
        ("segmentation", "benchmark"),
        ("alignment", "benchmark"),
        ("resource", "benchmark"),
        ("embedding", "benchmark"),
        ("memory", "benchmark"),
        ("cache", "benchmark"),
    ],
)
def test_monitor_resume_rejects_additional_protocol_changes(
    tmp_path,
    change,
    component,
):
    path = tmp_path / f"{change}.jsonl"
    dataset = _dataset()
    RepresentationMonitor(
        dataset,
        [MutableExtractor()],
        history_config=EvaluationHistoryConfig(storage="disk", path=path),
        **_monitor_options(),
    )
    original = path.read_bytes()
    extractor = MutableExtractor()
    options = _monitor_options()
    if change == "extractor":
        extractor.name = "changed"
    elif change == "metrics":
        options.update(
            metrics=[CallableMetric(name="mean", metric_fn=_mean_metric)],
            primary_metric="mean",
        )
    elif change == "compression":
        options["compression_config"] = EmbeddingCompressionConfig(
            enabled=True,
            method="prefix_truncate",
            n_components=2,
            assume_matryoshka=True,
        )
    elif change == "execution":
        options.update(
            execution=LocalBackend(),
            execution_config=ExecutionConfig(total_shards=2),
        )
    elif change == "scoring":
        options["scoring_config"] = OverlapScoringConfig(k=2)
    elif change == "views":
        options["target_view_config"] = TargetViewConfig(enabled=True)
    elif change == "separatix":
        options["separatix_config"] = SeparatixConfig(enabled=True, overlap_threshold=0.9)
    elif change == "segmentation":
        options["segmentation_config"] = SegmentationConfig(coverage_threshold=0.8)
    elif change == "alignment":
        options["structured_aligners"] = {
            "mutable": StructuredUnitAligner(
                "changed",
                lambda embeddings, annotation: (embeddings, annotation),
                cache_identity="monitor-test-aligner",
            )
        }
    elif change == "resource":
        options["resource_profiling_config"] = ResourceProfilingConfig(enabled=True)
    elif change == "embedding":
        options["embedding_config"] = EmbeddingConfig(batch_size=4)
    elif change == "memory":
        options["memory_config"] = MemoryConfig(subsample_rate=0.75)
    else:
        options["cache_config"] = CacheConfig(
            enabled=True,
            cache_dir=str(tmp_path / "cache"),
        )

    with pytest.raises(ValueError, match=rf"protocol mismatch.*{component}"):
        RepresentationMonitor(
            dataset,
            [extractor],
            history_config=EvaluationHistoryConfig(
                storage="disk",
                path=path,
                resume=True,
            ),
            **options,
        )
    assert path.read_bytes() == original


def test_disk_history_rejects_malformed_and_truncated_records(tmp_path):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"record_type":"manifest"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        EvaluationHistory.load(malformed)

    truncated = tmp_path / "truncated.jsonl"
    history = EvaluationHistory(EvaluationHistoryConfig(storage="disk", path=truncated))
    history.append(
        EvaluationRecord(
            context=EvaluationContext(epoch=0),
            status="success",
            evaluation_index=0,
            rows=[{}],
        )
    )
    truncated.write_text(
        truncated.read_text(encoding="utf-8").rstrip("\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="truncated"):
        EvaluationHistory.load(truncated)

    nonstandard = tmp_path / "nonstandard.jsonl"
    strict_history = EvaluationHistory(EvaluationHistoryConfig(storage="disk", path=nonstandard))
    strict_history.append(
        EvaluationRecord(
            context=EvaluationContext(epoch=0),
            status="success",
            evaluation_index=0,
            rows=[{}],
        )
    )
    payload = nonstandard.read_text(encoding="utf-8")
    payload = payload.replace('"error_message": null', '"error_message": NaN')
    nonstandard.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        EvaluationHistory.load(nonstandard)


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluation_index", True),
        ("evaluation_index", 0.5),
        ("evaluation_index", "0"),
        ("rows", {}),
        ("benchmark_result", []),
    ],
)
def test_disk_history_rejects_wrong_record_field_types(tmp_path, field, value):
    source = tmp_path / "source.jsonl"
    history = EvaluationHistory(EvaluationHistoryConfig(storage="disk", path=source))
    history.append(
        EvaluationRecord(
            context=EvaluationContext(epoch=0),
            status="success",
            evaluation_index=0,
            rows=[{"primary_score": 0.5}],
        )
    )
    lines = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    lines[1][field] = value
    target = tmp_path / f"wrong-{field}-{type(value).__name__}.jsonl"
    target.write_text(
        "".join(json.dumps(line, allow_nan=False) + "\n" for line in lines),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid record"):
        EvaluationHistory.load(target)


@pytest.mark.parametrize("version", [True, 1.0, 1])
def test_disk_history_rejects_non_v2_manifest_versions(tmp_path, version):
    source = tmp_path / "source.jsonl"
    EvaluationHistory(EvaluationHistoryConfig(storage="disk", path=source))
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["schema_version"] = version
    target = tmp_path / f"version-{type(version).__name__}-{version}.jsonl"
    target.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version|schema version"):
        EvaluationHistory.load(target)


def test_disk_history_rejects_unknown_fields_and_detail_invariants(tmp_path):
    summary_path = tmp_path / "summary.jsonl"
    summary = EvaluationHistory(
        EvaluationHistoryConfig(storage="disk", path=summary_path, detail="summary")
    )
    summary.append(
        EvaluationRecord(
            context=EvaluationContext(epoch=0),
            status="success",
            evaluation_index=0,
            rows=[{"primary_score": 0.5}],
        )
    )
    lines = [
        json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines()
    ]
    lines[1]["unexpected"] = True
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(
        "".join(json.dumps(line) + "\n" for line in lines),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid record"):
        EvaluationHistory.load(unknown)

    lines[1].pop("unexpected")
    lines[1]["benchmark_result"] = {"unexpected": "full payload"}
    summary_full = tmp_path / "summary-full.jsonl"
    summary_full.write_text(
        "".join(json.dumps(line) + "\n" for line in lines),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid record"):
        EvaluationHistory.load(summary_full)

    full_path = tmp_path / "full.jsonl"
    full = EvaluationHistory(
        EvaluationHistoryConfig(storage="disk", path=full_path, detail="full")
    )
    with pytest.raises(ValueError, match="benchmark_result"):
        full.append(
            EvaluationRecord(
                context=EvaluationContext(epoch=0),
                status="success",
                evaluation_index=0,
                rows=[{"primary_score": 0.5}],
            )
        )


def test_disk_history_rejects_unknown_or_missing_manifest_and_context_fields(tmp_path):
    source = tmp_path / "source.jsonl"
    history = EvaluationHistory(EvaluationHistoryConfig(storage="disk", path=source))
    history.append(
        EvaluationRecord(
            context=EvaluationContext(epoch=0),
            status="success",
            evaluation_index=0,
            rows=[{"primary_score": 0.5}],
        )
    )
    original = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
    ]
    mutations = []
    manifest_extra = deepcopy(original)
    manifest_extra[0]["unexpected"] = True
    mutations.append(("manifest-extra", manifest_extra))
    manifest_missing = deepcopy(original)
    manifest_missing[0].pop("created_at")
    mutations.append(("manifest-missing", manifest_missing))
    context_extra = deepcopy(original)
    context_extra[1]["context"]["unexpected"] = True
    mutations.append(("context-extra", context_extra))
    context_missing = deepcopy(original)
    context_missing[1]["context"].pop("metadata")
    mutations.append(("context-missing", context_missing))

    for name, lines in mutations:
        target = tmp_path / f"{name}.jsonl"
        target.write_text(
            "".join(json.dumps(line, allow_nan=False) + "\n" for line in lines),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            EvaluationHistory.load(target)


def test_evaluation_record_rejects_inconsistent_status_payloads():
    with pytest.raises(ValueError, match="empty result row"):
        EvaluationRecord(
            context=EvaluationContext(epoch=0),
            status="failure",
            evaluation_index=0,
            rows=[{"primary_score": 1.0}],
            error_type="RuntimeError",
            error_message="bad",
        )
    with pytest.raises(ValueError, match="failure fields"):
        EvaluationRecord(
            context=EvaluationContext(epoch=0),
            status="success",
            evaluation_index=0,
            rows=[{"primary_score": 1.0}],
            benchmark_result={},
            error_traceback="unexpected traceback",
        )

    full = EvaluationHistory(EvaluationHistoryConfig(detail="full"))
    with pytest.raises(ValueError, match="error_traceback"):
        full.append(
            EvaluationRecord(
                context=EvaluationContext(epoch=0),
                status="failure",
                evaluation_index=0,
                rows=[{}],
                error_type="RuntimeError",
                error_message="bad",
            )
        )


def test_monitor_error_policies_reporter_and_persistence_failures(
    fake_overlapindex,
    monkeypatch,
):
    raising = RepresentationMonitor(_dataset(), [FailingExtractor()], **_monitor_options())
    with pytest.raises(RuntimeError, match="exploded"):
        raising.evaluate(epoch=1)
    failure = list(raising.history.iter_records())[0]
    assert failure.status == "failure"
    assert failure.error_type == "RuntimeError"
    assert failure.error_traceback is None

    continuing = RepresentationMonitor(
        _dataset(),
        [FailingExtractor()],
        error_policy="continue",
        history_config=EvaluationHistoryConfig(detail="full"),
        **_monitor_options(),
    )
    assert continuing.evaluate(global_step=2) is None
    assert list(continuing.history.iter_records())[0].error_traceback

    class BadReporter:
        def report(self, record):
            raise RuntimeError("report failed")

    successful = RepresentationMonitor(
        _dataset(),
        [MutableExtractor()],
        reporters=[BadReporter()],
        **_monitor_options(),
    )
    with pytest.warns(RuntimeWarning, match="report failed"):
        assert successful.evaluate(snapshot_id="ok") is not None
    assert len(successful.history) == 1

    persistence_failure = RepresentationMonitor(
        _dataset(),
        [MutableExtractor()],
        error_policy="continue",
        **_monitor_options(),
    )

    def fail_append(record):
        raise OSError("history unavailable")

    monkeypatch.setattr(persistence_failure.history, "append", fail_append)
    with pytest.raises(OSError, match="history unavailable"):
        persistence_failure.evaluate(epoch=0)


def test_console_reporter_prints_success_and_failure():
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, precision=2)
    reporter.report(
        EvaluationRecord(
            context=EvaluationContext(epoch=1),
            status="success",
            evaluation_index=0,
            rows=[
                {
                    "output_name": "layer",
                    "hidden_layer": 2,
                    "primary_metric": "overlap",
                    "primary_score": 0.812,
                    "overlap_score": 0.812,
                    "separatix_skip_reason": "below threshold",
                },
                {
                    "output_name": "final",
                    "primary_metric": "overlap",
                    "primary_score": 0.9,
                    "overlap_score": 0.9,
                    "separatix_ran": True,
                    "separatix_recommendation": "linear_likely_sufficient",
                },
            ],
        )
    )
    reporter.report(
        EvaluationRecord(
            context=EvaluationContext(global_step=2),
            status="failure",
            evaluation_index=1,
            rows=[{}],
            error_type="RuntimeError",
            error_message="bad",
        )
    )

    output = stream.getvalue()
    assert "epoch=1" in output
    assert "layer=2" in output
    assert "overlap=0.81" in output
    assert "below threshold" in output
    assert "linear_likely_sufficient" in output
    assert "RuntimeError: bad" in output


def test_result_dataframe_can_include_invalid_aggregates():
    valid = _extractor_result("valid", 0.8, True)
    invalid = _extractor_result("invalid", 0.9, False)
    result = BenchmarkResult({}, [invalid, valid], [])

    assert result.to_dataframe()["extractor"].tolist() == ["valid"]
    frame = result.to_dataframe(include_invalid=True)
    assert frame["extractor"].tolist() == ["invalid", "valid"]
    assert pd.isna(frame.loc[0, "rank"])
    assert frame.loc[0, "aggregate_valid"] is np.False_ or not frame.loc[0, "aggregate_valid"]
    assert frame.loc[1, "rank"] == 1


def test_result_row_builder_covers_the_canonical_static_schema():
    result = BenchmarkResult({}, [_extractor_result("valid", 0.8, True)], [])

    row = result._tabular_rows(include_invalid=True)[0]

    assert set(RESULT_ROW_STATIC_COLUMNS).issubset(row)
    assert "metric.custom" in row


def test_monitor_supports_structured_and_segmentation_benchmarks(
    tmp_path,
    fake_overlapindex,
):
    annotations = [
        UnitAnnotation(labels=["x", "y"], unit_ids=[f"{name}:0", f"{name}:1"])
        for name in ("a", "b", "c", "d")
    ]
    structured_dataset = BenchmarkDataset.from_arrays(
        np.array(["a", "b", "c", "d"], dtype=object),
        ["left", "left", "right", "right"],
        modality="text",
        identity=DatasetIdentity.ephemeral(),
    ).with_unit_annotations(
        annotations,
        unit_type="token",
        task_family="sequence",
    )
    structured_values = [np.eye(2, dtype=float) for _ in range(4)]
    structured_extractor = CallableStructuredExtractor(
        "structured",
        transform_fn=lambda batch: structured_values[: len(batch)],
        output_specs=[
            StructuredOutputSpec(
                "tokens",
                unit_type="token",
                hidden_layer=2,
            )
        ],
    )
    structured = RepresentationMonitor(
        structured_dataset,
        [structured_extractor],
        cache_config=CacheConfig(enabled=False, cache_dir=str(tmp_path / "structured")),
        **_monitor_options(),
    )
    assert structured.evaluate(epoch=0) is not None
    structured_frame = structured.history.to_dataframe()
    assert structured_frame.loc[0, "task_family"] == "sequence"
    assert structured_frame.loc[0, "output_name"] == "tokens"
    assert structured_frame.loc[0, "hidden_layer"] == 2

    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    masks = np.array(
        [
            [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1]],
        ]
    )
    segmentation_dataset = SegmentationDataset.from_arrays(
        images,
        masks,
        class_metadata={
            0: {"background": True},
            1: {"is_thing": True},
            2: {"is_thing": False},
        },
        identity=DatasetIdentity.ephemeral(),
    )
    spatial_values = np.arange(24, dtype=float).reshape(2, 2, 2, 3)
    spatial_extractor = CallableSpatialExtractor(
        "spatial",
        transform_fn=lambda batch: spatial_values[: len(batch)],
        output_specs=[
            SpatialOutputSpec(
                "layer",
                SpatialLayout(2, 2),
                hidden_layer=3,
            )
        ],
    )
    segmentation = RepresentationMonitor(
        segmentation_dataset,
        [spatial_extractor],
        cache_config=CacheConfig(enabled=False, cache_dir=str(tmp_path / "segmentation")),
        segmentation_config=SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include_excluded",
        ),
        **_monitor_options(),
    )
    assert segmentation.evaluate(epoch=0) is not None
    segmentation_frame = segmentation.history.to_dataframe()
    assert segmentation_frame.loc[0, "output_name"] == "layer"
    assert segmentation_frame.loc[0, "hidden_layer"] == 3


def _extractor_result(name, score, aggregate_valid):
    metric = MetricResult(
        name="custom",
        score=score,
        metadata={"aggregate_valid": aggregate_valid},
    )
    return ExtractorResult(
        name=name,
        extractor_type="test",
        stability=None,
        separatix=None,
        embedding_metadata={"embedding_dim": 2},
        compression_metadata={"method": "none"},
        runtime={},
        warnings=[],
        recommendation="",
        metrics={"custom": metric},
        primary_metric_name="custom",
    )
