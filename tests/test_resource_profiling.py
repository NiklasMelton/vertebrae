from pathlib import Path

import numpy as np
import pytest

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CallableMetric,
    EmbeddingCompressionConfig,
    EmbeddingOutputSpec,
    ResourceProfilingConfig,
)
from vertebrae.config import CacheConfig, EmbeddingConfig, SeparatixConfig, StabilityConfig
from vertebrae.extractors import CallableExtractor, MultiOutputExtractor
from vertebrae.profiling import (
    KerasResourceProfileAdapter,
    ONNXResourceProfileAdapter,
    TorchResourceProfileAdapter,
)
from vertebrae.reports.markdown_report import render_markdown_report


class FakeResourceAdapter:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.events = []

    def metadata(self):
        return {
            "backend": "fake",
            "device": "fake:0",
            "asynchronous": True,
            "precision": "float32",
        }

    def synchronize(self):
        self.events.append("synchronize")
        return True

    def reset_peak_device_memory(self):
        self.events.append("reset")
        return True

    def peak_device_memory(self):
        self.events.append("peak")
        return {
            "status": "measured",
            "backend": "fake",
            "device": "fake:0",
            "peak_allocated_bytes": 123,
            "peak_reserved_bytes": 456,
        }

    def model_footprint(self):
        return {
            "status": "measured",
            "parameter_count": 10,
            "parameter_bytes": 40,
            "buffer_bytes": 8,
        }

    def deployment_artifacts(self):
        return [str(self.checkpoint), str(self.checkpoint)]


class FakeParameter:
    def __init__(self, count, itemsize):
        self.count = count
        self.itemsize = itemsize

    def numel(self):
        return self.count

    def element_size(self):
        return self.itemsize


def _dataset():
    return BenchmarkDataset.from_arrays(
        np.arange(24, dtype=np.float32).reshape(8, 3),
        np.asarray(["a"] * 4 + ["b"] * 4),
        modality="tabular",
    )


def _benchmark(dataset, extractor, **kwargs):
    return Benchmark(
        dataset,
        extractors=[extractor],
        embedding_config=EmbeddingConfig(batch_size=3),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        resource_profiling_config=ResourceProfilingConfig(enabled=True),
        **kwargs,
    )


def test_resource_profiling_config_validates_bounds():
    with pytest.raises(ValueError, match="sample_interval"):
        ResourceProfilingConfig(host_sample_interval_seconds=0)
    with pytest.raises(ValueError, match="quality_tolerance"):
        ResourceProfilingConfig(quality_tolerance=-0.1)


def test_streaming_profile_records_calls_memory_model_and_embedding(tmp_path, fake_overlapindex):
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"checkpoint")
    adapter = FakeResourceAdapter(checkpoint)
    extractor = CallableExtractor(
        "profiled",
        lambda values: np.asarray(values, dtype=np.float32),
        modality="tabular",
        streaming_safe=True,
        resource_profile_adapter=adapter,
    )

    result = _benchmark(_dataset(), extractor, cache_config=CacheConfig(enabled=False)).run()
    item = result.extractor_results[0]
    profile = item.resource_profile

    assert profile is not None
    assert profile.status == "measured"
    assert profile.inference.status == "measured"
    assert profile.inference.first_call_seconds is not None
    assert profile.inference.warm_call_count == 2
    assert profile.inference.materialized_samples == 8
    assert profile.inference.batch_sizes == [3, 3, 2]
    assert profile.inference.throughput_samples_per_second > 0
    assert profile.host_memory.peak_rss_bytes >= profile.host_memory.baseline_rss_bytes
    assert profile.device_memory.peak_allocated_bytes == 123
    assert profile.model.parameter_count == 10
    assert profile.model.in_memory_bytes == 48
    assert profile.model.checkpoint_bytes == len(b"checkpoint")
    assert len(profile.model.artifacts) == 1
    assert profile.embedding.raw_bytes == 8 * 3 * 4
    assert profile.embedding.evaluated_bytes == 8 * 3 * 4
    assert profile.context["synchronization_status"] == "synchronized"
    assert adapter.events[0] == "reset"
    assert "peak" in adapter.events

    payload = result.to_dict()["extractor_results"][0]["resource_profile"]
    assert payload["embedding"]["bytes_per_embedding"] == 12.0
    frame = result.to_dataframe()
    assert frame.loc[0, "parameter_bytes"] == 40
    assert frame.loc[0, "evaluated_embedding_bytes"] == 96
    report = render_markdown_report(result)
    assert "Resource profile for quality-similar candidates" in report
    assert "Peak host RSS increase" in report


def test_profile_respects_embedding_cache(tmp_path, fake_overlapindex):
    calls = []

    def transform(values):
        calls.append(len(values))
        return np.asarray(values, dtype=np.float32)

    extractor = CallableExtractor(
        "cached-profile",
        transform,
        modality="tabular",
        streaming_safe=True,
    )
    cache = CacheConfig(enabled=True, cache_dir=str(tmp_path / "cache"))
    _benchmark(_dataset(), extractor, cache_config=cache).run()
    measured_calls = list(calls)

    second = _benchmark(_dataset(), extractor, cache_config=cache).run()
    profile = second.extractor_results[0].resource_profile

    assert calls == measured_calls
    assert profile is not None
    assert profile.inference.status == "not_measured_cache_hit"
    assert profile.embedding.evaluated_bytes == 96


def test_disabled_profile_is_absent(fake_overlapindex):
    result = Benchmark(
        _dataset(),
        extractors=[CallableExtractor("plain", lambda values: values)],
        cache_config=CacheConfig(enabled=False),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
    ).run()

    assert result.extractor_results[0].resource_profile is None
    assert result.to_dataframe().loc[0, "resource_profile_status"] == "disabled"


def test_quality_cohort_respects_lower_is_better(fake_overlapindex):
    metric = CallableMetric(
        "cost",
        lambda embeddings, labels: float(np.asarray(embeddings).mean()),
        higher_is_better=False,
    )
    result = Benchmark(
        _dataset(),
        extractors=[
            CallableExtractor("best", lambda values: values),
            CallableExtractor("near", lambda values: np.asarray(values) + 0.005),
        ],
        metrics=[metric],
        primary_metric="cost",
        cache_config=CacheConfig(enabled=False),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        resource_profiling_config=ResourceProfilingConfig(
            enabled=True,
            quality_tolerance=0.01,
        ),
    ).run()

    assert result.ranked_results()[0].name == "best"
    assert [item.name for item in result.quality_cohort()] == ["best", "near"]
    assert [item.name for item in result.quality_cohort(tolerance=0)] == ["best"]


def test_multi_output_shares_inference_and_tracks_compressed_storage(fake_overlapindex):
    calls = []

    def transform_many(values):
        calls.append(len(values))
        array = np.asarray(values, dtype=np.float32)
        return {"wide": array, "narrow": array[:, :2]}

    result = Benchmark(
        _dataset(),
        extractors=[
            MultiOutputExtractor(
                "multi",
                [EmbeddingOutputSpec("wide"), EmbeddingOutputSpec("narrow")],
                transform_many,
                streaming_safe=True,
            )
        ],
        compression_config=EmbeddingCompressionConfig(
            enabled=True,
            method="prefix_truncate",
            n_components=1,
            assume_matryoshka=True,
        ),
        embedding_config=EmbeddingConfig(batch_size=3),
        cache_config=CacheConfig(enabled=False),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        resource_profiling_config=ResourceProfilingConfig(enabled=True),
    ).run()

    assert calls == [3, 3, 2]
    wide, narrow = result.extractor_results
    assert wide.resource_profile.inference is narrow.resource_profile.inference
    assert wide.resource_profile.embedding.raw_bytes == 8 * 3 * 4
    assert narrow.resource_profile.embedding.raw_bytes == 8 * 2 * 4
    assert wide.resource_profile.embedding.evaluated_bytes == 8 * 1 * 4
    assert narrow.resource_profile.embedding.evaluated_bytes == 8 * 1 * 4


def test_builtin_model_footprint_adapters_use_explicit_state(tmp_path):
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"12345")

    class TorchModel:
        def parameters(self):
            return [FakeParameter(4, 4), FakeParameter(2, 2)]

        def buffers(self):
            return [FakeParameter(3, 1)]

    torch_extractor = type(
        "TorchLike",
        (),
        {"model": TorchModel(), "device": None},
    )()
    torch_adapter = TorchResourceProfileAdapter(torch_extractor, [str(checkpoint)])
    assert torch_adapter.model_footprint() == {
        "status": "measured",
        "parameter_count": 6,
        "parameter_bytes": 20,
        "buffer_bytes": 3,
    }
    assert torch_adapter.deployment_artifacts() == (str(checkpoint),)

    weight = type(
        "Weight",
        (),
        {"shape": (2, 3), "dtype": np.dtype("float32")},
    )()
    keras_extractor = type("KerasLike", (), {"model": type("M", (), {"weights": [weight]})()})()
    keras_adapter = KerasResourceProfileAdapter(keras_extractor, [str(checkpoint)])
    assert keras_adapter.model_footprint()["parameter_count"] == 6
    assert keras_adapter.model_footprint()["parameter_bytes"] == 24

    onnx_extractor = type(
        "ONNXLike",
        (),
        {"model_path": checkpoint, "providers": ["CPUExecutionProvider"]},
    )()
    onnx_adapter = ONNXResourceProfileAdapter(onnx_extractor)
    assert onnx_adapter.deployment_artifacts() == (str(checkpoint),)
    assert onnx_adapter.metadata()["backend"] == "onnxruntime"
