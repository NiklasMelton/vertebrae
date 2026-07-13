import sys
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
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.config import CacheConfig, EmbeddingConfig, SeparatixConfig, StabilityConfig
from vertebrae.extractors import (
    CallableExtractor,
    GraphModelExtractor,
    HFAudioExtractor,
    HFMultimodalExtractor,
    HFTextExtractor,
    HFTimeSeriesExtractor,
    HFVideoExtractor,
    HFVisionExtractor,
    JAXFlaxExtractor,
    KerasExtractor,
    MultiOutputExtractor,
    OpenCLIPExtractor,
    SentenceTransformerExtractor,
    SigLIPExtractor,
    TFHubExtractor,
    TimmVisionExtractor,
    TorchvisionVisionExtractor,
)
from vertebrae.profiling import (
    AdapterOperationResult,
    BaseResourceProfileAdapter,
    DeploymentArtifact,
    DeviceMemoryMeasurement,
    JAXResourceProfileAdapter,
    KerasResourceProfileAdapter,
    ModelFootprintMeasurement,
    ONNXResourceProfileAdapter,
    ResourceAdapterMetadata,
    ResourceProfiler,
    TensorFlowResourceProfileAdapter,
    TorchResourceProfileAdapter,
)
from vertebrae.reports.markdown_report import render_markdown_report


class FakeResourceAdapter(BaseResourceProfileAdapter):
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.events = []

    def metadata(self):
        return ResourceAdapterMetadata(
            backend="fake",
            device="fake:0",
            asynchronous=True,
            weight_dtypes=("float32",),
        )

    def synchronize(self):
        self.events.append("synchronize")
        return AdapterOperationResult("succeeded")

    def reset_peak_device_memory(self):
        self.events.append("reset")
        return AdapterOperationResult("succeeded")

    def peak_device_memory(self):
        self.events.append("peak")
        return DeviceMemoryMeasurement(
            status="measured",
            backend="fake",
            device="fake:0",
            baseline_allocated_bytes=23,
            baseline_reserved_bytes=56,
            peak_allocated_bytes=123,
            peak_reserved_bytes=456,
            measurement_scope="profile_window",
        )

    def model_footprint(self):
        return ModelFootprintMeasurement(
            status="measured",
            parameter_count=10,
            parameter_bytes=40,
            trainable_parameter_count=8,
            trainable_parameter_bytes=32,
            buffer_bytes=8,
        )

    def deployment_artifacts(self):
        return [DeploymentArtifact(str(self.checkpoint)), DeploymentArtifact(str(self.checkpoint))]


class FakeParameter:
    def __init__(self, count, itemsize, *, device=None, dtype=None):
        self.count = count
        self.itemsize = itemsize
        self.device = device
        self.dtype = dtype

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
    assert profile.device_memory.peak_allocated_increase_bytes == 100
    assert profile.device_memory.measurement_scope == "profile_window"
    assert profile.model.parameter_count == 10
    assert profile.model.trainable_parameter_count == 8
    assert profile.model.in_memory_bytes == 48
    assert profile.model.checkpoint_bytes == len(b"checkpoint")
    assert profile.model.status == "measured"
    assert profile.model.parameter_status == "measured"
    assert profile.model.checkpoint_status == "measured"
    assert len(profile.model.artifacts) == 2
    assert profile.model.artifacts[1].status == "duplicate"
    assert profile.embedding.raw_bytes == 8 * 3 * 4
    assert profile.embedding.evaluated_bytes == 8 * 3 * 4
    assert profile.context["synchronization_status"] == "synchronized"
    assert adapter.events[:2] == ["synchronize", "reset"]
    assert "peak" in adapter.events

    payload = result.to_dict()["extractor_results"][0]["resource_profile"]
    assert payload["embedding"]["bytes_per_embedding"] == 12.0
    assert payload["device_memory"]["measurement_scope"] == "profile_window"
    frame = result.to_dataframe()
    assert frame.loc[0, "parameter_bytes"] == 40
    assert frame.loc[0, "trainable_parameter_bytes"] == 32
    assert frame.loc[0, "model_buffer_bytes"] == 8
    assert frame.loc[0, "model_in_memory_bytes"] == 48
    assert frame.loc[0, "device_memory_scope"] == "profile_window"
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
    torch_footprint = torch_adapter.model_footprint()
    assert torch_footprint.parameter_count == 6
    assert torch_footprint.parameter_bytes == 20
    assert torch_footprint.buffer_bytes == 3
    assert torch_footprint.in_memory_bytes == 23
    assert torch_adapter.deployment_artifacts() == (DeploymentArtifact(str(checkpoint)),)

    weight = type(
        "Weight",
        (),
        {"shape": (2, 3), "dtype": np.dtype("float32")},
    )()
    keras_extractor = type("KerasLike", (), {"model": type("M", (), {"weights": [weight]})()})()
    keras_adapter = KerasResourceProfileAdapter(keras_extractor, [str(checkpoint)])
    assert keras_adapter.model_footprint().parameter_count == 6
    assert keras_adapter.model_footprint().parameter_bytes == 24

    onnx_extractor = type(
        "ONNXLike",
        (),
        {"model_path": checkpoint, "providers": ["CPUExecutionProvider"]},
    )()
    onnx_adapter = ONNXResourceProfileAdapter(onnx_extractor)
    assert onnx_adapter.deployment_artifacts() == (
        DeploymentArtifact(str(checkpoint), role="checkpoint"),
    )
    assert onnx_adapter.metadata().backend == "onnxruntime"


def test_torch_adapter_resolves_model_device_and_reports_allocator_delta():
    events = []

    class Device:
        def __init__(self, value):
            self.value = str(value)
            self.type = self.value.split(":", 1)[0]

        def __str__(self):
            return self.value

    class Cuda:
        @staticmethod
        def synchronize(device):
            events.append(("sync", str(device)))

        @staticmethod
        def memory_allocated(device):
            return 100

        @staticmethod
        def memory_reserved(device):
            return 200

        @staticmethod
        def reset_peak_memory_stats(device):
            events.append(("reset", str(device)))

        @staticmethod
        def max_memory_allocated(device):
            return 160

        @staticmethod
        def max_memory_reserved(device):
            return 280

    torch = type("Torch", (), {"device": Device, "cuda": Cuda})()
    parameter = FakeParameter(4, 4, device="cuda:1", dtype="torch.float32")
    model = type(
        "Model", (), {"parameters": lambda self: [parameter], "buffers": lambda self: []}
    )()
    extractor = type("Extractor", (), {"device": None, "model": model})()
    adapter = TorchResourceProfileAdapter(extractor, torch_loader=lambda: torch)

    assert adapter.metadata().device == "cuda:1"
    assert adapter.metadata().device_resolution == "model_state"
    assert adapter.reset_peak_device_memory().status == "succeeded"
    assert adapter.synchronize().status == "succeeded"
    memory = adapter.peak_device_memory()
    assert memory.baseline_allocated_bytes == 100
    assert memory.peak_allocated_bytes == 160
    assert events == [("reset", "cuda:1"), ("sync", "cuda:1")]


def test_keras_footprint_counts_frozen_weights_as_parameters():
    trainable = type("Weight", (), {"shape": (2, 3), "dtype": np.dtype("float32")})()
    state = type("Weight", (), {"shape": (3,), "dtype": np.dtype("float64")})()
    model = type(
        "Model",
        (),
        {
            "trainable_weights": [trainable],
            "non_trainable_weights": [state],
            "weights": [trainable, state],
        },
    )()
    extractor = type("KerasLike", (), {"model": model})()

    footprint = KerasResourceProfileAdapter(extractor).model_footprint()

    assert footprint.parameter_count == 9
    assert footprint.parameter_bytes == 48
    assert footprint.trainable_parameter_count == 6
    assert footprint.trainable_parameter_bytes == 24
    assert footprint.buffer_bytes is None
    assert footprint.in_memory_bytes == 48


def test_fully_frozen_keras_model_keeps_total_parameter_footprint():
    frozen = type("Weight", (), {"shape": (4, 5), "dtype": np.dtype("float32")})()
    model = type("Model", (), {"trainable_weights": [], "weights": [frozen]})()
    extractor = type("KerasLike", (), {"model": model})()

    footprint = KerasResourceProfileAdapter(extractor).model_footprint()

    assert footprint.parameter_count == 20
    assert footprint.parameter_bytes == 80
    assert footprint.trainable_parameter_count == 0
    assert footprint.trainable_parameter_bytes == 0
    assert footprint.buffer_bytes is None


def test_tensorflow_adapter_reports_gpu_baseline_and_peak(monkeypatch):
    calls = []

    class Experimental:
        @staticmethod
        def get_memory_info(device):
            calls.append(("get", device))
            return {"current": 10, "peak": 35}

        @staticmethod
        def reset_memory_stats(device):
            calls.append(("reset", device))

    config = type(
        "Config",
        (),
        {
            "experimental": Experimental,
            "list_logical_devices": staticmethod(lambda kind: []),
        },
    )()
    tensorflow = type("TensorFlow", (), {"config": config})()
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", tensorflow)
    extractor = type("TFLike", (), {"model": None})()
    adapter = TensorFlowResourceProfileAdapter(extractor, profiling_device="GPU:0")

    assert adapter.reset_peak_device_memory().status == "succeeded"
    measurement = adapter.peak_device_memory()
    assert measurement.baseline_allocated_bytes == 10
    assert measurement.peak_allocated_bytes == 35
    assert calls == [("get", "GPU:0"), ("reset", "GPU:0"), ("get", "GPU:0")]


def test_tensorflow_adapter_does_not_guess_between_visible_devices(monkeypatch):
    logical = [type("Device", (), {"name": f"/device:GPU:{index}"})() for index in range(2)]
    config = type(
        "Config",
        (),
        {
            "experimental": object(),
            "list_logical_devices": staticmethod(lambda kind: logical),
        },
    )()
    tensorflow = type("TensorFlow", (), {"config": config})()
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", tensorflow)
    extractor = type("TFLike", (), {"model": None})()
    adapter = TensorFlowResourceProfileAdapter(extractor)

    assert adapter.reset_peak_device_memory().status == "unavailable"
    measurement = adapter.peak_device_memory()
    assert measurement.status == "unavailable"
    assert "ambiguous" in measurement.unavailable_reason


def test_loaded_model_device_overrides_conflicting_profiling_hint():
    weight = type(
        "Weight",
        (),
        {
            "shape": (1,),
            "dtype": np.dtype("float32"),
            "device": "/device:GPU:1",
        },
    )()
    model = type("Model", (), {"weights": [weight]})()

    class Extractor:
        def __init__(self):
            self.model = model

        def get_resource_profile_adapter(self):
            return TensorFlowResourceProfileAdapter(self, profiling_device="GPU:0")

    profile = ResourceProfiler(
        ResourceProfilingConfig(enabled=True), Extractor(), streaming=False
    ).finish(cache_hit=True)

    assert profile.context["device"] == "GPU:1"
    assert profile.context["device_resolution"] == ("model_state_conflicts_with_profiling_hint")
    assert any("overrides" in warning for warning in profile.warnings)


def test_adapter_failures_are_preserved_as_profile_warnings():
    class FailingAdapter(BaseResourceProfileAdapter):
        def synchronize(self):
            raise RuntimeError("device synchronization failed")

    extractor = type(
        "Extractor",
        (),
        {"get_resource_profile_adapter": lambda self: FailingAdapter()},
    )()
    profiler = ResourceProfiler(ResourceProfilingConfig(enabled=True), extractor, streaming=True)

    profiler.measure_call(lambda: np.ones((1, 1)), samples=1, call_type="transform")
    profile = profiler.finish()

    assert profile.context["synchronization_status"] == "host_observed"
    assert any(
        "synchronize" in warning and "RuntimeError" in warning for warning in profile.warnings
    )


def test_cache_hit_does_not_query_native_device_peak():
    events = []

    class Adapter(BaseResourceProfileAdapter):
        def metadata(self):
            return ResourceAdapterMetadata(backend="fake", device="cuda:0", asynchronous=True)

        def reset_peak_device_memory(self):
            events.append("reset")
            return AdapterOperationResult("succeeded")

        def peak_device_memory(self):
            events.append("peak")
            return DeviceMemoryMeasurement(
                status="measured",
                backend="fake",
                device="cuda:0",
                peak_allocated_bytes=10,
                measurement_scope="profile_window",
            )

    extractor = type("Extractor", (), {"get_resource_profile_adapter": lambda self: Adapter()})()
    profile = ResourceProfiler(
        ResourceProfilingConfig(enabled=True), extractor, streaming=False
    ).finish(cache_hit=True)

    assert events == []
    assert profile.device_memory.status == "not_measured_cache_hit"
    assert profile.device_memory.measurement_scope is None


def test_failed_device_reset_prevents_peak_query():
    events = []

    class Adapter(BaseResourceProfileAdapter):
        def metadata(self):
            return ResourceAdapterMetadata(backend="fake", device="cuda:0", asynchronous=True)

        def reset_peak_device_memory(self):
            events.append("reset")
            return AdapterOperationResult("unavailable", "reset failed")

        def peak_device_memory(self):
            events.append("peak")
            raise AssertionError("peak must not be queried without a scoped reset")

    extractor = type("Extractor", (), {"get_resource_profile_adapter": lambda self: Adapter()})()
    profiler = ResourceProfiler(ResourceProfilingConfig(enabled=True), extractor, streaming=False)
    profiler.measure_call(lambda: np.ones((1, 1)), samples=1, call_type="transform")
    profile = profiler.finish()

    assert events == ["reset"]
    assert profile.device_memory.status == "unavailable"
    assert profile.device_memory.measurement_scope is None
    assert "reset failed" in profile.device_memory.unavailable_reason


def test_synchronization_failure_is_accumulated_across_calls():
    outcomes = iter(["unavailable", "succeeded", "succeeded", "succeeded"])

    class Adapter(BaseResourceProfileAdapter):
        def metadata(self):
            return ResourceAdapterMetadata(backend="fake", device="cuda:0", asynchronous=True)

        def synchronize(self):
            return AdapterOperationResult(next(outcomes))

        def reset_peak_device_memory(self):
            return AdapterOperationResult("unavailable", "no peak API")

    extractor = type("Extractor", (), {"get_resource_profile_adapter": lambda self: Adapter()})()
    profiler = ResourceProfiler(ResourceProfilingConfig(enabled=True), extractor, streaming=True)
    profiler.measure_call(lambda: np.ones((1, 1)), samples=1, call_type="transform")
    profiler.measure_call(lambda: np.ones((1, 1)), samples=1, call_type="transform")

    profile = profiler.finish()

    assert profile.context["synchronization_status"] == "host_observed"
    assert any("host-observed" in warning for warning in profile.warnings)


@pytest.mark.parametrize("error", [KeyError("missing"), IndexError("bad"), NotImplementedError()])
def test_ordinary_adapter_exceptions_do_not_abort_scoring(error):
    class Adapter(BaseResourceProfileAdapter):
        def metadata(self):
            raise error

    extractor = type("Extractor", (), {"get_resource_profile_adapter": lambda self: Adapter()})()
    profiler = ResourceProfiler(ResourceProfilingConfig(enabled=True), extractor, streaming=False)
    profiler.measure_call(lambda: np.ones((1, 1)), samples=1, call_type="transform")

    profile = profiler.finish()

    assert any(type(error).__name__ in warning for warning in profile.warnings)


def test_torch_multi_device_model_synchronizes_all_devices_but_skips_peak():
    events = []

    class Device:
        def __init__(self, value):
            self.value = str(value)
            self.type = self.value.split(":", 1)[0]

        def __str__(self):
            return self.value

    class Cuda:
        @staticmethod
        def synchronize(device):
            events.append(("sync", str(device)))

    torch = type("Torch", (), {"device": Device, "cuda": Cuda})()
    parameters = [
        FakeParameter(1, 4, device="cuda:0", dtype="torch.float32"),
        FakeParameter(1, 4, device="cuda:1", dtype="torch.float32"),
    ]
    model = type("Model", (), {"parameters": lambda self: parameters, "buffers": lambda self: []})()
    extractor = type("Extractor", (), {"model": model, "device": None})()
    adapter = TorchResourceProfileAdapter(extractor, torch_loader=lambda: torch)

    assert adapter.synchronize().status == "succeeded"
    reset = adapter.reset_peak_device_memory()

    assert events == [("sync", "cuda:0"), ("sync", "cuda:1")]
    assert reset.status == "unavailable"
    assert "multi_device_model" in reset.reason


def test_torch_peak_is_invalidated_when_lazy_model_changes_device():
    class Device:
        def __init__(self, value):
            self.value = str(value)
            self.type = self.value.split(":", 1)[0]

        def __str__(self):
            return self.value

    class Cuda:
        @staticmethod
        def synchronize(device):
            return None

        @staticmethod
        def memory_allocated(device):
            return 10

        @staticmethod
        def memory_reserved(device):
            return 20

        @staticmethod
        def reset_peak_memory_stats(device):
            return None

    torch = type("Torch", (), {"device": Device, "cuda": Cuda})()
    extractor = type("Extractor", (), {"model": None, "device": None})()
    adapter = TorchResourceProfileAdapter(
        extractor,
        device_resolver=lambda module: "cuda:0",
        torch_loader=lambda: torch,
    )

    assert adapter.reset_peak_device_memory().status == "succeeded"
    parameter = FakeParameter(1, 4, device="cuda:1", dtype="torch.float32")
    extractor.model = type(
        "Model", (), {"parameters": lambda self: [parameter], "buffers": lambda self: []}
    )()

    measurement = adapter.peak_device_memory()

    assert measurement.status == "unavailable"
    assert "changed" in measurement.unavailable_reason


def test_keras_adapter_routes_tensorflow_torch_and_jax_backends(monkeypatch):
    class Backend:
        active = "tensorflow"

        @classmethod
        def backend(cls):
            return cls.active

    keras = type("Keras", (), {"backend": Backend})()
    monkeypatch.setitem(sys.modules, "keras", keras)

    tf_config = type(
        "Config",
        (),
        {
            "experimental": object(),
            "list_logical_devices": staticmethod(lambda kind: []),
        },
    )()
    monkeypatch.setitem(sys.modules, "tensorflow", type("TF", (), {"config": tf_config})())
    extractor = type("KerasLike", (), {"model": None})()
    adapter = KerasResourceProfileAdapter(extractor)
    assert adapter.metadata().backend == "keras-tensorflow"

    Backend.active = "torch"

    class Device:
        def __init__(self, value):
            self.value = str(value)
            self.type = self.value.split(":", 1)[0]

        def __str__(self):
            return self.value

    monkeypatch.setitem(sys.modules, "torch", type("Torch", (), {"device": Device})())
    torch_adapter = KerasResourceProfileAdapter(extractor, profiling_device="cpu")
    assert torch_adapter.metadata().backend == "keras-torch"
    assert torch_adapter.reset_peak_device_memory().status == "not_applicable"

    events = []
    Backend.active = "jax"
    jax = type(
        "Jax",
        (),
        {
            "effects_barrier": staticmethod(lambda: events.append("barrier")),
            "local_devices": staticmethod(lambda: ["gpu:0"]),
        },
    )()
    monkeypatch.setitem(sys.modules, "jax", jax)
    jax_adapter = KerasResourceProfileAdapter(extractor, profiling_device="gpu:0")

    assert jax_adapter.metadata().backend == "keras-jax"
    assert jax_adapter.synchronize().status == "succeeded"
    assert jax_adapter.reset_peak_device_memory().status == "unavailable"
    assert events == ["barrier"]


def test_profiling_only_configuration_does_not_change_extractor_recipe(tmp_path):
    transform = np.asarray
    left = CallableExtractor("callable", transform)
    right = CallableExtractor(
        "callable", transform, resource_profile_adapter=BaseResourceProfileAdapter()
    )
    assert left.recipe() == right.recipe()
    assert fingerprint_extractor_recipe(left.recipe()) == fingerprint_extractor_recipe(
        right.recipe()
    )

    model = type("Model", (), {"weights": []})()
    first = KerasExtractor(
        "keras", model, checkpoint_paths=[str(tmp_path / "a")], profiling_device="GPU:0"
    )
    second = KerasExtractor(
        "keras", model, checkpoint_paths=[str(tmp_path / "b")], profiling_device="GPU:1"
    )
    assert first.recipe() == second.recipe()
    assert fingerprint_extractor_recipe(first.recipe()) == fingerprint_extractor_recipe(
        second.recipe()
    )


def test_deployment_directories_require_explicit_recursion(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "weights.bin").write_bytes(b"weights")

    class ArtifactAdapter(BaseResourceProfileAdapter):
        def deployment_artifacts(self):
            return (
                DeploymentArtifact(str(bundle)),
                DeploymentArtifact(str(bundle), role="deployment_bundle", recursive=True),
            )

    extractor = type(
        "Extractor",
        (),
        {"get_resource_profile_adapter": lambda self: ArtifactAdapter()},
    )()
    profile = ResourceProfiler(
        ResourceProfilingConfig(enabled=True), extractor, streaming=False
    ).finish()

    assert profile.model.status == "partial"
    assert profile.model.checkpoint_status == "partial"
    assert profile.model.checkpoint_bytes == len(b"weights")
    assert profile.model.artifacts[0].status == "directory_requires_recursive"
    assert profile.model.artifacts[1].status == "measured"


@pytest.mark.parametrize(
    "factory, adapter_type",
    [
        (lambda: HFTextExtractor("hf-text", "model"), TorchResourceProfileAdapter),
        (lambda: HFVisionExtractor("hf-vision", "model"), TorchResourceProfileAdapter),
        (lambda: HFAudioExtractor("hf-audio", "model"), TorchResourceProfileAdapter),
        (
            lambda: HFTimeSeriesExtractor("hf-time-series", "model"),
            TorchResourceProfileAdapter,
        ),
        (lambda: HFVideoExtractor("hf-video", "model"), TorchResourceProfileAdapter),
        (
            lambda: HFMultimodalExtractor(
                "hf-multimodal",
                "model",
                input_modalities={"image": "image", "text": "text"},
                outputs=[{"name": "image", "source": "image", "model_output": "image_embeds"}],
            ),
            TorchResourceProfileAdapter,
        ),
        (
            lambda: SentenceTransformerExtractor("sentence", "model"),
            TorchResourceProfileAdapter,
        ),
        (lambda: TimmVisionExtractor("timm", "model"), TorchResourceProfileAdapter),
        (
            lambda: TorchvisionVisionExtractor("torchvision", "resnet18"),
            TorchResourceProfileAdapter,
        ),
        (lambda: OpenCLIPExtractor("openclip", "ViT-B-32"), TorchResourceProfileAdapter),
        (lambda: SigLIPExtractor("siglip", "model"), TorchResourceProfileAdapter),
        (
            lambda: GraphModelExtractor("graph", object(), lambda values: values),
            TorchResourceProfileAdapter,
        ),
        (lambda: TFHubExtractor("tfhub", "handle"), TensorFlowResourceProfileAdapter),
        (
            lambda: JAXFlaxExtractor("jax", lambda values: values, model=lambda values: values),
            JAXResourceProfileAdapter,
        ),
    ],
)
def test_owned_framework_extractors_expose_lazy_resource_adapters(factory, adapter_type):
    extractor = factory()

    adapter = extractor.get_resource_profile_adapter()

    assert isinstance(adapter, adapter_type)
    if hasattr(extractor, "_model"):
        assert extractor._model is None
