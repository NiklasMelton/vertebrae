"""Measured inference-resource profiles for local representation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import psutil

from vertebrae.config import ResourceProfilingConfig
from vertebrae.utils.memory import estimate_matrix_resident_bytes


@dataclass(frozen=True)
class AdapterOperationResult:
    """Outcome of an optional framework-specific profiling operation."""

    status: str
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "not_applicable", "unavailable"}:
            raise ValueError(
                "AdapterOperationResult.status must be 'succeeded', "
                "'not_applicable', or 'unavailable'."
            )


@dataclass(frozen=True)
class ResourceAdapterMetadata:
    """Framework, device, and synchronization context supplied by an adapter."""

    backend: Optional[str] = None
    device: Optional[str] = None
    device_resolution: str = "unavailable"
    asynchronous: bool = False
    synchronization_method: Optional[str] = None
    weight_dtypes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.device_resolution:
            raise ValueError("ResourceAdapterMetadata.device_resolution must be non-empty.")


@dataclass(frozen=True)
class DeviceMemoryMeasurement:
    """Framework allocator memory observed for one profiling session."""

    status: str
    backend: Optional[str] = None
    device: Optional[str] = None
    baseline_allocated_bytes: Optional[int] = None
    baseline_reserved_bytes: Optional[int] = None
    peak_allocated_bytes: Optional[int] = None
    peak_reserved_bytes: Optional[int] = None
    unavailable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"measured", "not_applicable", "unavailable"}:
            raise ValueError(
                "DeviceMemoryMeasurement.status must be 'measured', "
                "'not_applicable', or 'unavailable'."
            )
        for name in (
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            _validate_optional_nonnegative(name, getattr(self, name))
        if self.status == "measured" and self.peak_allocated_bytes is None:
            raise ValueError("Measured DeviceMemoryMeasurement requires peak_allocated_bytes.")


@dataclass(frozen=True)
class ModelFootprintMeasurement:
    """In-memory model state exposed by a framework adapter."""

    status: str
    parameter_count: Optional[int] = None
    parameter_bytes: Optional[int] = None
    buffer_bytes: Optional[int] = None
    in_memory_bytes: Optional[int] = None
    weight_dtypes: Tuple[str, ...] = ()
    unavailable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"measured", "unavailable"}:
            raise ValueError(
                "ModelFootprintMeasurement.status must be 'measured' or 'unavailable'."
            )
        for name in (
            "parameter_count",
            "parameter_bytes",
            "buffer_bytes",
            "in_memory_bytes",
        ):
            _validate_optional_nonnegative(name, getattr(self, name))
        if self.status == "measured" and all(
            value is None
            for value in (self.parameter_count, self.parameter_bytes, self.in_memory_bytes)
        ):
            raise ValueError(
                "Measured ModelFootprintMeasurement requires parameter or in-memory data."
            )


@dataclass(frozen=True)
class DeploymentArtifact:
    """Explicit local artifact whose deployment footprint should be measured."""

    path: str
    role: str = "checkpoint"
    recursive: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("DeploymentArtifact.path must be non-empty.")
        if not self.role:
            raise ValueError("DeploymentArtifact.role must be non-empty.")


@dataclass(frozen=True)
class ArtifactFootprint:
    """Measured footprint for one declared deployment artifact."""

    path: str
    resolved_path: str
    role: str
    recursive: bool
    status: str
    bytes: int = 0
    file_count: int = 0
    reason: Optional[str] = None


@dataclass
class InferenceProfile:
    status: str
    first_call_seconds: Optional[float] = None
    first_call_includes_fit: bool = False
    warm_call_count: int = 0
    warm_mean_seconds: Optional[float] = None
    warm_median_seconds: Optional[float] = None
    warm_p95_seconds: Optional[float] = None
    warm_max_seconds: Optional[float] = None
    warm_mean_seconds_per_sample: Optional[float] = None
    total_materialized_seconds: Optional[float] = None
    materialized_samples: int = 0
    throughput_samples_per_second: Optional[float] = None
    batch_sizes: List[int] = field(default_factory=list)


@dataclass
class HostMemoryProfile:
    status: str
    baseline_rss_bytes: Optional[int] = None
    peak_rss_bytes: Optional[int] = None
    peak_increase_bytes: Optional[int] = None
    sample_interval_seconds: Optional[float] = None


@dataclass
class DeviceMemoryProfile:
    status: str
    backend: Optional[str] = None
    device: Optional[str] = None
    baseline_allocated_bytes: Optional[int] = None
    baseline_reserved_bytes: Optional[int] = None
    peak_allocated_bytes: Optional[int] = None
    peak_reserved_bytes: Optional[int] = None
    peak_allocated_increase_bytes: Optional[int] = None
    peak_reserved_increase_bytes: Optional[int] = None
    unavailable_reason: Optional[str] = None


@dataclass
class ModelFootprint:
    status: str
    parameter_status: str = "unavailable"
    checkpoint_status: str = "unavailable"
    parameter_count: Optional[int] = None
    parameter_bytes: Optional[int] = None
    buffer_bytes: Optional[int] = None
    in_memory_bytes: Optional[int] = None
    checkpoint_bytes: Optional[int] = None
    weight_dtypes: List[str] = field(default_factory=list)
    artifacts: List[ArtifactFootprint] = field(default_factory=list)


@dataclass
class EmbeddingFootprint:
    raw_bytes: int
    evaluated_bytes: int
    bytes_per_embedding: float
    compression_savings_ratio: float


@dataclass
class ResourceProfile:
    status: str
    inference: InferenceProfile
    host_memory: HostMemoryProfile
    device_memory: DeviceMemoryProfile
    model: ModelFootprint
    embedding: Optional[EmbeddingFootprint]
    context: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseResourceProfileAdapter:
    """Default unavailable implementations for custom resource adapters."""

    def metadata(self) -> ResourceAdapterMetadata:
        return ResourceAdapterMetadata()

    def synchronize(self) -> AdapterOperationResult:
        return AdapterOperationResult("not_applicable")

    def reset_peak_device_memory(self) -> AdapterOperationResult:
        return AdapterOperationResult("not_applicable")

    def peak_device_memory(self) -> DeviceMemoryMeasurement:
        return DeviceMemoryMeasurement(
            status="unavailable",
            unavailable_reason="The adapter does not expose allocator memory.",
        )

    def model_footprint(self) -> ModelFootprintMeasurement:
        return ModelFootprintMeasurement(
            status="unavailable",
            unavailable_reason="The adapter does not expose model state.",
        )

    def deployment_artifacts(self) -> Sequence[DeploymentArtifact]:
        return ()


@dataclass
class _CallRecord:
    seconds: float
    samples: int
    call_type: str
    materialized: bool
    includes_fit: bool


class ResourceProfiler:
    """Observe real extractor calls without introducing repeat inference."""

    def __init__(
        self,
        config: ResourceProfilingConfig,
        extractor: Any,
        *,
        streaming: bool,
    ) -> None:
        self.config = config
        self.extractor = extractor
        self.streaming = streaming
        self.records: List[_CallRecord] = []
        self._baseline_rss: Optional[int] = None
        self._peak_rss: Optional[int] = None
        self._device_reset = False
        self._synchronized: Optional[bool] = None
        self._warnings: List[str] = []
        self.adapter = self._resolve_adapter(extractor)

    def measure_call(
        self,
        fn: Callable[[], Any],
        *,
        samples: int,
        call_type: str,
        materialized: bool = True,
        includes_fit: bool = False,
    ) -> Any:
        if not self.config.enabled:
            return fn()
        if self._baseline_rss is None and self.config.host_memory:
            self._baseline_rss = _rss_bytes()
            self._peak_rss = self._baseline_rss
        if self.adapter is not None and not self._device_reset and self.config.device_memory:
            self._adapter_call(
                "reset_peak_device_memory",
                self.adapter.reset_peak_device_memory,
                AdapterOperationResult("unavailable"),
                AdapterOperationResult,
            )
            self._device_reset = True

        synchronized_before = self._synchronize()
        sampler = _RssSampler(self.config.host_sample_interval_seconds)
        if self.config.host_memory:
            sampler.start()
        start = perf_counter()
        try:
            value = fn()
            synchronized_after = self._synchronize()
        finally:
            elapsed = perf_counter() - start
            if self.config.host_memory:
                sampler.stop()
                self._peak_rss = max(
                    int(self._peak_rss or 0),
                    sampler.peak_rss_bytes,
                    _rss_bytes(),
                )
        self._synchronized = bool(synchronized_before and synchronized_after)
        self.records.append(
            _CallRecord(
                seconds=float(elapsed),
                samples=int(samples),
                call_type=call_type,
                materialized=materialized,
                includes_fit=includes_fit,
            )
        )
        return value

    def mark_last_call_materialized(self) -> None:
        if self.records:
            self.records[-1].materialized = True

    def finish(self, *, cache_hit: bool = False) -> ResourceProfile:
        inference = self._inference_profile(cache_hit=cache_hit)
        host = self._host_profile()
        device = self._device_profile()
        model = self._model_profile()
        metadata = (
            self._adapter_call(
                "metadata",
                self.adapter.metadata,
                ResourceAdapterMetadata(),
                ResourceAdapterMetadata,
            )
            if self.adapter is not None
            else ResourceAdapterMetadata()
        )
        if self.records and self._synchronized is False and metadata.asynchronous:
            self._warnings.append(
                "Device execution could not be synchronized; latency is host-observed."
            )
        if inference.warm_call_count == 0 and inference.status == "measured":
            self._warnings.append(
                "Warm latency is unavailable because only one extractor call was measured."
            )
        return ResourceProfile(
            status=(
                "measured"
                if self.records
                else "not_measured_cache_hit"
                if cache_hit
                else "not_measured"
            ),
            inference=inference,
            host_memory=host,
            device_memory=device,
            model=model,
            embedding=None,
            context={
                "streaming": self.streaming,
                "call_types": [record.call_type for record in self.records],
                "planning_probe_calls": sum(
                    1 for record in self.records if not record.materialized
                ),
                "first_call_materialized": (self.records[0].materialized if self.records else None),
                "synchronization_status": (
                    "synchronized"
                    if self._synchronized
                    else "host_observed"
                    if self.records
                    else "not_measured"
                ),
                "preprocessing_included": True,
                "output_conversion_included": True,
                "backend": metadata.backend,
                "device": metadata.device,
                "device_resolution": metadata.device_resolution,
                "asynchronous": metadata.asynchronous,
                "synchronization_method": metadata.synchronization_method,
                "weight_dtypes": list(metadata.weight_dtypes),
            },
            warnings=sorted(set(self._warnings)),
        )

    def _resolve_adapter(self, extractor: Any) -> Any:
        factory = getattr(extractor, "get_resource_profile_adapter", None)
        if not callable(factory):
            return None
        try:
            return factory()
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._record_adapter_failure("get_resource_profile_adapter", exc)
            return None

    def _adapter_call(
        self,
        name: str,
        fn: Callable[[], Any],
        default: Any,
        expected_type: Optional[type] = None,
    ) -> Any:
        try:
            value = fn()
            if expected_type is not None and not isinstance(value, expected_type):
                raise TypeError(f"expected {expected_type.__name__}, got {type(value).__name__}")
            return value
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._record_adapter_failure(name, exc)
            return default

    def _record_adapter_failure(self, hook: str, exc: Exception) -> None:
        self._warnings.append(
            f"Resource profiling adapter hook '{hook}' failed with " f"{type(exc).__name__}: {exc}"
        )

    def _synchronize(self) -> bool:
        if self.adapter is None:
            return True
        result = self._adapter_call(
            "synchronize",
            self.adapter.synchronize,
            AdapterOperationResult("unavailable"),
            AdapterOperationResult,
        )
        return result.status in {"succeeded", "not_applicable"}

    def _inference_profile(self, *, cache_hit: bool) -> InferenceProfile:
        if not self.records:
            return InferenceProfile(
                status="not_measured_cache_hit" if cache_hit else "not_measured"
            )
        first = self.records[0]
        warm = self.records[1:]
        warm_seconds = np.asarray([item.seconds for item in warm], dtype=float)
        materialized = [item for item in self.records if item.materialized]
        total_seconds = float(sum(item.seconds for item in materialized))
        samples = int(sum(item.samples for item in materialized))
        warm_samples = int(sum(item.samples for item in warm))
        return InferenceProfile(
            status="measured",
            first_call_seconds=first.seconds,
            first_call_includes_fit=first.includes_fit,
            warm_call_count=len(warm),
            warm_mean_seconds=float(warm_seconds.mean()) if len(warm_seconds) else None,
            warm_median_seconds=float(np.median(warm_seconds)) if len(warm_seconds) else None,
            warm_p95_seconds=float(np.percentile(warm_seconds, 95)) if len(warm_seconds) else None,
            warm_max_seconds=float(warm_seconds.max()) if len(warm_seconds) else None,
            warm_mean_seconds_per_sample=(
                float(warm_seconds.sum() / warm_samples) if warm_samples else None
            ),
            total_materialized_seconds=total_seconds,
            materialized_samples=samples,
            throughput_samples_per_second=(samples / total_seconds if total_seconds > 0 else None),
            batch_sizes=[item.samples for item in materialized],
        )

    def _host_profile(self) -> HostMemoryProfile:
        if not self.config.host_memory:
            return HostMemoryProfile(status="disabled")
        if self._baseline_rss is None or self._peak_rss is None:
            return HostMemoryProfile(status="not_measured")
        return HostMemoryProfile(
            status="measured",
            baseline_rss_bytes=self._baseline_rss,
            peak_rss_bytes=self._peak_rss,
            peak_increase_bytes=max(0, self._peak_rss - self._baseline_rss),
            sample_interval_seconds=self.config.host_sample_interval_seconds,
        )

    def _device_profile(self) -> DeviceMemoryProfile:
        metadata = (
            self._adapter_call(
                "metadata",
                self.adapter.metadata,
                ResourceAdapterMetadata(),
                ResourceAdapterMetadata,
            )
            if self.adapter is not None
            else ResourceAdapterMetadata()
        )
        if not self.config.device_memory:
            return DeviceMemoryProfile(
                status="disabled", backend=metadata.backend, device=metadata.device
            )
        if self.adapter is None:
            return DeviceMemoryProfile(
                status="unavailable",
                unavailable_reason="The extractor does not provide a resource adapter.",
            )
        payload = self._adapter_call(
            "peak_device_memory",
            self.adapter.peak_device_memory,
            DeviceMemoryMeasurement(
                status="unavailable",
                unavailable_reason="The adapter device-memory hook failed.",
            ),
            DeviceMemoryMeasurement,
        )
        if payload.status == "unavailable":
            self._warnings.append(
                "Peak device memory is unavailable for this extractor backend"
                + (f": {payload.unavailable_reason}" if payload.unavailable_reason else ".")
            )
        return DeviceMemoryProfile(
            status=payload.status,
            backend=payload.backend or metadata.backend,
            device=payload.device or metadata.device,
            baseline_allocated_bytes=payload.baseline_allocated_bytes,
            baseline_reserved_bytes=payload.baseline_reserved_bytes,
            peak_allocated_bytes=payload.peak_allocated_bytes,
            peak_reserved_bytes=payload.peak_reserved_bytes,
            peak_allocated_increase_bytes=_nonnegative_difference(
                payload.peak_allocated_bytes, payload.baseline_allocated_bytes
            ),
            peak_reserved_increase_bytes=_nonnegative_difference(
                payload.peak_reserved_bytes, payload.baseline_reserved_bytes
            ),
            unavailable_reason=payload.unavailable_reason,
        )

    def _model_profile(self) -> ModelFootprint:
        payload = (
            self._adapter_call(
                "model_footprint",
                self.adapter.model_footprint,
                ModelFootprintMeasurement(
                    status="unavailable",
                    unavailable_reason="The adapter model-footprint hook failed.",
                ),
                ModelFootprintMeasurement,
            )
            if self.adapter is not None
            else ModelFootprintMeasurement(status="unavailable")
        )
        declarations = (
            self._adapter_call(
                "deployment_artifacts",
                self.adapter.deployment_artifacts,
                (),
            )
            if self.adapter is not None
            else ()
        )
        if not isinstance(declarations, Sequence) or any(
            not isinstance(item, DeploymentArtifact) for item in declarations
        ):
            self._warnings.append(
                "Resource profiling adapter hook 'deployment_artifacts' returned an invalid "
                "typed payload."
            )
            declarations = ()
        artifacts = _artifact_footprint(declarations)
        measured_artifacts = [item for item in artifacts if item.status in {"measured", "partial"}]
        checkpoint_bytes = sum(item.bytes for item in measured_artifacts)
        parameter_status = "measured" if payload.status == "measured" else "unavailable"
        if not artifacts:
            checkpoint_status = "unavailable"
        elif all(item.status in {"measured", "duplicate"} for item in artifacts):
            checkpoint_status = "measured"
        elif measured_artifacts:
            checkpoint_status = "partial"
        else:
            checkpoint_status = "unavailable"
        if parameter_status == "measured" and checkpoint_status == "measured":
            status = "measured"
        elif parameter_status == "measured" or checkpoint_status in {"measured", "partial"}:
            status = "partial"
        else:
            status = "unavailable"
        if status == "unavailable":
            self._warnings.append("Model parameter and checkpoint footprint is unavailable.")
        if any(item.status not in {"measured", "duplicate"} for item in artifacts):
            self._warnings.append(
                "One or more explicit deployment artifacts could not be fully measured."
            )
        in_memory = payload.in_memory_bytes
        if in_memory is None and payload.parameter_bytes is not None:
            in_memory = payload.parameter_bytes + int(payload.buffer_bytes or 0)
        return ModelFootprint(
            status=status,
            parameter_status=parameter_status,
            checkpoint_status=checkpoint_status,
            parameter_count=payload.parameter_count,
            parameter_bytes=payload.parameter_bytes,
            buffer_bytes=payload.buffer_bytes,
            in_memory_bytes=in_memory,
            checkpoint_bytes=(checkpoint_bytes if measured_artifacts else None),
            weight_dtypes=list(payload.weight_dtypes),
            artifacts=artifacts,
        )


def with_embedding_footprint(
    profile: Optional[ResourceProfile],
    raw_embeddings: Any,
    evaluated_embeddings: Any,
) -> Optional[ResourceProfile]:
    if profile is None:
        return None
    raw_bytes = estimate_matrix_resident_bytes(raw_embeddings)
    evaluated_bytes = estimate_matrix_resident_bytes(evaluated_embeddings)
    n_samples = int(getattr(evaluated_embeddings, "shape", (0,))[0])
    footprint = EmbeddingFootprint(
        raw_bytes=raw_bytes,
        evaluated_bytes=evaluated_bytes,
        bytes_per_embedding=(evaluated_bytes / n_samples if n_samples else 0.0),
        compression_savings_ratio=(1.0 - (evaluated_bytes / raw_bytes) if raw_bytes else 0.0),
    )
    return replace(profile, embedding=footprint)


class TorchResourceProfileAdapter(BaseResourceProfileAdapter):
    """Resource hooks for a Torch-backed extractor without forcing model loading."""

    def __init__(
        self,
        extractor: Any,
        artifacts: Sequence[Union[str, DeploymentArtifact]] = (),
        *,
        model_getter: Optional[Callable[[], Any]] = None,
        device_resolver: Optional[Callable[[Any], Any]] = None,
        torch_loader: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.extractor = extractor
        self._artifacts = _normalize_artifacts(artifacts)
        self._model_getter = model_getter or self._default_model_getter
        self._device_resolver = device_resolver
        self._torch_loader = torch_loader
        self._baseline_allocated: Optional[int] = None
        self._baseline_reserved: Optional[int] = None

    def metadata(self) -> ResourceAdapterMetadata:
        torch = self._load_torch()
        device, source = self._resolved_device(torch)
        model = self._model_getter()
        return ResourceAdapterMetadata(
            backend="torch",
            device=str(device),
            device_resolution=source,
            asynchronous=device.type in {"cuda", "mps"},
            synchronization_method=(
                "torch.cuda.synchronize"
                if device.type == "cuda"
                else "torch.mps.synchronize"
                if device.type == "mps"
                else "not_applicable"
            ),
            weight_dtypes=_torch_weight_dtypes(model),
        )

    def synchronize(self) -> AdapterOperationResult:
        torch = self._load_torch()
        device, _ = self._resolved_device(torch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            return AdapterOperationResult("succeeded")
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
            return AdapterOperationResult("succeeded")
        return AdapterOperationResult("not_applicable")

    def reset_peak_device_memory(self) -> AdapterOperationResult:
        torch = self._load_torch()
        device, _ = self._resolved_device(torch)
        if device.type != "cuda":
            return AdapterOperationResult("not_applicable")
        self._baseline_allocated = int(torch.cuda.memory_allocated(device))
        self._baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        return AdapterOperationResult("succeeded")

    def peak_device_memory(self) -> DeviceMemoryMeasurement:
        torch = self._load_torch()
        device, _ = self._resolved_device(torch)
        if device.type == "cpu":
            return DeviceMemoryMeasurement(
                status="not_applicable",
                backend="torch",
                device=str(device),
                unavailable_reason="CPU memory is reported through process RSS.",
            )
        if device.type != "cuda":
            return DeviceMemoryMeasurement(
                status="unavailable",
                backend="torch",
                device=str(device),
                unavailable_reason=(
                    "Torch does not expose allocator peak counters for " f"{device.type}."
                ),
            )
        return DeviceMemoryMeasurement(
            status="measured",
            backend="torch",
            device=str(device),
            baseline_allocated_bytes=self._baseline_allocated,
            baseline_reserved_bytes=self._baseline_reserved,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )

    def model_footprint(self) -> ModelFootprintMeasurement:
        return _torch_model_footprint(self._model_getter())

    def deployment_artifacts(self) -> Sequence[DeploymentArtifact]:
        return self._artifacts

    def _load_torch(self) -> Any:
        if self._torch_loader is not None:
            return self._torch_loader()
        loader = getattr(self.extractor, "_load_torch", None)
        if callable(loader):
            return loader()
        import torch

        return torch

    def _default_model_getter(self) -> Any:
        model = getattr(self.extractor, "model", None)
        return model if model is not None else getattr(self.extractor, "_model", None)

    def _resolved_device(self, torch: Any) -> Tuple[Any, str]:
        if self._device_resolver is not None:
            value = self._device_resolver(torch)
            if value is not None:
                return torch.device(value), "extractor_resolver"
        explicit = getattr(self.extractor, "device", None)
        if explicit is not None:
            return torch.device(explicit), "explicit"
        model_device = _torch_model_device(self._model_getter())
        if model_device is not None:
            return torch.device(model_device), "model_state"
        resolver = getattr(self.extractor, "_device", None)
        if callable(resolver):
            return torch.device(resolver(torch)), "extractor_resolver"
        return torch.device("cpu"), "default_cpu"


class TensorFlowResourceProfileAdapter(BaseResourceProfileAdapter):
    """Resource hooks shared by Keras and TensorFlow Hub extractors."""

    def __init__(
        self,
        extractor: Any,
        artifacts: Sequence[Union[str, DeploymentArtifact]] = (),
        *,
        model_getter: Optional[Callable[[], Any]] = None,
        backend: str = "tensorflow",
        profiling_device: Optional[str] = None,
    ) -> None:
        self.extractor = extractor
        self._artifacts = _normalize_artifacts(artifacts)
        self._model_getter = model_getter or self._default_model_getter
        self.backend = backend
        self.profiling_device = profiling_device
        self._baseline_allocated: Optional[int] = None

    def metadata(self) -> ResourceAdapterMetadata:
        device, source = self._resolved_device()
        model = self._model_getter()
        return ResourceAdapterMetadata(
            backend=self.backend,
            device=device,
            device_resolution=source,
            asynchronous=False,
            synchronization_method="extractor_output_to_numpy",
            weight_dtypes=_tensorflow_weight_dtypes(model),
        )

    def synchronize(self) -> AdapterOperationResult:
        return AdapterOperationResult("succeeded")

    def reset_peak_device_memory(self) -> AdapterOperationResult:
        tf = self._load_tensorflow()
        device, _ = self._resolved_device(tf)
        if device is None:
            return AdapterOperationResult(
                "unavailable",
                "TensorFlow device resolution is ambiguous across visible devices.",
            )
        if "GPU" not in device.upper():
            return AdapterOperationResult("not_applicable")
        experimental = getattr(tf.config, "experimental", None)
        if experimental is None or not hasattr(experimental, "get_memory_info"):
            return AdapterOperationResult(
                "unavailable", "TensorFlow memory-info APIs are unavailable."
            )
        info = experimental.get_memory_info(device)
        self._baseline_allocated = int(info.get("current", 0))
        if not hasattr(experimental, "reset_memory_stats"):
            return AdapterOperationResult(
                "unavailable", "TensorFlow reset_memory_stats is unavailable."
            )
        experimental.reset_memory_stats(device)
        return AdapterOperationResult("succeeded")

    def peak_device_memory(self) -> DeviceMemoryMeasurement:
        tf = self._load_tensorflow()
        device, _ = self._resolved_device(tf)
        if device is None:
            return DeviceMemoryMeasurement(
                status="unavailable",
                backend=self.backend,
                unavailable_reason=(
                    "TensorFlow device resolution is ambiguous across visible devices."
                ),
            )
        if "GPU" not in device.upper():
            return DeviceMemoryMeasurement(
                status="not_applicable",
                backend=self.backend,
                device=device,
                unavailable_reason="CPU memory is reported through process RSS.",
            )
        experimental = getattr(tf.config, "experimental", None)
        if experimental is None or not hasattr(experimental, "get_memory_info"):
            return DeviceMemoryMeasurement(
                status="unavailable",
                backend=self.backend,
                device=device,
                unavailable_reason="TensorFlow memory-info APIs are unavailable.",
            )
        info = experimental.get_memory_info(device)
        return DeviceMemoryMeasurement(
            status="measured",
            backend=self.backend,
            device=device,
            baseline_allocated_bytes=self._baseline_allocated,
            peak_allocated_bytes=int(info.get("peak", 0)),
        )

    def model_footprint(self) -> ModelFootprintMeasurement:
        return _tensorflow_model_footprint(self._model_getter())

    def deployment_artifacts(self) -> Sequence[DeploymentArtifact]:
        return self._artifacts

    def _load_tensorflow(self) -> Any:
        import tensorflow as tf

        return tf

    def _default_model_getter(self) -> Any:
        model = getattr(self.extractor, "model", None)
        return model if model is not None else getattr(self.extractor, "_model", None)

    def _resolved_device(self, tf: Any = None) -> Tuple[Optional[str], str]:
        if self.profiling_device:
            return _normalize_tensorflow_device(self.profiling_device), "explicit_profiling_hint"
        device = _tensorflow_model_device(self._model_getter())
        if device is not None:
            return device, "model_state"
        tf = tf or self._load_tensorflow()
        logical = list(tf.config.list_logical_devices("GPU"))
        if len(logical) == 1:
            return _normalize_tensorflow_device(logical[0].name), "single_visible_gpu"
        if not logical:
            return "CPU:0", "default_cpu"
        return None, "ambiguous_visible_devices"


class KerasResourceProfileAdapter(TensorFlowResourceProfileAdapter):
    """Backward-compatible name for the typed TensorFlow-family adapter."""

    def __init__(
        self,
        extractor: Any,
        artifacts: Sequence[Union[str, DeploymentArtifact]] = (),
        *,
        profiling_device: Optional[str] = None,
    ) -> None:
        super().__init__(
            extractor,
            artifacts,
            backend="keras",
            profiling_device=profiling_device,
        )


class ONNXResourceProfileAdapter(BaseResourceProfileAdapter):
    def __init__(
        self,
        extractor: Any,
        external_artifacts: Sequence[Union[str, DeploymentArtifact]] = (),
    ) -> None:
        self.extractor = extractor
        self._external_artifacts = _normalize_artifacts(external_artifacts, role="external_weights")

    def metadata(self) -> ResourceAdapterMetadata:
        session = getattr(self.extractor, "_session", None)
        providers = (
            list(session.get_providers())
            if session is not None and callable(getattr(session, "get_providers", None))
            else list(getattr(self.extractor, "providers", None) or ())
        )
        provider = providers[0] if providers else "CPUExecutionProvider"
        return ResourceAdapterMetadata(
            backend="onnxruntime",
            device=provider,
            device_resolution="active_provider" if session is not None else "configured_provider",
            asynchronous=False,
            synchronization_method="InferenceSession.run",
        )

    def synchronize(self) -> AdapterOperationResult:
        return AdapterOperationResult("succeeded")

    def reset_peak_device_memory(self) -> AdapterOperationResult:
        return AdapterOperationResult(
            "unavailable", "ONNX Runtime does not expose a portable allocator peak API."
        )

    def peak_device_memory(self) -> DeviceMemoryMeasurement:
        metadata = self.metadata()
        return DeviceMemoryMeasurement(
            status="unavailable",
            backend=metadata.backend,
            device=metadata.device,
            unavailable_reason="ONNX Runtime does not expose a portable allocator peak API.",
        )

    def model_footprint(self) -> ModelFootprintMeasurement:
        return ModelFootprintMeasurement(
            status="unavailable",
            unavailable_reason="ONNX parameters are not parsed without the optional onnx package.",
        )

    def deployment_artifacts(self) -> Sequence[DeploymentArtifact]:
        return (
            DeploymentArtifact(str(self.extractor.model_path), role="checkpoint"),
            *self._external_artifacts,
        )


class _RssSampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.peak_rss_bytes = _rss_bytes()
        self._stop = Event()
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.interval * 2))
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())


def _rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def _torch_model_footprint(model: Any) -> ModelFootprintMeasurement:
    if model is None:
        return ModelFootprintMeasurement(
            status="unavailable", unavailable_reason="The model has not been loaded."
        )
    parameters = _deduplicated_values(
        model.parameters() if callable(getattr(model, "parameters", None)) else ()
    )
    buffers = _deduplicated_values(
        model.buffers() if callable(getattr(model, "buffers", None)) else ()
    )
    if not parameters and not buffers:
        return ModelFootprintMeasurement(
            status="unavailable", unavailable_reason="The model exposes no parameters or buffers."
        )
    parameter_count = sum(int(value.numel()) for value in parameters)
    parameter_bytes = sum(int(value.numel()) * int(value.element_size()) for value in parameters)
    buffer_bytes = sum(int(value.numel()) * int(value.element_size()) for value in buffers)
    dtypes = tuple(
        sorted({str(value.dtype) for value in [*parameters, *buffers] if hasattr(value, "dtype")})
    )
    return ModelFootprintMeasurement(
        status="measured",
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        buffer_bytes=buffer_bytes,
        in_memory_bytes=parameter_bytes + buffer_bytes,
        weight_dtypes=dtypes,
    )


def _torch_model_device(model: Any) -> Any:
    if model is None:
        return None
    for method_name in ("parameters", "buffers"):
        method = getattr(model, method_name, None)
        if callable(method):
            for value in method():
                device = getattr(value, "device", None)
                if device is not None:
                    return device
    return getattr(model, "device", None)


def _torch_weight_dtypes(model: Any) -> Tuple[str, ...]:
    if model is None:
        return ()
    values: List[Any] = []
    for method_name in ("parameters", "buffers"):
        method = getattr(model, method_name, None)
        if callable(method):
            values.extend(method())
    return tuple(sorted({str(value.dtype) for value in values if hasattr(value, "dtype")}))


def _tensorflow_model_footprint(model: Any) -> ModelFootprintMeasurement:
    if model is None:
        return ModelFootprintMeasurement(
            status="unavailable", unavailable_reason="The model has not been loaded."
        )
    trainable = _deduplicated_values(
        getattr(model, "trainable_weights", ()) or getattr(model, "trainable_variables", ())
    )
    all_weights = _deduplicated_values(
        getattr(model, "weights", ()) or getattr(model, "variables", ())
    )
    if (
        not trainable
        and all_weights
        and not hasattr(model, "trainable_weights")
        and not hasattr(model, "trainable_variables")
    ):
        trainable = list(all_weights)
    trainable_ids = {id(value) for value in trainable}
    buffers = [value for value in all_weights if id(value) not in trainable_ids]
    if not trainable and not buffers:
        return ModelFootprintMeasurement(
            status="unavailable", unavailable_reason="The model exposes no variables."
        )
    parameter_count, parameter_bytes = _tensorflow_values_size(trainable)
    _, buffer_bytes = _tensorflow_values_size(buffers)
    dtypes = _tensorflow_weight_dtypes(model)
    return ModelFootprintMeasurement(
        status="measured",
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        buffer_bytes=buffer_bytes,
        in_memory_bytes=parameter_bytes + buffer_bytes,
        weight_dtypes=dtypes,
    )


def _tensorflow_values_size(values: Sequence[Any]) -> Tuple[int, int]:
    count = 0
    size = 0
    for value in values:
        shape = tuple(int(item) for item in getattr(value, "shape", ()))
        elements = int(np.prod(shape, dtype=np.int64)) if shape else 0
        dtype = getattr(value, "dtype", np.dtype("float32"))
        try:
            itemsize = np.dtype(getattr(dtype, "as_numpy_dtype", dtype)).itemsize
        except TypeError:
            itemsize = 0
        count += elements
        size += elements * itemsize
    return count, size


def _tensorflow_weight_dtypes(model: Any) -> Tuple[str, ...]:
    if model is None:
        return ()
    values = getattr(model, "weights", ()) or getattr(model, "variables", ())
    return tuple(sorted({str(getattr(value, "dtype", "unknown")) for value in values}))


def _tensorflow_model_device(model: Any) -> Optional[str]:
    if model is None:
        return None
    values = getattr(model, "weights", ()) or getattr(model, "variables", ())
    devices = {
        _normalize_tensorflow_device(str(value.device))
        for value in values
        if getattr(value, "device", None)
    }
    return next(iter(devices)) if len(devices) == 1 else None


def _normalize_tensorflow_device(value: str) -> str:
    upper = value.upper()
    if "GPU:" in upper:
        return "GPU:" + upper.rsplit("GPU:", 1)[1]
    if "CPU:" in upper:
        return "CPU:" + upper.rsplit("CPU:", 1)[1]
    return value


def _normalize_artifacts(
    artifacts: Sequence[Union[str, DeploymentArtifact]],
    *,
    role: str = "checkpoint",
) -> Tuple[DeploymentArtifact, ...]:
    return tuple(
        item if isinstance(item, DeploymentArtifact) else DeploymentArtifact(str(item), role=role)
        for item in artifacts
    )


def _artifact_footprint(declarations: Sequence[DeploymentArtifact]) -> List[ArtifactFootprint]:
    results: List[ArtifactFootprint] = []
    measured_files: set[str] = set()
    for declaration in declarations:
        raw_path = Path(declaration.path).expanduser()
        try:
            resolved = raw_path.resolve()
        except OSError as exc:
            results.append(
                _artifact_result(declaration, raw_path.absolute(), "unreadable", reason=str(exc))
            )
            continue
        if resolved.is_file():
            key = str(resolved)
            if key in measured_files:
                results.append(
                    _artifact_result(
                        declaration,
                        resolved,
                        "duplicate",
                        reason="The resolved file was already counted.",
                    )
                )
                continue
            try:
                size = int(resolved.stat().st_size)
            except OSError as exc:
                results.append(
                    _artifact_result(declaration, resolved, "unreadable", reason=str(exc))
                )
                continue
            measured_files.add(key)
            results.append(
                _artifact_result(declaration, resolved, "measured", size=size, file_count=1)
            )
            continue
        if resolved.is_dir():
            if not declaration.recursive:
                results.append(
                    _artifact_result(
                        declaration,
                        resolved,
                        "directory_requires_recursive",
                        reason="Set recursive=True to measure a deployment bundle directory.",
                    )
                )
                continue
            total = 0
            count = 0
            skipped = 0
            try:
                children = sorted(resolved.rglob("*"))
            except OSError as exc:
                results.append(
                    _artifact_result(declaration, resolved, "unreadable", reason=str(exc))
                )
                continue
            for child in children:
                try:
                    child_resolved = child.resolve()
                    if not _is_within(child_resolved, resolved) or not child_resolved.is_file():
                        if child.is_symlink():
                            skipped += 1
                        continue
                    key = str(child_resolved)
                    if key in measured_files:
                        continue
                    total += int(child_resolved.stat().st_size)
                    count += 1
                    measured_files.add(key)
                except OSError:
                    skipped += 1
            results.append(
                _artifact_result(
                    declaration,
                    resolved,
                    "partial" if skipped else "measured",
                    size=total,
                    file_count=count,
                    reason=(
                        f"Skipped {skipped} unreadable or out-of-root paths." if skipped else None
                    ),
                )
            )
            continue
        results.append(
            _artifact_result(declaration, resolved, "missing", reason="The path does not exist.")
        )
    return results


def _artifact_result(
    declaration: DeploymentArtifact,
    resolved: Path,
    status: str,
    *,
    size: int = 0,
    file_count: int = 0,
    reason: Optional[str] = None,
) -> ArtifactFootprint:
    return ArtifactFootprint(
        path=declaration.path,
        resolved_path=str(resolved),
        role=declaration.role,
        recursive=declaration.recursive,
        status=status,
        bytes=size,
        file_count=file_count,
        reason=reason,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _deduplicated_values(values: Any) -> List[Any]:
    result = []
    seen: set[int] = set()
    for value in values:
        identity = id(value)
        if identity not in seen:
            seen.add(identity)
            result.append(value)
    return result


def _nonnegative_difference(value: Optional[int], baseline: Optional[int]) -> Optional[int]:
    if value is None or baseline is None:
        return None
    return max(0, int(value) - int(baseline))


def _validate_optional_nonnegative(name: str, value: Optional[int]) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0 when provided.")
