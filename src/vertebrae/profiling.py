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
    measurement_scope: Optional[str] = None
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
        if self.status == "measured" and self.measurement_scope != "profile_window":
            raise ValueError(
                "Measured DeviceMemoryMeasurement requires "
                "measurement_scope='profile_window'."
            )
        if self.status != "measured" and self.measurement_scope is not None:
            raise ValueError(
                "Unmeasured DeviceMemoryMeasurement must not declare a measurement_scope."
            )


@dataclass(frozen=True)
class ModelFootprintMeasurement:
    """In-memory model state exposed by a framework adapter."""

    status: str
    parameter_count: Optional[int] = None
    parameter_bytes: Optional[int] = None
    trainable_parameter_count: Optional[int] = None
    trainable_parameter_bytes: Optional[int] = None
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
            "trainable_parameter_count",
            "trainable_parameter_bytes",
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
        if (
            self.parameter_count is not None
            and self.trainable_parameter_count is not None
            and self.trainable_parameter_count > self.parameter_count
        ):
            raise ValueError("trainable_parameter_count must not exceed parameter_count.")
        if (
            self.parameter_bytes is not None
            and self.trainable_parameter_bytes is not None
            and self.trainable_parameter_bytes > self.parameter_bytes
        ):
            raise ValueError("trainable_parameter_bytes must not exceed parameter_bytes.")


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
    measurement_scope: Optional[str] = None
    unavailable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status == "measured" and self.measurement_scope != "profile_window":
            raise ValueError(
                "Measured DeviceMemoryProfile requires measurement_scope='profile_window'."
            )
        if self.status != "measured" and self.measurement_scope is not None:
            raise ValueError("Unmeasured DeviceMemoryProfile must not declare a scope.")


@dataclass
class ModelFootprint:
    status: str
    parameter_status: str = "unavailable"
    checkpoint_status: str = "unavailable"
    parameter_count: Optional[int] = None
    parameter_bytes: Optional[int] = None
    trainable_parameter_count: Optional[int] = None
    trainable_parameter_bytes: Optional[int] = None
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
        self._device_reset_attempted = False
        self._device_reset_result: Optional[AdapterOperationResult] = None
        self._synchronized: Optional[bool] = None
        self._warnings: List[str] = []
        self._adapter_metadata: Optional[ResourceAdapterMetadata] = None
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
        synchronized_before = self._synchronize()
        self._accumulate_synchronization(synchronized_before)
        if (
            self.adapter is not None
            and not self._device_reset_attempted
            and self.config.device_memory
        ):
            self._device_reset_result = self._adapter_call(
                "reset_peak_device_memory",
                self.adapter.reset_peak_device_memory,
                AdapterOperationResult(
                    "unavailable", "The adapter device-memory reset hook failed."
                ),
                AdapterOperationResult,
            )
            self._device_reset_attempted = True

        sampler = _RssSampler(self.config.host_sample_interval_seconds)
        if self.config.host_memory:
            sampler.start()
        start = perf_counter()
        try:
            value = fn()
            synchronized_after = self._synchronize()
            self._accumulate_synchronization(synchronized_after)
        finally:
            elapsed = perf_counter() - start
            if self.config.host_memory:
                sampler.stop()
                self._peak_rss = max(
                    int(self._peak_rss or 0),
                    sampler.peak_rss_bytes,
                    _rss_bytes(),
                )
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
        device = self._device_profile(cache_hit=cache_hit)
        model = self._model_profile()
        metadata = self._metadata()
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
                "resource_adapter": (
                    self.adapter.__class__.__module__
                    + "."
                    + self.adapter.__class__.__name__
                    if self.adapter is not None
                    else None
                ),
                "profiling_device_hint": (
                    getattr(self.adapter, "profiling_device", None)
                    if self.adapter is not None
                    else None
                ),
            },
            warnings=sorted(set(self._warnings)),
        )

    def _resolve_adapter(self, extractor: Any) -> Any:
        factory = getattr(extractor, "get_resource_profile_adapter", None)
        if not callable(factory):
            return None
        try:
            return factory()
        except Exception as exc:
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
        except Exception as exc:
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

    def _accumulate_synchronization(self, synchronized: bool) -> None:
        self._synchronized = (
            synchronized
            if self._synchronized is None
            else bool(self._synchronized and synchronized)
        )

    def _metadata(self) -> ResourceAdapterMetadata:
        if self._adapter_metadata is None:
            self._adapter_metadata = (
                self._adapter_call(
                    "metadata",
                    self.adapter.metadata,
                    ResourceAdapterMetadata(),
                    ResourceAdapterMetadata,
                )
                if self.adapter is not None
                else ResourceAdapterMetadata()
            )
            if (
                self._adapter_metadata.device_resolution
                == "model_state_conflicts_with_profiling_hint"
            ):
                self._warnings.append(
                    "The loaded model device overrides a conflicting profiling_device hint."
                )
        return self._adapter_metadata

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

    def _device_profile(self, *, cache_hit: bool) -> DeviceMemoryProfile:
        metadata = self._metadata()
        if not self.config.device_memory:
            return DeviceMemoryProfile(
                status="disabled", backend=metadata.backend, device=metadata.device
            )
        if self.adapter is None:
            return DeviceMemoryProfile(
                status="unavailable",
                unavailable_reason="The extractor does not provide a resource adapter.",
            )
        if not self.records:
            return DeviceMemoryProfile(
                status="not_measured_cache_hit" if cache_hit else "not_measured",
                backend=metadata.backend,
                device=metadata.device,
                unavailable_reason=(
                    "Inference was served from cache; no device-memory window was measured."
                    if cache_hit
                    else "No extractor calls were measured."
                ),
            )
        reset = self._device_reset_result
        if reset is None:
            return DeviceMemoryProfile(
                status="unavailable",
                backend=metadata.backend,
                device=metadata.device,
                unavailable_reason="No device-memory measurement boundary was established.",
            )
        if reset.status != "succeeded":
            status = (
                "not_applicable"
                if reset.status == "not_applicable" and _is_cpu_device(metadata.device)
                else "unavailable"
            )
            reason = reset.reason or (
                "CPU memory is reported through process RSS."
                if status == "not_applicable"
                else "The backend cannot establish a scoped device-memory window."
            )
            if status == "unavailable":
                self._warnings.append(f"Peak device memory is unavailable: {reason}")
            return DeviceMemoryProfile(
                status=status,
                backend=metadata.backend,
                device=metadata.device,
                unavailable_reason=reason,
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
            measurement_scope=payload.measurement_scope,
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
            trainable_parameter_count=payload.trainable_parameter_count,
            trainable_parameter_bytes=payload.trainable_parameter_bytes,
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
        self._reset_device: Optional[str] = None

    def metadata(self) -> ResourceAdapterMetadata:
        torch = self._load_torch()
        devices, source = self._resolved_devices(torch)
        model = self._model_getter()
        device_names = [str(device) for device in devices]
        asynchronous = any(device.type in {"cuda", "mps"} for device in devices)
        return ResourceAdapterMetadata(
            backend="torch",
            device=",".join(device_names) if device_names else None,
            device_resolution=source,
            asynchronous=asynchronous,
            synchronization_method=(
                "torch.synchronize_each_active_device"
                if len(devices) > 1
                else "torch.cuda.synchronize"
                if devices and devices[0].type == "cuda"
                else "torch.mps.synchronize"
                if devices and devices[0].type == "mps"
                else "not_applicable"
            ),
            weight_dtypes=_torch_weight_dtypes(model),
        )

    def synchronize(self) -> AdapterOperationResult:
        torch = self._load_torch()
        devices, _ = self._resolved_devices(torch)
        synchronized = False
        mps_synchronized = False
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                synchronized = True
            elif device.type == "mps" and hasattr(torch, "mps") and not mps_synchronized:
                torch.mps.synchronize()
                synchronized = True
                mps_synchronized = True
        return AdapterOperationResult("succeeded" if synchronized else "not_applicable")

    def reset_peak_device_memory(self) -> AdapterOperationResult:
        torch = self._load_torch()
        devices, _ = self._resolved_devices(torch)
        if len(devices) > 1:
            return AdapterOperationResult(
                "unavailable",
                "multi_device_model: scoped allocator peaks are unavailable for Torch "
                "models spanning multiple devices.",
            )
        device = devices[0]
        if device.type == "cpu":
            return AdapterOperationResult(
                "not_applicable", "CPU memory is reported through process RSS."
            )
        if device.type != "cuda":
            return AdapterOperationResult(
                "unavailable",
                f"Torch does not expose resettable allocator peak counters for {device.type}.",
            )
        self._baseline_allocated = int(torch.cuda.memory_allocated(device))
        self._baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        self._reset_device = str(device)
        return AdapterOperationResult("succeeded")

    def peak_device_memory(self) -> DeviceMemoryMeasurement:
        torch = self._load_torch()
        devices, _ = self._resolved_devices(torch)
        if len(devices) > 1:
            return DeviceMemoryMeasurement(
                status="unavailable",
                backend="torch",
                device=",".join(str(device) for device in devices),
                unavailable_reason=(
                    "multi_device_model: scoped allocator peaks are unavailable for Torch "
                    "models spanning multiple devices."
                ),
            )
        device = devices[0]
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
        if self._reset_device != str(device):
            return DeviceMemoryMeasurement(
                status="unavailable",
                backend="torch",
                device=str(device),
                unavailable_reason=(
                    "The active Torch device changed after the allocator measurement "
                    "boundary was established."
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
            measurement_scope="profile_window",
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

    def _resolved_devices(self, torch: Any) -> Tuple[Tuple[Any, ...], str]:
        model_devices = _torch_model_devices(self._model_getter())
        if model_devices:
            devices = tuple(torch.device(value) for value in model_devices)
            source = "multi_device_model" if len(devices) > 1 else "model_state"
            if len(devices) == 1:
                hinted: Any = None
                if self._device_resolver is not None:
                    hinted = self._device_resolver(torch)
                elif getattr(self.extractor, "device", None) is not None:
                    hinted = self.extractor.device
                if hinted is not None and str(torch.device(hinted)) != str(devices[0]):
                    source = "model_state_conflicts_with_profiling_hint"
            return devices, source
        if self._device_resolver is not None:
            value = self._device_resolver(torch)
            if value is not None:
                return (torch.device(value),), "extractor_resolver"
        explicit = getattr(self.extractor, "device", None)
        if explicit is not None:
            return (torch.device(explicit),), "explicit"
        resolver = getattr(self.extractor, "_device", None)
        if callable(resolver):
            return (torch.device(resolver(torch)),), "extractor_resolver"
        return (torch.device("cpu"),), "default_cpu"


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
        self._reset_device: Optional[str] = None

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
        self._reset_device = device
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
        if self._reset_device != device:
            return DeviceMemoryMeasurement(
                status="unavailable",
                backend=self.backend,
                device=device,
                unavailable_reason=(
                    "The active TensorFlow device changed after the allocator measurement "
                    "boundary was established."
                ),
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
            measurement_scope="profile_window",
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
        model_devices = _tensorflow_model_devices(self._model_getter())
        hint = (
            _normalize_tensorflow_device(self.profiling_device)
            if self.profiling_device
            else None
        )
        if len(model_devices) == 1:
            device = model_devices[0]
            return (
                device,
                "model_state_conflicts_with_profiling_hint"
                if hint is not None and hint != device
                else "model_state",
            )
        if hint is not None:
            return hint, "explicit_profiling_hint"
        if len(model_devices) > 1:
            return None, "ambiguous_model_devices"
        tf = tf or self._load_tensorflow()
        logical = list(tf.config.list_logical_devices("GPU"))
        if len(logical) == 1:
            return _normalize_tensorflow_device(logical[0].name), "single_visible_gpu"
        if not logical:
            return "CPU:0", "default_cpu"
        return None, "ambiguous_visible_devices"


class JAXResourceProfileAdapter(BaseResourceProfileAdapter):
    """Synchronization and device metadata for JAX-backed extractors."""

    def __init__(
        self,
        extractor: Any,
        *,
        values_getter: Optional[Callable[[], Any]] = None,
        jax_loader: Optional[Callable[[], Any]] = None,
        backend: str = "jax",
        profiling_device: Optional[str] = None,
    ) -> None:
        self.extractor = extractor
        self._values_getter = values_getter or (lambda: ())
        self._jax_loader = jax_loader
        self.backend = backend
        self.profiling_device = profiling_device

    def metadata(self) -> ResourceAdapterMetadata:
        jax = self._load_jax()
        values = _jax_tree_leaves(jax, self._values_getter())
        devices = _jax_value_devices(values)
        device: Optional[str]
        if devices:
            device = ",".join(devices)
            source = "multi_device_model" if len(devices) > 1 else "model_state"
            if (
                len(devices) == 1
                and self.profiling_device is not None
                and self.profiling_device != devices[0]
            ):
                source = "model_state_conflicts_with_profiling_hint"
        elif self.profiling_device:
            device = self.profiling_device
            source = "explicit_profiling_hint"
        else:
            local_devices = list(jax.local_devices())
            device = str(local_devices[0]) if len(local_devices) == 1 else None
            source = "single_local_device" if device is not None else "ambiguous_local_devices"
        asynchronous = bool(
            device and not all(_is_cpu_device(item) for item in devices or [device])
        )
        return ResourceAdapterMetadata(
            backend=self.backend,
            device=device,
            device_resolution=source,
            asynchronous=asynchronous,
            synchronization_method="jax.effects_barrier",
            weight_dtypes=_keras_weight_dtypes(values),
        )

    def synchronize(self) -> AdapterOperationResult:
        jax = self._load_jax()
        barrier = getattr(jax, "effects_barrier", None)
        if not callable(barrier):
            return AdapterOperationResult(
                "unavailable", "JAX effects_barrier is unavailable."
            )
        barrier()
        return AdapterOperationResult("succeeded")

    def reset_peak_device_memory(self) -> AdapterOperationResult:
        metadata = self.metadata()
        if metadata.device and all(
            _is_cpu_device(item) for item in metadata.device.split(",")
        ):
            return AdapterOperationResult(
                "not_applicable", "CPU memory is reported through process RSS."
            )
        return AdapterOperationResult(
            "unavailable",
            "JAX does not expose a portable resettable allocator peak window.",
        )

    def peak_device_memory(self) -> DeviceMemoryMeasurement:
        metadata = self.metadata()
        return DeviceMemoryMeasurement(
            status="unavailable",
            backend=self.backend,
            device=metadata.device,
            unavailable_reason=(
                "JAX does not expose a portable resettable allocator peak window."
            ),
        )

    def model_footprint(self) -> ModelFootprintMeasurement:
        jax = self._load_jax()
        values = _jax_tree_leaves(jax, self._values_getter())
        if not values:
            return ModelFootprintMeasurement(
                status="unavailable", unavailable_reason="The JAX parameter tree is empty."
            )
        parameter_count, parameter_bytes = _tensorflow_values_size(values)
        return ModelFootprintMeasurement(
            status="measured",
            parameter_count=parameter_count,
            parameter_bytes=parameter_bytes,
            buffer_bytes=None,
            in_memory_bytes=parameter_bytes,
            weight_dtypes=_keras_weight_dtypes(values),
        )

    def _load_jax(self) -> Any:
        if self._jax_loader is not None:
            value = self._jax_loader()
            if value is not None:
                return value
        existing = getattr(self.extractor, "_jax", None)
        if existing is not None:
            return existing
        import jax

        return jax


class KerasResourceProfileAdapter(BaseResourceProfileAdapter):
    """Backend-aware resource hooks for multi-backend Keras models."""

    def __init__(
        self,
        extractor: Any,
        artifacts: Sequence[Union[str, DeploymentArtifact]] = (),
        *,
        profiling_device: Optional[str] = None,
    ) -> None:
        self.extractor = extractor
        self._artifacts = _normalize_artifacts(artifacts)
        self.profiling_device = profiling_device
        self._delegate: Optional[BaseResourceProfileAdapter] = None
        self._backend: Optional[str] = None

    def metadata(self) -> ResourceAdapterMetadata:
        backend = self._backend_name()
        payload = self._backend_adapter().metadata()
        return replace(
            payload,
            backend=f"keras-{backend}",
            weight_dtypes=_keras_weight_dtypes(self._weights()),
        )

    def synchronize(self) -> AdapterOperationResult:
        return self._backend_adapter().synchronize()

    def reset_peak_device_memory(self) -> AdapterOperationResult:
        return self._backend_adapter().reset_peak_device_memory()

    def peak_device_memory(self) -> DeviceMemoryMeasurement:
        payload = self._backend_adapter().peak_device_memory()
        return replace(payload, backend=f"keras-{self._backend_name()}")

    def model_footprint(self) -> ModelFootprintMeasurement:
        return _tensorflow_model_footprint(self._model())

    def deployment_artifacts(self) -> Sequence[DeploymentArtifact]:
        return self._artifacts

    def _backend_adapter(self) -> BaseResourceProfileAdapter:
        backend = self._backend_name()
        if self._delegate is not None:
            return self._delegate
        if backend == "tensorflow":
            self._delegate = TensorFlowResourceProfileAdapter(
                self.extractor,
                model_getter=self._model,
                backend="keras-tensorflow",
                profiling_device=self.profiling_device,
            )
        elif backend == "torch":
            self._delegate = TorchResourceProfileAdapter(
                self.extractor,
                model_getter=self._model,
                device_resolver=(
                    (lambda torch: self.profiling_device)
                    if self.profiling_device is not None
                    else None
                ),
            )
        elif backend == "jax":
            self._delegate = JAXResourceProfileAdapter(
                self.extractor,
                values_getter=self._weights,
                backend="keras-jax",
                profiling_device=self.profiling_device,
            )
        else:
            self._delegate = _UnavailableBackendResourceProfileAdapter(
                backend=f"keras-{backend}",
                reason=f"Keras backend '{backend}' is not supported for device profiling.",
            )
        return self._delegate

    def _backend_name(self) -> str:
        if self._backend is None:
            keras = self._load_keras()
            backend_fn = getattr(getattr(keras, "backend", None), "backend", None)
            if not callable(backend_fn):
                raise RuntimeError("Keras does not expose backend.backend().")
            self._backend = str(backend_fn()).lower()
        return self._backend

    def _load_keras(self) -> Any:
        loader = getattr(self.extractor, "_load_keras", None)
        if callable(loader):
            return loader()
        try:
            import keras

            return keras
        except ImportError:
            from tensorflow import keras

            return keras

    def _model(self) -> Any:
        model = getattr(self.extractor, "model", None)
        return model if model is not None else getattr(self.extractor, "_model", None)

    def _weights(self) -> Any:
        model = self._model()
        if model is None:
            return ()
        return getattr(model, "weights", ()) or getattr(model, "variables", ())


class _UnavailableBackendResourceProfileAdapter(BaseResourceProfileAdapter):
    def __init__(self, *, backend: str, reason: str) -> None:
        self.backend = backend
        self.reason = reason

    def metadata(self) -> ResourceAdapterMetadata:
        return ResourceAdapterMetadata(backend=self.backend)

    def synchronize(self) -> AdapterOperationResult:
        return AdapterOperationResult("unavailable", self.reason)

    def reset_peak_device_memory(self) -> AdapterOperationResult:
        return AdapterOperationResult("unavailable", self.reason)

    def peak_device_memory(self) -> DeviceMemoryMeasurement:
        return DeviceMemoryMeasurement(
            status="unavailable", backend=self.backend, unavailable_reason=self.reason
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
    trainable = [value for value in parameters if getattr(value, "requires_grad", True)]
    trainable_parameter_count = sum(int(value.numel()) for value in trainable)
    trainable_parameter_bytes = sum(
        int(value.numel()) * int(value.element_size()) for value in trainable
    )
    buffer_bytes = sum(int(value.numel()) * int(value.element_size()) for value in buffers)
    dtypes = tuple(
        sorted({str(value.dtype) for value in [*parameters, *buffers] if hasattr(value, "dtype")})
    )
    return ModelFootprintMeasurement(
        status="measured",
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        trainable_parameter_count=trainable_parameter_count,
        trainable_parameter_bytes=trainable_parameter_bytes,
        buffer_bytes=buffer_bytes,
        in_memory_bytes=parameter_bytes + buffer_bytes,
        weight_dtypes=dtypes,
    )


def _torch_model_devices(model: Any) -> Tuple[Any, ...]:
    if model is None:
        return ()
    devices: Dict[str, Any] = {}
    for method_name in ("parameters", "buffers"):
        method = getattr(model, method_name, None)
        if callable(method):
            for value in method():
                device = getattr(value, "device", None)
                if device is not None:
                    devices[str(device)] = device
    for value in getattr(model, "weights", ()) or getattr(model, "variables", ()):
        unwrapped = getattr(value, "value", value)
        device = getattr(unwrapped, "device", None)
        if device is not None:
            devices[str(device)] = device
    model_device = getattr(model, "device", None)
    if not devices and model_device is not None:
        devices[str(model_device)] = model_device
    return tuple(devices[key] for key in sorted(devices))


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
    if not trainable and all_weights and not any(
        hasattr(model, name) for name in ("trainable_weights", "trainable_variables")
    ):
        trainable = list(all_weights)
    if not all_weights:
        return ModelFootprintMeasurement(
            status="unavailable", unavailable_reason="The model exposes no variables."
        )
    parameter_count, parameter_bytes = _tensorflow_values_size(all_weights)
    trainable_parameter_count, trainable_parameter_bytes = _tensorflow_values_size(trainable)
    dtypes = _tensorflow_weight_dtypes(model)
    return ModelFootprintMeasurement(
        status="measured",
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        trainable_parameter_count=trainable_parameter_count,
        trainable_parameter_bytes=trainable_parameter_bytes,
        buffer_bytes=None,
        in_memory_bytes=parameter_bytes,
        weight_dtypes=dtypes,
    )


def _tensorflow_values_size(values: Sequence[Any]) -> Tuple[int, int]:
    count = 0
    size = 0
    for value in values:
        raw_shape = getattr(value, "shape", None)
        if raw_shape is None:
            continue
        shape = tuple(int(item) for item in raw_shape)
        elements = int(np.prod(shape, dtype=np.int64))
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
    return _keras_weight_dtypes(values)


def _tensorflow_model_devices(model: Any) -> Tuple[str, ...]:
    if model is None:
        return ()
    values = getattr(model, "weights", ()) or getattr(model, "variables", ())
    devices = set()
    for value in values:
        unwrapped = getattr(value, "value", value)
        device = getattr(unwrapped, "device", None)
        if device:
            devices.add(_normalize_tensorflow_device(str(device)))
    return tuple(sorted(devices))


def _keras_weight_dtypes(values: Any) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(getattr(getattr(value, "value", value), "dtype", "unknown"))
                for value in values
            }
        )
    )


def _jax_value_devices(values: Any) -> Tuple[str, ...]:
    devices: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        unwrapped = getattr(value, "value", value)
        value_devices = getattr(unwrapped, "devices", None)
        if callable(value_devices):
            devices.update(str(device) for device in value_devices())
            return
        device = getattr(unwrapped, "device", None)
        if device is not None:
            devices.add(str(device() if callable(device) else device))

    visit(values)
    return tuple(sorted(devices))


def _jax_tree_leaves(jax: Any, values: Any) -> List[Any]:
    tree_util = getattr(jax, "tree_util", None)
    tree_leaves = getattr(tree_util, "tree_leaves", None)
    if callable(tree_leaves):
        return list(tree_leaves(values))
    if isinstance(values, dict):
        return [item for value in values.values() for item in _jax_tree_leaves(jax, value)]
    if isinstance(values, (list, tuple)):
        return [item for value in values for item in _jax_tree_leaves(jax, value)]
    return [] if values is None else [values]


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


def _is_cpu_device(device: Optional[str]) -> bool:
    return bool(device and "cpu" in device.lower())


def _validate_optional_nonnegative(name: str, value: Optional[int]) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0 when provided.")
