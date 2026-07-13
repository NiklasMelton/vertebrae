"""Measured inference-resource profiles for local representation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import psutil

from vertebrae.config import ResourceProfilingConfig
from vertebrae.utils.memory import estimate_matrix_resident_bytes


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
    peak_allocated_bytes: Optional[int] = None
    peak_reserved_bytes: Optional[int] = None


@dataclass
class ModelFootprint:
    status: str
    parameter_count: Optional[int] = None
    parameter_bytes: Optional[int] = None
    buffer_bytes: Optional[int] = None
    in_memory_bytes: Optional[int] = None
    checkpoint_bytes: Optional[int] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)


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
        self.adapter = _resolve_adapter(extractor)
        self.records: List[_CallRecord] = []
        self._baseline_rss: Optional[int] = None
        self._peak_rss: Optional[int] = None
        self._device_reset = False
        self._synchronized: Optional[bool] = None
        self._warnings: List[str] = []

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
            self._device_reset = bool(
                _safe_adapter_call(self.adapter.reset_peak_device_memory, False)
            )

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
        metadata = _safe_adapter_call(self.adapter.metadata, {}) if self.adapter is not None else {}
        if self.records and self._synchronized is False and metadata.get("asynchronous", False):
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
                **metadata,
            },
            warnings=sorted(set(self._warnings)),
        )

    def _synchronize(self) -> bool:
        if self.adapter is None:
            return True
        return bool(_safe_adapter_call(self.adapter.synchronize, False))

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
        metadata = _safe_adapter_call(self.adapter.metadata, {}) if self.adapter is not None else {}
        if not self.config.device_memory:
            return DeviceMemoryProfile(status="disabled", **_device_identity(metadata))
        if self.adapter is None:
            return DeviceMemoryProfile(status="unavailable")
        payload = _safe_adapter_call(
            self.adapter.peak_device_memory,
            {"status": "unavailable"},
        )
        status = str(payload.get("status", "unavailable"))
        if status == "unavailable":
            self._warnings.append("Peak device memory is unavailable for this extractor backend.")
        return DeviceMemoryProfile(
            status=status,
            backend=payload.get("backend", metadata.get("backend")),
            device=payload.get("device", metadata.get("device")),
            peak_allocated_bytes=_optional_int(payload.get("peak_allocated_bytes")),
            peak_reserved_bytes=_optional_int(payload.get("peak_reserved_bytes")),
        )

    def _model_profile(self) -> ModelFootprint:
        payload = (
            _safe_adapter_call(self.adapter.model_footprint, {"status": "unavailable"})
            if self.adapter is not None
            else {"status": "unavailable"}
        )
        artifacts = _artifact_footprint(
            _safe_adapter_call(self.adapter.deployment_artifacts, ())
            if self.adapter is not None
            else ()
        )
        checkpoint_bytes = sum(item["bytes"] for item in artifacts if item["status"] == "measured")
        parameter_bytes = _optional_int(payload.get("parameter_bytes"))
        buffer_bytes = _optional_int(payload.get("buffer_bytes"))
        in_memory = payload.get("in_memory_bytes")
        if in_memory is None and parameter_bytes is not None:
            in_memory = parameter_bytes + int(buffer_bytes or 0)
        measured = payload.get("status") == "measured" or any(
            item["status"] == "measured" for item in artifacts
        )
        if not measured:
            self._warnings.append("Model parameter and checkpoint footprint is unavailable.")
        if any(item["status"] == "missing" for item in artifacts):
            self._warnings.append("One or more explicit deployment artifacts do not exist.")
        return ModelFootprint(
            status="measured" if measured else "unavailable",
            parameter_count=_optional_int(payload.get("parameter_count")),
            parameter_bytes=parameter_bytes,
            buffer_bytes=buffer_bytes,
            in_memory_bytes=_optional_int(in_memory),
            checkpoint_bytes=(checkpoint_bytes if measured and artifacts else None),
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


class TorchResourceProfileAdapter:
    def __init__(self, extractor: Any, artifacts: Sequence[str] = ()) -> None:
        self.extractor = extractor
        self._artifacts = tuple(artifacts)

    def metadata(self) -> Dict[str, Any]:
        device = self.extractor.device or "cpu"
        return {
            "backend": "torch",
            "device": str(device),
            "asynchronous": str(device).startswith(("cuda", "mps")),
        }

    def synchronize(self) -> bool:
        torch = self.extractor._load_torch()
        device = str(self.extractor.device or "cpu")
        if device.startswith("cuda"):
            torch.cuda.synchronize(self.extractor.device)
        elif device.startswith("mps") and hasattr(torch, "mps"):
            torch.mps.synchronize()
        return True

    def reset_peak_device_memory(self) -> bool:
        torch = self.extractor._load_torch()
        device = str(self.extractor.device or "cpu")
        if not device.startswith("cuda"):
            return False
        torch.cuda.reset_peak_memory_stats(self.extractor.device)
        return True

    def peak_device_memory(self) -> Dict[str, Any]:
        torch = self.extractor._load_torch()
        device = str(self.extractor.device or "cpu")
        if not device.startswith("cuda"):
            return {"status": "unavailable", "backend": "torch", "device": device}
        return {
            "status": "measured",
            "backend": "torch",
            "device": device,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.extractor.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.extractor.device)),
        }

    def model_footprint(self) -> Dict[str, Any]:
        return _iterable_model_footprint(self.extractor.model)

    def deployment_artifacts(self) -> Sequence[str]:
        return self._artifacts


class KerasResourceProfileAdapter:
    def __init__(self, extractor: Any, artifacts: Sequence[str] = ()) -> None:
        self.extractor = extractor
        self._artifacts = tuple(artifacts)

    def metadata(self) -> Dict[str, Any]:
        return {"backend": "keras", "device": None, "asynchronous": False}

    def synchronize(self) -> bool:
        return True

    def reset_peak_device_memory(self) -> bool:
        return False

    def peak_device_memory(self) -> Dict[str, Any]:
        return {"status": "unavailable", "backend": "keras"}

    def model_footprint(self) -> Dict[str, Any]:
        weights = list(getattr(self.extractor.model, "weights", ()))
        if not weights:
            return {"status": "unavailable"}
        count = 0
        size = 0
        for weight in weights:
            shape = tuple(int(item) for item in getattr(weight, "shape", ()))
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 0
            dtype = getattr(weight, "dtype", np.dtype("float32"))
            try:
                itemsize = np.dtype(getattr(dtype, "as_numpy_dtype", dtype)).itemsize
            except TypeError:
                itemsize = 0
            count += elements
            size += elements * itemsize
        return {
            "status": "measured",
            "parameter_count": count,
            "parameter_bytes": size,
            "buffer_bytes": 0,
        }

    def deployment_artifacts(self) -> Sequence[str]:
        return self._artifacts


class ONNXResourceProfileAdapter:
    def __init__(self, extractor: Any) -> None:
        self.extractor = extractor

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "onnxruntime",
            "device": (self.extractor.providers or [None])[0],
            "asynchronous": False,
        }

    def synchronize(self) -> bool:
        return True

    def reset_peak_device_memory(self) -> bool:
        return False

    def peak_device_memory(self) -> Dict[str, Any]:
        return {"status": "unavailable", **_device_identity(self.metadata())}

    def model_footprint(self) -> Dict[str, Any]:
        return {"status": "unavailable"}

    def deployment_artifacts(self) -> Sequence[str]:
        return (str(self.extractor.model_path),)


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


def _resolve_adapter(extractor: Any) -> Any:
    factory = getattr(extractor, "get_resource_profile_adapter", None)
    return factory() if callable(factory) else None


def _rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def _safe_adapter_call(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return default


def _iterable_model_footprint(model: Any) -> Dict[str, Any]:
    parameters = list(model.parameters()) if callable(getattr(model, "parameters", None)) else []
    buffers = list(model.buffers()) if callable(getattr(model, "buffers", None)) else []
    if not parameters and not buffers:
        return {"status": "unavailable"}
    parameter_count = sum(int(value.numel()) for value in parameters)
    parameter_bytes = sum(int(value.numel()) * int(value.element_size()) for value in parameters)
    buffer_bytes = sum(int(value.numel()) * int(value.element_size()) for value in buffers)
    return {
        "status": "measured",
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
    }


def _artifact_footprint(paths: Sequence[str]) -> List[Dict[str, Any]]:
    files: Dict[str, Path] = {}
    missing: Dict[str, Path] = {}
    for raw in paths:
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        if resolved.is_file():
            files[str(resolved)] = resolved
        elif resolved.is_dir():
            for child in resolved.rglob("*"):
                if child.is_file():
                    files[str(child.resolve())] = child.resolve()
        else:
            missing[str(resolved)] = resolved
    measured = [
        {"path": key, "bytes": int(path.stat().st_size), "status": "measured"}
        for key, path in sorted(files.items())
    ]
    return measured + [{"path": key, "bytes": 0, "status": "missing"} for key in sorted(missing)]


def _device_identity(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {"backend": metadata.get("backend"), "device": metadata.get("device")}


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)
