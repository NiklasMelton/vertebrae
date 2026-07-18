"""Framework-neutral monitoring for repeated labeled benchmark evaluations."""

import json
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    TextIO,
    Union,
)

from vertebrae._version import __version__
from vertebrae.benchmark import Benchmark
from vertebrae.cache.fingerprint import hash_json_exact
from vertebrae.config import CacheConfig
from vertebrae.results import (
    BenchmarkResult,
    benchmark_result_columns,
    null_benchmark_result_row,
)
from vertebrae.utils.serialization import json_dumps_strict, make_json_safe

_HISTORY_SCHEMA = "vertebrae.evaluation_history"
_HISTORY_SCHEMA_VERSION = 2
_IDENTIFIER_FIELDS = ("snapshot_id", "epoch", "global_step", "timestamp", "checkpoint")
_HISTORY_CONTEXT_COLUMNS = (
    "evaluation_index",
    "status",
    "snapshot_id",
    "epoch",
    "global_step",
    "timestamp",
    "checkpoint",
    "recorded_at",
    "error_type",
    "error_message",
)
_CONTEXT_KEYS = frozenset(
    {
        "snapshot_id",
        "epoch",
        "global_step",
        "timestamp",
        "checkpoint",
        "metadata",
        "recorded_at",
    }
)
_RECORD_KEYS = frozenset(
    {
        "context",
        "status",
        "evaluation_index",
        "rows",
        "benchmark_result",
        "error_type",
        "error_message",
        "error_traceback",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "record_type",
        "schema",
        "schema_version",
        "detail",
        "vertebrae_version",
        "created_at",
        "monitor_metadata",
    }
)
_MONITOR_METADATA_KEYS = frozenset({"protocol", "protocol_hash", "metric_names"})


@dataclass(frozen=True)
class EvaluationContext:
    """Normalized caller-provided coordinates for one benchmark evaluation."""

    snapshot_id: Optional[str] = None
    epoch: Optional[int] = None
    global_step: Optional[int] = None
    timestamp: Optional[Union[str, datetime]] = None
    checkpoint: Optional[Union[str, Path]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    recorded_at: Union[str, datetime] = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_id",
            _normalize_optional_string(self.snapshot_id, "snapshot_id"),
        )
        object.__setattr__(self, "epoch", _normalize_optional_index(self.epoch, "epoch"))
        object.__setattr__(
            self,
            "global_step",
            _normalize_optional_index(self.global_step, "global_step"),
        )
        object.__setattr__(
            self,
            "timestamp",
            _normalize_optional_timestamp(self.timestamp, "timestamp"),
        )
        object.__setattr__(
            self,
            "checkpoint",
            _normalize_optional_path(self.checkpoint, "checkpoint"),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _normalize_required_timestamp(self.recorded_at, "recorded_at"),
        )
        object.__setattr__(self, "metadata", _normalize_context_metadata(self.metadata))
        if not any(getattr(self, name) is not None for name in _IDENTIFIER_FIELDS):
            raise ValueError(
                "At least one evaluation identifier is required: snapshot_id, epoch, "
                "global_step, timestamp, or checkpoint."
            )

    def identity_payload(self) -> Dict[str, Any]:
        """Return the explicitly populated fields used for duplicate detection."""

        return {
            name: getattr(self, name)
            for name in _IDENTIFIER_FIELDS
            if getattr(self, name) is not None
        }

    def identity_signature(self) -> str:
        """Return a deterministic signature for the evaluation coordinates."""

        return hash_json_exact(self.identity_payload())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the context to strict JSON-compatible data."""

        return make_json_safe(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationContext":
        """Restore a context from persisted history data."""

        _require_exact_keys(value, _CONTEXT_KEYS, "evaluation context")
        _require_optional_json_string(value["snapshot_id"], "context.snapshot_id")
        _require_optional_json_integer(value["epoch"], "context.epoch")
        _require_optional_json_integer(value["global_step"], "context.global_step")
        _require_optional_json_string(value["timestamp"], "context.timestamp")
        _require_optional_json_string(value["checkpoint"], "context.checkpoint")
        _require_json_mapping(value["metadata"], "context.metadata")
        _require_json_string(value["recorded_at"], "context.recorded_at")
        return cls(
            snapshot_id=value["snapshot_id"],
            epoch=value["epoch"],
            global_step=value["global_step"],
            timestamp=value["timestamp"],
            checkpoint=value["checkpoint"],
            metadata=dict(value["metadata"]),
            recorded_at=value["recorded_at"],
        )


@dataclass(frozen=True)
class EvaluationHistoryConfig:
    """Storage settings for representation-monitoring history."""

    storage: str = "memory"
    path: Optional[Union[str, Path]] = None
    detail: str = "summary"
    resume: bool = False

    def __post_init__(self) -> None:
        if self.storage not in {"memory", "disk"}:
            raise ValueError("storage must be either 'memory' or 'disk'.")
        if self.detail not in {"summary", "full"}:
            raise ValueError("detail must be either 'summary' or 'full'.")
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be a bool.")
        if self.storage == "memory":
            if self.path is not None:
                raise ValueError("path is only valid when storage='disk'.")
            if self.resume:
                raise ValueError("resume is only valid when storage='disk'.")
            return
        normalized = _normalize_optional_path(self.path, "path")
        if normalized is None:
            raise ValueError("path is required when storage='disk'.")
        object.__setattr__(self, "path", normalized)


@dataclass
class EvaluationRecord:
    """One committed monitoring evaluation and its operational summary rows."""

    context: EvaluationContext
    status: str
    evaluation_index: int
    rows: List[Dict[str, Any]]
    benchmark_result: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, EvaluationContext):
            raise TypeError("EvaluationRecord.context must be an EvaluationContext.")
        if self.status not in {"success", "failure"}:
            raise ValueError("EvaluationRecord.status must be 'success' or 'failure'.")
        if (
            isinstance(self.evaluation_index, bool)
            or not isinstance(self.evaluation_index, Integral)
            or self.evaluation_index < 0
        ):
            raise ValueError("evaluation_index must be a nonnegative integer.")
        self.evaluation_index = int(self.evaluation_index)
        if not isinstance(self.rows, list) or any(
            not isinstance(row, Mapping) for row in self.rows
        ):
            raise TypeError("rows must be a list of mappings.")
        normalized_rows = make_json_safe([dict(row) for row in self.rows])
        if not isinstance(normalized_rows, list):
            raise TypeError("rows must serialize to a list.")
        self.rows = [dict(row) for row in normalized_rows]
        if self.benchmark_result is not None:
            if not isinstance(self.benchmark_result, Mapping):
                raise TypeError("benchmark_result must be a mapping when provided.")
            normalized_result = make_json_safe(dict(self.benchmark_result))
            if not isinstance(normalized_result, dict):
                raise TypeError("benchmark_result must serialize to a mapping.")
            self.benchmark_result = normalized_result
        for name in ("error_type", "error_message", "error_traceback"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string when provided.")
        if self.status == "success":
            if (
                self.error_type is not None
                or self.error_message is not None
                or self.error_traceback is not None
            ):
                raise ValueError("Successful evaluation records cannot contain failure fields.")
        else:
            if self.rows != [{}]:
                raise ValueError("Failed evaluation records must contain one empty result row.")
            if not self.error_type or not self.error_message:
                raise ValueError("Failed evaluation records require error_type and error_message.")
            if self.benchmark_result is not None:
                raise ValueError("Failed evaluation records cannot contain benchmark_result.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this record to strict JSON-compatible data."""

        return make_json_safe(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationRecord":
        """Restore a record from persisted history data."""

        _require_exact_keys(value, _RECORD_KEYS, "evaluation record")
        _require_json_mapping(value["context"], "record.context")
        status = _require_json_string(value["status"], "record.status")
        evaluation_index = _require_json_integer(
            value["evaluation_index"],
            "record.evaluation_index",
        )
        rows = _require_json_list(value["rows"], "record.rows")
        if any(not isinstance(row, Mapping) for row in rows):
            raise TypeError("record.rows must contain only JSON objects.")
        if status == "failure" and rows != [{}]:
            raise ValueError("Failed persisted records must contain one empty result row.")
        benchmark_result = value["benchmark_result"]
        if benchmark_result is not None:
            _require_json_mapping(benchmark_result, "record.benchmark_result")
        for name in ("error_type", "error_message", "error_traceback"):
            _require_optional_json_string(value[name], f"record.{name}")
        return cls(
            context=EvaluationContext.from_dict(value["context"]),
            status=status,
            evaluation_index=evaluation_index,
            rows=[dict(row) for row in rows],
            benchmark_result=(
                dict(benchmark_result) if benchmark_result is not None else None
            ),
            error_type=value["error_type"],
            error_message=value["error_message"],
            error_traceback=value["error_traceback"],
        )


class EvaluationReporter(Protocol):
    """Observer protocol invoked after a monitoring record is committed."""

    def report(self, record: EvaluationRecord) -> None:
        """Report one committed evaluation."""


class ConsoleReporter:
    """Print compact monitoring results without third-party console dependencies."""

    def __init__(self, stream: Optional[TextIO] = None, precision: int = 4) -> None:
        if isinstance(precision, bool) or not isinstance(precision, Integral) or precision < 0:
            raise ValueError("precision must be a nonnegative integer.")
        self.stream: Optional[TextIO] = stream
        self.precision: int = int(precision)

    def report(self, record: EvaluationRecord) -> None:
        """Print one compact line per result row."""

        stream = self.stream or sys.stdout
        identifiers = ", ".join(
            f"{name}={getattr(record.context, name)}"
            for name in _IDENTIFIER_FIELDS
            if getattr(record.context, name) is not None
        )
        print(
            f"[vertebrae] evaluation={record.evaluation_index} "
            f"status={record.status} {identifiers}",
            file=stream,
        )
        if record.status == "failure":
            print(
                f"  {record.error_type or 'Error'}: {record.error_message or 'unknown failure'}",
                file=stream,
            )
            return
        for row in record.rows:
            output = row.get("output_name") or row.get("extractor") or "output"
            layer = row.get("hidden_layer")
            layer_text = f" layer={layer}" if layer is not None else ""
            primary_name = row.get("primary_metric") or "primary"
            primary_score = _format_score(row.get("primary_score"), self.precision)
            overlap_score = _format_score(row.get("overlap_score"), self.precision)
            separatix = (
                row.get("separatix_recommendation")
                or row.get("separatix_skip_reason")
                or ("ran" if row.get("separatix_ran") else "disabled")
            )
            print(
                f"  {output}{layer_text} {primary_name}={primary_score} "
                f"overlap={overlap_score} separatix={separatix}",
                file=stream,
            )


class EvaluationHistory:
    """Memory- or local JSONL-backed representation-monitoring history."""

    def __init__(
        self,
        config: Optional[EvaluationHistoryConfig] = None,
        *,
        monitor_metadata: Optional[Mapping[str, Any]] = None,
        _read_only: bool = False,
    ) -> None:
        self.config = config or EvaluationHistoryConfig()
        self.monitor_metadata = make_json_safe(dict(monitor_metadata or {}))
        self._read_only = _read_only
        self._records: List[EvaluationRecord] = []
        self._signatures: set[str] = set()
        self._observed_result_columns: set[str] = set()
        self._observed_metadata_columns: set[str] = set()
        self._next_index = 0
        self._record_count = 0
        if self.config.storage == "disk":
            self._initialize_disk()

    @property
    def detail(self) -> str:
        """Return the configured history detail level."""

        return self.config.detail

    @property
    def next_evaluation_index(self) -> int:
        """Return the index that must be assigned to the next record."""

        return self._next_index

    def __len__(self) -> int:
        return self._record_count

    def contains_context(self, context: EvaluationContext) -> bool:
        """Return whether these evaluation coordinates were already recorded."""

        return context.identity_signature() in self._signatures

    def append(self, record: EvaluationRecord) -> None:
        """Commit one record to the configured history."""

        if self._read_only:
            raise RuntimeError("Loaded evaluation histories are read-only.")
        if self.detail == "summary" and (
            record.benchmark_result is not None or record.error_traceback is not None
        ):
            record = replace(
                record,
                benchmark_result=None,
                error_traceback=None,
            )
        _validate_record_detail(record, self.detail)
        if record.evaluation_index != self._next_index:
            raise ValueError(
                "Evaluation record index does not continue history: expected "
                f"{self._next_index}, got {record.evaluation_index}."
            )
        if self.config.storage == "memory":
            self._records.append(record)
        else:
            payload = {"record_type": "evaluation", **record.to_dict()}
            line = json_dumps_strict(payload, sort_keys=True) + "\n"
            with self._path().open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
        self._register(record)

    def iter_records(self) -> Iterator[EvaluationRecord]:
        """Iterate records, streaming disk-backed histories."""

        if self.config.storage == "memory":
            yield from self._records
            return
        yield from self._iter_disk_records()

    def to_dataframe(self) -> Any:
        """Return one tidy row per evaluation result variant."""

        import pandas as pd

        records = list(self.iter_records())
        return pd.DataFrame(
            _history_rows(records, self._result_columns()),
            columns=self._dataframe_columns(),
        )

    def latest_dataframe(self) -> Any:
        """Return rows from only the most recently committed evaluation."""

        import pandas as pd

        latest = None
        for record in self.iter_records():
            latest = record
        rows = [] if latest is None else _history_rows([latest], self._result_columns())
        return pd.DataFrame(rows, columns=self._dataframe_columns())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationHistory":
        """Open an existing disk history for read-only inspection."""

        target = Path(path)
        manifest = _read_manifest(target)
        config = EvaluationHistoryConfig(
            storage="disk",
            path=target,
            detail=str(manifest["detail"]),
            resume=True,
        )
        return cls(
            config,
            monitor_metadata=manifest.get("monitor_metadata") or {},
            _read_only=True,
        )

    def _initialize_disk(self) -> None:
        path = self._path()
        if path.exists() and path.stat().st_size > 0:
            if not self.config.resume and not self._read_only:
                raise FileExistsError(
                    f"Evaluation history already exists and is nonempty: {path}. "
                    "Set resume=True to append."
                )
            manifest = _read_manifest(path)
            _validate_manifest(manifest, expected_detail=self.config.detail)
            stored_metadata = dict(manifest["monitor_metadata"])
            if not self._read_only:
                _validate_monitor_compatibility(stored_metadata, self.monitor_metadata)
            self.monitor_metadata = stored_metadata
            for record in self._iter_disk_records():
                self._register(record)
            return
        if self._read_only:
            raise FileNotFoundError(f"Evaluation history does not exist or is empty: {path}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "record_type": "manifest",
            "schema": _HISTORY_SCHEMA,
            "schema_version": _HISTORY_SCHEMA_VERSION,
            "detail": self.config.detail,
            "vertebrae_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "monitor_metadata": self.monitor_metadata,
        }
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json_dumps_strict(manifest, sort_keys=True) + "\n")
            handle.flush()

    def _iter_disk_records(self) -> Iterator[EvaluationRecord]:
        path = self._path()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(
                        f"Evaluation history contains a truncated line at {line_number}."
                    )
                if not line.strip():
                    raise ValueError(f"Evaluation history contains a blank line at {line_number}.")
                try:
                    payload = _json_loads_strict(line)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Evaluation history contains invalid JSON at line {line_number}."
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Evaluation history line {line_number} must contain a JSON object."
                    )
                if line_number == 1:
                    _validate_manifest(payload, expected_detail=self.config.detail)
                    continue
                if payload.get("record_type") != "evaluation":
                    raise ValueError(
                        f"Evaluation history line {line_number} is not an evaluation record."
                    )
                record_payload = dict(payload)
                record_payload.pop("record_type", None)
                try:
                    record = EvaluationRecord.from_dict(record_payload)
                    _validate_record_detail(record, self.detail)
                    yield record
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Evaluation history has an invalid record at line {line_number}."
                    ) from exc

    def _register(self, record: EvaluationRecord) -> None:
        if record.evaluation_index != self._next_index:
            raise ValueError(
                "Evaluation history indices must be contiguous and zero-based; expected "
                f"{self._next_index}, got {record.evaluation_index}."
            )
        self._signatures.add(record.context.identity_signature())
        for row in record.rows:
            self._observed_result_columns.update(row)
        self._observed_metadata_columns.update(
            f"context_metadata.{name}" for name in record.context.metadata
        )
        self._next_index += 1
        self._record_count += 1

    def _result_columns(self) -> List[str]:
        metric_names = self.monitor_metadata.get("metric_names", [])
        if not isinstance(metric_names, list) or any(
            not isinstance(name, str) for name in metric_names
        ):
            metric_names = []
        return benchmark_result_columns(
            metric_names=metric_names,
            observed_columns=list(self._observed_result_columns),
        )

    def _dataframe_columns(self) -> List[str]:
        return [
            *_HISTORY_CONTEXT_COLUMNS,
            *sorted(self._observed_metadata_columns),
            *self._result_columns(),
        ]

    def _path(self) -> Path:
        if self.config.path is None:
            raise RuntimeError("Disk history has no configured path.")
        return Path(self.config.path)


class RepresentationMonitor:
    """Repeatedly run a fresh labeled benchmark over live extractor objects."""

    def __init__(
        self,
        dataset: Any,
        extractors: Iterable[Any],
        *,
        history_config: Optional[EvaluationHistoryConfig] = None,
        reporters: Iterable[EvaluationReporter] = (),
        error_policy: str = "raise",
        cache_config: Optional[CacheConfig] = None,
        **benchmark_options: Any,
    ) -> None:
        if error_policy not in {"raise", "continue"}:
            raise ValueError("error_policy must be either 'raise' or 'continue'.")
        if "dataset" in benchmark_options or "extractors" in benchmark_options:
            raise ValueError("dataset and extractors must be passed to RepresentationMonitor.")
        self.dataset = dataset
        self.extractors = list(extractors)
        if not self.extractors:
            raise ValueError("At least one extractor must be provided.")
        self.reporters = list(reporters)
        for reporter in self.reporters:
            if not callable(getattr(reporter, "report", None)):
                raise TypeError("Every reporter must provide report(record).")
        self.error_policy = error_policy
        self.benchmark_options = _snapshot_benchmark_options(benchmark_options)
        configured_cache = cache_config or CacheConfig(enabled=False)
        if not isinstance(configured_cache, CacheConfig):
            raise TypeError("cache_config must be a CacheConfig.")
        self.cache_config = replace(configured_cache, force_recompute=True)
        benchmark = self._new_benchmark()
        monitor_metadata = _monitor_metadata(benchmark, error_policy)
        self.history = EvaluationHistory(
            history_config,
            monitor_metadata=monitor_metadata,
        )

    def evaluate(
        self,
        *,
        snapshot_id: Optional[str] = None,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        timestamp: Optional[Union[str, datetime]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[BenchmarkResult]:
        """Run and record one fresh benchmark evaluation."""

        context = EvaluationContext(
            snapshot_id=snapshot_id,
            epoch=epoch,
            global_step=global_step,
            timestamp=timestamp,
            checkpoint=checkpoint,
            metadata={} if metadata is None else metadata,
        )
        if self.history.contains_context(context):
            warnings.warn(
                "Evaluation context duplicates previously recorded identifiers; "
                "the new evaluation will still be appended.",
                UserWarning,
                stacklevel=2,
            )
        evaluation_index = self.history.next_evaluation_index
        try:
            result = self._new_benchmark().run()
        except Exception as exc:
            record = EvaluationRecord(
                context=context,
                status="failure",
                evaluation_index=evaluation_index,
                rows=[{}],
                error_type=type(exc).__name__,
                error_message=str(exc),
                error_traceback=(traceback.format_exc() if self.history.detail == "full" else None),
            )
            self.history.append(record)
            self._report(record)
            if self.error_policy == "continue":
                return None
            raise
        record = EvaluationRecord(
            context=context,
            status="success",
            evaluation_index=evaluation_index,
            rows=result._tabular_rows(include_invalid=True),
            benchmark_result=(result.to_dict() if self.history.detail == "full" else None),
        )
        self.history.append(record)
        self._report(record)
        return result

    def _report(self, record: EvaluationRecord) -> None:
        for reporter in self.reporters:
            try:
                reporter.report(record)
            except Exception as exc:
                warnings.warn(
                    f"Evaluation reporter {type(reporter).__name__} failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _new_benchmark(self) -> Benchmark:
        return Benchmark(
            dataset=self.dataset,
            extractors=self.extractors,
            cache_config=self.cache_config,
            **self.benchmark_options,
        )


def _history_rows(
    records: Iterable[EvaluationRecord],
    result_columns: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        base = {
            "evaluation_index": record.evaluation_index,
            "status": record.status,
            "snapshot_id": record.context.snapshot_id,
            "epoch": record.context.epoch,
            "global_step": record.context.global_step,
            "timestamp": record.context.timestamp,
            "checkpoint": record.context.checkpoint,
            "recorded_at": record.context.recorded_at,
            "error_type": record.error_type,
            "error_message": record.error_message,
        }
        for name in sorted(record.context.metadata):
            value = record.context.metadata[name]
            base[f"context_metadata.{name}"] = value
        result_rows = (
            [null_benchmark_result_row(result_columns)]
            if record.status == "failure"
            else record.rows or [{}]
        )
        for result_row in result_rows:
            combined = dict(base)
            for name in sorted(result_row):
                if name not in combined:
                    combined[name] = result_row[name]
            rows.append(combined)
    return rows


def _monitor_metadata(benchmark: Benchmark, error_policy: str) -> Dict[str, Any]:
    protocol = _monitor_protocol(benchmark, error_policy)
    return make_json_safe(
        {
            "protocol": protocol,
            "protocol_hash": hash_json_exact(protocol),
            "metric_names": [metric.name for metric in benchmark.metrics],
        }
    )


def _snapshot_benchmark_options(value: Mapping[str, Any]) -> Dict[str, Any]:
    options = dict(value)
    for name in ("compression_configs", "metrics"):
        if name in options and options[name] is not None:
            options[name] = tuple(options[name])
    if options.get("structured_aligners") is not None:
        options["structured_aligners"] = dict(options["structured_aligners"])
    return options


def _monitor_protocol(benchmark: Benchmark, error_policy: str) -> Dict[str, Any]:
    dataset = benchmark.dataset
    dataset_summary = dataset.summary() if callable(getattr(dataset, "summary", None)) else {}
    identity_key = (
        dataset.identity_key() if callable(getattr(dataset, "identity_key", None)) else None
    )
    structured_aligners = {
        name: aligner.recipe()
        for name, aligner in sorted(benchmark.structured_aligners.items())
    }
    execution = benchmark.execution
    execution_identity = (
        None
        if execution is None
        else f"{type(execution).__module__}.{type(execution).__qualname__}"
    )
    return make_json_safe(
        {
            "dataset": {
                "identity_key": identity_key,
                "summary": dataset_summary,
            },
            "extractors": [
                _monitoring_extractor_recipe(extractor.recipe())
                for extractor in benchmark.extractors
            ],
            "benchmark": {
                "scoring_config": asdict(benchmark.scoring_config),
                "stability_config": asdict(benchmark.stability_config),
                "label_view_config": asdict(benchmark.label_view_config),
                "target_view_config": asdict(benchmark.target_view_config),
                "separatix_config": asdict(benchmark.separatix_config),
                "compression_configs": [
                    asdict(config) for config in benchmark.compression_configs
                ],
                "embedding_config": asdict(benchmark.embedding_config),
                "memory_config": asdict(benchmark.memory_config),
                "execution_backend": execution_identity,
                "execution_config": asdict(benchmark.execution_config),
                "segmentation_config": asdict(benchmark.segmentation_config),
                "structured_aligners": structured_aligners,
                "metrics": [metric.recipe() for metric in benchmark.metrics],
                "primary_metric": benchmark.primary_metric,
                "resource_profiling_config": asdict(
                    benchmark.resource_profiling_config
                ),
                "cache_policy": {
                    "enabled": benchmark.cache_config.enabled,
                    "force_recompute": True,
                },
            },
            "error_policy": error_policy,
        }
    )


def _monitoring_extractor_recipe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _monitoring_extractor_recipe(item)
            for key, item in value.items()
            if key not in {"cache_safe", "path_identities"}
        }
    if isinstance(value, list):
        return [_monitoring_extractor_recipe(item) for item in value]
    if isinstance(value, tuple):
        return [_monitoring_extractor_recipe(item) for item in value]
    return value


def _read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Evaluation history does not exist or is empty: {path}.")
    with path.open("r", encoding="utf-8") as handle:
        line = handle.readline()
    if not line.endswith("\n"):
        raise ValueError("Evaluation history manifest is truncated.")
    try:
        manifest = _json_loads_strict(line)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Evaluation history manifest contains invalid JSON.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Evaluation history manifest must contain a JSON object.")
    _validate_manifest(manifest)
    return dict(manifest)


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_detail: Optional[str] = None,
) -> None:
    _require_exact_keys(manifest, _MANIFEST_KEYS, "evaluation history manifest")
    if manifest["record_type"] != "manifest" or manifest["schema"] != _HISTORY_SCHEMA:
        raise ValueError("Evaluation history manifest has an unsupported schema.")
    try:
        schema_version = _require_json_integer(
            manifest["schema_version"],
            "manifest.schema_version",
        )
    except TypeError as exc:
        raise ValueError("Evaluation history manifest has an invalid schema_version.") from exc
    if schema_version != _HISTORY_SCHEMA_VERSION:
        if schema_version == 1:
            raise ValueError(
                "Evaluation history schema version 1 is unsupported; recreate the "
                "monitoring history with the current unreleased format."
            )
        raise ValueError(
            "Evaluation history manifest has unsupported schema version "
            f"{schema_version}; expected {_HISTORY_SCHEMA_VERSION}."
        )
    detail = manifest["detail"]
    if detail not in {"summary", "full"}:
        raise ValueError("Evaluation history manifest has an invalid detail level.")
    version = manifest["vertebrae_version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Evaluation history manifest has an invalid Vertebrae version.")
    try:
        _normalize_required_timestamp(manifest["created_at"], "manifest.created_at")
    except (TypeError, ValueError) as exc:
        raise ValueError("Evaluation history manifest has an invalid creation time.") from exc
    monitor_metadata = manifest["monitor_metadata"]
    if not isinstance(monitor_metadata, Mapping):
        raise ValueError("Evaluation history manifest has invalid monitor metadata.")
    _validate_monitor_metadata(monitor_metadata)
    if expected_detail is not None and detail != expected_detail:
        raise ValueError(
            f"Evaluation history detail mismatch: expected {expected_detail!r}, got {detail!r}."
        )


def _validate_monitor_metadata(value: Mapping[str, Any]) -> None:
    if not value:
        return
    _require_exact_keys(value, _MONITOR_METADATA_KEYS, "manifest.monitor_metadata")
    protocol = value["protocol"]
    protocol_hash = value["protocol_hash"]
    metric_names = value["metric_names"]
    if not isinstance(protocol, Mapping):
        raise ValueError("Evaluation history monitor protocol must be a mapping.")
    if not isinstance(protocol_hash, str) or not protocol_hash.strip():
        raise ValueError("Evaluation history monitor protocol hash must be a string.")
    if hash_json_exact(protocol) != protocol_hash:
        raise ValueError("Evaluation history monitor protocol hash does not match its payload.")
    if not isinstance(metric_names, list) or any(
        not isinstance(name, str) or not name for name in metric_names
    ):
        raise ValueError("Evaluation history metric names must be a list of strings.")


def _validate_monitor_compatibility(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    if stored == current:
        return
    stored_protocol = stored.get("protocol")
    current_protocol = current.get("protocol")
    if isinstance(stored_protocol, Mapping) and isinstance(current_protocol, Mapping):
        differing = sorted(
            key
            for key in set(stored_protocol) | set(current_protocol)
            if stored_protocol.get(key) != current_protocol.get(key)
        )
        detail = ", ".join(differing) or "metadata"
        raise ValueError(
            "Evaluation history monitoring protocol mismatch; differing components: "
            f"{detail}."
        )
    raise ValueError("Evaluation history monitor metadata does not match the existing file.")


def _validate_record_detail(record: EvaluationRecord, detail: str) -> None:
    if record.status == "success":
        if not record.rows:
            raise ValueError("Successful evaluation records require at least one result row.")
        if detail == "full" and record.benchmark_result is None:
            raise ValueError("Full successful records require benchmark_result.")
    elif record.rows != [{}]:
        raise ValueError("Failed evaluation records must contain one empty result row.")
    if detail == "summary":
        if record.benchmark_result is not None or record.error_traceback is not None:
            raise ValueError("Summary records cannot contain full result or traceback payloads.")
    elif record.status == "failure" and not record.error_traceback:
        raise ValueError("Full failed records require error_traceback.")


def _json_loads_strict(value: str) -> Any:
    def reject_constant(constant: str) -> Any:
        raise ValueError(f"Non-standard JSON constant {constant!r} is not allowed.")

    def reject_duplicate_keys(pairs: List[Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} is not allowed.")
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    owner: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a JSON object.")
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{owner} fields do not match the schema; missing={missing}, extra={extra}."
        )


def _require_json_mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a JSON object.")
    return value


def _require_json_list(value: Any, owner: str) -> List[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{owner} must be a JSON array.")
    return value


def _require_json_string(value: Any, owner: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{owner} must be a JSON string.")
    return value


def _require_optional_json_string(value: Any, owner: str) -> Optional[str]:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{owner} must be a JSON string or null.")
    return value


def _require_json_integer(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner} must be a JSON integer.")
    return value


def _require_optional_json_integer(value: Any, owner: str) -> Optional[int]:
    if value is None:
        return None
    return _require_json_integer(value, owner)


def _normalize_context_metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping.")
    _validate_metadata_keys(value, "metadata", set())
    normalized = make_json_safe(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError("metadata must serialize to a mapping.")
    return normalized


def _validate_metadata_keys(value: Any, path: str, active: set[int]) -> None:
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            return
        active.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} keys must be strings.")
                _validate_metadata_keys(item, f"{path}.{key}", active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in active:
            return
        active.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_metadata_keys(item, f"{path}[{index}]", active)
        finally:
            active.remove(identity)


def _normalize_optional_index(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer when provided.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return int(value)


def _normalize_optional_string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string when provided.")
    return value.strip()


def _normalize_optional_path(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a string or Path when provided.")
    normalized = str(value)
    if not normalized.strip():
        raise ValueError(f"{name} must be nonblank when provided.")
    return normalized


def _normalize_optional_timestamp(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _normalize_required_timestamp(value, name)


def _normalize_required_timestamp(value: Any, name: str) -> str:
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 timestamp.") from exc
    else:
        raise TypeError(f"{name} must be a datetime or ISO-8601 string.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include timezone information.")
    return parsed.astimezone(timezone.utc).isoformat()


def _format_score(value: Any, precision: int) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "ConsoleReporter",
    "EvaluationContext",
    "EvaluationHistory",
    "EvaluationHistoryConfig",
    "EvaluationRecord",
    "EvaluationReporter",
    "RepresentationMonitor",
]
