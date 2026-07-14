"""Artifact-backed orchestration for labeled :class:`vertebrae.Benchmark` runs."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from uuid import uuid4

from vertebrae._version import __version__
from vertebrae.cache import ArtifactStore, ArtifactStoreConfig, create_artifact_store
from vertebrae.cache.factory import create_artifact_store_from_config
from vertebrae.cache.fingerprint import hash_json
from vertebrae.execution.base import BenchmarkExecutionError
from vertebrae.execution.distributed import (
    benchmark_result_from_artifacts,
    compress_embedding_artifacts,
    diagnose_embedding_artifacts,
    embedding_artifact_key,
    groups_artifact_key,
    labels_artifact_key,
    materialize_embedding_shards,
    materialize_group_artifact,
    materialize_label_artifact,
    merge_embedding_shards,
    plan_compression_job,
    plan_embedding_shard_jobs,
    run_stability_artifacts,
    score_embedding_artifacts,
    separatix_artifact_key,
    stability_artifact_key,
)
from vertebrae.execution.jobs import (
    EmbeddingMergeJob,
    ScoringJob,
    SeparatixJob,
    StabilityJob,
)
from vertebrae.execution.local import LocalBackend
from vertebrae.profiling import resource_profile_like_from_dict
from vertebrae.reports.recommendations import recommendations_for_benchmark
from vertebrae.results import BenchmarkResult, ExtractorResult
from vertebrae.scoring.metrics import MetricResult
from vertebrae.scoring.separatix import SeparatixResult
from vertebrae.utils.serialization import make_json_safe


@dataclass(frozen=True)
class _NestedBenchmarkJob:
    """Whole-extractor materialization job for structured or segmentation inputs."""

    benchmark_kwargs: dict[str, Any]
    result_key: str
    store_config: ArtifactStoreConfig


def run_artifact_backed_benchmark(benchmark: Any) -> BenchmarkResult:
    """Run a benchmark through its explicit execution backend."""

    if getattr(benchmark.dataset, "modality", None) == "segmentation" or (
        benchmark.dataset.unit_annotations()
        and any(
            callable(getattr(extractor, "transform_structured", None))
            for extractor in benchmark.extractors
        )
    ):
        return _run_nested_materialization_benchmark(benchmark)
    return _run_standard_benchmark(benchmark)


def _run_standard_benchmark(benchmark: Any) -> BenchmarkResult:
    benchmark._validate_view_config_compatibility()
    datasets, label_warnings, target_warnings = benchmark._evaluation_datasets()
    benchmark._validate_output_view_mapping()
    store = create_artifact_store(
        benchmark.cache_config.cache_dir,
        **benchmark.cache_config.storage_options,
    )
    run_id = uuid4().hex
    run_prefix = f"runs/{run_id}"
    extractor_results: list[ExtractorResult] = []
    effective_shards: list[int] = []
    try:
        for extractor in benchmark.extractors:
            for evaluation_dataset in datasets:
                prepared, warnings, _, probe_plan = benchmark._prepare_dataset_for_extractor(
                    extractor,
                    evaluation_dataset,
                )
                requested_shards = benchmark.execution_config.total_shards
                shard_count = min(requested_shards, len(prepared.y))
                effective_shards.append(shard_count)
                canonical_key = embedding_artifact_key(prepared, extractor)
                raw_key = (
                    canonical_key
                    if benchmark.cache_config.enabled
                    else f"{run_prefix}/{canonical_key}"
                )
                manifest = _cached_embedding_manifest(
                    store,
                    raw_key,
                    use_cache=(
                        benchmark.cache_config.enabled
                        and not benchmark.cache_config.force_recompute
                    ),
                )
                if manifest is None:
                    if probe_plan is None:
                        extractor.fit(prepared.X, prepared.y)
                    jobs = plan_embedding_shard_jobs(
                        prepared,
                        extractor,
                        total_shards=shard_count,
                        batch_size=benchmark.embedding_config.batch_size,
                        resource_profiling_config=benchmark.resource_profiling_config,
                        output_key=raw_key,
                        fit_extractor=False,
                        streaming_enabled=benchmark.embedding_config.streaming_enabled,
                    )
                    manifests = _run_stage(
                        benchmark,
                        "embedding",
                        jobs,
                        lambda execution, jobs=jobs: materialize_embedding_shards(
                            jobs,
                            store=store,
                            execution=execution,
                        ),
                    )
                    manifest = merge_embedding_shards(
                        EmbeddingMergeJob(
                            shard_keys=tuple(item["output_key"] for item in manifests),
                            output_key=raw_key,
                            n_samples=len(prepared.y),
                        ),
                        store,
                    )
                    if not benchmark.execution_config.retain_intermediate_artifacts:
                        for item in manifests:
                            store.delete_prefix(item["output_key"])
                output_manifests = _normalized_output_manifests(
                    store,
                    manifest,
                    extractor=extractor,
                    dataset=prepared,
                )
                for output_manifest in output_manifests:
                    output_name = output_manifest.get("output_name")
                    scoring_dataset = prepared
                    if benchmark._has_output_view_mappings():
                        if not benchmark._output_has_view_mapping(output_name):
                            continue
                        scoring_dataset = benchmark._mapped_output_dataset(
                            dataset=prepared,
                            output_name=output_name,
                            label_view_warnings=label_warnings,
                            target_view_warnings=target_warnings,
                        )
                        if scoring_dataset is None:
                            continue
                    extractor_results.extend(
                        _score_output_manifest(
                            benchmark,
                            store,
                            output_manifest,
                            scoring_dataset,
                            run_prefix=run_prefix,
                            warnings=warnings,
                        )
                    )
        if not extractor_results:
            raise ValueError("No valid benchmark outputs were available for scoring.")
        result = BenchmarkResult(
            dataset_summary=benchmark.dataset.summary(),
            extractor_results=extractor_results,
            recommendations=recommendations_for_benchmark(
                extractor_results,
                quality_tolerance=(
                    benchmark.resource_profiling_config.quality_tolerance
                    if benchmark.resource_profiling_config.enabled
                    else None
                ),
            ),
            metadata=_result_metadata(
                benchmark,
                run_id,
                effective_shards,
                label_warnings,
                target_warnings,
            ),
        )
        return result
    finally:
        if (
            not benchmark.cache_config.enabled
            and not benchmark.execution_config.retain_intermediate_artifacts
        ):
            store.delete_prefix(run_prefix)


def _score_output_manifest(
    benchmark: Any,
    store: ArtifactStore,
    raw_manifest: dict[str, Any],
    dataset: Any,
    *,
    run_prefix: str,
    warnings: list[str],
) -> list[ExtractorResult]:
    raw_key = raw_manifest["output_key"]
    aligned_identity = raw_manifest.get("dataset_identity_key")
    labels_key = labels_artifact_key(dataset)
    groups_key: Optional[str] = None
    if not benchmark.cache_config.enabled:
        labels_key = f"{run_prefix}/{labels_key}"
    materialize_label_artifact(
        dataset,
        store,
        key=labels_key,
        aligned_embedding_identity_key=aligned_identity,
    )
    if callable(getattr(dataset, "groups", None)) and dataset.groups() is not None:
        groups_key = groups_artifact_key(dataset)
        if not benchmark.cache_config.enabled:
            groups_key = f"{run_prefix}/{groups_key}"
        materialize_group_artifact(
            dataset,
            store,
            key=groups_key,
            aligned_embedding_identity_key=aligned_identity,
        )

    results: list[ExtractorResult] = []
    scoring_config = benchmark._resolved_scoring_config(dataset)
    for compression_config in benchmark.compression_configs:
        evaluated_key = raw_key
        if compression_config.enabled and compression_config.method != "none":
            compression_job = plan_compression_job(raw_key, compression_config)
            if not benchmark.cache_config.enabled:
                compression_hash = hash_json(_compression_identity(raw_key, compression_config))
                compression_job = type(compression_job)(
                    embedding_key=raw_key,
                    output_key=f"{run_prefix}/compressed/{compression_hash}",
                    compression_config=compression_config,
                    resources=compression_job.resources,
                )
            evaluated_key = compression_job.output_key
            if not (
                benchmark.cache_config.enabled
                and not benchmark.cache_config.force_recompute
                and store.exists(evaluated_key)
            ):
                _run_stage(
                    benchmark,
                    "compression",
                    [compression_job],
                    lambda execution, job=compression_job: compress_embedding_artifacts(
                        [job], store, execution
                    ),
                )

        score_protocol = hash_json(
            {
                "scoring_config": make_json_safe(scoring_config),
                "metrics": [metric.recipe() for metric in benchmark.metrics],
                "primary_metric": benchmark.primary_metric,
            }
        )
        score_key = f"{evaluated_key}/scores/{score_protocol}"
        score_job = ScoringJob(
            embedding_key=evaluated_key,
            labels_key=labels_key,
            groups_key=groups_key,
            output_key=score_key,
            scoring_config=scoring_config,
            metrics=benchmark.metrics,
            primary_metric=benchmark.primary_metric,
        )
        _run_stage(
            benchmark,
            "scoring",
            [score_job],
            lambda execution, job=score_job: score_embedding_artifacts([job], store, execution),
        )

        stability_key = None
        if benchmark.stability_config.enabled and benchmark.stability_config.mode != "none":
            stability_key = stability_artifact_key(
                evaluated_key,
                benchmark.stability_config,
            )
            stability_job = StabilityJob(
                embedding_key=evaluated_key,
                labels_key=labels_key,
                output_key=stability_key,
                scoring_config=scoring_config,
                stability_config=benchmark.stability_config,
            )
            _run_stage(
                benchmark,
                "diagnostics",
                [stability_job],
                lambda execution, job=stability_job: run_stability_artifacts(
                    [job], store, execution
                ),
            )

        diagnostic_key = None
        if benchmark.separatix_config.enabled:
            diagnostic_key = (
                f"{separatix_artifact_key(evaluated_key)}-"
                f"{hash_json(make_json_safe(benchmark.separatix_config))}"
            )
            diagnostic_job = SeparatixJob(
                embedding_key=evaluated_key,
                labels_key=labels_key,
                groups_key=groups_key,
                score_key=score_key,
                output_key=diagnostic_key,
                separatix_config=benchmark.separatix_config,
            )
            _run_stage(
                benchmark,
                "diagnostics",
                [diagnostic_job],
                lambda execution, job=diagnostic_job: diagnose_embedding_artifacts(
                    [job], store, execution
                ),
            )

        payload = benchmark_result_from_artifacts(
            score_key,
            store,
            stability_key=stability_key,
            separatix_key=diagnostic_key,
        )
        result = _extractor_result_from_payload(payload["extractor_results"][0])
        result.warnings = sorted(set(result.warnings + warnings))
        if not compression_config.enabled or compression_config.method == "none":
            result.compression_metadata.setdefault("method", "none")
        results.append(result)
    return results


def _normalized_output_manifests(
    store: ArtifactStore,
    manifest: dict[str, Any],
    *,
    extractor: Any,
    dataset: Any,
) -> list[dict[str, Any]]:
    outputs = (
        manifest.get("outputs")
        if manifest.get("artifact_type") == "multi_output_embedding"
        else None
    )
    manifests = [dict(item) for item in outputs] if outputs else [dict(manifest)]
    for item in manifests:
        name = extractor.name
        if item.get("output_name"):
            name = f"{name}:{item['output_name']}"
        recipe = dict(item.get("extractor_recipe") or extractor.recipe())
        recipe["name"] = name
        item.update(
            {
                "extractor_recipe": recipe,
                "extractor_name": name,
                "extractor_type": getattr(extractor, "extractor_type", "unknown"),
                "modality": dataset.modality,
                "cache_key": item["output_key"],
            }
        )
        store.put_json(item["output_key"], item)
    return manifests


def _cached_embedding_manifest(
    store: ArtifactStore,
    key: str,
    *,
    use_cache: bool,
) -> Optional[dict[str, Any]]:
    if not use_cache:
        return None
    try:
        manifest = store.get_json(key)
    except FileNotFoundError:
        return None
    if manifest.get("artifact_type") == "multi_output_embedding":
        outputs = manifest.get("outputs", [])
        if outputs and all(store.exists(item["output_key"]) for item in outputs):
            return manifest
        return None
    return manifest if store.exists(key) else None


def _run_stage(
    benchmark: Any,
    stage: str,
    jobs: Iterable[Any],
    call: Any,
) -> Any:
    job_list = list(jobs)
    execution = (
        benchmark.execution
        if stage in benchmark.execution_config.dispatch_stages
        else LocalBackend()
    )
    identity = ", ".join(
        str(getattr(job, "output_key", getattr(job, "result_key", type(job).__name__)))
        for job in job_list
    )
    try:
        return call(execution)
    except Exception as exc:
        if isinstance(exc, BenchmarkExecutionError):
            raise
        raise BenchmarkExecutionError(execution, stage, identity or "empty job set", exc) from exc


def _run_nested_materialization_benchmark(benchmark: Any) -> BenchmarkResult:
    store = create_artifact_store(
        benchmark.cache_config.cache_dir,
        **benchmark.cache_config.storage_options,
    )
    run_id = uuid4().hex
    run_prefix = f"runs/{run_id}"
    jobs = []
    for index, extractor in enumerate(benchmark.extractors):
        result_key = f"{run_prefix}/nested/{index:05d}"
        jobs.append(
            _NestedBenchmarkJob(
                benchmark_kwargs=_nested_benchmark_kwargs(benchmark, extractor),
                result_key=result_key,
                store_config=store.config(),
            )
        )
    try:
        payloads = _run_stage(
            benchmark,
            "embedding",
            jobs,
            lambda selected: selected.map(_run_nested_benchmark_job, jobs),
        )
        extractor_results = [
            _extractor_result_from_payload(item)
            for payload in payloads
            for item in payload["extractor_results"]
        ]
        summary = dict(payloads[0]["dataset_summary"])
        summary_key = (
            "segmentation_outputs"
            if getattr(benchmark.dataset, "modality", None) == "segmentation"
            else "structured_outputs"
        )
        summary[summary_key] = [
            item for payload in payloads for item in payload["dataset_summary"].get(summary_key, [])
        ]
        return BenchmarkResult(
            dataset_summary=summary,
            extractor_results=extractor_results,
            recommendations=recommendations_for_benchmark(extractor_results),
            metadata=_result_metadata(benchmark, run_id, [1] * len(jobs), [], []),
        )
    finally:
        if not benchmark.execution_config.retain_intermediate_artifacts:
            store.delete_prefix(run_prefix)


def _nested_benchmark_kwargs(benchmark: Any, extractor: Any) -> dict[str, Any]:
    return {
        "dataset": benchmark.dataset,
        "extractors": [extractor],
        "scoring_config": benchmark._explicit_scoring_config,
        "stability_config": benchmark.stability_config,
        "label_view_config": benchmark.label_view_config,
        "target_view_config": benchmark.target_view_config,
        "separatix_config": benchmark.separatix_config,
        "cache_config": benchmark.cache_config,
        "compression_configs": benchmark.compression_configs,
        "embedding_config": benchmark.embedding_config,
        "memory_config": benchmark.memory_config,
        "segmentation_config": benchmark.segmentation_config,
        "structured_aligners": benchmark.structured_aligners,
        "metrics": [metric for metric in benchmark.metrics if metric.name != "overlap"],
        "primary_metric": benchmark.primary_metric,
        "resource_profiling_config": benchmark.resource_profiling_config,
    }


def _compression_identity(raw_key: str, compression_config: Any) -> dict[str, Any]:
    return {"raw": raw_key, "config": make_json_safe(compression_config)}


def _run_nested_benchmark_job(job: _NestedBenchmarkJob) -> dict[str, Any]:
    from vertebrae.benchmark import Benchmark

    result = Benchmark(**job.benchmark_kwargs).run()
    payload = result.to_dict()
    create_artifact_store_from_config(job.store_config).put_json(job.result_key, payload)
    return payload


def _extractor_result_from_payload(value: dict[str, Any]) -> ExtractorResult:
    metrics = {
        name: MetricResult(**dict(metric)) for name, metric in value.get("metrics", {}).items()
    }
    separatix = value.get("separatix")
    profile = value.get("resource_profile")
    return ExtractorResult(
        name=value["name"],
        extractor_type=value["extractor_type"],
        stability=value.get("stability"),
        separatix=SeparatixResult(**dict(separatix)) if separatix else None,
        embedding_metadata=dict(value.get("embedding_metadata", {})),
        compression_metadata=dict(value.get("compression_metadata", {})),
        runtime=dict(value.get("runtime", {})),
        warnings=list(value.get("warnings", [])),
        recommendation=value.get("recommendation", "aggregate_unavailable"),
        metrics=metrics,
        primary_metric_name=value.get("primary_metric_name", "overlap"),
        label_view=value.get("label_view"),
        target_view=value.get("target_view"),
        weakest_class=value.get("weakest_class"),
        weakest_class_score=value.get("weakest_class_score"),
        resource_profile=(
            resource_profile_like_from_dict(dict(profile)) if profile is not None else None
        ),
    )


def _result_metadata(
    benchmark: Any,
    run_id: str,
    effective_shards: list[int],
    label_warnings: list[str],
    target_warnings: list[str],
) -> dict[str, Any]:
    metadata = {
        "vertebrae_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scoring_config": asdict(benchmark.scoring_config),
        "stability_config": asdict(benchmark.stability_config),
        "label_view_config": asdict(benchmark.label_view_config),
        "target_view_config": asdict(benchmark.target_view_config),
        "separatix_config": asdict(benchmark.separatix_config),
        "cache_config": asdict(benchmark.cache_config),
        "compression_configs": [asdict(config) for config in benchmark.compression_configs],
        "embedding_config": asdict(benchmark.embedding_config),
        "memory_config": asdict(benchmark.memory_config),
        "metrics": [metric.recipe() for metric in benchmark.metrics],
        "primary_metric": benchmark.primary_metric,
        "resource_profiling_config": asdict(benchmark.resource_profiling_config),
        "label_view_warnings": label_warnings,
        "target_view_warnings": target_warnings,
        "execution": {
            "backend": type(benchmark.execution).__name__,
            "config": asdict(benchmark.execution_config),
            "requested_total_shards": benchmark.execution_config.total_shards,
            "effective_total_shards": effective_shards,
            "dispatched_stages": list(benchmark.execution_config.dispatch_stages),
            "artifact_backed": True,
            "run_id": run_id,
        },
    }
    if getattr(benchmark.dataset, "modality", None) == "segmentation":
        metadata["segmentation_config"] = asdict(benchmark.segmentation_config)
    return metadata
