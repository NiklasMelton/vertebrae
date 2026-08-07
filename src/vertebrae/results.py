"""Structured benchmark results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from vertebrae.profiling import DistributedResourceProfile, ResourceProfile, ResourceProfileLike
from vertebrae.scoring.metrics import MetricResult
from vertebrae.scoring.separatix import SeparatixResult, probe_summary_for_result
from vertebrae.utils.serialization import make_json_safe

RESULT_ROW_STATIC_COLUMNS = (
    "rank",
    "extractor",
    "parent_extractor",
    "output_name",
    "hidden_layer",
    "pooling",
    "extractor_type",
    "primary_metric",
    "primary_score",
    "primary_higher_is_better",
    "aggregate_valid",
    "overlap_score",
    "overlap_macro",
    "overlap_weighted",
    "target_type",
    "target_names",
    "target_view",
    "label_view",
    "weakest_class",
    "weakest_class_score",
    "probe_status",
    "best_probe",
    "probe_metric",
    "probe_score",
    "probe_metrics",
    "probe_linear_score",
    "probe_nonlinear_score",
    "probe_nonlinear_delta",
    "probe_comparison_confidence",
    "probe_evaluation_mode",
    "probe_sampled",
    "probe_n_samples_original",
    "probe_n_samples_used",
    "probe_grouped",
    "probe_n_groups",
    "probe_skip_reason",
    "probe_alignment_status",
    "probe_evaluation_plan_id",
    "probe_cv_method",
    "probe_cohort_size",
    "probe_n_splits",
    "probe_effective_train_size_summary",
    "probe_effective_train_size_mean",
    "stability_mode",
    "stability_repeats",
    "stability_interval_level",
    "stability_mean",
    "stability_std",
    "stability_min",
    "stability_max",
    "stability_interval",
    "stability_interval_lower",
    "stability_interval_upper",
    "stability_interval_width",
    "embedding_dim",
    "task_family",
    "alignment_mode",
    "alignment_recipe",
    "compression_method",
    "compression_precision",
    "compressed_dim",
    "recommendation",
    "warnings",
    "warning_count",
    "separatix_ran",
    "separatix_recommendation",
    "separatix_confidence",
    "separatix_skip_reason",
    "separatix_guidance_status",
    "separatix_minimum_family",
    "separatix_plausible_families",
    "separatix_guidance_decision_method",
    "separatix_selected_family",
    "separatix_selected_probe",
    "separatix_selected_recipe_id",
    "separatix_mlp_override",
    "separatix_paired_status",
    "separatix_paired_method",
    "separatix_family_guidance",
    "resource_profile_status",
    "first_call_seconds",
    "warm_median_seconds",
    "warm_p95_seconds",
    "throughput_samples_per_second",
    "resource_batch_sizes",
    "synchronization_status",
    "peak_host_rss_bytes",
    "peak_host_rss_increase_bytes",
    "peak_device_allocated_bytes",
    "baseline_device_allocated_bytes",
    "peak_device_allocated_increase_bytes",
    "peak_device_reserved_bytes",
    "device_memory_status",
    "device_memory_scope",
    "device_memory_unavailable_reason",
    "model_footprint_status",
    "parameter_footprint_status",
    "checkpoint_footprint_status",
    "parameter_count",
    "parameter_bytes",
    "trainable_parameter_count",
    "trainable_parameter_bytes",
    "model_buffer_bytes",
    "model_in_memory_bytes",
    "model_weight_dtypes",
    "checkpoint_bytes",
    "raw_embedding_bytes",
    "evaluated_embedding_bytes",
    "raw_persisted_embedding_bytes",
    "evaluated_persisted_embedding_bytes",
    "evaluated_persisted_embedding_status",
    "embedding_bytes_per_sample",
    "resource_profile_scope",
    "worker_first_call_median_seconds",
    "worker_first_call_p95_seconds",
    "worker_first_call_max_seconds",
    "aggregate_compute_throughput_samples_per_second",
    "max_worker_peak_rss_bytes",
    "max_worker_peak_device_allocated_bytes",
)
RESULT_RUNTIME_STAGES = (
    "embedding_seconds",
    "compression_seconds",
    "scoring_seconds",
    "separatix_seconds",
    "stability_seconds",
)


def benchmark_result_columns(
    metric_names: Optional[List[str]] = None,
    observed_columns: Optional[List[str]] = None,
) -> List[str]:
    """Return deterministic canonical columns for benchmark and monitoring rows."""

    columns = list(RESULT_ROW_STATIC_COLUMNS)
    columns.extend(f"metric.{name}" for name in metric_names or [])
    columns.extend(f"runtime.{name}" for name in RESULT_RUNTIME_STAGES)
    known = set(columns)
    columns.extend(sorted(name for name in observed_columns or [] if name not in known))
    return columns


def null_benchmark_result_row(columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return a null row matching the canonical result schema."""

    return {name: None for name in (columns or benchmark_result_columns())}


@dataclass
class ExtractorResult:
    """Result data for one evaluated extractor.

    Attributes:
        name: Extractor name.
        extractor_type: Extractor family/type metadata.
        metrics: All normalized metric results, including ``overlap`` when enabled.
        primary_metric_name: Name of the metric used for ranking.
        stability: Optional stability-analysis summary.
        separatix: Optional Separatix diagnostic summary.
        embedding_metadata: Metadata for the embedding artifact.
        runtime: Runtime timing metadata by benchmark stage.
        warnings: Warnings produced during evaluation.
        recommendation: Recommendation label for this extractor.
        label_view: Label-view metadata for the scoring target.
        target_view: Target-view metadata for the scoring target.
        weakest_class: Class with the lowest per-class score when available.
        weakest_class_score: Score for `weakest_class` when available.
        resource_profile: Optional measured extraction and footprint profile.
    """

    name: str
    extractor_type: str
    stability: Optional[Dict[str, Any]]
    separatix: Optional[SeparatixResult]
    embedding_metadata: Dict[str, Any]
    compression_metadata: Dict[str, Any]
    runtime: Dict[str, Any]
    warnings: List[str]
    recommendation: str
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    primary_metric_name: str = "overlap"
    label_view: Optional[Dict[str, Any]] = None
    target_view: Optional[Dict[str, Any]] = None
    weakest_class: Optional[str] = None
    weakest_class_score: Optional[float] = None
    resource_profile: Optional[ResourceProfileLike] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the extractor result to a JSON-safe dictionary.

        Returns:
            JSON-compatible result data.
        """

        return make_json_safe(self)

    @property
    def overlap(self) -> Optional[MetricResult]:
        """Return the overlap metric result without duplicating serialized state."""

        metric = self.metrics.get("overlap")
        return metric if metric and metric.kind == "overlap_index" else None

    @property
    def primary_score(self) -> float:
        """Return the aggregate score used for ranking."""

        return self.metrics[self.primary_metric_name].score


@dataclass
class BenchmarkResult:
    """Aggregated result for a benchmark run.

    Attributes:
        dataset_summary: Summary of the labeled dataset.
        extractor_results: Per-extractor results.
        recommendations: Practitioner-facing benchmark recommendations.
        metadata: Reproducibility metadata for the run.
    """

    dataset_summary: Dict[str, Any]
    extractor_results: List[ExtractorResult]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the benchmark result to a JSON-safe dictionary.

        Returns:
            JSON-compatible benchmark data.
        """

        return make_json_safe(self)

    def ranked_results(self) -> List[ExtractorResult]:
        """Return extractor results sorted by the selected aggregate metric.

        Returns:
            Sorted extractor results.
        """

        valid = [
            item
            for item in self.extractor_results
            if bool(_primary_metric(item).metadata.get("aggregate_valid", True))
        ]
        return sorted(valid, key=_rankable_score, reverse=True)

    def quality_cohort(self, tolerance: Optional[float] = None) -> List[ExtractorResult]:
        """Return candidates within an absolute primary-score tolerance of the best."""

        ranked = self.ranked_results()
        if not ranked:
            return []
        if tolerance is None:
            tolerance = float(
                self.metadata.get("resource_profiling_config", {}).get("quality_tolerance", 0.01)
            )
        if tolerance < 0:
            raise ValueError("quality cohort tolerance must be >= 0.")
        best = _rankable_score(ranked[0])
        return [item for item in ranked if best - _rankable_score(item) <= tolerance]

    def to_dataframe(self, *, include_invalid: bool = False) -> Any:
        """Convert benchmark results into a pandas DataFrame.

        Args:
            include_invalid: Whether results with an invalid primary aggregate should
                remain in the table with a null rank. The default preserves the
                ranked, valid-only benchmark view.

        Returns:
            A pandas DataFrame with one row per extractor.
        """

        import pandas as pd

        return pd.DataFrame(self._tabular_rows(include_invalid=include_invalid))

    def _tabular_rows(self, *, include_invalid: bool = False) -> List[Dict[str, Any]]:
        """Build JSON-safe-oriented tabular rows without pandas dtype coercion."""

        rows = []
        ranked = self.ranked_results()
        ranks = {id(item): rank for rank, item in enumerate(ranked, start=1)}
        items = self.extractor_results if include_invalid else ranked
        for item in items:
            rank = ranks.get(id(item))
            primary = _primary_metric(item)
            aggregate_valid = bool(primary.metadata.get("aggregate_valid", True))
            profile = item.resource_profile
            distributed_profile = (
                profile if isinstance(profile, DistributedResourceProfile) else None
            )
            local_profile = profile if isinstance(profile, ResourceProfile) else None
            inference = local_profile.inference if local_profile else None
            host_memory = local_profile.host_memory if local_profile else None
            device_memory = local_profile.device_memory if local_profile else None
            model = profile.model if profile else None
            embedding = profile.embedding if profile else None
            probe = probe_summary_for_result(item.separatix)
            primary_probe_metric = probe.get("primary_metric") or {}
            comparison = probe.get("comparison") or {}
            evaluation = probe.get("evaluation") or {}
            sampling = evaluation.get("sampling") or {}
            effective_train_size = evaluation.get("effective_train_size_summary") or {}
            family_guidance = (item.separatix.family_guidance if item.separatix else {}) or {}
            stability = item.stability or {}
            stability_summary = stability.get("summary") or {}
            output_identity = _output_identity(item)
            row = {
                "rank": rank,
                "extractor": item.name,
                "parent_extractor": output_identity["parent_extractor"],
                "output_name": output_identity["output_name"],
                "hidden_layer": output_identity["hidden_layer"],
                "pooling": output_identity["pooling"],
                "extractor_type": item.extractor_type,
                "primary_metric": item.primary_metric_name,
                "primary_score": item.primary_score,
                "primary_higher_is_better": primary.higher_is_better,
                "aggregate_valid": aggregate_valid,
                "overlap_score": item.overlap.score if item.overlap else None,
                "overlap_macro": item.overlap.macro_score if item.overlap else None,
                "overlap_weighted": item.overlap.weighted_score if item.overlap else None,
                "target_type": _result_target_metadata(item).get("target_type", "single_label"),
                "target_names": _result_target_metadata(item).get("target_names"),
                "target_view": (item.target_view or {}).get("name"),
                "label_view": (item.label_view or {}).get("name"),
                "weakest_class": item.weakest_class,
                "weakest_class_score": item.weakest_class_score,
                "probe_status": probe.get("status"),
                "best_probe": probe.get("best_probe"),
                "probe_metric": primary_probe_metric.get("name"),
                "probe_score": primary_probe_metric.get("value"),
                "probe_metrics": probe.get("metrics", {}),
                "probe_linear_score": comparison.get("linear_value"),
                "probe_nonlinear_score": comparison.get("nonlinear_value"),
                "probe_nonlinear_delta": comparison.get("delta"),
                "probe_comparison_confidence": comparison.get("confidence"),
                "probe_evaluation_mode": evaluation.get("mode"),
                "probe_sampled": sampling.get("sampled"),
                "probe_n_samples_original": sampling.get("n_original"),
                "probe_n_samples_used": sampling.get("n_used"),
                "probe_grouped": evaluation.get("grouped"),
                "probe_n_groups": evaluation.get("n_groups"),
                "probe_skip_reason": probe.get("skip_reason"),
                "probe_alignment_status": evaluation.get("alignment_status"),
                "probe_evaluation_plan_id": evaluation.get("evaluation_plan_id"),
                "probe_cv_method": evaluation.get("cv_method"),
                "probe_cohort_size": evaluation.get("cohort_size", evaluation.get("n_samples")),
                "probe_n_splits": evaluation.get("n_splits"),
                "probe_effective_train_size_summary": effective_train_size,
                "probe_effective_train_size_mean": effective_train_size.get("mean"),
                "stability_mode": stability.get("mode"),
                "stability_repeats": stability.get("repeats"),
                "stability_interval_level": stability.get("interval_level"),
                "stability_mean": stability_summary.get("mean"),
                "stability_std": stability_summary.get("std"),
                "stability_min": stability_summary.get("min"),
                "stability_max": stability_summary.get("max"),
                "stability_interval": (
                    [
                        stability_summary.get("lower"),
                        stability_summary.get("upper"),
                    ]
                    if stability_summary
                    else None
                ),
                "stability_interval_lower": stability_summary.get("lower"),
                "stability_interval_upper": stability_summary.get("upper"),
                "stability_interval_width": stability_summary.get("width"),
                "embedding_dim": item.embedding_metadata.get("embedding_dim"),
                "task_family": (item.embedding_metadata.get("structured", {}) or {}).get(
                    "task_family"
                ),
                "alignment_mode": (item.embedding_metadata.get("structured", {}) or {}).get(
                    "alignment_mode"
                ),
                "alignment_recipe": (item.embedding_metadata.get("structured", {}) or {}).get(
                    "alignment_recipe"
                ),
                "compression_method": item.compression_metadata.get("method", "none"),
                "compression_precision": item.compression_metadata.get("precision"),
                "compressed_dim": item.compression_metadata.get(
                    "compressed_dim",
                    item.embedding_metadata.get("embedding_dim"),
                ),
                "recommendation": item.recommendation,
                "warnings": list(item.warnings),
                "warning_count": len(item.warnings),
                "separatix_ran": bool(item.separatix and item.separatix.ran),
                "separatix_recommendation": (
                    item.separatix.recommendation if item.separatix else None
                ),
                "separatix_confidence": (item.separatix.confidence if item.separatix else None),
                "separatix_skip_reason": (
                    item.separatix.skipped_reason if item.separatix else None
                ),
                "separatix_guidance_status": family_guidance.get("status"),
                "separatix_minimum_family": family_guidance.get("minimum_recommended_family"),
                "separatix_plausible_families": family_guidance.get("plausible_families", []),
                "separatix_guidance_decision_method": family_guidance.get("decision_method"),
                "separatix_selected_family": family_guidance.get("selected_family"),
                "separatix_selected_probe": family_guidance.get("selected_probe"),
                "separatix_selected_recipe_id": family_guidance.get("selected_recipe_id"),
                "separatix_mlp_override": family_guidance.get("mlp_override"),
                "separatix_paired_status": family_guidance.get("paired_status"),
                "separatix_paired_method": family_guidance.get("paired_method"),
                "separatix_family_guidance": family_guidance,
                "resource_profile_status": profile.status if profile else "disabled",
                "first_call_seconds": (inference.first_call_seconds if inference else None),
                "warm_median_seconds": (inference.warm_median_seconds if inference else None),
                "warm_p95_seconds": inference.warm_p95_seconds if inference else None,
                "throughput_samples_per_second": (
                    inference.throughput_samples_per_second if inference else None
                ),
                "resource_batch_sizes": inference.batch_sizes if inference else [],
                "synchronization_status": (
                    profile.context.get("synchronization_status") if profile else None
                ),
                "peak_host_rss_bytes": (host_memory.peak_rss_bytes if host_memory else None),
                "peak_host_rss_increase_bytes": (
                    host_memory.peak_increase_bytes if host_memory else None
                ),
                "peak_device_allocated_bytes": (
                    device_memory.peak_allocated_bytes if device_memory else None
                ),
                "baseline_device_allocated_bytes": (
                    device_memory.baseline_allocated_bytes if device_memory else None
                ),
                "peak_device_allocated_increase_bytes": (
                    device_memory.peak_allocated_increase_bytes if device_memory else None
                ),
                "peak_device_reserved_bytes": (
                    device_memory.peak_reserved_bytes if device_memory else None
                ),
                "device_memory_status": device_memory.status if device_memory else None,
                "device_memory_scope": (device_memory.measurement_scope if device_memory else None),
                "device_memory_unavailable_reason": (
                    device_memory.unavailable_reason if device_memory else None
                ),
                "model_footprint_status": model.status if model else None,
                "parameter_footprint_status": model.parameter_status if model else None,
                "checkpoint_footprint_status": model.checkpoint_status if model else None,
                "parameter_count": model.parameter_count if model else None,
                "parameter_bytes": model.parameter_bytes if model else None,
                "trainable_parameter_count": (model.trainable_parameter_count if model else None),
                "trainable_parameter_bytes": (model.trainable_parameter_bytes if model else None),
                "model_buffer_bytes": model.buffer_bytes if model else None,
                "model_in_memory_bytes": model.in_memory_bytes if model else None,
                "model_weight_dtypes": model.weight_dtypes if model else [],
                "checkpoint_bytes": model.checkpoint_bytes if model else None,
                "raw_embedding_bytes": embedding.raw_bytes if embedding else None,
                "evaluated_embedding_bytes": (embedding.evaluated_bytes if embedding else None),
                "raw_persisted_embedding_bytes": (
                    embedding.raw_persisted.bytes if embedding and embedding.raw_persisted else None
                ),
                "evaluated_persisted_embedding_bytes": (
                    embedding.evaluated_persisted.bytes
                    if embedding and embedding.evaluated_persisted
                    else None
                ),
                "evaluated_persisted_embedding_status": (
                    embedding.evaluated_persisted.status
                    if embedding and embedding.evaluated_persisted
                    else None
                ),
                "embedding_bytes_per_sample": (
                    embedding.bytes_per_embedding if embedding else None
                ),
                "resource_profile_scope": (
                    "distributed_shards" if distributed_profile else "local" if profile else None
                ),
                "worker_first_call_median_seconds": (
                    distributed_profile.worker_first_calls.median_seconds
                    if distributed_profile
                    else None
                ),
                "worker_first_call_p95_seconds": (
                    distributed_profile.worker_first_calls.p95_seconds
                    if distributed_profile
                    else None
                ),
                "worker_first_call_max_seconds": (
                    distributed_profile.worker_first_calls.max_seconds
                    if distributed_profile
                    else None
                ),
                "aggregate_compute_throughput_samples_per_second": (
                    distributed_profile.aggregate_compute_throughput_samples_per_second
                    if distributed_profile
                    else None
                ),
                "max_worker_peak_rss_bytes": (
                    distributed_profile.max_worker_peak_rss_bytes if distributed_profile else None
                ),
                "max_worker_peak_device_allocated_bytes": (
                    distributed_profile.max_worker_peak_device_allocated_bytes
                    if distributed_profile
                    else None
                ),
            }
            for metric_name, metric in item.metrics.items():
                row[f"metric.{metric_name}"] = metric.score
            for runtime_name, runtime_value in item.runtime.items():
                row[f"runtime.{runtime_name}"] = runtime_value
            rows.append(row)
        return rows

    def save_json(self, path: str) -> None:
        """Save the benchmark result as JSON."""

        from vertebrae.reports.json_report import save_json_report

        save_json_report(self, str(Path(path)))

    def save_markdown(self, path: str) -> None:
        """Save the benchmark result as a Markdown report."""

        from vertebrae.reports.markdown_report import save_markdown_report

        save_markdown_report(self, str(Path(path)))


def _primary_metric(item: ExtractorResult) -> MetricResult:
    return item.metrics[item.primary_metric_name]


def _rankable_score(item: ExtractorResult) -> float:
    metric = _primary_metric(item)
    return metric.score if metric.higher_is_better else -metric.score


def _result_target_metadata(item: ExtractorResult) -> Dict[str, Any]:
    if item.overlap is not None:
        return item.overlap.metadata
    return _primary_metric(item).metadata


def _output_identity(item: ExtractorResult) -> Dict[str, Any]:
    metadata = item.embedding_metadata
    output_name = metadata.get("output_name")
    recipe = metadata.get("recipe") or {}
    materialization = metadata.get("structured") or metadata.get("segmentation") or {}
    materialization_recipe = materialization.get("output_recipe") or {}
    extractor_recipe = (
        metadata.get("extractor_recipe") or metadata.get("source_extractor_recipe") or recipe
    )
    parent_extractor = (
        metadata.get("parent_extractor_name")
        or extractor_recipe.get("name")
        or metadata.get("extractor_name")
        or item.name
    )
    hidden_layer = recipe.get("hidden_layer", materialization_recipe.get("hidden_layer"))
    pooling = recipe.get("pooling", materialization_recipe.get("pooling"))
    if output_name is None:
        output_name = recipe.get("output_name")

    if output_name is not None and (hidden_layer is None or pooling is None):
        for collection_name in ("outputs", "structured_outputs", "spatial_outputs"):
            for output in extractor_recipe.get(collection_name, []) or []:
                if output.get("name") != output_name:
                    continue
                if hidden_layer is None:
                    hidden_layer = output.get("hidden_layer")
                if pooling is None:
                    pooling = output.get("pooling")
                break

    return {
        "parent_extractor": parent_extractor,
        "output_name": output_name,
        "hidden_layer": hidden_layer,
        "pooling": pooling,
    }
