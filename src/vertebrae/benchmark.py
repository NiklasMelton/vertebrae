"""Benchmark runner."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import chain
from time import perf_counter
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

from vertebrae._version import __version__
from vertebrae.cache import ArtifactStore, create_artifact_store
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe, hash_json_exact
from vertebrae.cache.keys import named_output_artifact_keys
from vertebrae.compression import (
    compress_embedding_artifact_key,
    compress_embeddings,
    compression_variant_name,
)
from vertebrae.config import (
    CacheConfig,
    ContinuousOverlapScoringConfig,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    ExecutionConfig,
    LabelViewConfig,
    MemoryConfig,
    OverlapScoringConfig,
    ResourceProfilingConfig,
    SegmentationConfig,
    SeparatixConfig,
    StabilityConfig,
    TargetViewConfig,
    overlap_scoring_config_recipe,
)
from vertebrae.execution.base import ExecutionBackend
from vertebrae.execution.jobs import SampleBatch
from vertebrae.extractors.base import EmbeddingOutput
from vertebrae.profiling import ResourceProfiler, with_embedding_footprint
from vertebrae.reports.recommendations import (
    recommendation_for_extractor,
    recommendations_for_benchmark,
)
from vertebrae.results import BenchmarkResult, ExtractorResult
from vertebrae.scoring.metrics import MetricResult, OverlapMetric, as_embedding_metric
from vertebrae.scoring.separatix import SeparatixResult, SeparatixScorer
from vertebrae.scoring.stability import run_stability_analysis
from vertebrae.utils.labels import (
    MULTI_LABEL_TARGET,
    REGRESSION_TARGET,
    canonical_metric_targets,
    label_view_suffix,
    metric_labels,
    target_view_suffix,
)
from vertebrae.utils.memory import (
    EmbeddingMemoryEstimate,
    IncrementalMatrixReferenceStager,
    IncrementalMatrixStager,
    assert_within_memory,
    estimate_embedding_from_probe,
    estimate_matrix_resident_bytes,
    estimate_metadata_resident_bytes,
    estimate_metadata_scoring_input_bytes,
    largest_fitting_subsample_rate,
    resolve_memory_budget,
)
from vertebrae.utils.semantic_labels import (
    canonical_semantic_array,
    label_display,
    semantic_label_key,
)
from vertebrae.utils.validation import (
    ensure_numeric_matrix,
    is_sparse_matrix,
    sparse_storage_format,
)


class Benchmark:
    """Run one or more extractors against a labeled dataset.

    Args:
        dataset: Dataset object with inputs and labels.
        extractors: Optional iterable of extractors to evaluate.
        scoring_config: OverlapIndex scoring configuration.
        stability_config: Stability-analysis configuration.
        cache_config: Embedding cache configuration.
        embedding_config: Embedding batching and streaming configuration.
        execution: Optional local, Ray, or Dask execution backend. When omitted,
            the benchmark uses the direct in-process path.
        execution_config: Artifact-backed execution and sharding settings used
            only when an explicit backend is provided.
    """

    def __init__(
        self,
        dataset: Any,
        extractors: Optional[Iterable[Any]] = None,
        scoring_config: Optional[
            Union[OverlapScoringConfig, ContinuousOverlapScoringConfig]
        ] = None,
        stability_config: Optional[StabilityConfig] = None,
        label_view_config: Optional[LabelViewConfig] = None,
        target_view_config: Optional[TargetViewConfig] = None,
        separatix_config: Optional[SeparatixConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        compression_config: Optional[EmbeddingCompressionConfig] = None,
        compression_configs: Optional[Iterable[EmbeddingCompressionConfig]] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        execution: Optional[Any] = None,
        execution_config: Optional[ExecutionConfig] = None,
        segmentation_config: Optional[SegmentationConfig] = None,
        structured_aligners: Optional[dict[str, Any]] = None,
        metrics: Optional[Iterable[Any]] = None,
        primary_metric: Optional[str] = None,
        resource_profiling_config: Optional[ResourceProfilingConfig] = None,
    ) -> None:
        self.dataset = dataset
        self.extractors = list(extractors or [])
        self._explicit_scoring_config = scoring_config
        self.scoring_config = scoring_config or _default_scoring_config_for_dataset(dataset)
        self.stability_config = stability_config or StabilityConfig()
        self.label_view_config = label_view_config or LabelViewConfig()
        self.target_view_config = target_view_config or TargetViewConfig()
        self.separatix_config = separatix_config or SeparatixConfig()
        self.cache_config = cache_config or CacheConfig()
        if compression_config is not None and compression_configs is not None:
            raise ValueError("Provide compression_config or compression_configs, not both.")
        default_compressions = (
            [compression_config]
            if compression_config is not None
            else [EmbeddingCompressionConfig()]
        )
        self.compression_configs = list(compression_configs or default_compressions)
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.memory_config = memory_config or MemoryConfig()
        if execution is None and execution_config is not None:
            raise ValueError("execution_config requires an explicit execution backend.")
        if execution is not None and not isinstance(execution, ExecutionBackend):
            raise TypeError("execution must implement submit(), gather(), status(), and map().")
        self.execution = execution
        self.execution_config = execution_config or ExecutionConfig()
        self.segmentation_config = segmentation_config or SegmentationConfig()
        self.structured_aligners = dict(structured_aligners or {})
        self.metrics = [as_embedding_metric(metric) for metric in (metrics or [])]
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("Metric names must be unique within a benchmark.")
        overlap_metrics = [metric for metric in self.metrics if metric.name == "overlap"]
        if len(overlap_metrics) > 1 or any(
            not isinstance(metric, OverlapMetric) for metric in overlap_metrics
        ):
            raise ValueError("Only one built-in OverlapMetric may use the name 'overlap'.")
        provided_overlap = bool(overlap_metrics)
        if not provided_overlap:
            self.metrics.insert(0, OverlapMetric(config=self.scoring_config))
        self.overlap_metric = next(metric for metric in self.metrics if metric.name == "overlap")
        if not isinstance(self.overlap_metric, OverlapMetric):
            raise ValueError("The overlap metric must be an OverlapMetric instance.")
        if provided_overlap and self.overlap_metric.config is not None:
            if scoring_config is not None and scoring_config != self.overlap_metric.config:
                raise ValueError(
                    "Provide overlap scoring options through OverlapMetric or "
                    "scoring_config, not both."
                )
            self.scoring_config = self.overlap_metric.config
            self._explicit_scoring_config = self.overlap_metric.config
        elif provided_overlap:
            # A config-less OverlapMetric inherits the benchmark's resolved
            # configuration instead of silently using scorer defaults.
            unresolved_overlap = self.overlap_metric
            self.overlap_metric = unresolved_overlap.with_config(self.scoring_config)
            self.metrics = [
                self.overlap_metric if metric is unresolved_overlap else metric
                for metric in self.metrics
            ]
        available_metrics = [metric.name for metric in self.metrics]
        self.primary_metric = primary_metric or available_metrics[0]
        self.resource_profiling_config = resource_profiling_config or ResourceProfilingConfig()
        self._resource_profiler: Optional[ResourceProfiler] = None
        if self.primary_metric not in available_metrics:
            raise ValueError(
                f"primary_metric {self.primary_metric!r} is not among configured metrics: "
                f"{available_metrics}."
            )

    def add_extractor(self, extractor: Any) -> "Benchmark":
        """Add an extractor to this benchmark.

        Args:
            extractor: Feature extractor implementing the vertebrae protocol.

        Returns:
            This benchmark instance for fluent chaining.
        """

        self.extractors.append(extractor)
        return self

    def run(self) -> BenchmarkResult:
        """Run feature extraction, scoring, diagnostics, and reporting aggregation.

        Returns:
            Aggregated benchmark result.

        Raises:
            ValueError: If no extractors are configured or dataset validation fails.
        """

        self.dataset.validate()
        if (
            self.dataset.metadata.get("target_type") == REGRESSION_TARGET
            and getattr(self.dataset, "modality", None) == "segmentation"
        ):
            raise ValueError(
                "Segmentation evaluation does not currently support regression targets."
            )
        if not self.extractors:
            raise ValueError("At least one extractor must be provided.")
        is_segmentation = getattr(self.dataset, "modality", None) == "segmentation"
        unit_annotations = getattr(self.dataset, "unit_annotations", None)
        is_structured = (
            callable(unit_annotations)
            and bool(unit_annotations())
            and any(
                callable(getattr(extractor, "transform_structured", None))
                for extractor in self.extractors
            )
        )
        self._validate_view_config_compatibility()
        if is_segmentation:
            self._validate_special_output_configuration("spatial_output_specs")
        elif is_structured:
            self._validate_special_output_configuration("structured_output_specs")
        if self.execution is not None:
            from vertebrae.execution.benchmark_runner import run_artifact_backed_benchmark

            return run_artifact_backed_benchmark(self)
        if is_segmentation:
            return self._run_segmentation()
        if is_structured:
            return self._run_structured()

        evaluation_datasets, label_view_warnings, target_view_warnings = self._evaluation_datasets()
        self._validate_output_view_mapping()
        extractor_results: List[ExtractorResult] = []
        for extractor in self.extractors:
            result = self._run_extractor(
                extractor,
                evaluation_datasets,
                label_view_warnings,
                target_view_warnings,
            )
            if isinstance(result, list):
                extractor_results.extend(result)
            else:
                extractor_results.append(result)
        if not extractor_results:
            raise ValueError("Benchmark produced no scoreable extractor outputs.")
        recommendations = recommendations_for_benchmark(
            extractor_results,
            quality_tolerance=(
                self.resource_profiling_config.quality_tolerance
                if self.resource_profiling_config.enabled
                else None
            ),
        )
        return BenchmarkResult(
            dataset_summary=self.dataset.summary(),
            extractor_results=extractor_results,
            recommendations=recommendations,
            metadata=self._result_metadata(
                scoring_config=self.scoring_config,
                extra={
                    "label_view_warnings": label_view_warnings,
                    "target_view_warnings": target_view_warnings,
                },
            ),
        )

    def _run_segmentation(self) -> BenchmarkResult:
        from vertebrae.cache import create_artifact_store
        from vertebrae.extractors import PrecomputedExtractor
        from vertebrae.segmentation import iter_materialize_segmentation_outputs

        extractor_results: List[ExtractorResult] = []
        materialization_summaries = []
        store = create_artifact_store(
            self.cache_config.cache_dir,
            **self.cache_config.storage_options,
        )
        for extractor in self.extractors:
            if not callable(getattr(extractor, "transform_spatial", None)):
                raise ValueError(
                    f"Extractor '{extractor.name}' does not support spatial outputs. "
                    "Use an extractor implementing transform_spatial(...)."
                )
            profiler = ResourceProfiler(
                self.resource_profiling_config,
                extractor,
                streaming=True,
            )
            materializations = iter_materialize_segmentation_outputs(
                dataset=self.dataset,
                extractor=extractor,
                config=self.segmentation_config,
                batch_size=self.embedding_config.batch_size,
                resource_profiler=(profiler if self.resource_profiling_config.enabled else None),
                memory_config=self.memory_config,
            )
            try:
                first_materialization = next(materializations)
            except StopIteration as exc:
                raise ValueError(
                    f"Spatial extractor {extractor.name!r} produced no outputs."
                ) from exc
            source_profile = profiler.finish() if self.resource_profiling_config.enabled else None
            materialization_iter = chain((first_materialization,), materializations)
            del first_materialization
            for materialization in materialization_iter:
                precomputed = PrecomputedExtractor(
                    name=_qualified_output_name(extractor.name, materialization.name)
                )
                precomputed.cache_embeddings = self._cache_embeddings_enabled(  # type: ignore[attr-defined]
                    extractor
                ) and bool(materialization.metadata.get("cache_safe", True))
                scoring_config = _classification_scoring_config(self.scoring_config)
                if self.segmentation_config.background_mode == "include_excluded":
                    exclusions = _normalized_excluded_classes(scoring_config.exclude_classes)
                    for background_label in materialization.metadata.get("background_labels", []):
                        if not _label_is_excluded_exact(background_label, exclusions):
                            exclusions.append(background_label)
                    scoring_config = replace(
                        scoring_config,
                        exclude_classes=exclusions,
                    )
                result = Benchmark(
                    dataset=materialization.dataset,
                    extractors=[precomputed],
                    scoring_config=scoring_config,
                    stability_config=self.stability_config,
                    target_view_config=self.target_view_config,
                    separatix_config=self.separatix_config,
                    cache_config=self.cache_config,
                    compression_configs=self.compression_configs,
                    embedding_config=self.embedding_config,
                    memory_config=self.memory_config,
                    metrics=[metric for metric in self.metrics if metric.name != "overlap"],
                    primary_metric=self.primary_metric,
                    resource_profiling_config=self.resource_profiling_config,
                ).run()
                for item in result.extractor_results:
                    if source_profile is not None and item.resource_profile is not None:
                        item.resource_profile = replace(
                            source_profile,
                            embedding=item.resource_profile.embedding,
                        )
                    item.extractor_type = getattr(extractor, "extractor_type", "spatial")
                    item.embedding_metadata["segmentation"] = materialization.metadata
                    item.embedding_metadata["source_extractor_recipe"] = extractor.recipe()
                    item.embedding_metadata["resolved_scoring_config"] = (
                        overlap_scoring_config_recipe(scoring_config)
                    )
                    provenance = _selected_provenance(
                        materialization.provenance,
                        item.embedding_metadata,
                    )
                    provenance_key = f"{item.embedding_metadata['cache_key']}/provenance"
                    item.embedding_metadata["provenance_key"] = provenance_key
                    item.embedding_metadata["provenance_rows"] = len(provenance)
                    if item.embedding_metadata.get("cache_eligible", False):
                        store.put_json(
                            provenance_key,
                            {"rows": provenance},
                        )
                    extractor_results.append(item)
                materialization_summaries.append(
                    {
                        "extractor": extractor.name,
                        "output": materialization.name,
                        **materialization.metadata,
                    }
                )
                del result, materialization

        if not extractor_results:
            raise ValueError("Segmentation benchmark produced no scoreable outputs.")
        recommendations = recommendations_for_benchmark(
            extractor_results,
            quality_tolerance=(
                self.resource_profiling_config.quality_tolerance
                if self.resource_profiling_config.enabled
                else None
            ),
        )
        return BenchmarkResult(
            dataset_summary={
                **self.dataset.summary(),
                "segmentation_outputs": materialization_summaries,
            },
            extractor_results=extractor_results,
            recommendations=recommendations,
            metadata=self._result_metadata(
                scoring_config=scoring_config,
                extra={
                    "segmentation_config": asdict(self.segmentation_config),
                    "resolved_scoring_configs": {
                        item.name: item.embedding_metadata["resolved_scoring_config"]
                        for item in extractor_results
                    },
                },
            ),
        )

    def _run_structured(self) -> BenchmarkResult:
        from vertebrae.extractors import PrecomputedExtractor
        from vertebrae.structured import iter_materialize_structured_outputs

        extractor_results: List[ExtractorResult] = []
        materialization_summaries = []
        store = create_artifact_store(
            self.cache_config.cache_dir,
            **self.cache_config.storage_options,
        )
        for extractor in self.extractors:
            if not callable(getattr(extractor, "transform_structured", None)):
                raise ValueError(
                    f"Extractor '{extractor.name}' does not support structured outputs."
                )
            profiler = ResourceProfiler(
                self.resource_profiling_config,
                extractor,
                streaming=True,
            )
            materializations = iter_materialize_structured_outputs(
                dataset=self.dataset,
                extractor=extractor,
                batch_size=self.embedding_config.batch_size,
                aligners=self.structured_aligners,
                resource_profiler=(profiler if self.resource_profiling_config.enabled else None),
                memory_config=self.memory_config,
            )
            try:
                first_materialization = next(materializations)
            except StopIteration as exc:
                raise ValueError(
                    f"Structured extractor {extractor.name!r} produced no outputs."
                ) from exc
            source_profile = profiler.finish() if self.resource_profiling_config.enabled else None
            materialization_iter = chain((first_materialization,), materializations)
            del first_materialization
            for materialization in materialization_iter:
                output_dataset: Any = materialization.dataset
                if self._output_has_view_mapping(materialization.name):
                    output_dataset = self._mapped_output_dataset(
                        dataset=materialization.dataset,
                        output_name=materialization.name,
                        label_view_warnings=[],
                        target_view_warnings=[],
                    )
                    if output_dataset is None:
                        del materialization
                        continue
                precomputed = PrecomputedExtractor(
                    name=_qualified_output_name(extractor.name, materialization.name)
                )
                precomputed.cache_embeddings = self._cache_embeddings_enabled(  # type: ignore[attr-defined]
                    extractor
                ) and bool(materialization.metadata.get("cache_safe", True))
                result = Benchmark(
                    dataset=output_dataset,
                    extractors=[precomputed],
                    scoring_config=self._resolved_scoring_config(output_dataset),
                    stability_config=self.stability_config,
                    target_view_config=(
                        TargetViewConfig()
                        if materialization.name in self.target_view_config.output_views
                        else self.target_view_config
                    ),
                    separatix_config=self.separatix_config,
                    cache_config=self.cache_config,
                    compression_configs=self.compression_configs,
                    embedding_config=self.embedding_config,
                    memory_config=self.memory_config,
                    metrics=[metric for metric in self.metrics if metric.name != "overlap"],
                    primary_metric=self.primary_metric,
                    resource_profiling_config=self.resource_profiling_config,
                ).run()
                for item in result.extractor_results:
                    if source_profile is not None and item.resource_profile is not None:
                        item.resource_profile = replace(
                            source_profile,
                            embedding=item.resource_profile.embedding,
                        )
                    item.extractor_type = getattr(extractor, "extractor_type", "structured")
                    item.embedding_metadata["structured"] = materialization.metadata
                    item.embedding_metadata["source_extractor_recipe"] = extractor.recipe()
                    item.embedding_metadata["resolved_scoring_config"] = (
                        overlap_scoring_config_recipe(self._resolved_scoring_config(output_dataset))
                    )
                    provenance = _selected_provenance(
                        materialization.provenance,
                        item.embedding_metadata,
                    )
                    provenance_key = f"{item.embedding_metadata['cache_key']}/provenance"
                    item.embedding_metadata["provenance_key"] = provenance_key
                    item.embedding_metadata["provenance_rows"] = len(provenance)
                    if item.embedding_metadata.get("cache_eligible", False):
                        store.put_json(provenance_key, {"rows": provenance})
                    extractor_results.append(item)
                materialization_summaries.append(
                    {
                        "extractor": extractor.name,
                        "output": materialization.name,
                        **materialization.metadata,
                    }
                )
                del result, output_dataset, materialization

        if not extractor_results:
            raise ValueError("Structured benchmark produced no scoreable outputs.")
        recommendations = recommendations_for_benchmark(
            extractor_results,
            quality_tolerance=(
                self.resource_profiling_config.quality_tolerance
                if self.resource_profiling_config.enabled
                else None
            ),
        )
        return BenchmarkResult(
            dataset_summary={
                **self.dataset.summary(),
                "structured_outputs": materialization_summaries,
            },
            extractor_results=extractor_results,
            recommendations=recommendations,
            metadata=self._result_metadata(
                scoring_config=self.scoring_config,
                extra={
                    "structured_aligners": {
                        name: aligner.recipe() for name, aligner in self.structured_aligners.items()
                    },
                    "resolved_scoring_configs": {
                        item.name: item.embedding_metadata["resolved_scoring_config"]
                        for item in extractor_results
                    },
                },
            ),
        )

    def _run_extractor(
        self,
        extractor: Any,
        evaluation_datasets: List[Any],
        label_view_warnings: List[str],
        target_view_warnings: List[str],
    ) -> Union[ExtractorResult, List[ExtractorResult]]:
        if self._has_output_view_mappings() and self._supports_named_outputs(extractor):
            return self._run_extractor_with_output_views(
                extractor,
                label_view_warnings,
                target_view_warnings,
            )
        results: List[ExtractorResult] = []
        for dataset in evaluation_datasets:
            result = self._run_extractor_on_dataset(extractor, dataset)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        if len(results) == 1:
            return results[0]
        return results

    def _run_extractor_with_output_views(
        self,
        extractor: Any,
        label_view_warnings: List[str],
        target_view_warnings: List[str],
    ) -> Union[ExtractorResult, List[ExtractorResult]]:
        warnings: List[str] = []
        profiler = self._start_resource_profiler(extractor)
        runtime = {}
        start = perf_counter()
        dataset, subsampling_warnings, subsampling_metadata, probe_plan = (
            self._prepare_dataset_for_extractor(extractor, self.dataset)
        )
        warnings.extend(subsampling_warnings)
        store = create_artifact_store(
            self.cache_config.cache_dir,
            **self.cache_config.storage_options,
        )
        variants = self._get_or_compute_embedding_variants(
            extractor,
            dataset,
            store,
            subsampling_metadata,
            probe_plan,
        )
        self._attach_resource_profile(profiler, variants)
        runtime["embedding_seconds"] = perf_counter() - start
        results: List[ExtractorResult] = []
        mapped_outputs = 0
        for variant in variants:
            output_name = variant["metadata"].get("output_name")
            if not self._output_has_view_mapping(output_name):
                continue
            mapped_outputs += 1
            scoring_dataset = self._mapped_output_dataset(
                dataset=dataset,
                output_name=output_name,
                label_view_warnings=label_view_warnings,
                target_view_warnings=target_view_warnings,
            )
            if scoring_dataset is None:
                continue
            results.extend(
                self._score_embedding_variant(
                    extractor=extractor,
                    variant=variant,
                    dataset=scoring_dataset,
                    store=store,
                    warnings=warnings,
                    runtime=runtime,
                )
            )
        if mapped_outputs == 0:
            return []
        if not results:
            raise ValueError("No valid mapped hierarchy label views were available for scoring.")
        if len(results) == 1:
            self._resource_profiler = None
            return results[0]
        self._resource_profiler = None
        return results

    def _run_extractor_on_dataset(
        self,
        extractor: Any,
        evaluation_dataset: Any,
    ) -> Union[ExtractorResult, List[ExtractorResult]]:
        warnings: List[str] = []
        profiler = self._start_resource_profiler(extractor)
        runtime = {}
        start = perf_counter()
        dataset, subsampling_warnings, subsampling_metadata, probe_plan = (
            self._prepare_dataset_for_extractor(extractor, evaluation_dataset)
        )
        warnings.extend(subsampling_warnings)
        store = create_artifact_store(
            self.cache_config.cache_dir,
            **self.cache_config.storage_options,
        )
        variants = self._get_or_compute_embedding_variants(
            extractor,
            dataset,
            store,
            subsampling_metadata,
            probe_plan,
        )
        self._attach_resource_profile(profiler, variants)
        runtime["embedding_seconds"] = perf_counter() - start
        results: List[ExtractorResult] = []
        for variant in variants:
            results.extend(
                self._score_embedding_variant(
                    extractor=extractor,
                    variant=variant,
                    dataset=dataset,
                    store=store,
                    warnings=warnings,
                    runtime=runtime,
                )
            )
        if len(results) == 1:
            self._resource_profiler = None
            return results[0]
        self._resource_profiler = None
        return results

    def _score_embedding_variant(
        self,
        extractor: Any,
        variant: dict,
        dataset: Any,
        store: ArtifactStore,
        warnings: List[str],
        runtime: dict,
    ) -> List[ExtractorResult]:
        embeddings = variant["embeddings"]
        scoring_config = self._resolved_scoring_config(dataset)
        embedding_metadata = dict(variant["metadata"])
        base_resource_profile = embedding_metadata.pop("_resource_profile", None)
        embedding_metadata["label_view"] = dataset.active_label_view()
        embedding_metadata["target_view"] = dataset.active_target_view()
        results: List[ExtractorResult] = []
        for compression_config in self.compression_configs:
            variant_warnings = list(warnings)
            variant_runtime = dict(runtime)
            compression_start = perf_counter()
            compressed_embeddings, compression_metadata = (
                self._get_or_compute_compressed_embeddings(
                    embeddings=embeddings,
                    embedding_metadata=embedding_metadata,
                    labels=dataset.y,
                    store=store,
                    config=compression_config,
                )
            )
            variant_runtime["compression_seconds"] = perf_counter() - compression_start
            variant_warnings.extend(compression_metadata.get("warnings", []))

            score_start = perf_counter()
            scoring_metadata = dict(embedding_metadata)
            scoring_metadata["embedding_dim"] = compression_metadata.get(
                "compressed_dim",
                embedding_metadata.get("embedding_dim"),
            )
            scoring_metadata["shape"] = [
                embedding_metadata.get("n_samples"),
                scoring_metadata["embedding_dim"],
            ]
            scoring_metadata["sparse"] = compression_metadata.get(
                "output_sparse",
                embedding_metadata.get("sparse"),
            )
            self._admit_scoring_memory(scoring_metadata, dataset)
            target_metadata = dict(dataset.metadata)
            target_metadata["target_type"] = dataset.metadata.get("target_type", "auto")
            groups = dataset.groups() if callable(getattr(dataset, "groups", None)) else None
            canonical_labels = canonical_metric_targets(
                dataset.y,
                label_names=dataset.metadata.get("label_names"),
                target_type=target_metadata["target_type"],
                target_names=dataset.metadata.get("target_names"),
            )
            canonical_groups = None if groups is None else canonical_semantic_array(groups)
            metric_results: dict[str, MetricResult] = {}
            for metric in self.metrics:
                metric_result = metric.score(
                    compressed_embeddings,
                    canonical_labels,
                    target_metadata=target_metadata,
                    groups=canonical_groups,
                )
                metric_result.metadata = {**target_metadata, **metric_result.metadata}
                metric_results[metric.name] = metric_result
                variant_warnings.extend(metric_result.warnings)
            variant_runtime["scoring_seconds"] = perf_counter() - score_start
            overlap = metric_results["overlap"]

            separatix_start = perf_counter()
            separatix = None
            separatix = self._run_separatix_diagnostic(
                compressed_embeddings,
                dataset.y,
                overlap.score,
                scoring_config=scoring_config,
                target_type=target_metadata["target_type"],
                label_names=dataset.metadata.get("label_names"),
                target_names=dataset.metadata.get("target_names"),
                groups=groups,
            )
            variant_runtime["separatix_seconds"] = perf_counter() - separatix_start
            if separatix:
                variant_warnings.extend(separatix.warnings)

            stability_start = perf_counter()
            stability = run_stability_analysis(
                compressed_embeddings,
                dataset.y,
                scoring_config,
                self.stability_config,
                label_names=dataset.metadata.get("label_names"),
                target_type=target_metadata["target_type"],
                target_names=dataset.metadata.get("target_names"),
            )
            variant_runtime["stability_seconds"] = perf_counter() - stability_start
            if stability:
                variant_warnings.extend(stability.get("warnings", []))

            weakest_class, weakest_score = _weakest_class(
                overlap.per_class_scores,
                excluded_classes=overlap.metadata.get("exclude_classes"),
                label_catalog=overlap.metadata.get("label_catalog"),
            )
            recommendation = (
                recommendation_for_extractor(
                    overlap.score,
                    stability,
                    weakest_score,
                    target_type=overlap.metadata.get("target_type", "single_label"),
                )
                if self.primary_metric == "overlap"
                and overlap.metadata.get("aggregate_valid", True)
                else "aggregate_unavailable"
                if self.primary_metric == "overlap"
                else f"ranked_by_{self.primary_metric}"
            )
            result_name = _qualified_result_name(
                embedding_metadata.get("extractor_name", extractor.name),
                dataset.active_target_view(),
                dataset.active_label_view(),
            )
            results.append(
                ExtractorResult(
                    name=_variant_extractor_name(result_name, compression_metadata),
                    extractor_type=embedding_metadata.get(
                        "extractor_type",
                        getattr(extractor, "extractor_type", "unknown"),
                    ),
                    stability=stability,
                    separatix=separatix,
                    embedding_metadata=embedding_metadata,
                    compression_metadata=compression_metadata,
                    runtime=variant_runtime,
                    warnings=sorted(set(variant_warnings)),
                    label_view=dataset.active_label_view(),
                    target_view=dataset.active_target_view(),
                    weakest_class=weakest_class,
                    weakest_class_score=weakest_score,
                    recommendation=recommendation,
                    metrics=metric_results,
                    primary_metric_name=self.primary_metric,
                    resource_profile=with_embedding_footprint(
                        base_resource_profile,
                        embeddings,
                        compressed_embeddings,
                        store=store,
                        raw_key=embedding_metadata.get("cache_key"),
                        evaluated_key=compression_metadata.get(
                            "cache_key",
                            embedding_metadata.get("cache_key"),
                        ),
                        persisted_storage=self.resource_profiling_config.persisted_storage,
                    ),
                )
            )
        return results

    def _run_separatix_diagnostic(
        self,
        embeddings: Any,
        labels: Any,
        overlap_score: float,
        scoring_config: Union[OverlapScoringConfig, ContinuousOverlapScoringConfig],
        target_type: str,
        label_names: Optional[Any] = None,
        target_names: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Optional[SeparatixResult]:
        if not self.separatix_config.enabled:
            return None
        scorer = SeparatixScorer(
            config=self.separatix_config,
            overlap_config=scoring_config,
        )
        threshold = (
            self.separatix_config.regression_overlap_threshold
            if target_type == REGRESSION_TARGET
            else self.separatix_config.overlap_threshold
        )
        if overlap_score < threshold:
            return scorer.skipped_result(
                reason=(
                    "Skipped Separatix because overlap score "
                    f"{overlap_score:.4f} is below the configured threshold "
                    f"{threshold:.4f}."
                ),
                overlap_score=overlap_score,
                threshold=threshold,
            )
        (
            diagnostic_embeddings,
            diagnostic_labels,
            diagnostic_groups,
            diagnostic_label_names,
        ) = self._diagnostic_inputs(
            embeddings,
            labels,
            groups,
            target_type=target_type,
            scoring_config=scoring_config,
            label_names=label_names,
        )
        diagnostic_shape: Any = getattr(diagnostic_labels, "shape", ())
        if int(diagnostic_shape[0]) == 0 or (
            target_type == MULTI_LABEL_TARGET and int(diagnostic_shape[1]) == 0
        ):
            return scorer.skipped_result(
                reason="Skipped Separatix because all classes were excluded from diagnostics.",
                overlap_score=overlap_score,
                threshold=threshold,
            )
        try:
            return scorer.score(
                diagnostic_embeddings,
                diagnostic_labels,
                label_names=diagnostic_label_names,
                target_type=target_type,
                target_names=target_names,
                groups=diagnostic_groups,
            )
        except ValueError as exc:
            if diagnostic_groups is None:
                raise
            return scorer.skipped_result(
                reason=f"Skipped grouped Separatix diagnostic: {exc}",
                overlap_score=overlap_score,
                threshold=threshold,
            )

    def _diagnostic_inputs(
        self,
        embeddings: Any,
        labels: Any,
        groups: Optional[Any],
        target_type: str,
        scoring_config: Union[OverlapScoringConfig, ContinuousOverlapScoringConfig],
        label_names: Optional[Any] = None,
    ) -> Tuple[Any, Any, Optional[Any], Optional[Any]]:
        if target_type == REGRESSION_TARGET:
            return embeddings, labels, groups, label_names
        excluded = _normalized_excluded_classes(_scoring_excluded_classes(scoring_config))
        if target_type == MULTI_LABEL_TARGET:
            label_array, label_metadata = metric_labels(
                labels,
                label_names=label_names,
                target_type=MULTI_LABEL_TARGET,
            )
            names = tuple(label_metadata.get("label_names") or ())
            if not excluded:
                return embeddings, label_array, groups, names
            if len(names) != label_array.shape[1]:
                raise ValueError("Multi-label exclusions require one label name per target column.")
            excluded_keys = {semantic_label_key(item) for item in excluded}
            known_keys = {semantic_label_key(item) for item in names}
            unknown = excluded_keys - known_keys
            if unknown:
                missing = [item for item in excluded if semantic_label_key(item) in unknown]
                raise ValueError(
                    "exclude_classes contains labels absent from the multi-label target: "
                    f"{missing!r}."
                )
            keep = [
                index
                for index, name in enumerate(names)
                if semantic_label_key(name) not in excluded_keys
            ]
            selected_labels = label_array[:, keep]
            active_rows = np.asarray(selected_labels.sum(axis=1)).reshape(-1) > 0
            return (
                embeddings[active_rows],
                selected_labels[active_rows],
                None if groups is None else np.asarray(groups)[active_rows],
                tuple(names[index] for index in keep),
            )
        if not excluded:
            return embeddings, labels, groups, label_names
        label_array = np.asarray(labels)
        mask = np.asarray(
            [not _label_is_excluded_exact(label, excluded) for label in label_array],
            dtype=bool,
        )
        filtered_groups = None if groups is None else np.asarray(groups)[mask]
        return embeddings[mask], label_array[mask], filtered_groups, label_names

    def _prepare_dataset_for_extractor(
        self,
        extractor: Any,
        dataset: Any,
    ) -> Tuple[Any, List[str], dict, Optional[Tuple[SampleBatch, Any, Any]]]:
        warnings: List[str] = []
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None
        metadata: dict[str, Any] = {
            "subsampled": False,
            "subsample_reason": None,
            "requested_subsample_rate": self.memory_config.subsample_rate,
            "effective_subsample_rate": 1.0,
            "manual_subsample_rate": 1.0,
            "automatic_subsample_rate": 1.0,
            "cumulative_subsample_rate": 1.0,
            "original_n_samples": int(len(dataset.y)),
            "parent_n_samples": int(len(dataset.y)),
        }
        recipe = extractor.recipe()
        if self.cache_config.enabled and recipe.get("cache_safe") is False:
            warnings.append(
                f"Extractor '{extractor.name}' has no stable cache identity; embedding and "
                "derived compression cache reuse is bypassed. Provide cache_identity to opt in."
            )
        if self.memory_config.subsample_rate < 1.0:
            dataset, user_metadata, warning = self._subsample_dataset(
                dataset,
                rate=self.memory_config.subsample_rate,
                reason="user_requested",
            )
            metadata.update(user_metadata)
            metadata["manual_subsample_rate"] = user_metadata["effective_subsample_rate"]
            warnings.append(warning)

        if self._should_stream_embeddings(extractor):
            auto_rate, probe_plan = self._auto_subsample_rate_for_streaming_estimate(
                extractor,
                dataset,
            )
            if auto_rate < 1.0:
                dataset, auto_metadata, warning = self._subsample_dataset(
                    dataset,
                    rate=auto_rate,
                    reason="memory_limit",
                )
                metadata.update(auto_metadata)
                metadata["automatic_subsample_rate"] = auto_metadata["effective_subsample_rate"]
                warnings.append(warning)
                probe_plan = None

        metadata["final_n_samples"] = int(len(dataset.y))
        metadata["cumulative_subsample_rate"] = int(len(dataset.y)) / int(
            metadata["original_n_samples"]
        )
        metadata["effective_subsample_rate"] = metadata["cumulative_subsample_rate"]

        return dataset, warnings, metadata, probe_plan

    def _evaluation_datasets(self) -> Tuple[List[Any], List[str], List[str]]:
        target_view_warnings = self._target_view_warnings()
        datasets = self._target_view_datasets(target_view_warnings)
        if (
            self.label_view_config.output_levels
            and self.dataset.metadata.get("label_hierarchy") is None
        ):
            raise ValueError(
                "LabelViewConfig requires dataset label hierarchy metadata. "
                "Use BenchmarkDataset.with_label_hierarchy(...)."
            )
        if self._requires_label_hierarchy() and any(
            dataset.metadata.get("target_type") == REGRESSION_TARGET for dataset in datasets
        ):
            raise ValueError("LabelViewConfig is not supported for regression targets.")
        if not self.label_view_config.enabled:
            return datasets, [], target_view_warnings
        if (
            self._requires_label_hierarchy()
            and self.dataset.metadata.get("label_hierarchy") is None
        ):
            raise ValueError(
                "LabelViewConfig requires dataset label hierarchy metadata. "
                "Use BenchmarkDataset.with_label_hierarchy(...)."
            )
        label_view_warnings: List[str] = []
        resolved = []
        seen = set()
        for base_dataset in datasets:
            for level in self.label_view_config.hierarchy_levels:
                try:
                    dataset = base_dataset.label_view(level)
                except ValueError as exc:
                    if not self.label_view_config.skip_invalid_levels:
                        raise
                    label_view_warnings.append(f"Skipped hierarchy level {level!r}: {exc}")
                    continue
                view_key = (
                    dataset.active_target_view().get("key"),
                    dataset.active_label_view().get("key"),
                )
                if view_key in seen:
                    continue
                seen.add(view_key)
                resolved.append(dataset)
        if not resolved:
            raise ValueError("No valid hierarchy label views were available for benchmarking.")
        return resolved, label_view_warnings, target_view_warnings

    def _requires_label_hierarchy(self) -> bool:
        return bool(self.label_view_config.enabled or self.label_view_config.output_levels)

    def _requires_target_views(self) -> bool:
        return bool(self.target_view_config.enabled or self.target_view_config.output_views)

    def _validate_view_config_compatibility(self) -> None:
        if self._requires_label_hierarchy() and self._requires_target_views():
            raise ValueError(
                "LabelViewConfig and TargetViewConfig cannot be combined in one benchmark yet."
            )

    def _validate_special_output_configuration(self, spec_method: str) -> None:
        workflow = "segmentation" if spec_method == "spatial_output_specs" else "structured"
        output_names: set[str] = set()
        for extractor in self.extractors:
            method = getattr(extractor, spec_method, None)
            if not callable(method):
                raise ValueError(
                    f"Extractor '{extractor.name}' does not declare {workflow} output specs."
                )
            specs = list(method())
            raw_names = [getattr(spec, "name", None) for spec in specs]
            if not raw_names or any(
                not isinstance(name, str) or not name.strip() for name in raw_names
            ):
                raise ValueError(
                    f"Extractor '{extractor.name}' must declare non-empty {workflow} output names."
                )
            names = [name for name in raw_names if isinstance(name, str)]
            if len(names) != len(set(names)):
                raise ValueError(
                    f"Extractor '{extractor.name}' declares duplicate {workflow} output names."
                )
            output_names.update(names)

        if workflow == "segmentation" and (
            self._requires_label_hierarchy() or self._requires_target_views()
        ):
            raise ValueError(
                "LabelViewConfig and TargetViewConfig are not supported for segmentation "
                "materialization. Configure background handling through SegmentationConfig."
            )
        if workflow == "structured" and self._requires_label_hierarchy():
            raise ValueError(
                "LabelViewConfig is not supported for structured unit materialization because "
                "unit annotations do not declare a hierarchy. Use named target views instead."
            )

        requested_outputs = set(self.target_view_config.output_views)
        unknown = sorted(requested_outputs - output_names)
        if unknown:
            raise ValueError(
                f"Configured {workflow} output mappings contain unknown outputs: {unknown}."
            )
        if requested_outputs:
            available = set(self.dataset.target_view_names())
            missing = sorted(set(self.target_view_config.output_views.values()) - available)
            if missing:
                raise ValueError(
                    "TargetViewConfig.output_views contains unknown target views: " f"{missing}."
                )
        if workflow == "structured":
            unknown_aligners = sorted(set(self.structured_aligners) - output_names)
            if unknown_aligners:
                raise ValueError(
                    f"Structured aligners contain unknown output names: {unknown_aligners}."
                )

    def _result_metadata(
        self,
        *,
        scoring_config: Union[OverlapScoringConfig, ContinuousOverlapScoringConfig],
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "vertebrae_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scoring_config": overlap_scoring_config_recipe(scoring_config),
            "stability_config": asdict(self.stability_config),
            "label_view_config": asdict(self.label_view_config),
            "target_view_config": asdict(self.target_view_config),
            "separatix_config": asdict(self.separatix_config),
            "cache_config": asdict(self.cache_config),
            "compression_configs": [asdict(config) for config in self.compression_configs],
            "embedding_config": asdict(self.embedding_config),
            "memory_config": asdict(self.memory_config),
            "metrics": [metric.recipe() for metric in self.metrics],
            "primary_metric": self.primary_metric,
            "resource_profiling_config": asdict(self.resource_profiling_config),
        }
        metadata.update(extra or {})
        return metadata

    def _validate_output_view_mapping(self) -> None:
        if not self._has_output_view_mappings():
            return
        output_names = {
            spec.name
            for extractor in self.extractors
            if self._supports_named_outputs(extractor)
            for spec in self._output_specs(extractor)
        }
        requested_outputs = set(self.label_view_config.output_levels) | set(
            self.target_view_config.output_views
        )
        unknown = sorted(requested_outputs - output_names)
        if unknown:
            raise ValueError(
                "Configured output view mappings contain unknown output names: " f"{unknown}."
            )
        if self.target_view_config.output_views:
            available = (
                set(self.dataset.target_view_names())
                if callable(getattr(self.dataset, "target_view_names", None))
                else set()
            )
            if not available:
                raise ValueError(
                    "TargetViewConfig.output_views requires dataset target view metadata. "
                    "Use BenchmarkDataset.with_target_views(...)."
                )
            missing = sorted(set(self.target_view_config.output_views.values()) - available)
            if missing:
                raise ValueError(
                    "TargetViewConfig.output_views contains unknown target views: " f"{missing}."
                )

    def _mapped_output_dataset(
        self,
        dataset: Any,
        output_name: str,
        label_view_warnings: List[str],
        target_view_warnings: List[str],
    ) -> Optional[Any]:
        mapped = dataset
        if output_name in self.target_view_config.output_views:
            view_name = self.target_view_config.output_views[output_name]
            try:
                mapped = mapped.target_view(view_name)
            except ValueError as exc:
                if not self.target_view_config.skip_invalid_views:
                    raise
                target_view_warnings.append(
                    f"Skipped output {output_name!r} target view {view_name!r}: {exc}"
                )
                return None
        if output_name in self.label_view_config.output_levels:
            level = self.label_view_config.output_levels[output_name]
            try:
                mapped = mapped.label_view(level)
            except ValueError as exc:
                if not self.label_view_config.skip_invalid_levels:
                    raise
                label_view_warnings.append(
                    f"Skipped output {output_name!r} hierarchy level {level!r}: {exc}"
                )
                return None
        return mapped

    def _subsample_dataset(self, dataset: Any, rate: float, reason: str) -> Tuple[Any, dict, str]:
        indices = dataset.stratified_subsample_indices(
            rate=rate,
            random_state=self.memory_config.subsample_random_state,
            min_samples_per_class=self.memory_config.min_subsample_samples_per_class,
        )
        parent_n_samples = int(len(dataset.y))
        subset = dataset.subset(
            indices,
            metadata={
                "subsampled": True,
                "subsample_reason": reason,
                "requested_subsample_rate": rate,
                "effective_subsample_rate": len(indices) / parent_n_samples,
            },
        )
        effective_rate = len(indices) / parent_n_samples
        metadata = {
            "subsampled": True,
            "subsample_reason": reason,
            "requested_subsample_rate": rate,
            "effective_subsample_rate": effective_rate,
            "parent_n_samples": parent_n_samples,
            "sample_indices": subset.metadata.get("sample_indices", indices.tolist()),
            "n_samples_after_subsampling": int(len(indices)),
        }
        regression = dataset.metadata.get("target_type") == REGRESSION_TARGET
        if reason == "memory_limit":
            if regression:
                warning = (
                    "Embedding memory estimate exceeded the configured budget; using a "
                    "target-preserving regression subsample with effective rate "
                    f"{effective_rate:.3f} "
                    f"({len(indices)}/{parent_n_samples} samples)."
                )
            else:
                warning = (
                    "Embedding memory estimate exceeded the configured budget; using a "
                    f"class-stratified subsample with effective rate {effective_rate:.3f} "
                    f"({len(indices)}/{parent_n_samples} samples)."
                )
        else:
            if regression:
                warning = (
                    "Using user-requested target-preserving regression subsample with effective "
                    "rate "
                    f"{effective_rate:.3f} ({len(indices)}/{parent_n_samples} samples)."
                )
            else:
                warning = (
                    "Using user-requested class-stratified subsample with effective rate "
                    f"{effective_rate:.3f} ({len(indices)}/{parent_n_samples} samples)."
                )
        return subset, metadata, warning

    def _auto_subsample_rate_for_streaming_estimate(
        self,
        extractor: Any,
        dataset: Any,
    ) -> Tuple[float, Optional[Tuple[SampleBatch, Any, Any]]]:
        if self._embedding_cache_available(extractor, dataset):
            return 1.0, None
        if not self.memory_config.auto_subsample_on_memory_exceeded:
            return 1.0, None
        reuse_probe = bool(getattr(extractor, "already_fitted", False))
        if reuse_probe:
            probe_extractor = extractor
        else:
            try:
                probe_extractor = deepcopy(extractor)
            except Exception:
                # Fitting the live instance here could train on rows an automatic
                # subsample later discards. If cloning is unsupported, defer to the
                # normal one-fit path and its memory admission error.
                return 1.0, None
            probe_extractor.fit(dataset.X, dataset.y)
        first_batch = next(
            dataset.iter_batches(
                batch_size=min(self.embedding_config.batch_size, len(dataset.y)),
                shard=None,
            )
        )
        if self._supports_transform_many(probe_extractor):
            first_outputs = self._embed_batch_many(
                probe_extractor,
                first_batch,
                materialized=False,
                profiled=reuse_probe,
            )
            estimates, aggregate = self._estimate_multi_output_memory(
                first_outputs,
                n_samples=len(dataset.y),
                scoring_row_multiplier=self._scoring_row_multiplier(dataset),
            )
            required = max(
                aggregate["scoring_input_bytes"],
                aggregate["resident_bytes"],
            )
            try:
                self._admit_multi_embedding_plan(aggregate)
            except ValueError:
                rate = largest_fitting_subsample_rate(required, self.memory_config)
                if rate <= 0.0:
                    probe_plan = (
                        first_batch,
                        first_outputs,
                        {"per_output": estimates, **aggregate},
                    )
                    return 1.0, probe_plan if reuse_probe else None
                return min(1.0, rate), None
            probe_plan = (
                first_batch,
                first_outputs,
                {"per_output": estimates, **aggregate},
            )
            return 1.0, probe_plan if reuse_probe else None

        first_embeddings = self._embed_batch(
            probe_extractor,
            first_batch,
            materialized=False,
            profiled=reuse_probe,
        )
        estimate = estimate_embedding_from_probe(
            first_embeddings,
            n_samples=len(dataset.y),
            batch_size=self.embedding_config.batch_size,
            memory_config=self.memory_config,
            scoring_row_multiplier=self._scoring_row_multiplier(dataset),
        )
        required = estimate.scoring_input_bytes
        if estimate.strategy == "in_memory":
            required = max(required, estimate.resident_bytes)
        try:
            self._admit_embedding_plan(estimate)
        except ValueError:
            rate = largest_fitting_subsample_rate(required, self.memory_config)
            if rate <= 0.0:
                single_probe_plan = (first_batch, first_embeddings, estimate)
                return 1.0, single_probe_plan if reuse_probe else None
            return min(1.0, rate), None
        single_probe_plan = (first_batch, first_embeddings, estimate)
        return 1.0, single_probe_plan if reuse_probe else None

    def _embedding_cache_available(self, extractor: Any, dataset: Any) -> bool:
        if not self._cache_embeddings_enabled(extractor) or self.cache_config.force_recompute:
            return False
        store = create_artifact_store(
            self.cache_config.cache_dir,
            **self.cache_config.storage_options,
        )
        base_key = (
            f"embeddings/{dataset.identity_key()}/"
            f"{fingerprint_extractor_recipe(extractor.recipe())}"
        )
        if self._supports_transform_many(extractor):
            specs = self._output_specs(extractor)
            cache_keys = named_output_artifact_keys(base_key, (spec.name for spec in specs))
            return all(store.exists(cache_keys[spec.name]) for spec in specs)
        return store.exists(base_key)

    def _get_or_compute_embedding_variants(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        subsampling_metadata: Optional[dict] = None,
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
    ) -> List[dict]:
        if not self._supports_transform_many(extractor):
            embeddings, metadata = self._get_or_compute_embeddings(
                extractor,
                dataset,
                store,
                subsampling_metadata,
                probe_plan,
            )
            return [{"embeddings": embeddings, "metadata": metadata}]
        return self._get_or_compute_multi_embeddings(
            extractor,
            dataset,
            store,
            subsampling_metadata,
            probe_plan,
        )

    def _get_or_compute_embeddings(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        subsampling_metadata: Optional[dict] = None,
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
    ) -> Any:
        recipe = extractor.recipe()
        dataset_key = dataset.identity_key()
        extractor_key = fingerprint_extractor_recipe(recipe)
        cache_key = f"embeddings/{dataset_key}/{extractor_key}"
        embedding_cache_enabled = self._cache_embeddings_enabled(extractor)
        if (
            embedding_cache_enabled
            and not self.cache_config.force_recompute
            and store.exists(cache_key)
        ):
            self._admit_cached_embedding_load(store.get_json(cache_key))
            embeddings, metadata = store.get_artifact(cache_key)
            self._admit_cached_embedding_load(metadata)
            metadata = dict(metadata)
            metadata["cache_hit"] = True
            metadata["cache_status"] = "hit"
            metadata.update(subsampling_metadata or {})
            return embeddings, metadata

        if self._should_stream_embeddings(extractor):
            embeddings, metadata = self._stream_embeddings(
                extractor,
                dataset,
                store,
                cache_key,
                recipe,
                probe_plan,
                subsampling_metadata,
            )
            return embeddings, metadata

        embeddings = self._measure_resource_call(
            lambda: extractor.fit_transform(dataset.X, dataset.y),
            samples=len(dataset.y),
            call_type="fit_transform",
            includes_fit=True,
        )
        embeddings = ensure_numeric_matrix(
            embeddings,
            f"Extractor '{extractor.name}' embeddings",
            allow_sparse=True,
        )
        self._admit_resident_embedding(embeddings)
        if embeddings.shape[0] != len(dataset.y):
            raise ValueError(
                f"Extractor '{extractor.name}' returned {embeddings.shape[0]} embeddings "
                f"for {len(dataset.y)} labels."
            )
        single_output_spec = self._single_output_spec(extractor)
        metadata = self._embedding_metadata(
            extractor=extractor,
            dataset=dataset,
            embeddings=embeddings,
            cache_key=cache_key,
            recipe=recipe,
            output_name=single_output_spec.name if single_output_spec is not None else None,
            output_metadata=(
                dict(single_output_spec.metadata) if single_output_spec is not None else None
            ),
        )
        metadata.update(subsampling_metadata or {})
        if embedding_cache_enabled:
            store.put_artifact(cache_key, embeddings, metadata)
        return embeddings, metadata

    def _get_or_compute_multi_embeddings(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        subsampling_metadata: Optional[dict] = None,
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
    ) -> List[dict]:
        recipe = extractor.recipe()
        base_key = f"embeddings/{dataset.identity_key()}/{fingerprint_extractor_recipe(recipe)}"
        specs = self._output_specs(extractor)
        cache_keys = named_output_artifact_keys(base_key, (spec.name for spec in specs))
        embedding_cache_enabled = self._cache_embeddings_enabled(extractor)
        if embedding_cache_enabled and not self.cache_config.force_recompute:
            if all(store.exists(cache_key) for cache_key in cache_keys.values()):
                return self._load_cached_multi_embeddings(
                    extractor=extractor,
                    dataset=dataset,
                    store=store,
                    specs=specs,
                    cache_keys=cache_keys,
                    subsampling_metadata=subsampling_metadata,
                )

        if self._should_stream_embeddings(extractor):
            variants = self._stream_multi_embeddings(
                extractor=extractor,
                dataset=dataset,
                store=store,
                cache_keys=cache_keys,
                recipe=recipe,
                probe_plan=probe_plan,
                subsampling_metadata=subsampling_metadata,
            )
            return variants

        extractor.fit(dataset.X, dataset.y)
        raw_outputs = self._measure_resource_call(
            lambda: extractor.transform_many(dataset.X),
            samples=len(dataset.y),
            call_type="transform_many",
        )
        outputs = self._validate_multi_outputs(
            extractor=extractor,
            outputs=raw_outputs,
            expected_rows=len(dataset.y),
            context="embeddings",
        )
        self._admit_multi_resident_embeddings(outputs)
        variants = []
        for output in outputs:
            cache_key = cache_keys[output.name]
            metadata = self._embedding_metadata(
                extractor=extractor,
                dataset=dataset,
                embeddings=output.embeddings,
                cache_key=cache_key,
                recipe=self._qualified_output_recipe(recipe, output),
                extractor_name=_qualified_output_name(extractor.name, output.name),
                parent_extractor_name=extractor.name,
                output_name=output.name,
                extractor_recipe=recipe,
                output_metadata=output.metadata,
            )
            metadata.update(subsampling_metadata or {})
            if embedding_cache_enabled:
                store.put_artifact(cache_key, output.embeddings, metadata)
            variants.append({"embeddings": output.embeddings, "metadata": metadata})
        return variants

    def _load_cached_multi_embeddings(
        self,
        *,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        specs: List[Any],
        cache_keys: dict[str, str],
        subsampling_metadata: Optional[dict],
    ) -> List[dict]:
        """Preflight and load cached named outputs without retaining aggregate RAM."""

        preflight = {spec.name: dict(store.get_json(cache_keys[spec.name])) for spec in specs}
        resident_sizes = []
        for metadata in preflight.values():
            self._admit_cached_embedding_load(metadata)
            self._admit_scoring_memory(metadata, dataset)
            resident = estimate_metadata_resident_bytes(metadata)
            if resident is not None:
                resident_sizes.append(resident)
        aggregate_resident = sum(resident_sizes) if len(resident_sizes) == len(specs) else None
        force_disk = False
        if aggregate_resident is not None:
            budget = resolve_memory_budget(self.memory_config).max_memory_bytes
            force_disk = aggregate_resident > budget
            if force_disk and not self.memory_config.allow_disk_spill:
                assert_within_memory(
                    aggregate_resident,
                    self.memory_config,
                    purpose="Cached multi-output embedding artifacts",
                )

        if not force_disk:
            variants = []
            for spec in specs:
                embeddings, metadata = store.get_artifact(cache_keys[spec.name])
                self._admit_cached_embedding_load(metadata)
                metadata = self._cached_embedding_metadata(
                    metadata,
                    subsampling_metadata=subsampling_metadata,
                )
                variants.append({"embeddings": embeddings, "metadata": metadata})
            return variants

        committed_metadata: dict[str, dict] = {}
        variants = []
        with (
            IncrementalMatrixStager(
                self.memory_config,
                purpose=f"Extractor '{extractor.name}' cached multi-output staging",
            ) as stager,
            IncrementalMatrixReferenceStager(
                self.memory_config,
                purpose=f"Extractor '{extractor.name}' cached multi-output staging",
                matrix_stager=stager,
            ) as reference_stager,
        ):
            for spec in specs:
                embeddings, metadata = store.get_artifact(cache_keys[spec.name])
                self._admit_cached_embedding_load(metadata)
                embeddings = ensure_numeric_matrix(
                    embeddings,
                    f"Extractor '{extractor.name}' cached output '{spec.name}'",
                    allow_sparse=True,
                )
                if embeddings.shape[0] != len(dataset.y):
                    raise ValueError(
                        f"Cached extractor output '{spec.name}' has {embeddings.shape[0]} "
                        f"rows for {len(dataset.y)} labels."
                    )
                for row_index in range(len(dataset.y)):
                    reference_stager.append(
                        spec.name,
                        row_index,
                        stager.append(
                            spec.name,
                            embeddings[row_index : row_index + 1],
                        ),
                    )
                committed_metadata[spec.name] = self._cached_embedding_metadata(
                    metadata,
                    subsampling_metadata=subsampling_metadata,
                )
                del embeddings

            for spec in specs:
                assembly = reference_stager.assemble(
                    spec.name,
                    expected_rows=len(dataset.y),
                    purpose=f"Extractor '{extractor.name}' cached output '{spec.name}'",
                    force_disk=True,
                )
                metadata = committed_metadata[spec.name]
                metadata["materialization"] = {
                    "strategy": assembly.strategy,
                    "required_bytes": assembly.required_bytes,
                    "budget_bytes": assembly.budget_bytes,
                    "staging_strategy": assembly.staging_strategy,
                }
                variants.append({"embeddings": assembly.matrix, "metadata": metadata})
        return variants

    @staticmethod
    def _cached_embedding_metadata(
        metadata: dict,
        *,
        subsampling_metadata: Optional[dict],
    ) -> dict:
        committed = dict(metadata)
        committed["cache_hit"] = True
        committed["cache_status"] = "hit"
        committed.update(subsampling_metadata or {})
        return committed

    def _get_or_compute_compressed_embeddings(
        self,
        embeddings: Any,
        embedding_metadata: dict,
        labels: Any,
        store: ArtifactStore,
        config: EmbeddingCompressionConfig,
    ) -> Tuple[Any, dict]:
        if not config.enabled or config.method == "none":
            compression_result = compress_embeddings(embeddings, config=config, y=labels)
            return compression_result.embeddings, compression_result.metadata

        source_key = embedding_metadata["cache_key"]
        compression_key = compress_embedding_artifact_key(source_key, config)
        cache_eligible = bool(embedding_metadata.get("cache_eligible", False))
        if (
            cache_eligible
            and not self.cache_config.force_recompute
            and store.exists(compression_key)
        ):
            self._admit_cached_embedding_load(store.get_json(compression_key))
            compressed_embeddings, metadata = store.get_artifact(compression_key)
            self._admit_cached_embedding_load(metadata)
            metadata = dict(metadata)
            metadata["cache_hit"] = True
            metadata["cache_status"] = "hit"
            return compressed_embeddings, metadata

        compression_result = compress_embeddings(embeddings, config=config, y=labels)
        metadata = dict(compression_result.metadata)
        metadata["cache_key"] = compression_key
        metadata["cache_hit"] = False
        metadata["cache_eligible"] = cache_eligible
        metadata["source_cache_status"] = embedding_metadata.get("cache_status")
        metadata["cache_status"] = (
            "miss" if cache_eligible else embedding_metadata.get("cache_status", "disabled")
        )
        if cache_eligible:
            store.put_artifact(compression_key, compression_result.embeddings, metadata)
        return compression_result.embeddings, metadata

    def _should_stream_embeddings(self, extractor: Any) -> bool:
        if not self.embedding_config.streaming_enabled:
            return False
        return bool(getattr(extractor, "streaming_safe", False))

    def _stream_embeddings(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        cache_key: str,
        recipe: dict,
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
        subsampling_metadata: Optional[dict] = None,
    ) -> Tuple[Any, dict]:
        n_samples = len(dataset.y)
        embedding_cache_enabled = self._cache_embeddings_enabled(extractor)
        batch_iterator = iter(
            dataset.iter_batches(
                batch_size=self.embedding_config.batch_size,
                shard=None,
            )
        )
        if probe_plan is None:
            extractor.fit(dataset.X, dataset.y)
            try:
                first_batch = next(batch_iterator)
            except StopIteration as exc:
                raise ValueError("At least one sample is required for embedding.") from exc
            first_embeddings = self._embed_batch(extractor, first_batch)
            memory_estimate = estimate_embedding_from_probe(
                first_embeddings,
                n_samples=n_samples,
                batch_size=self.embedding_config.batch_size,
                memory_config=self.memory_config,
                scoring_row_multiplier=self._scoring_row_multiplier(dataset),
            )
            self._admit_embedding_plan(memory_estimate)
        else:
            first_batch, first_embeddings, memory_estimate = probe_plan
            if self._resource_profiler is not None:
                self._resource_profiler.mark_last_call_materialized()
            try:
                skipped_batch = next(batch_iterator)
            except StopIteration as exc:
                raise ValueError("At least one sample is required for embedding.") from exc
            if not np.array_equal(skipped_batch.indices, first_batch.indices):
                raise ValueError("Reusable embedding probe does not match streaming batch order.")
        batch_pairs = _prepend_batch(
            first_batch.indices,
            first_embeddings,
            self._embedding_batches_from(extractor, batch_iterator),
        )
        single_output_spec = self._single_output_spec(extractor)
        if embedding_cache_enabled:
            metadata = self._embedding_metadata(
                extractor=extractor,
                dataset=dataset,
                embeddings=first_embeddings,
                cache_key=cache_key,
                recipe=recipe,
                output_name=(single_output_spec.name if single_output_spec is not None else None),
                output_metadata=(
                    dict(single_output_spec.metadata) if single_output_spec is not None else None
                ),
            )
            metadata["streamed"] = True
            metadata["stream_batch_size"] = self.embedding_config.batch_size
            metadata["memory_estimate"] = memory_estimate.to_dict()
            metadata.update(subsampling_metadata or {})
            store.put_artifact_batches(
                cache_key,
                batch_pairs,
                n_samples=n_samples,
                metadata=metadata,
                require_complete=True,
            )
            embeddings, metadata = store.get_artifact(cache_key)
        else:
            if memory_estimate.strategy == "stream_to_disk":
                raise ValueError(
                    "Embedding artifact is estimated to exceed the memory budget, but "
                    "embedding caching is disabled for this extractor. Enable caching for "
                    "the extractor or increase MemoryConfig.max_memory_bytes."
                )
            embeddings = _combine_embedding_batches(
                batch_pairs,
                n_samples=n_samples,
            )
            metadata = self._embedding_metadata(
                extractor=extractor,
                dataset=dataset,
                embeddings=embeddings,
                cache_key=cache_key,
                recipe=recipe,
                output_name=(single_output_spec.name if single_output_spec is not None else None),
                output_metadata=(
                    dict(single_output_spec.metadata) if single_output_spec is not None else None
                ),
            )
            metadata["streamed"] = True
            metadata["stream_batch_size"] = self.embedding_config.batch_size
            metadata["memory_estimate"] = memory_estimate.to_dict()
            metadata.update(subsampling_metadata or {})
        return embeddings, metadata

    def _stream_multi_embeddings(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        cache_keys: dict[str, str],
        recipe: dict,
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
        subsampling_metadata: Optional[dict] = None,
    ) -> List[dict]:
        n_samples = len(dataset.y)
        embedding_cache_enabled = self._cache_embeddings_enabled(extractor)
        batch_iterator = iter(
            dataset.iter_batches(
                batch_size=self.embedding_config.batch_size,
                shard=None,
            )
        )
        if probe_plan is None:
            extractor.fit(dataset.X, dataset.y)
            try:
                first_batch = next(batch_iterator)
            except StopIteration as exc:
                raise ValueError("At least one sample is required for embedding.") from exc
            first_outputs = self._embed_batch_many(extractor, first_batch)
            estimates, aggregate = self._estimate_multi_output_memory(
                first_outputs,
                n_samples=n_samples,
                scoring_row_multiplier=self._scoring_row_multiplier(dataset),
            )
            self._admit_multi_embedding_plan(aggregate)
        else:
            first_batch, first_outputs, estimate_info = probe_plan
            if self._resource_profiler is not None:
                self._resource_profiler.mark_last_call_materialized()
            estimates = estimate_info["per_output"]
            aggregate = estimate_info
            try:
                skipped_batch = next(batch_iterator)
            except StopIteration as exc:
                raise ValueError("At least one sample is required for embedding.") from exc
            if not np.array_equal(skipped_batch.indices, first_batch.indices):
                raise ValueError("Reusable embedding probe does not match streaming batch order.")

        output_metadata = {output.name: dict(output.metadata) for output in first_outputs}
        output_contracts = {
            output.name: {
                "recipe": hash_json_exact(dict(output.recipe)),
                "metadata": hash_json_exact(dict(output.metadata)),
            }
            for output in first_outputs
        }
        output_recipe = {
            output.name: self._qualified_output_recipe(recipe, output) for output in first_outputs
        }

        def stage_outputs(
            stager: IncrementalMatrixStager,
            reference_stager: IncrementalMatrixReferenceStager,
            batch: SampleBatch,
            outputs: List[EmbeddingOutput],
        ) -> None:
            indices = _strict_batch_indices(batch.indices)
            if len(indices) and (indices.min() < 0 or indices.max() >= n_samples):
                raise ValueError("Embedding batch indices are outside the dataset row range.")
            for output in outputs:
                contract = {
                    "recipe": hash_json_exact(dict(output.recipe)),
                    "metadata": hash_json_exact(dict(output.metadata)),
                }
                if contract != output_contracts[output.name]:
                    raise ValueError(
                        f"Extractor '{extractor.name}' output '{output.name}' changed its "
                        "recipe or metadata between streaming batches."
                    )
                for row_index, sample_index in enumerate(indices):
                    position = int(sample_index)
                    reference_stager.append(
                        output.name,
                        position,
                        stager.append(
                            output.name,
                            output.embeddings[row_index : row_index + 1],
                        ),
                    )

        variants = []
        with (
            IncrementalMatrixStager(
                self.memory_config,
                purpose=f"Extractor '{extractor.name}' local multi-output staging",
            ) as stager,
            IncrementalMatrixReferenceStager(
                self.memory_config,
                purpose=f"Extractor '{extractor.name}' local multi-output staging",
                matrix_stager=stager,
            ) as reference_stager,
        ):
            stage_outputs(stager, reference_stager, first_batch, first_outputs)
            for batch in batch_iterator:
                outputs = self._embed_batch_many(extractor, batch)
                stage_outputs(stager, reference_stager, batch, outputs)

            for output_name in output_metadata:
                assembly = reference_stager.assemble(
                    output_name,
                    expected_rows=n_samples,
                    purpose=f"Extractor '{extractor.name}' output '{output_name}' embeddings",
                    force_disk=aggregate["strategy"] == "stream_to_disk",
                )
                embeddings = assembly.matrix
                cache_key = cache_keys[output_name]
                metadata = self._embedding_metadata(
                    extractor=extractor,
                    dataset=dataset,
                    embeddings=embeddings,
                    cache_key=cache_key,
                    recipe=output_recipe[output_name],
                    extractor_name=_qualified_output_name(extractor.name, output_name),
                    parent_extractor_name=extractor.name,
                    output_name=output_name,
                    extractor_recipe=recipe,
                    output_metadata=output_metadata[output_name],
                )
                metadata["streamed"] = True
                metadata["stream_batch_size"] = self.embedding_config.batch_size
                metadata["memory_estimate"] = estimates[output_name].to_dict()
                metadata["multi_output_memory_estimate"] = {
                    key: value for key, value in aggregate.items() if key != "per_output"
                }
                metadata["materialization"] = {
                    "strategy": assembly.strategy,
                    "required_bytes": assembly.required_bytes,
                    "budget_bytes": assembly.budget_bytes,
                    "staging_strategy": assembly.staging_strategy,
                }
                metadata.update(subsampling_metadata or {})
                if embedding_cache_enabled:
                    store.put_artifact(cache_key, embeddings, metadata)
                variants.append({"embeddings": embeddings, "metadata": metadata})
        return variants

    def _embedding_batches_from(
        self,
        extractor: Any,
        batches: Iterator[SampleBatch],
    ) -> Iterator[Tuple[np.ndarray, Any]]:
        for batch in batches:
            yield batch.indices, self._embed_batch(extractor, batch)

    def _embed_batch(
        self,
        extractor: Any,
        batch: SampleBatch,
        *,
        materialized: bool = True,
        profiled: bool = True,
    ) -> Any:
        def call() -> Any:
            return extractor.transform(batch.X)

        embeddings = (
            self._measure_resource_call(
                call,
                samples=len(batch.indices),
                call_type="transform",
                materialized=materialized,
            )
            if profiled
            else call()
        )
        embeddings = ensure_numeric_matrix(
            embeddings,
            f"Extractor '{extractor.name}' batch embeddings",
            allow_sparse=True,
        )
        if embeddings.shape[0] != len(batch.indices):
            raise ValueError(
                f"Extractor '{extractor.name}' returned {embeddings.shape[0]} embeddings "
                f"for a batch with {len(batch.indices)} samples."
            )
        return embeddings

    def _embed_batch_many(
        self,
        extractor: Any,
        batch: SampleBatch,
        *,
        materialized: bool = True,
        profiled: bool = True,
    ) -> List[EmbeddingOutput]:
        def call() -> Any:
            return extractor.transform_many(batch.X)

        outputs = (
            self._measure_resource_call(
                call,
                samples=len(batch.indices),
                call_type="transform_many",
                materialized=materialized,
            )
            if profiled
            else call()
        )
        return self._validate_multi_outputs(
            extractor=extractor,
            outputs=outputs,
            expected_rows=len(batch.indices),
            context="batch embeddings",
        )

    def _start_resource_profiler(self, extractor: Any) -> ResourceProfiler:
        profiler = ResourceProfiler(
            self.resource_profiling_config,
            extractor,
            streaming=self._should_stream_embeddings(extractor),
        )
        self._resource_profiler = profiler
        return profiler

    def _attach_resource_profile(self, profiler: ResourceProfiler, variants: List[dict]) -> None:
        profile = (
            profiler.finish(cache_hit=all(item["metadata"].get("cache_hit") for item in variants))
            if self.resource_profiling_config.enabled
            else None
        )
        for variant in variants:
            variant["metadata"]["_resource_profile"] = profile

    def _measure_resource_call(
        self,
        fn: Any,
        *,
        samples: int,
        call_type: str,
        materialized: bool = True,
        includes_fit: bool = False,
    ) -> Any:
        if self._resource_profiler is None:
            return fn()
        return self._resource_profiler.measure_call(
            fn,
            samples=samples,
            call_type=call_type,
            materialized=materialized,
            includes_fit=includes_fit,
        )

    def _embedding_metadata(
        self,
        extractor: Any,
        dataset: Any,
        embeddings: Any,
        cache_key: str,
        recipe: dict,
        extractor_name: Optional[str] = None,
        parent_extractor_name: Optional[str] = None,
        output_name: Optional[str] = None,
        extractor_recipe: Optional[dict] = None,
        output_metadata: Optional[dict] = None,
    ) -> dict:
        sparse_embeddings = is_sparse_matrix(embeddings)
        cache_eligible = self._cache_embeddings_enabled(extractor)
        recipe_cache_safe = recipe.get("cache_safe")
        cache_status = (
            "miss"
            if cache_eligible
            else "bypassed_unsafe_identity"
            if recipe_cache_safe is False
            else "disabled"
        )
        return {
            "extractor_name": extractor_name or extractor.name,
            "parent_extractor_name": parent_extractor_name,
            "output_name": output_name,
            "extractor_type": getattr(extractor, "extractor_type", "unknown"),
            "modality": getattr(extractor, "modality", dataset.modality),
            "cache_hit": False,
            "cache_eligible": cache_eligible,
            "cache_status": cache_status,
            "cache_key": cache_key,
            "shape": list(embeddings.shape),
            "n_samples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "sparse": sparse_embeddings,
            "nnz": int(embeddings.nnz) if sparse_embeddings else None,
            "storage_format": (sparse_storage_format(embeddings) if sparse_embeddings else "dense"),
            "streamed": False,
            "memory_estimate": None,
            "recipe": recipe,
            "extractor_recipe": extractor_recipe or recipe,
            "output_metadata": output_metadata or {},
            "label_view": dataset.active_label_view(),
        }

    def _admit_embedding_plan(self, estimate: EmbeddingMemoryEstimate) -> None:
        batch_required = (
            estimate.batch_embedding_bytes
            + self.memory_config.model_memory_bytes
            + self.memory_config.raw_batch_memory_bytes
        )
        assert_within_memory(
            batch_required,
            self.memory_config,
            purpose="Embedding batch",
        )
        if estimate.strategy == "in_memory":
            assert_within_memory(
                estimate.resident_bytes,
                self.memory_config,
                purpose="Resident embedding artifact",
            )
        assert_within_memory(
            estimate.scoring_input_bytes,
            self.memory_config,
            purpose="Scoring input",
        )

    def _admit_resident_embedding(self, embeddings: Any) -> None:
        required = estimate_matrix_resident_bytes(embeddings)
        assert_within_memory(
            required,
            self.memory_config,
            purpose="Resident embedding artifact",
        )

    def _admit_multi_resident_embeddings(self, outputs: List[EmbeddingOutput]) -> None:
        required = sum(estimate_matrix_resident_bytes(output.embeddings) for output in outputs)
        assert_within_memory(
            required,
            self.memory_config,
            purpose="Resident embedding artifacts",
        )

    def _admit_cached_embedding_load(self, metadata: dict) -> None:
        required = estimate_metadata_resident_bytes(metadata)
        if required is not None:
            assert_within_memory(
                required,
                self.memory_config,
                purpose="Cached embedding artifact load",
            )

    def _admit_scoring_memory(self, metadata: dict, dataset: Any) -> None:
        required = estimate_metadata_scoring_input_bytes(
            metadata,
            scoring_row_multiplier=self._scoring_row_multiplier(dataset),
        )
        if required is not None:
            assert_within_memory(
                required,
                self.memory_config,
                purpose="Scoring input",
            )

    @staticmethod
    def _scoring_row_multiplier(dataset: Any) -> float:
        if dataset.metadata.get("target_type") != MULTI_LABEL_TARGET:
            return 1.0
        summary = dataset.summary()
        return max(1.0, float(summary.get("mean_label_cardinality", 1.0)))

    def _resolved_scoring_config(
        self,
        dataset: Any,
    ) -> Union[OverlapScoringConfig, ContinuousOverlapScoringConfig]:
        target_type = dataset.metadata.get("target_type", "auto")
        if self._explicit_scoring_config is None:
            return _default_scoring_config_for_dataset(dataset)
        if target_type == REGRESSION_TARGET:
            if not isinstance(self._explicit_scoring_config, ContinuousOverlapScoringConfig):
                raise ValueError("Regression target views require ContinuousOverlapScoringConfig.")
            return self._explicit_scoring_config
        if isinstance(self._explicit_scoring_config, ContinuousOverlapScoringConfig):
            raise ValueError(
                "Classification and multi-label target views require OverlapScoringConfig."
            )
        return self._explicit_scoring_config

    def _supports_transform_many(self, extractor: Any) -> bool:
        if not callable(getattr(extractor, "transform_many", None)):
            return False
        if not callable(getattr(extractor, "output_specs", None)):
            return False
        return len(list(extractor.output_specs())) > 1

    def _supports_named_outputs(self, extractor: Any) -> bool:
        if not callable(getattr(extractor, "transform_many", None)):
            return False
        if not callable(getattr(extractor, "output_specs", None)):
            return False
        return len(list(extractor.output_specs())) >= 1

    def _has_output_view_mappings(self) -> bool:
        return bool(self.label_view_config.output_levels or self.target_view_config.output_views)

    def _output_has_view_mapping(self, output_name: Any) -> bool:
        return bool(
            output_name in self.label_view_config.output_levels
            or output_name in self.target_view_config.output_views
        )

    def _target_view_warnings(self) -> List[str]:
        return []

    def _target_view_datasets(self, warnings: List[str]) -> List[Any]:
        if not self.target_view_config.enabled:
            return [self.dataset]
        if not callable(getattr(self.dataset, "target_view_names", None)):
            raise ValueError(
                "TargetViewConfig requires dataset target view metadata. "
                "Use BenchmarkDataset.with_target_views(...)."
            )
        available_names = self.dataset.target_view_names()
        if not available_names:
            raise ValueError(
                "TargetViewConfig requires dataset target view metadata. "
                "Use BenchmarkDataset.with_target_views(...)."
            )
        requested = list(self.target_view_config.views) or available_names
        datasets = []
        seen = set()
        for name in requested:
            try:
                dataset = self.dataset.target_view(name)
            except ValueError as exc:
                if not self.target_view_config.skip_invalid_views:
                    raise
                warnings.append(f"Skipped target view {name!r}: {exc}")
                continue
            view_key = dataset.active_target_view().get("key")
            if view_key in seen:
                continue
            seen.add(view_key)
            datasets.append(dataset)
        if not datasets:
            raise ValueError("No valid target views were available for benchmarking.")
        return datasets

    def _single_output_spec(self, extractor: Any) -> Optional[Any]:
        if not callable(getattr(extractor, "output_specs", None)):
            return None
        specs = list(extractor.output_specs())
        if len(specs) != 1:
            return None
        return specs[0]

    def _output_specs(self, extractor: Any) -> List[Any]:
        specs = list(extractor.output_specs())
        if not specs:
            raise ValueError(f"Extractor '{extractor.name}' returned no output specs.")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Extractor '{extractor.name}' output names must be unique.")
        return specs

    def _validate_multi_outputs(
        self,
        extractor: Any,
        outputs: Any,
        expected_rows: int,
        context: str,
    ) -> List[EmbeddingOutput]:
        materialized = list(outputs)
        expected_names = [spec.name for spec in self._output_specs(extractor)]
        actual_names = [output.name for output in materialized]
        if actual_names and any(not isinstance(name, str) or not name for name in actual_names):
            raise ValueError(
                f"Extractor '{extractor.name}' returned a blank or non-string output name."
            )
        if len(actual_names) != len(set(actual_names)):
            raise ValueError(
                f"Extractor '{extractor.name}' returned duplicate output names for {context}."
            )
        if len(actual_names) != len(expected_names) or set(actual_names) != set(expected_names):
            raise ValueError(
                f"Extractor '{extractor.name}' returned outputs {sorted(actual_names)} for "
                f"{context}, expected {sorted(expected_names)}."
            )
        validated = []
        for output in materialized:
            embeddings = ensure_numeric_matrix(
                output.embeddings,
                f"Extractor '{extractor.name}' output '{output.name}' {context}",
                allow_sparse=True,
            )
            if embeddings.shape[0] != expected_rows:
                raise ValueError(
                    f"Extractor '{extractor.name}' output '{output.name}' returned "
                    f"{embeddings.shape[0]} embeddings for {expected_rows} labels."
                )
            validated.append(
                EmbeddingOutput(
                    name=output.name,
                    embeddings=embeddings,
                    recipe=dict(output.recipe),
                    metadata=dict(output.metadata),
                )
            )
        return sorted(validated, key=lambda item: expected_names.index(item.name))

    def _estimate_multi_output_memory(
        self,
        outputs: List[EmbeddingOutput],
        n_samples: int,
        scoring_row_multiplier: float = 1.0,
    ) -> Tuple[dict[str, EmbeddingMemoryEstimate], dict[str, Any]]:
        estimates = {
            output.name: estimate_embedding_from_probe(
                output.embeddings,
                n_samples=n_samples,
                batch_size=self.embedding_config.batch_size,
                memory_config=self.memory_config,
                scoring_row_multiplier=scoring_row_multiplier,
            )
            for output in outputs
        }
        resident_bytes = sum(estimate.resident_bytes for estimate in estimates.values())
        aggregate_exceeds_budget = (
            resident_bytes > resolve_memory_budget(self.memory_config).max_memory_bytes
        )
        aggregate = {
            "resident_bytes": resident_bytes,
            "dense_scoring_bytes": max(
                estimate.dense_scoring_bytes for estimate in estimates.values()
            ),
            "scoring_input_bytes": max(
                estimate.scoring_input_bytes for estimate in estimates.values()
            ),
            "batch_embedding_bytes": sum(
                estimate.batch_embedding_bytes for estimate in estimates.values()
            ),
            "strategy": (
                "stream_to_disk"
                if aggregate_exceeds_budget
                or any(estimate.strategy == "stream_to_disk" for estimate in estimates.values())
                else "in_memory"
            ),
        }
        return estimates, aggregate

    def _admit_multi_embedding_plan(self, aggregate: dict[str, Any]) -> None:
        batch_required = (
            int(aggregate["batch_embedding_bytes"])
            + self.memory_config.model_memory_bytes
            + self.memory_config.raw_batch_memory_bytes
        )
        assert_within_memory(
            batch_required,
            self.memory_config,
            purpose="Embedding batch",
        )
        if aggregate["strategy"] == "in_memory" or not self.memory_config.allow_disk_spill:
            assert_within_memory(
                int(aggregate["resident_bytes"]),
                self.memory_config,
                purpose="Resident embedding artifacts",
            )
        assert_within_memory(
            int(aggregate["scoring_input_bytes"]),
            self.memory_config,
            purpose="Scoring input",
        )

    def _cache_embeddings_enabled(self, extractor: Any) -> bool:
        if not self.cache_config.enabled or not bool(getattr(extractor, "cache_embeddings", True)):
            return False
        recipe = extractor.recipe()
        return recipe.get("cache_safe") is not False

    def _qualified_output_recipe(self, recipe: dict, output: EmbeddingOutput) -> dict:
        qualified = dict(recipe)
        qualified.pop("outputs", None)
        qualified.update(output.recipe)
        qualified["output_name"] = output.name
        return qualified


def _default_scoring_config_for_dataset(
    dataset: Any,
) -> Union[OverlapScoringConfig, ContinuousOverlapScoringConfig]:
    target_type = getattr(dataset, "metadata", {}).get("target_type")
    if target_type == REGRESSION_TARGET:
        return ContinuousOverlapScoringConfig()
    return OverlapScoringConfig()


def _classification_scoring_config(
    config: Union[OverlapScoringConfig, ContinuousOverlapScoringConfig],
) -> OverlapScoringConfig:
    if isinstance(config, OverlapScoringConfig):
        return config
    return OverlapScoringConfig(
        k=config.k,
        kmeans_kwargs=config.kmeans_kwargs,
        offline_chunk_size=config.offline_chunk_size,
        normalize_embeddings=config.normalize_embeddings,
        max_dense_bytes=config.max_dense_bytes,
    )


def _scoring_excluded_classes(
    config: Union[OverlapScoringConfig, ContinuousOverlapScoringConfig],
) -> Any:
    if isinstance(config, OverlapScoringConfig):
        return config.exclude_classes
    return None


def _weakest_class(
    per_class_scores: dict,
    excluded_classes: Optional[Any] = None,
    label_catalog: Optional[Any] = None,
) -> Any:
    excluded = _normalized_excluded_classes(excluded_classes)
    numeric_scores = {
        label: float(score)
        for label, score in per_class_scores.items()
        if isinstance(score, (int, float, np.number))
        and not _label_is_excluded(label, excluded, label_catalog)
    }
    if not numeric_scores:
        return None, None
    label, score = min(numeric_scores.items(), key=lambda item: item[1])
    return label_display(label, label_catalog or []), score


def _normalized_excluded_classes(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _label_is_excluded(
    label: Any,
    excluded: List[Any],
    label_catalog: Optional[Any] = None,
) -> bool:
    label_value = label.item() if hasattr(label, "item") else label
    catalog_keys = {
        item.get("key")
        for item in (label_catalog or [])
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }
    label_key = (
        label_value
        if isinstance(label_value, str) and label_value in catalog_keys
        else semantic_label_key(label_value)
    )
    for item in excluded:
        item_value = item.item() if hasattr(item, "item") else item
        item_key = (
            item_value
            if isinstance(item_value, str) and item_value in catalog_keys
            else semantic_label_key(item_value)
        )
        if label_key == item_key:
            return True
    return False


def _label_is_excluded_exact(label: Any, excluded: List[Any]) -> bool:
    label_value = label.item() if hasattr(label, "item") else label
    label_key = semantic_label_key(label_value)
    return any(
        label_key == semantic_label_key(item.item() if hasattr(item, "item") else item)
        for item in excluded
    )


def _variant_extractor_name(name: str, compression_metadata: dict) -> str:
    return compression_variant_name(name, compression_metadata)


def _qualified_result_name(
    name: str,
    target_view: Optional[dict],
    label_view: Optional[dict],
) -> str:
    return f"{name}{target_view_suffix(target_view)}{label_view_suffix(label_view)}"


def _qualified_output_name(parent_name: str, output_name: str) -> str:
    return f"{parent_name}:{output_name}"


def _combine_embedding_batches(
    batches: Iterable[Tuple[np.ndarray, Any]],
    n_samples: int,
) -> Any:
    collected = list(batches)
    if not collected:
        raise ValueError("At least one embedding batch is required.")
    first = collected[0][1]
    expected_contract = _embedding_batch_contract(first)
    written = np.zeros(n_samples, dtype=bool)
    if is_sparse_matrix(first):
        from scipy import sparse

        rows = []
        row_indices = []
        for indices, batch in collected:
            if not is_sparse_matrix(batch):
                raise ValueError("Cannot mix sparse and dense embedding batches.")
            _validate_embedding_batch_contract(batch, expected_contract)
            _check_batch_indices(indices, batch.shape[0], written)
            rows.append(batch)
            row_indices.append(_strict_batch_indices(indices))
        if not bool(np.all(written)):
            missing = np.flatnonzero(~written)
            raise ValueError(
                f"Embedding batches did not cover all samples; missing {missing[:10]}."
            )
        stacked = sparse.vstack(rows, format=expected_contract["storage_format"])
        encountered = np.concatenate(row_indices)
        inverse = np.empty(n_samples, dtype=int)
        inverse[encountered] = np.arange(n_samples, dtype=int)
        return stacked[inverse]

    first_arr = np.asarray(first)
    output = np.empty((n_samples, first_arr.shape[1]), dtype=first_arr.dtype)
    for indices, batch in collected:
        if is_sparse_matrix(batch):
            raise ValueError("Cannot mix sparse and dense embedding batches.")
        arr = np.asarray(batch)
        _validate_embedding_batch_contract(arr, expected_contract)
        _check_batch_indices(indices, arr.shape[0], written)
        output[_strict_batch_indices(indices)] = arr
    if not bool(np.all(written)):
        missing = np.flatnonzero(~written)
        raise ValueError(f"Embedding batches did not cover all samples; missing {missing[:10]}.")
    return output


def _check_batch_indices(indices: np.ndarray, n_rows: int, written: np.ndarray) -> None:
    indices = _strict_batch_indices(indices)
    if len(indices) != n_rows:
        raise ValueError("Batch index count must match embedding row count.")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(written)):
        raise ValueError("Embedding batch indices are outside the dataset row range.")
    if np.any(written[indices]):
        duplicates = indices[written[indices]]
        raise ValueError(f"Duplicate embedding rows for sample indices {duplicates[:10]}.")
    written[indices] = True


def _strict_batch_indices(indices: Any) -> np.ndarray:
    raw = np.asarray(indices)
    if raw.ndim != 1:
        raise ValueError("Embedding batch indices must be one-dimensional.")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("Embedding batch indices must contain non-boolean integers.")
    return raw.astype(int, copy=False)


def _embedding_batch_contract(batch: Any) -> dict[str, Any]:
    sparse = is_sparse_matrix(batch)
    shape = getattr(batch, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("Embedding batches must be two-dimensional matrices.")
    return {
        "embedding_dim": int(shape[1]),
        "dtype": str(batch.dtype),
        "sparse": sparse,
        "storage_format": sparse_storage_format(batch) if sparse else "dense",
    }


def _validate_embedding_batch_contract(batch: Any, expected: dict[str, Any]) -> None:
    actual = _embedding_batch_contract(batch)
    if actual != expected:
        raise ValueError(
            "Embedding batches changed matrix contract; expected feature width "
            f"{expected['embedding_dim']}, dtype {expected['dtype']}, sparse="
            f"{expected['sparse']}, format {expected['storage_format']}, but received "
            f"width {actual['embedding_dim']}, dtype {actual['dtype']}, sparse="
            f"{actual['sparse']}, format {actual['storage_format']}."
        )


def _selected_provenance(
    rows: List[dict[str, Any]],
    metadata: dict[str, Any],
) -> List[dict[str, Any]]:
    expected_rows = int(metadata.get("n_samples", len(rows)))
    indices = metadata.get("sample_indices")
    if indices is None:
        if expected_rows != len(rows):
            raise ValueError(
                "Materialized provenance cannot be aligned with the evaluated embeddings."
            )
        return list(rows)
    resolved = _strict_batch_indices(indices)
    if len(resolved) != expected_rows:
        raise ValueError(
            "Materialized provenance index count does not match evaluated embedding rows."
        )
    if len(resolved) and (resolved.min() < 0 or resolved.max() >= len(rows)):
        raise ValueError("Materialized provenance indices are outside the available rows.")
    if len(set(resolved.tolist())) != len(resolved):
        raise ValueError("Materialized provenance indices must be unique.")
    return [rows[int(index)] for index in resolved]


def _prepend_batch(
    indices: np.ndarray,
    embeddings: Any,
    remaining: Iterator[Tuple[np.ndarray, Any]],
) -> Iterator[Tuple[np.ndarray, Any]]:
    yield indices, embeddings
    yield from remaining
