"""Training-free query--gallery benchmark runner."""

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vertebrae.compression import compression_variant_name
from vertebrae.compression.paired import compress_embedding_pair
from vertebrae.config import (
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    MemoryConfig,
    ResourceProfilingConfig,
    RetrievalConfig,
)
from vertebrae.datasets.retrieval import RetrievalDataset
from vertebrae.profiling import (
    ResourceProfile,
    ResourceProfileLike,
    ResourceProfiler,
    resource_profile_columns,
    with_embedding_footprint,
)
from vertebrae.reports._markdown import markdown_table_row, markdown_text
from vertebrae.scoring.retrieval import RetrievalScorer, RetrievalScoreResult
from vertebrae.utils.embedding_batches import encode_endpoint_batches, endpoint_n_rows
from vertebrae.utils.serialization import make_json_safe


@dataclass
class RetrievalExtractorResult:
    name: str
    extractor_type: str
    forward: RetrievalScoreResult
    reverse: Optional[RetrievalScoreResult]
    primary_score: float
    compression_metadata: Dict[str, Any]
    runtime: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    recipe: Dict[str, Any] = field(default_factory=dict)
    resource_profiles: Dict[str, ResourceProfileLike] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return make_json_safe(self)


@dataclass
class RetrievalBenchmarkResult:
    dataset_summary: Dict[str, Any]
    extractor_results: List[RetrievalExtractorResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ranked_results(self) -> List[RetrievalExtractorResult]:
        return sorted(self.extractor_results, key=lambda item: item.primary_score, reverse=True)

    def quality_cohort(self, tolerance: Optional[float] = None) -> List[RetrievalExtractorResult]:
        ranked = self.ranked_results()
        if not ranked:
            return []
        if tolerance is None:
            tolerance = float(
                self.metadata.get("resource_profiling_config", {}).get("quality_tolerance", 0.01)
            )
        if tolerance < 0:
            raise ValueError("quality cohort tolerance must be >= 0.")
        return [
            item for item in ranked if ranked[0].primary_score - item.primary_score <= tolerance
        ]

    def to_dataframe(self) -> Any:
        import pandas as pd

        rows = []
        for rank, item in enumerate(self.ranked_results(), start=1):
            row = {
                "rank": rank,
                "extractor": item.name,
                "extractor_type": item.extractor_type,
                "primary_metric": item.forward.primary_metric,
                "primary_score": item.primary_score,
                **item.forward.metrics,
                "compression_method": item.compression_metadata.get("method", "none"),
            }
            if item.reverse:
                row.update({f"reverse_{key}": value for key, value in item.reverse.metrics.items()})
            for endpoint, profile in item.resource_profiles.items():
                row.update(resource_profile_columns(profile, prefix=f"{endpoint}_"))
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return make_json_safe(self)

    def save_json(self, path: str) -> None:
        from vertebrae.reports.json_report import save_json_report

        save_json_report(self, path)

    def save_markdown(self, path: str) -> None:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_retrieval_markdown_report(self), encoding="utf-8")


class RetrievalBenchmark:
    """Compare frozen extractors under an explicit exact retrieval protocol."""

    def __init__(
        self,
        dataset: RetrievalDataset,
        extractors: Iterable[Any],
        retrieval_config: Optional[RetrievalConfig] = None,
        compression_config: Optional[EmbeddingCompressionConfig] = None,
        compression_configs: Optional[Iterable[EmbeddingCompressionConfig]] = None,
        query_branch: Optional[str] = None,
        gallery_branch: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        resource_profiling_config: Optional[ResourceProfilingConfig] = None,
    ) -> None:
        if compression_config is not None and compression_configs is not None:
            raise ValueError("Provide compression_config or compression_configs, not both.")
        self.dataset = dataset
        self.extractors = list(extractors)
        self.config = retrieval_config or RetrievalConfig()
        self.compression_configs = list(
            compression_configs or [compression_config or EmbeddingCompressionConfig()]
        )
        self.query_branch = query_branch
        self.gallery_branch = gallery_branch
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.memory_config = memory_config or MemoryConfig()
        self.resource_profiling_config = resource_profiling_config or ResourceProfilingConfig()
        if (query_branch is None) != (gallery_branch is None):
            raise ValueError("query_branch and gallery_branch must be provided together.")

    def run(self) -> RetrievalBenchmarkResult:
        self.dataset.validated()
        if not self.extractors:
            raise ValueError("RetrievalBenchmark requires at least one extractor.")
        query_ids = list(self.dataset.query_id_values())
        gallery_ids = list(self.dataset.gallery_id_values())
        relevance = self.dataset.normalized_relevance()
        exclusions = self.dataset.normalized_exclusions()
        query_modality, gallery_modality = self.dataset.protocol_modalities()
        results: List[RetrievalExtractorResult] = []
        for extractor in self.extractors:
            extract_start = perf_counter()
            if self.query_branch is None:
                self._fit_standard_extractor(extractor)
            query_embeddings, query_profile = self._encode_profiled(
                extractor,
                self.dataset.query_values(),
                self.query_branch,
                query_modality,
                side="query",
                process_first_inference=True,
            )
            gallery_embeddings, gallery_profile = self._encode_profiled(
                extractor,
                self.dataset.gallery_values(),
                self.gallery_branch,
                gallery_modality,
                side="gallery",
                process_first_inference=False,
            )
            extract_seconds = perf_counter() - extract_start
            for compression_config in self.compression_configs:
                query_compressed, gallery_compressed, compression_metadata = _compress_pair(
                    query_embeddings, gallery_embeddings, compression_config
                )
                score_start = perf_counter()
                scorer = RetrievalScorer(self.config)
                forward = scorer.score(
                    query_compressed,
                    gallery_compressed,
                    relevance,
                    query_ids=query_ids,
                    gallery_ids=gallery_ids,
                    exclusions=exclusions,
                )
                reverse = None
                if self.config.bidirectional:
                    reverse_relevance, reverse_exclusions = _transpose_relations(
                        relevance,
                        exclusions,
                        len(gallery_ids),
                    )
                    if any(
                        not any(
                            (gallery_index, query_index) not in reverse_exclusions
                            for query_index in values
                        )
                        for gallery_index, values in reverse_relevance.items()
                    ):
                        raise ValueError(
                            "bidirectional retrieval requires every gallery item to have an "
                            "eligible reverse relevance relation."
                        )
                    reverse = scorer.score(
                        gallery_compressed,
                        query_compressed,
                        reverse_relevance,
                        query_ids=gallery_ids,
                        gallery_ids=query_ids,
                        exclusions=reverse_exclusions,
                    )
                primary = (
                    forward.score
                    if reverse is None
                    else float((forward.score + reverse.score) / 2.0)
                )
                results.append(
                    RetrievalExtractorResult(
                        name=_variant_name(extractor.name, compression_metadata),
                        extractor_type=getattr(extractor, "extractor_type", "unknown"),
                        forward=forward,
                        reverse=reverse,
                        primary_score=primary,
                        compression_metadata=compression_metadata,
                        runtime={
                            "extraction_seconds": extract_seconds,
                            "scoring_seconds": perf_counter() - score_start,
                        },
                        warnings=sorted(
                            set(
                                forward.warnings
                                + (reverse.warnings if reverse else [])
                                + list(compression_metadata.get("warnings", []))
                            )
                        ),
                        recipe=getattr(extractor, "recipe", lambda: {})(),
                        resource_profiles=_endpoint_profiles(
                            query=with_embedding_footprint(
                                query_profile,
                                query_embeddings,
                                query_compressed,
                                persisted_storage=self.resource_profiling_config.persisted_storage,
                            ),
                            gallery=with_embedding_footprint(
                                gallery_profile,
                                gallery_embeddings,
                                gallery_compressed,
                                persisted_storage=self.resource_profiling_config.persisted_storage,
                            ),
                        )
                        if self.resource_profiling_config.enabled
                        else {},
                    )
                )
        return RetrievalBenchmarkResult(
            dataset_summary=self.dataset.summary(),
            extractor_results=results,
            metadata={
                "retrieval_config": asdict(self.config),
                "compression_configs": [asdict(config) for config in self.compression_configs],
                "embedding_config": asdict(self.embedding_config),
                "memory_config": asdict(self.memory_config),
                "resource_profiling_config": asdict(self.resource_profiling_config),
            },
        )

    def _encode_profiled(
        self,
        extractor: Any,
        values: Any,
        branch: Optional[str],
        modality: str,
        *,
        side: str,
        process_first_inference: bool,
    ) -> Tuple[Any, Optional[ResourceProfile]]:
        streaming = bool(
            self.embedding_config.streaming_enabled
            and (branch is not None or getattr(extractor, "streaming_safe", False))
        )
        profiler = ResourceProfiler(
            self.resource_profiling_config,
            extractor,
            streaming=streaming,
            context={
                "endpoint": side,
                "modality": modality,
                "branch": branch,
                "process_first_inference": process_first_inference,
                "configured_batch_size": self.embedding_config.batch_size,
                "measurement_scope": "local_endpoint",
            },
        )
        embeddings = encode_endpoint_batches(
            values,
            batch_size=(self.embedding_config.batch_size if streaming else endpoint_n_rows(values)),
            encode=lambda batch: self._encode(extractor, batch, branch, modality),
            owner=f"Retriever '{extractor.name}' {side} embeddings",
            profiler=profiler if self.resource_profiling_config.enabled else None,
            call_type=f"encode_retrieval_{side}",
            memory_config=self.memory_config,
        )
        profile = profiler.finish() if self.resource_profiling_config.enabled else None
        return embeddings, profile

    def _encode(self, extractor: Any, values: Any, branch: Optional[str], modality: str) -> Any:
        if branch is not None:
            encode = getattr(extractor, "encode_retrieval", None)
            if not callable(encode):
                raise TypeError(
                    "A query_branch/gallery_branch requires an extractor implementing "
                    "encode_retrieval()."
                )
            return encode(values, branch=branch, modality=modality)
        if getattr(extractor, "modality", None) not in {modality, "multimodal", "embeddings"}:
            raise ValueError(
                f"Extractor {extractor.name!r} does not support retrieval modality {modality!r}."
            )
        return extractor.transform(values)

    def _fit_standard_extractor(self, extractor: Any) -> None:
        try:
            extractor.fit(self.dataset.gallery_values(), y=None)
        except TypeError as exc:
            raise TypeError(
                "Standard retrieval extractors must support unsupervised fit(X, y=None) or "
                "be pre-fitted. Use a retrieval branch adapter for asymmetric models."
            ) from exc


def _compress_pair(
    query: Any, gallery: Any, config: EmbeddingCompressionConfig
) -> tuple[Any, Any, Dict[str, Any]]:
    gallery_result, query_result, metadata = compress_embedding_pair(
        gallery,
        query,
        config,
        fit_name="gallery embeddings",
        paired_name="query embeddings",
    )
    return (
        query_result,
        gallery_result,
        {
            **metadata,
            "fit_side": "gallery",
        },
    )


def _transpose_relations(
    relevance: Dict[int, Dict[int, float]], exclusions: set[tuple[int, int]], n_gallery: int
) -> tuple[Dict[int, Dict[int, float]], set[tuple[int, int]]]:
    transposed: Dict[int, Dict[int, float]] = {}
    for query, values in relevance.items():
        for gallery, grade in values.items():
            transposed.setdefault(gallery, {})[query] = grade
    for index in range(n_gallery):
        transposed.setdefault(index, {})
    return transposed, {(gallery, query) for query, gallery in exclusions}


def _variant_name(name: str, compression: Dict[str, Any]) -> str:
    return compression_variant_name(name, compression)


def _endpoint_profiles(**profiles: Optional[ResourceProfile]) -> Dict[str, ResourceProfileLike]:
    return {name: profile for name, profile in profiles.items() if profile is not None}


def render_retrieval_markdown_report(result: RetrievalBenchmarkResult) -> str:
    data = result.to_dict()
    lines = ["# vertebrae retrieval report", "", "## Dataset summary", ""]
    for key, value in data["dataset_summary"].items():
        lines.append(f"- {markdown_text(key)}: {markdown_text(value)}")
    primary_metric = (
        result.ranked_results()[0].forward.primary_metric if result.extractor_results else "score"
    )
    lines.extend(
        [
            "",
            "## Ranking",
            "",
            markdown_table_row(
                ["rank", "extractor", "primary score", primary_metric, "mrr", "map"]
            ),
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, item in enumerate(result.ranked_results(), start=1):
        metrics = item.forward.metrics
        primary = metrics.get(primary_metric, float("nan"))
        mrr = metrics.get("mrr", float("nan"))
        mean_average_precision = metrics.get("map", float("nan"))
        lines.append(
            markdown_table_row(
                [
                    rank,
                    item.name,
                    f"{item.primary_score:.4f}",
                    f"{primary:.4f}",
                    f"{mrr:.4f}",
                    f"{mean_average_precision:.4f}",
                ]
            )
        )
        lines.append("")
        lines.append(f"Forward metrics for {markdown_text(item.name)}: {markdown_text(metrics)}")
        if item.reverse:
            lines.append(f"Reverse metrics: {markdown_text(item.reverse.metrics)}")
        for warning in item.warnings:
            lines.append(f"Warning: {markdown_text(warning)}")
    cohort = [item for item in result.quality_cohort() if item.resource_profiles]
    if cohort:
        lines.extend(
            [
                "",
                "## Resources for quality-similar candidates",
                "",
                "These candidates fall within the configured quality tolerance. Distributed "
                "throughput is aggregate compute throughput, not cluster wall-clock throughput.",
                "",
            ]
        )
        for item in cohort:
            lines.extend([f"### {markdown_text(item.name)}", ""])
            for endpoint in ("query", "gallery"):
                profile = item.resource_profiles.get(endpoint)
                if profile is not None:
                    lines.extend(_endpoint_resource_markdown(endpoint, profile))
    return "\n".join(lines) + "\n"


def _endpoint_resource_markdown(endpoint: str, profile: ResourceProfileLike) -> List[str]:
    columns = resource_profile_columns(profile)
    safe_endpoint = markdown_text(endpoint)
    lines = [
        f"- {safe_endpoint} scope: {markdown_text(columns.get('resource_profile_scope'))}",
        f"- {safe_endpoint} logical embedding bytes: "
        f"{markdown_text(columns.get('embedding_logical_bytes'))}",
        f"- {safe_endpoint} persisted embedding bytes: "
        f"{markdown_text(columns.get('embedding_persisted_bytes'))}",
    ]
    if columns.get("resource_profile_scope") == "distributed_shards":
        lines.extend(
            [
                f"- {safe_endpoint} worker-first median seconds: "
                f"{markdown_text(columns.get('worker_first_call_median_seconds'))}",
                f"- {safe_endpoint} worker-first p95 seconds: "
                f"{markdown_text(columns.get('worker_first_call_p95_seconds'))}",
                f"- {safe_endpoint} aggregate compute throughput (samples/s): "
                f"{markdown_text(columns.get('aggregate_compute_throughput_samples_per_second'))}",
                f"- {safe_endpoint} maximum worker RSS bytes: "
                f"{markdown_text(columns.get('max_worker_peak_rss_bytes'))}",
                f"- {safe_endpoint} maximum worker device bytes: "
                f"{markdown_text(columns.get('max_worker_peak_device_allocated_bytes'))}",
                f"- {safe_endpoint} model in-memory bytes: "
                f"{markdown_text(profile.model.in_memory_bytes if profile.model else None)}",
                f"- {safe_endpoint} checkpoint bytes: "
                f"{markdown_text(profile.model.checkpoint_bytes if profile.model else None)}",
            ]
        )
    else:
        modality_branch = markdown_text(f"{columns.get('modality')}/{columns.get('branch')}")
        lines.extend(
            [
                f"- {safe_endpoint} first call seconds: "
                f"{markdown_text(columns.get('first_call_seconds'))}",
                f"- {safe_endpoint} warm median seconds: "
                f"{markdown_text(columns.get('warm_median_seconds'))}",
                f"- {safe_endpoint} throughput (samples/s): "
                f"{markdown_text(columns.get('throughput_samples_per_second'))}",
                f"- {safe_endpoint} batches: {markdown_text(columns.get('batch_sizes'))}",
                f"- {safe_endpoint} cache: {markdown_text(columns.get('cache_status'))}",
                f"- {safe_endpoint} synchronization: "
                f"{markdown_text(columns.get('synchronization_status'))}",
                f"- {safe_endpoint} modality/branch: {modality_branch}",
                f"- {safe_endpoint} measurement scope: "
                f"{markdown_text(columns.get('measurement_scope'))}",
                f"- {safe_endpoint} contained process-first measured inference: "
                f"{markdown_text(columns.get('process_first_inference'))}",
            ]
        )
    return lines + [""]
