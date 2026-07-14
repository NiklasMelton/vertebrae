"""Training-free zero-shot semantic-alignment benchmark runner."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vertebrae.cache import create_artifact_store
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe, hash_json
from vertebrae.compression.naming import compression_variant_name
from vertebrae.compression.paired import compress_embedding_pair
from vertebrae.config import (
    CacheConfig,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    OverlapScoringConfig,
    ResourceProfilingConfig,
    ZeroShotConfig,
)
from vertebrae.datasets.zero_shot import ZeroShotDataset
from vertebrae.profiling import (
    ResourceProfile,
    ResourceProfileLike,
    ResourceProfiler,
    resource_profile_columns,
    with_embedding_footprint,
)
from vertebrae.scoring.metrics import MetricResult, OverlapMetric
from vertebrae.scoring.zero_shot import ZeroShotScorer, ZeroShotScoreResult
from vertebrae.utils.embedding_batches import encode_endpoint_batches, endpoint_n_rows
from vertebrae.utils.semantic_labels import (
    label_display,
    portable_json,
    semantic_label_catalog,
    semantic_label_keys,
)


@dataclass
class ZeroShotExtractorResult:
    """One frozen extractor evaluated under a fixed prompt protocol."""

    name: str
    extractor_type: str
    zero_shot: ZeroShotScoreResult
    overlap: MetricResult
    primary_score: float
    compression_metadata: Dict[str, Any]
    runtime: Dict[str, float]
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)
    cache_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recipe: Dict[str, Any] = field(default_factory=dict)
    resource_profiles: Dict[str, ResourceProfileLike] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return portable_json(
            {
                "name": self.name,
                "extractor_type": self.extractor_type,
                "zero_shot": self.zero_shot.to_dict(),
                "overlap": _metric_result_to_dict(self.overlap),
                "primary_score": self.primary_score,
                "compression_metadata": self.compression_metadata,
                "runtime": self.runtime,
                "embedding_metadata": self.embedding_metadata,
                "cache_metadata": self.cache_metadata,
                "warnings": self.warnings,
                "recipe": self.recipe,
                "resource_profiles": self.resource_profiles,
            }
        )


@dataclass
class ZeroShotBenchmarkResult:
    """Aggregated zero-shot comparison results."""

    dataset_summary: Dict[str, Any]
    extractor_results: List[ZeroShotExtractorResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ranked_results(self) -> List[ZeroShotExtractorResult]:
        return sorted(self.extractor_results, key=lambda item: item.primary_score, reverse=True)

    def quality_cohort(self, tolerance: Optional[float] = None) -> List[ZeroShotExtractorResult]:
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
                "primary_metric": item.zero_shot.primary_metric,
                "primary_score": item.primary_score,
                **item.zero_shot.metrics,
                "overlap_score": item.overlap.score,
                "overlap_macro": item.overlap.macro_score,
                "correct_class_margin": item.zero_shot.diagnostics["correct_class_margin"]["mean"],
                "compression_method": item.compression_metadata.get("method", "none"),
                "compressed_dim": item.compression_metadata.get("compressed_dim"),
            }
            for endpoint, profile in item.resource_profiles.items():
                row.update(resource_profile_columns(profile, prefix=f"{endpoint}_"))
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return portable_json(
            {
                "dataset_summary": self.dataset_summary,
                "extractor_results": [item.to_dict() for item in self.extractor_results],
                "metadata": self.metadata,
            }
        )

    def save_json(self, path: str) -> None:
        from vertebrae.reports.json_report import save_json_report

        save_json_report(self, path)

    def save_markdown(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_zero_shot_markdown_report(self), encoding="utf-8")


@dataclass(frozen=True)
class ZeroShotCandidate:
    """One extractor plus its explicit frozen sample and text endpoints."""

    extractor: Any
    sample_branch: str
    text_branch: str

    def __post_init__(self) -> None:
        if not self.sample_branch or not self.text_branch:
            raise ValueError("ZeroShotCandidate branches must be non-empty strings.")


class ZeroShotBenchmark:
    """Compare text-aligned frozen backbones without a learned task head."""

    def __init__(
        self,
        dataset: ZeroShotDataset,
        extractors: Iterable[Any],
        *,
        sample_branch: Optional[str] = None,
        text_branch: Optional[str] = None,
        zero_shot_config: Optional[ZeroShotConfig] = None,
        scoring_config: Optional[OverlapScoringConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        compression_config: Optional[EmbeddingCompressionConfig] = None,
        compression_configs: Optional[Iterable[EmbeddingCompressionConfig]] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        resource_profiling_config: Optional[ResourceProfilingConfig] = None,
    ) -> None:
        if compression_config is not None and compression_configs is not None:
            raise ValueError("Provide compression_config or compression_configs, not both.")
        self.dataset = dataset
        raw_extractors = list(extractors)
        self.candidates = _resolve_candidates(raw_extractors, sample_branch, text_branch)
        self.extractors = [candidate.extractor for candidate in self.candidates]
        self.sample_branch = sample_branch
        self.text_branch = text_branch
        self.config = zero_shot_config or ZeroShotConfig()
        self.scoring_config = scoring_config or OverlapScoringConfig()
        self.cache_config = cache_config or CacheConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.resource_profiling_config = resource_profiling_config or ResourceProfilingConfig()
        self.compression_configs = list(
            compression_configs or [compression_config or EmbeddingCompressionConfig()]
        )

    def run(self) -> ZeroShotBenchmarkResult:
        self.dataset.validated()
        if not self.extractors:
            raise ValueError("ZeroShotBenchmark requires at least one extractor.")
        store = (
            create_artifact_store(self.cache_config.cache_dir, **self.cache_config.storage_options)
            if self.cache_config.enabled
            else None
        )
        prompts, prompt_labels, template_ids = self.dataset.prompt_rows()
        class_labels = [spec.label for spec in self.dataset.class_specs]
        label_catalog = semantic_label_catalog(class_labels)
        overlap_labels = semantic_label_keys(self.dataset.labels.tolist())
        results = []
        for candidate in self.candidates:
            extractor = candidate.extractor
            recipe: Dict[str, Any] = getattr(extractor, "recipe", lambda: {})()
            sample_start = perf_counter()
            sample_profiler = self._endpoint_profiler(
                extractor,
                side="samples",
                branch=candidate.sample_branch,
                modality=self.dataset.modality,
                process_first_inference=True,
            )
            sample_embeddings, sample_cache = self._cached_encode(
                store,
                extractor,
                self.dataset.samples,
                side="samples",
                branch=candidate.sample_branch,
                modality=self.dataset.modality,
                identity=self.dataset.dataset.identity_key(),
                recipe=recipe,
                profiler=sample_profiler,
            )
            sample_profile = (
                sample_profiler.finish(cache_hit=bool(sample_cache["hit"]))
                if self.resource_profiling_config.enabled
                else None
            )
            sample_seconds = perf_counter() - sample_start
            prompt_start = perf_counter()
            prompt_profiler = self._endpoint_profiler(
                extractor,
                side="prompts",
                branch=candidate.text_branch,
                modality="text",
                process_first_inference=not bool(
                    sample_profile is not None and sample_profile.status == "measured"
                ),
            )
            prompt_embeddings, prompt_cache = self._cached_encode(
                store,
                extractor,
                prompts,
                side="prompts",
                branch=candidate.text_branch,
                modality="text",
                identity=self.dataset.protocol_fingerprint(),
                recipe=recipe,
                profiler=prompt_profiler,
            )
            prompt_profile = (
                prompt_profiler.finish(cache_hit=bool(prompt_cache["hit"]))
                if self.resource_profiling_config.enabled
                else None
            )
            prompt_seconds = perf_counter() - prompt_start
            for compression_config in self.compression_configs:
                compression_start = perf_counter()
                compressed_samples, compressed_prompts, compression_metadata = _compress_pair(
                    sample_embeddings, prompt_embeddings, compression_config
                )
                compression_metadata["fit_side"] = "samples"
                compression_seconds = perf_counter() - compression_start
                score_start = perf_counter()
                zero_shot = ZeroShotScorer(self.config).score(
                    compressed_samples,
                    compressed_prompts,
                    self.dataset.labels,
                    class_labels=class_labels,
                    prompt_labels=prompt_labels,
                    template_ids=template_ids,
                    sample_ids=_source_sample_ids(self.dataset),
                )
                overlap = OverlapMetric(config=self.scoring_config).score(
                    compressed_samples,
                    overlap_labels,
                    target_metadata={"target_type": "single_label"},
                )
                zero_shot.metadata["label_catalog"] = label_catalog
                score_seconds = perf_counter() - score_start
                warnings = sorted(
                    set(
                        zero_shot.warnings
                        + overlap.warnings
                        + list(compression_metadata.get("warnings", []))
                    )
                )
                results.append(
                    ZeroShotExtractorResult(
                        name=_variant_name(extractor.name, compression_metadata),
                        extractor_type=getattr(extractor, "extractor_type", "unknown"),
                        zero_shot=zero_shot,
                        overlap=overlap,
                        primary_score=zero_shot.score,
                        compression_metadata=compression_metadata,
                        runtime={
                            "sample_embedding_seconds": sample_seconds,
                            "prompt_embedding_seconds": prompt_seconds,
                            "compression_seconds": compression_seconds,
                            "scoring_seconds": score_seconds,
                        },
                        embedding_metadata={
                            "sample_branch": candidate.sample_branch,
                            "text_branch": candidate.text_branch,
                            "sample_modality": self.dataset.modality,
                            "source_dataset_identity_key": self.dataset.dataset.identity_key(),
                            "protocol_fingerprint": self.dataset.protocol_fingerprint(),
                            "sample_embedding_dim": int(sample_embeddings.shape[1]),
                            "prompt_embedding_dim": int(prompt_embeddings.shape[1]),
                        },
                        cache_metadata={"samples": sample_cache, "prompts": prompt_cache},
                        warnings=warnings + sample_cache["warnings"] + prompt_cache["warnings"],
                        recipe={
                            **recipe,
                            "zero_shot_sample_branch": candidate.sample_branch,
                            "zero_shot_text_branch": candidate.text_branch,
                        },
                        resource_profiles=(
                            _endpoint_profiles(
                                samples=with_embedding_footprint(
                                    sample_profile,
                                    sample_embeddings,
                                    compressed_samples,
                                    store=store,
                                    raw_key=sample_cache.get("key"),
                                    evaluated_key=(
                                        sample_cache.get("key")
                                        if not compression_metadata.get("applied")
                                        else None
                                    ),
                                    persisted_storage=(
                                        self.resource_profiling_config.persisted_storage
                                    ),
                                ),
                                prompts=with_embedding_footprint(
                                    prompt_profile,
                                    prompt_embeddings,
                                    compressed_prompts,
                                    store=store,
                                    raw_key=prompt_cache.get("key"),
                                    evaluated_key=(
                                        prompt_cache.get("key")
                                        if not compression_metadata.get("applied")
                                        else None
                                    ),
                                    persisted_storage=(
                                        self.resource_profiling_config.persisted_storage
                                    ),
                                ),
                            )
                            if self.resource_profiling_config.enabled
                            else {}
                        ),
                    )
                )
        return ZeroShotBenchmarkResult(
            dataset_summary=self.dataset.summary(),
            extractor_results=results,
            metadata={
                "zero_shot_config": asdict(self.config),
                "overlap_scoring_config": asdict(self.scoring_config),
                "compression_configs": [asdict(config) for config in self.compression_configs],
                "cache_config": asdict(self.cache_config),
                "embedding_config": asdict(self.embedding_config),
                "resource_profiling_config": asdict(self.resource_profiling_config),
                "protocol": self.dataset.protocol_recipe(),
                "interpretation": (
                    "Zero-shot scores measure frozen semantic text alignment. Overlap is "
                    "reported as contextual evidence for sample-embedding target structure; "
                    "the metrics are not combined."
                ),
            },
        )

    def _cached_encode(
        self,
        store: Any,
        extractor: Any,
        values: Any,
        *,
        side: str,
        branch: str,
        modality: str,
        identity: str,
        recipe: Dict[str, Any],
        profiler: ResourceProfiler,
    ) -> Tuple[Any, Dict[str, Any]]:
        cache_safe = recipe.get("cache_safe") is not False
        cache_metadata: Dict[str, Any] = {
            "enabled": store is not None,
            "hit": False,
            "warnings": [],
        }
        if not cache_safe:
            cache_metadata["enabled"] = False
            cache_metadata["warnings"].append(
                "Skipped zero-shot embedding cache because this callable extractor has no "
                "portable callable identity or explicit cache_identity."
            )
        key = _embedding_key(identity, recipe, side, branch)
        cache_metadata["key"] = key if cache_safe else None
        if (
            cache_safe
            and store is not None
            and not self.cache_config.force_recompute
            and store.exists(key)
        ):
            cache_metadata["hit"] = True
            return store.get_array(key), cache_metadata
        encoder = getattr(extractor, "encode_retrieval", None)
        if not callable(encoder):
            raise TypeError(
                "ZeroShotBenchmark requires text-aligned extractors implementing "
                "encode_retrieval()."
            )
        embeddings = encode_endpoint_batches(
            values,
            batch_size=(
                self.embedding_config.batch_size
                if self.embedding_config.streaming_enabled
                else endpoint_n_rows(values)
            ),
            encode=lambda batch: encoder(batch, branch=branch, modality=modality),
            owner=f"Zero-shot {side} embeddings",
            profiler=profiler if self.resource_profiling_config.enabled else None,
            call_type=f"encode_zero_shot_{side}",
        )
        if cache_safe and store is not None:
            store.put_array(key, embeddings)
            store.put_json(
                key,
                {
                    "artifact_type": "zero_shot_embedding",
                    "side": side,
                    "branch": branch,
                    "modality": modality,
                    "identity": identity,
                    "recipe_hash": fingerprint_extractor_recipe(recipe),
                    "n_samples": int(embeddings.shape[0]),
                    "embedding_dim": int(embeddings.shape[1]),
                },
            )
        return embeddings, cache_metadata

    def _endpoint_profiler(
        self,
        extractor: Any,
        *,
        side: str,
        branch: str,
        modality: str,
        process_first_inference: bool,
    ) -> ResourceProfiler:
        return ResourceProfiler(
            self.resource_profiling_config,
            extractor,
            streaming=self.embedding_config.streaming_enabled,
            context={
                "endpoint": side,
                "modality": modality,
                "branch": branch,
                "process_first_inference": process_first_inference,
                "configured_batch_size": self.embedding_config.batch_size,
                "measurement_scope": "local_endpoint",
            },
        )


def _embedding_key(identity: str, recipe: Dict[str, Any], side: str, branch: str) -> str:
    recipe_hash = fingerprint_extractor_recipe(recipe)
    branch_hash = hash_json(branch)[:16]
    return f"zero_shot/embeddings/{identity}/{recipe_hash}/{side}/{branch_hash}"


def _compress_pair(
    samples: Any,
    prompts: Any,
    config: EmbeddingCompressionConfig,
) -> tuple[Any, Any, Dict[str, Any]]:
    return compress_embedding_pair(samples, prompts, config)


def _variant_name(name: str, compression: Dict[str, Any]) -> str:
    return compression_variant_name(name, compression)


def _endpoint_profiles(**profiles: Optional[ResourceProfile]) -> Dict[str, ResourceProfileLike]:
    return {name: profile for name, profile in profiles.items() if profile is not None}


def _resolve_candidates(
    extractors: List[Any], sample_branch: Optional[str], text_branch: Optional[str]
) -> List[ZeroShotCandidate]:
    if not extractors:
        return []
    candidates: List[ZeroShotCandidate] = []
    for item in extractors:
        if isinstance(item, ZeroShotCandidate):
            candidates.append(item)
        else:
            if not sample_branch or not text_branch:
                raise ValueError(
                    "Legacy raw extractors require both sample_branch and text_branch; "
                    "use ZeroShotCandidate for per-extractor branches."
                )
            candidates.append(ZeroShotCandidate(item, sample_branch, text_branch))
    return candidates


def _source_sample_ids(dataset: ZeroShotDataset) -> List[Any]:
    values = dataset.dataset.metadata.get("sample_indices")
    if values is None:
        return list(range(len(dataset.labels)))
    if len(values) != len(dataset.labels):
        raise ValueError("Dataset sample_indices metadata must align with zero-shot samples.")
    return list(values)


def render_zero_shot_markdown_report(result: ZeroShotBenchmarkResult) -> str:
    """Render a concise report without conflating alignment and overlap scores."""

    data = result.to_dict()
    lines = ["# vertebrae zero-shot report", "", "## Protocol", ""]
    summary = data["dataset_summary"]
    lines.extend(
        [
            f"- Sample modality: {summary['sample_modality']}",
            f"- Samples: {summary['n_samples']}",
            f"- Classes: {summary['n_classes']}",
            f"- Fixed prompts: {summary['n_prompts']}",
            "- Interpretation: zero-shot measures semantic text alignment; overlap is "
            "contextual evidence for frozen sample-embedding structure.",
            "",
            "## Ranking",
            "",
            "| rank | extractor | primary metric | primary score | accuracy | macro F1 | "
            "balanced accuracy | overlap macro | compression |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, item in enumerate(result.ranked_results(), start=1):
        metrics = item.zero_shot.metrics
        lines.append(
            f"| {rank} | {item.name} | {item.zero_shot.primary_metric} | "
            f"{item.primary_score:.4f} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {metrics['balanced_accuracy']:.4f} | "
            f"{item.overlap.macro_score:.4f} | "
            f"{item.compression_metadata.get('method', 'none')} |"
        )
    cohort = [item for item in result.quality_cohort() if item.resource_profiles]
    if cohort:
        from vertebrae.retrieval import _endpoint_resource_markdown

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
            lines.extend([f"### {item.name}", ""])
            for endpoint in ("samples", "prompts"):
                profile = item.resource_profiles.get(endpoint)
                if profile is not None:
                    lines.extend(_endpoint_resource_markdown(endpoint, profile))
    lines.extend(["", "## Per-extractor details", ""])
    for item in result.ranked_results():
        lines.extend(
            [
                f"### {item.name}",
                "",
                "- Correct-class margin: "
                f"{item.zero_shot.diagnostics['correct_class_margin']['mean']:.4f}",
                f"- Top-score ties: {item.zero_shot.diagnostics['n_top_score_ties']}",
                f"- Overlap score: {item.overlap.score:.4f}",
                "- Per-class metrics:",
            ]
        )
        for label, metrics in item.zero_shot.per_class.items():
            catalog = item.zero_shot.metadata.get("label_catalog", [])
            lines.append(
                f"  - {label_display(label, catalog)}: precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
                f"support={int(metrics['support'])}"
            )
        for warning in item.warnings:
            lines.append(f"- Warning: {warning}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _metric_result_to_dict(result: MetricResult) -> Dict[str, Any]:
    return portable_json(
        {
            "name": result.name,
            "score": result.score,
            "higher_is_better": result.higher_is_better,
            "kind": result.kind,
            "diagnostics": result.diagnostics,
            "warnings": result.warnings,
            "metadata": result.metadata,
        }
    )
