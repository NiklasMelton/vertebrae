"""Distributed embedding primitives."""

from dataclasses import asdict, replace
from functools import partial
from itertools import chain
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
from uuid import uuid4

import numpy as np

from vertebrae._version import __version__
from vertebrae.cache import (
    ArtifactStore,
    ArtifactStoreConfig,
    create_artifact_store_from_config,
)
from vertebrae.cache.fingerprint import (
    fingerprint_extractor_recipe,
    hash_json_exact,
)
from vertebrae.cache.keys import named_output_artifact_key, named_output_artifact_keys
from vertebrae.compression import (
    compress_embedding_artifact_key,
    compress_embeddings,
    compression_variant_name,
)
from vertebrae.compression.paired import compress_embedding_pair
from vertebrae.config import (
    MemoryConfig,
    OverlapScoringConfig,
    ResourceProfilingConfig,
    RetrievalConfig,
    SegmentationConfig,
    overlap_scoring_config_recipe,
)
from vertebrae.execution.jobs import (
    CompressionJob,
    EmbeddingMergeJob,
    EmbeddingShardJob,
    RetrievalCompressionJob,
    RetrievalEmbeddingShardJob,
    RetrievalScoringJob,
    ScoringJob,
    SeparatixJob,
    ShardSpec,
    StabilityJob,
)
from vertebrae.extractors._identity import (
    derived_cache_reuse_decision,
    extractor_cache_reuse_decision,
)
from vertebrae.extractors.base import EmbeddingOutput
from vertebrae.profiling import (
    ResourceProfiler,
    aggregate_distributed_resource_profiles,
    distributed_resource_profile_from_dict,
    resource_profile_from_dict,
    with_distributed_embedding_footprint,
    with_embedding_footprint,
)
from vertebrae.scoring.separatix import SeparatixResult, SeparatixScorer
from vertebrae.utils.embedding_batches import encode_endpoint_batches, take_endpoint_rows
from vertebrae.utils.labels import (
    MULTI_LABEL_TARGET,
    REGRESSION_TARGET,
    canonical_metric_targets,
    label_view_suffix,
    labels_to_jsonable,
    metric_labels,
    target_summary,
    target_view_suffix,
)
from vertebrae.utils.memory import (
    IncrementalMatrixReferenceStager,
    IncrementalMatrixStager,
)
from vertebrae.utils.semantic_labels import (
    SemanticLabelKey,
    canonical_semantic_array,
    label_display,
    semantic_label_key,
)
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import (
    ensure_numeric_matrix,
    is_sparse_matrix,
    sparse_storage_format,
)


def embedding_artifact_key(dataset: Any, extractor: Any) -> str:
    """Build the canonical embedding artifact key.

    Args:
        dataset: Dataset object with an `identity_key()` method.
        extractor: Feature extractor with a serializable `recipe()`.

    Returns:
        Artifact key for the complete embedding matrix.
    """

    dataset_key = dataset.identity_key()
    extractor_key = fingerprint_extractor_recipe(extractor.recipe())
    return f"embeddings/{dataset_key}/{extractor_key}"


def retrieval_embedding_artifact_key(
    dataset: Any, extractor: Any, side: str, branch: Optional[str] = None
) -> str:
    """Build the canonical endpoint embedding key for a retrieval dataset."""
    if side not in {"query", "gallery"}:
        raise ValueError("side must be 'query' or 'gallery'.")
    recipe = extractor.recipe()
    recipe_hash = fingerprint_extractor_recipe(recipe)
    branch_key = "default" if branch is None else hash_json_exact({"branch": branch})
    return f"retrieval/embeddings/{dataset.identity_key()}/{recipe_hash}/{side}/{branch_key}"


def _execution_artifact_identity(
    canonical_key: str,
    recipe: dict[str, Any],
    *,
    output_key: Optional[str] = None,
    run_prefix: Optional[str] = None,
) -> tuple[str, bool, str]:
    """Resolve reusable versus run-scoped artifact identity for an execution job."""

    cache_eligible, cache_status = extractor_cache_reuse_decision(recipe)
    if output_key is not None:
        if not cache_eligible and not output_key.startswith("runs/"):
            raise ValueError(
                "Cache-ineligible extractors require a run-scoped output key under 'runs/'. "
                "Enable embedding caching and provide a safe cache_identity to opt into "
                "canonical reuse."
            )
        return (
            output_key,
            cache_eligible,
            cache_status,
        )
    if cache_eligible:
        return canonical_key, True, "miss"
    resolved_prefix = run_prefix or f"runs/{uuid4().hex}"
    if not resolved_prefix.startswith("runs/"):
        raise ValueError("run_prefix must be scoped beneath 'runs/'.")
    return (
        f"{resolved_prefix}/{canonical_key}",
        False,
        cache_status,
    )


def _embedding_metadata_from_array_manifest(array_manifest: Any) -> dict[str, Any]:
    """Build the minimal resident-memory contract used during composite commit."""

    sparse = array_manifest.storage_format == "npz"
    return {
        "shape": list(array_manifest.shape),
        "dtype": array_manifest.dtype,
        "sparse": sparse,
        "nnz": array_manifest.nnz if sparse else None,
    }


def retrieval_embedding_shard_key(base_key: str, shard: ShardSpec) -> str:
    """Build the deterministic key for one retrieval endpoint shard."""
    return embedding_shard_key(base_key, shard)


def embedding_shard_key(base_key: str, shard: ShardSpec) -> str:
    """Build an artifact key for one embedding shard.

    Args:
        base_key: Complete embedding artifact key.
        shard: Deterministic shard assignment.

    Returns:
        Artifact key for the shard.
    """

    return f"{base_key}/shards/{shard.shard_index:05d}-of-{shard.total_shards:05d}"


def embedding_output_key(base_key: str, output_name: str) -> str:
    """Build an artifact key for one named embedding output."""

    return named_output_artifact_key(base_key, output_name)


def embedding_output_shard_key(shard_key: str, output_name: str) -> str:
    """Build an artifact key for one named embedding shard output."""

    return named_output_artifact_key(shard_key, output_name)


def labels_artifact_key(dataset: Any) -> str:
    """Build the canonical label artifact key.

    Args:
        dataset: Dataset object with an `identity_key()` method.

    Returns:
        Artifact key for labels.
    """

    digest = _labels_content_digest(dataset)
    return f"labels/{dataset.identity_key()}/v2-{digest}"


def groups_artifact_key(dataset: Any) -> str:
    """Build the canonical independence-group artifact key."""

    digest = _groups_content_digest(dataset)
    return f"groups/{dataset.identity_key()}/v2-{digest}"


def scoring_artifact_key(
    embedding_key: str,
    seed: Any = None,
    *,
    labels_key: str,
    groups_key: Optional[str],
    scoring_config: Any,
    metrics: Any,
    metric: Any = None,
    primary_metric: str,
    run_prefix: Optional[str] = None,
) -> str:
    """Build a scoring artifact key.

    Args:
        embedding_key: Complete embedding artifact key.
        seed: Optional scoring seed.

    Returns:
        Artifact key for scoring output.
    """

    _require_artifact_identity(embedding_key, "embedding_key")
    _require_artifact_identity(labels_key, "labels_key")
    _validate_optional_artifact_identity(groups_key, "groups_key")
    if scoring_config is None:
        raise ValueError("scoring_config is required for a scoring artifact identity.")
    if metrics is None:
        raise ValueError("metrics is required; pass an empty sequence for overlap-only scoring.")
    if not isinstance(primary_metric, str) or not primary_metric:
        raise ValueError("primary_metric must be a non-empty string.")
    metric_recipes = _configured_metric_recipes(scoring_config, metrics, metric)
    identity = {
        "identity_schema": 2,
        "embedding_key": embedding_key,
        "labels_key": labels_key,
        "groups_key": groups_key,
        "scoring_config": overlap_scoring_config_recipe(scoring_config),
        "metric_recipes": metric_recipes,
        "primary_metric": primary_metric,
        "seed": seed,
    }
    digest = hash_json_exact(identity)
    if _metric_recipes_cache_safe(metric_recipes):
        return f"{embedding_key}/scores/v2-{digest}"
    resolved_prefix = run_prefix or f"runs/{uuid4().hex}"
    if not resolved_prefix.startswith("runs/"):
        raise ValueError("run_prefix must be scoped beneath 'runs/'.")
    return f"{resolved_prefix}/scores/v2-{digest}"


def stability_artifact_key(
    embedding_key: str,
    *,
    labels_key: str,
    scoring_config: Any,
    stability_config: Any,
) -> str:
    """Build a configuration-specific stability artifact key."""

    _require_artifact_identity(embedding_key, "embedding_key")
    _require_artifact_identity(labels_key, "labels_key")
    if scoring_config is None:
        raise ValueError("scoring_config is required for a stability artifact identity.")
    if stability_config is None:
        raise ValueError("stability_config is required for a stability artifact identity.")
    identity = {
        "identity_schema": 2,
        "embedding_key": embedding_key,
        "labels_key": labels_key,
        "scoring_config": overlap_scoring_config_recipe(scoring_config),
        "stability_config": make_json_safe(stability_config),
    }
    return f"{embedding_key}/diagnostics/stability-v2-{hash_json_exact(identity)}"


def retrieval_scoring_artifact_key(
    query_embedding_key: str,
    gallery_embedding_key: str,
    *,
    relevance_key: str,
    exclusions_key: Optional[str],
    retrieval_config: Any,
) -> str:
    """Build a stable key for a query--gallery retrieval score artifact."""

    _require_artifact_identity(query_embedding_key, "query_embedding_key")
    _require_artifact_identity(gallery_embedding_key, "gallery_embedding_key")
    _require_artifact_identity(relevance_key, "relevance_key")
    _validate_optional_artifact_identity(exclusions_key, "exclusions_key")
    if retrieval_config is None:
        raise ValueError("retrieval_config is required for a retrieval score identity.")

    identity = {
        "identity_schema": 2,
        "query": query_embedding_key,
        "gallery": gallery_embedding_key,
        "relevance": relevance_key,
        "exclusions": exclusions_key,
        "retrieval_config": retrieval_config,
    }
    return f"retrieval/scores/v2-{hash_json_exact(identity)}"


def retrieval_compression_artifact_key(
    query_embedding_key: str, gallery_embedding_key: str, config: Any
) -> str:
    """Build a paired compression artifact prefix."""
    from vertebrae.compression import compression_recipe_hash

    identity = hash_json_exact(
        {"identity_schema": 2, "query": query_embedding_key, "gallery": gallery_embedding_key}
    )
    return f"retrieval/compressions/{identity}/{compression_recipe_hash(config)}"


def separatix_artifact_key(
    embedding_key: str,
    *,
    labels_key: str,
    groups_key: Optional[str],
    score_key: str,
    separatix_config: Any,
) -> str:
    """Build a Separatix diagnostic artifact key."""

    _require_artifact_identity(embedding_key, "embedding_key")
    _require_artifact_identity(labels_key, "labels_key")
    _validate_optional_artifact_identity(groups_key, "groups_key")
    _require_artifact_identity(score_key, "score_key")
    if separatix_config is None:
        raise ValueError("separatix_config is required for a Separatix artifact identity.")
    identity = {
        "identity_schema": 2,
        "embedding_key": embedding_key,
        "labels_key": labels_key,
        "groups_key": groups_key,
        "score_key": score_key,
        "separatix_config": make_json_safe(separatix_config),
    }
    return f"{embedding_key}/diagnostics/separatix-v2-{hash_json_exact(identity)}"


def _require_artifact_identity(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty artifact key.")


def _validate_optional_artifact_identity(value: Any, name: str) -> None:
    if value is not None:
        _require_artifact_identity(value, name)


def _labels_content_digest(dataset: Any) -> str:
    metadata = dataset.metadata
    values = np.asarray(dataset.y)
    return hash_json_exact(
        {
            "identity_schema": 2,
            "values": labels_to_jsonable(
                dataset.y,
                label_names=metadata.get("label_names"),
                target_type=metadata.get("target_type", "auto"),
                target_names=metadata.get("target_names"),
            ),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "target_type": metadata.get("target_type", "auto"),
            "label_names": metadata.get("label_names"),
            "target_names": metadata.get("target_names"),
            "target_view": dataset.active_target_view(),
            "label_view": dataset.active_label_view(),
        }
    )


def _groups_content_digest(dataset: Any) -> str:
    groups = dataset.groups() if callable(getattr(dataset, "groups", None)) else None
    if groups is None:
        raise ValueError("Cannot build a group artifact key for a dataset without groups.")
    values = np.asarray(groups)
    return hash_json_exact(
        {
            "identity_schema": 2,
            "values": values,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "group_name": dataset.metadata.get("group_name", "group"),
        }
    )


def _configured_metric_recipes(
    scoring_config: Any,
    metrics: Any,
    metric: Any = None,
) -> list[dict[str, Any]]:
    """Return the exact recipes for the metrics a scoring job will execute."""

    return [
        make_json_safe(metric.recipe())
        for metric in _configured_metrics(
            scoring_config,
            metrics,
            metric,
        )
    ]


def _configured_metrics(
    scoring_config: Any,
    metrics: Any,
    metric: Any = None,
) -> list[Any]:
    """Resolve metric adapters and bind the scoring job's overlap configuration."""

    from vertebrae.scoring.metrics import OverlapMetric, as_embedding_metric

    configured = [as_embedding_metric(metric) for metric in (metrics or [])]
    if metric is not None:
        configured.append(as_embedding_metric(metric))
    resolved: list[Any] = []
    for configured_metric in configured:
        if isinstance(configured_metric, OverlapMetric):
            if configured_metric.config is None:
                configured_metric = configured_metric.with_config(scoring_config)
            elif overlap_scoring_config_recipe(
                configured_metric.config
            ) != overlap_scoring_config_recipe(scoring_config):
                raise ValueError(
                    "OverlapMetric.config conflicts with ScoringJob.scoring_config. "
                    "Use one identical overlap configuration for artifact identity and scoring."
                )
        resolved.append(configured_metric)
    configured = resolved
    if not any(configured_metric.name == "overlap" for configured_metric in configured):
        configured.insert(0, OverlapMetric(config=scoring_config))
    names = [configured_metric.name for configured_metric in configured]
    if len(names) != len(set(names)):
        raise ValueError("Metric names must be unique within a scoring job.")
    return configured


def _metric_recipes_cache_safe(metric_recipes: Iterable[dict[str, Any]]) -> bool:
    """Return whether every metric has a portable, reusable identity."""

    return all(
        recipe.get("cache_safe") is not False and recipe.get("portable") is not False
        for recipe in metric_recipes
    )


def materialize_segmentation_artifacts(
    dataset: Any,
    extractor: Any,
    store: ArtifactStore,
    segmentation_config: Any = None,
    batch_size: int = 16,
    resource_profiling_config: Optional[ResourceProfilingConfig] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> dict[str, Any]:
    """Materialize spatial segmentation outputs into standard artifact boundaries."""

    from vertebrae.segmentation import iter_materialize_segmentation_outputs

    recipe = extractor.recipe()
    resolved_segmentation_config = (
        segmentation_config if segmentation_config is not None else SegmentationConfig()
    )
    segmentation_config_dict = asdict(resolved_segmentation_config)
    segmentation_config_hash = hash_json_exact(segmentation_config_dict)
    resource_config = resource_profiling_config or ResourceProfilingConfig()
    resolved_memory_config = memory_config or MemoryConfig()
    profiler = ResourceProfiler(
        resource_config,
        extractor,
        streaming=True,
        context={"measurement_scope": "artifact_materialization", "modality": "segmentation"},
    )
    base_key = (
        f"segmentation/{dataset.identity_key()}/"
        f"{fingerprint_extractor_recipe(recipe)}/{segmentation_config_hash}"
    )
    outputs = []
    materializations = iter_materialize_segmentation_outputs(
        dataset,
        extractor,
        config=resolved_segmentation_config,
        batch_size=batch_size,
        resource_profiler=profiler if resource_config.enabled else None,
        memory_config=resolved_memory_config,
    )
    try:
        first_materialization = next(materializations)
    except StopIteration as exc:
        raise ValueError("Segmentation materialization produced no outputs.") from exc
    shared_profile = profiler.finish() if resource_config.enabled else None
    pending_materializations = chain((first_materialization,), materializations)
    del first_materialization
    for materialization in pending_materializations:
        output_key = named_output_artifact_key(base_key, materialization.name)
        labels_key = f"{output_key}/labels"
        groups_key = f"{output_key}/groups"
        provenance_key = f"{output_key}/provenance"
        embeddings = materialization.dataset.X
        labels = materialization.dataset.y
        groups = materialization.dataset.groups()
        if groups is None:
            raise ValueError("Segmentation materialization must define image groups.")
        _validate_materialized_rows(
            embeddings,
            labels,
            groups,
            materialization.provenance,
            workflow="Segmentation",
            output_name=materialization.name,
        )
        embedding_manifest = {
            "artifact_type": "segmentation_embedding",
            "vertebrae_version": __version__,
            "output_key": output_key,
            "dataset_identity_key": materialization.dataset.identity_key(),
            "source_dataset_identity_key": dataset.identity_key(),
            "extractor_recipe": recipe,
            "recipe_hash": fingerprint_extractor_recipe(recipe),
            "segmentation_config": make_json_safe(segmentation_config_dict),
            "segmentation_config_hash": segmentation_config_hash,
            "output_name": materialization.name,
            "n_samples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "sparse": False,
            "nnz": None,
            "storage_format": "dense",
            "modality": "segmentation",
            "segmentation": materialization.metadata,
            "labels_key": labels_key,
            "groups_key": groups_key,
            "provenance_key": provenance_key,
            "resource_profiling_config": asdict(resource_config),
            "memory_config": asdict(resolved_memory_config),
        }

        def finalize_embedding_metadata(
            metadata: dict[str, Any],
            _array_manifest: Any,
            artifact_stat: Any,
            embedded_values: Any = embeddings,
        ) -> dict[str, Any]:
            profile = with_embedding_footprint(
                shared_profile,
                embedded_values,
                embedded_values,
                raw_stat=artifact_stat,
                evaluated_stat=artifact_stat,
                persisted_storage=resource_config.persisted_storage,
            )
            if profile is not None:
                metadata["resource_profile"] = make_json_safe(profile)
            return metadata

        store.put_artifact(
            output_key,
            embeddings,
            embedding_manifest,
            metadata_finalizer=finalize_embedding_metadata,
        )
        embedding_manifest = store.get_json(output_key)
        label_summary = target_summary(
            labels,
            target_type=materialization.dataset.metadata.get("target_type", "auto"),
            target_names=materialization.dataset.metadata.get("target_names"),
        )
        label_manifest = {
            "artifact_type": "labels",
            "vertebrae_version": __version__,
            "output_key": labels_key,
            "dataset_identity_key": materialization.dataset.identity_key(),
            "n_samples": int(len(labels)),
            "target_type": label_summary["target_type"],
            "class_counts": make_json_safe(label_summary["class_counts"]),
            "n_classes": label_summary["n_classes"],
            "target_view": materialization.dataset.active_target_view(),
            "label_view": materialization.dataset.active_label_view(),
        }
        store.put_labels_artifact(
            labels_key,
            labels,
            label_manifest,
            target_type=materialization.dataset.metadata.get("target_type", "auto"),
            target_names=materialization.dataset.metadata.get("target_names"),
        )
        stored_groups, group_encoding = _group_artifact_values(groups)
        group_manifest = {
            "artifact_type": "groups",
            "vertebrae_version": __version__,
            "output_key": groups_key,
            "dataset_identity_key": materialization.dataset.identity_key(),
            "n_samples": int(len(groups)),
            "n_groups": int(len({semantic_label_key(value) for value in groups})),
            "group_name": "image_id",
            "group_value_encoding": group_encoding,
        }
        store.put_labels_artifact(
            groups_key,
            stored_groups,
            group_manifest,
            target_type="single_label",
        )
        store.put_json(provenance_key, {"rows": materialization.provenance})
        outputs.append(embedding_manifest)
        del embeddings, labels, groups, materialization
    bundle = {
        "artifact_type": "segmentation_embedding_bundle",
        "vertebrae_version": __version__,
        "output_key": base_key,
        "dataset_identity_key": dataset.identity_key(),
        "extractor_recipe": recipe,
        "segmentation_config": make_json_safe(segmentation_config_dict),
        "segmentation_config_hash": segmentation_config_hash,
        "resource_profiling_config": asdict(resource_config),
        "memory_config": asdict(resolved_memory_config),
        "resource_profile": (
            make_json_safe(shared_profile) if shared_profile is not None else None
        ),
        "outputs": outputs,
    }
    store.put_json(base_key, bundle)
    return bundle


def materialize_structured_artifacts(
    dataset: Any,
    extractor: Any,
    store: ArtifactStore,
    batch_size: int = 16,
    aligners: Optional[dict[str, Any]] = None,
    resource_profiling_config: Optional[ResourceProfilingConfig] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> dict[str, Any]:
    """Materialize structured unit outputs into standard artifact boundaries."""

    from vertebrae.structured import iter_materialize_structured_outputs

    recipe = extractor.recipe()
    aligner_recipes: dict[str, Any] = {}
    for name, aligner in sorted((aligners or {}).items()):
        recipe_method = getattr(aligner, "recipe", None)
        if not callable(recipe_method):
            raise TypeError(f"Structured aligner {name!r} must define recipe().")
        aligner_recipes[name] = make_json_safe(recipe_method())
    structured_identity = {
        "identity_schema": 2,
        "extractor_recipe": recipe,
        "aligners": aligner_recipes,
    }
    cache_safe = recipe.get("cache_safe") is not False and all(
        aligner_recipe.get("cache_safe") is not False for aligner_recipe in aligner_recipes.values()
    )
    resource_config = resource_profiling_config or ResourceProfilingConfig()
    resolved_memory_config = memory_config or MemoryConfig()
    profiler = ResourceProfiler(
        resource_config,
        extractor,
        streaming=True,
        context={"measurement_scope": "artifact_materialization", "modality": "structured"},
    )
    canonical_key = f"structured/{dataset.identity_key()}/{hash_json_exact(structured_identity)}"
    base_key, cache_eligible, cache_status = _execution_artifact_identity(
        canonical_key,
        {"cache_safe": cache_safe},
    )
    outputs = []
    materializations = iter_materialize_structured_outputs(
        dataset,
        extractor,
        batch_size=batch_size,
        aligners=aligners,
        resource_profiler=profiler if resource_config.enabled else None,
        memory_config=resolved_memory_config,
    )
    try:
        first_materialization = next(materializations)
    except StopIteration as exc:
        raise ValueError("Structured materialization produced no outputs.") from exc
    shared_profile = profiler.finish() if resource_config.enabled else None
    pending_materializations = chain((first_materialization,), materializations)
    del first_materialization
    for materialization in pending_materializations:
        output_key = named_output_artifact_key(base_key, materialization.name)
        labels_key = f"{output_key}/labels"
        groups_key = f"{output_key}/groups"
        provenance_key = f"{output_key}/provenance"
        embeddings = materialization.dataset.X
        labels = materialization.dataset.y
        groups = materialization.dataset.groups()
        if groups is None:
            raise ValueError("Structured materialization must define parent groups.")
        _validate_materialized_rows(
            embeddings,
            labels,
            groups,
            materialization.provenance,
            workflow="Structured",
            output_name=materialization.name,
        )
        sparse_embeddings = is_sparse_matrix(embeddings)
        embedding_manifest = {
            "artifact_type": "structured_embedding",
            "vertebrae_version": __version__,
            "output_key": output_key,
            "dataset_identity_key": materialization.dataset.identity_key(),
            "source_dataset_identity_key": dataset.identity_key(),
            "extractor_recipe": recipe,
            "recipe_hash": fingerprint_extractor_recipe(recipe),
            "output_name": materialization.name,
            "n_samples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "sparse": sparse_embeddings,
            "nnz": int(embeddings.nnz) if sparse_embeddings else None,
            "storage_format": (sparse_storage_format(embeddings) if sparse_embeddings else "dense"),
            "modality": materialization.dataset.modality,
            "structured": materialization.metadata,
            "unit_type": materialization.metadata.get("unit_type"),
            "task_family": materialization.metadata.get("task_family"),
            "alignment_mode": materialization.metadata.get("alignment_mode"),
            "alignment_recipe": materialization.metadata.get("alignment_recipe"),
            "aligner_recipes": aligner_recipes,
            "cache_eligible": cache_eligible,
            "cache_status": cache_status,
            "labels_key": labels_key,
            "groups_key": groups_key,
            "provenance_key": provenance_key,
            "resource_profiling_config": asdict(resource_config),
            "memory_config": asdict(resolved_memory_config),
        }

        def finalize_embedding_metadata(
            metadata: dict[str, Any],
            _array_manifest: Any,
            artifact_stat: Any,
            embedded_values: Any = embeddings,
        ) -> dict[str, Any]:
            profile = with_embedding_footprint(
                shared_profile,
                embedded_values,
                embedded_values,
                raw_stat=artifact_stat,
                evaluated_stat=artifact_stat,
                persisted_storage=resource_config.persisted_storage,
            )
            if profile is not None:
                metadata["resource_profile"] = make_json_safe(profile)
            return metadata

        store.put_artifact(
            output_key,
            embeddings,
            embedding_manifest,
            metadata_finalizer=finalize_embedding_metadata,
        )
        embedding_manifest = store.get_json(output_key)
        label_names = materialization.dataset.metadata.get("label_names")
        target_type = materialization.dataset.metadata.get("target_type", "auto")
        target_names = materialization.dataset.metadata.get("target_names")
        label_summary = target_summary(
            labels,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        label_manifest = {
            "artifact_type": "labels",
            "vertebrae_version": __version__,
            "output_key": labels_key,
            "dataset_identity_key": materialization.dataset.identity_key(),
            "n_samples": int(len(labels)),
            "target_type": label_summary["target_type"],
            "class_counts": make_json_safe(label_summary["class_counts"]),
            "n_classes": label_summary["n_classes"],
            "target_view": materialization.dataset.active_target_view(),
            "label_view": materialization.dataset.active_label_view(),
        }
        for label_key in (
            "labelset_counts",
            "mean_label_cardinality",
            "label_density",
            "n_targets",
            "target_names",
            "target_means",
            "target_variances",
            "constant_targets",
            "nonconstant_targets",
        ):
            if label_key in label_summary:
                label_manifest[label_key] = make_json_safe(label_summary[label_key])
        store.put_labels_artifact(
            labels_key,
            labels,
            label_manifest,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        stored_groups, group_encoding = _group_artifact_values(groups)
        group_manifest = {
            "artifact_type": "groups",
            "vertebrae_version": __version__,
            "output_key": groups_key,
            "dataset_identity_key": materialization.dataset.identity_key(),
            "n_samples": int(len(groups)),
            "n_groups": int(len({semantic_label_key(value) for value in groups})),
            "group_name": "parent_id",
            "group_value_encoding": group_encoding,
        }
        store.put_labels_artifact(
            groups_key,
            stored_groups,
            group_manifest,
            target_type="single_label",
        )
        store.put_json(provenance_key, {"rows": materialization.provenance})
        outputs.append(embedding_manifest)
        del embeddings, labels, groups, materialization
    bundle = {
        "artifact_type": "structured_embedding_bundle",
        "vertebrae_version": __version__,
        "output_key": base_key,
        "dataset_identity_key": dataset.identity_key(),
        "extractor_recipe": recipe,
        "aligner_recipes": aligner_recipes,
        "structured_identity": structured_identity,
        "cache_eligible": cache_eligible,
        "cache_status": cache_status,
        "resource_profiling_config": asdict(resource_config),
        "memory_config": asdict(resolved_memory_config),
        "resource_profile": (
            make_json_safe(shared_profile) if shared_profile is not None else None
        ),
        "outputs": outputs,
    }
    store.put_json(base_key, bundle)
    return bundle


def _validate_materialized_rows(
    embeddings: Any,
    labels: Any,
    groups: Any,
    provenance: Any,
    *,
    workflow: str,
    output_name: str,
) -> None:
    """Validate every row-aligned artifact before publishing any output files."""

    matrix = ensure_numeric_matrix(
        embeddings,
        f"{workflow} output {output_name!r} embeddings",
        allow_sparse=True,
    )
    n_rows = int(matrix.shape[0])
    aligned = {
        "labels": labels,
        "groups": groups,
        "provenance": provenance,
    }
    for name, values in aligned.items():
        try:
            length = len(values)
        except TypeError as exc:
            raise TypeError(
                f"{workflow} output {output_name!r} {name} must be row-aligned."
            ) from exc
        if length != n_rows:
            raise ValueError(
                f"{workflow} output {output_name!r} has {n_rows} embedding rows but "
                f"{length} {name} rows."
            )


def plan_embedding_shard_jobs(
    dataset: Any,
    extractor: Any,
    total_shards: int,
    batch_size: int = 128,
    resource_profiling_config: Optional[ResourceProfilingConfig] = None,
    output_key: Optional[str] = None,
    streaming_enabled: bool = True,
    memory_config: Optional[MemoryConfig] = None,
    run_prefix: Optional[str] = None,
) -> list[EmbeddingShardJob]:
    """Create deterministic embedding shard jobs.

    Args:
        dataset: Dataset to embed.
        extractor: Feature extractor to run on each shard.
        total_shards: Number of non-overlapping shards to create.
        batch_size: Number of samples per transform batch.

    Returns:
        Embedding shard jobs with canonical output keys.
    """

    if isinstance(total_shards, bool) or not isinstance(total_shards, (int, np.integer)):
        raise ValueError("total_shards must be an integer >= 1.")
    if int(total_shards) < 1:
        raise ValueError("total_shards must be >= 1.")
    n_samples = int(len(dataset.y))
    if n_samples < 1:
        raise ValueError("Cannot plan embedding shards for an empty dataset.")
    extractor_streaming_safe = bool(getattr(extractor, "streaming_safe", False))
    effective_streaming = bool(streaming_enabled and extractor_streaming_safe)
    planned_shards = min(int(total_shards), n_samples) if effective_streaming else 1
    base_key, cache_eligible, cache_status = _execution_artifact_identity(
        embedding_artifact_key(dataset, extractor),
        extractor.recipe(),
        output_key=output_key,
        run_prefix=run_prefix,
    )
    return [
        EmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            shard=shard,
            output_key=embedding_shard_key(base_key, shard),
            batch_size=batch_size,
            streaming_enabled=effective_streaming,
            cache_eligible=cache_eligible,
            cache_status=cache_status,
            memory_config=memory_config or MemoryConfig(),
            resource_profiling_config=(resource_profiling_config or ResourceProfilingConfig()),
        )
        for shard in (
            ShardSpec(total_shards=planned_shards, shard_index=i) for i in range(planned_shards)
        )
    ]


def materialize_embedding_shards(
    jobs: Iterable[EmbeddingShardJob],
    store: ArtifactStore,
    execution: Any,
) -> list[dict[str, Any]]:
    """Materialize embedding shards with an execution backend.

    Args:
        jobs: Embedding shard jobs.
        store: Artifact store receiving shard outputs.
        execution: Backend with a `map()` method, such as `LocalBackend`.

    Returns:
        Shard manifests in job order.
    """

    return execution.map(
        partial(_materialize_embedding_shard_job, store_config=store.config()),
        jobs,
    )


def materialize_and_merge_embeddings(
    dataset: Any,
    extractor: Any,
    store: ArtifactStore,
    execution: Any,
    total_shards: int,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Materialize sharded embeddings and merge them into one artifact.

    Args:
        dataset: Dataset to embed.
        extractor: Feature extractor to run.
        store: Artifact store for shard and merged artifacts.
        execution: Backend used to materialize shard jobs.
        total_shards: Number of deterministic sample shards.
        batch_size: Number of samples per transform batch.

    Returns:
        Merged embedding manifest.
    """

    extractor.fit(dataset.X, dataset.y)
    jobs = plan_embedding_shard_jobs(
        dataset=dataset,
        extractor=extractor,
        total_shards=total_shards,
        batch_size=batch_size,
    )
    shard_manifests = materialize_embedding_shards(jobs, store=store, execution=execution)
    base_key = jobs[0].output_key.rsplit("/shards/", 1)[0]
    return merge_embedding_shards(
        EmbeddingMergeJob(
            shard_keys=tuple(manifest["output_key"] for manifest in shard_manifests),
            output_key=base_key,
            n_samples=len(dataset.y),
        ),
        store=store,
    )


def materialize_label_artifact(
    dataset: Any,
    store: ArtifactStore,
    key: Any = None,
    aligned_embedding_identity_key: Optional[str] = None,
) -> dict[str, Any]:
    """Materialize dataset labels as an artifact.

    Args:
        dataset: Dataset with labels.
        store: Artifact store receiving labels.
        key: Optional artifact key. Defaults to `labels_artifact_key(dataset)`.

    Returns:
        JSON-compatible label manifest.
    """

    output_key = key or labels_artifact_key(dataset)
    labels = target_summary(
        dataset.y,
        label_names=dataset.metadata.get("label_names"),
        target_type=dataset.metadata.get("target_type", "auto"),
        target_names=dataset.metadata.get("target_names"),
    )
    manifest = {
        "artifact_type": "labels",
        "vertebrae_version": __version__,
        "output_key": output_key,
        "dataset_identity_key": dataset.identity_key(),
        "aligned_embedding_identity_key": aligned_embedding_identity_key,
        "content_digest": _labels_content_digest(dataset),
        "n_samples": int(len(dataset.y)),
        "dtype": str(np.asarray(dataset.y).dtype),
        "target_type": labels["target_type"],
        "class_counts": make_json_safe(labels["class_counts"]),
        "n_classes": labels["n_classes"],
        "target_view": make_json_safe(dataset.active_target_view()),
        "label_view": make_json_safe(dataset.active_label_view()),
    }
    for label_key in (
        "labelset_counts",
        "mean_label_cardinality",
        "label_density",
        "n_targets",
        "target_names",
        "target_means",
        "target_variances",
        "constant_targets",
        "nonconstant_targets",
    ):
        if label_key in labels:
            manifest[label_key] = make_json_safe(labels[label_key])
    store.put_labels_artifact(
        output_key,
        dataset.y,
        manifest,
        label_names=dataset.metadata.get("label_names"),
        target_type=dataset.metadata.get("target_type", "auto"),
        target_names=dataset.metadata.get("target_names"),
    )
    return store.get_json(output_key)


def materialize_group_artifact(
    dataset: Any,
    store: ArtifactStore,
    key: Any = None,
    aligned_embedding_identity_key: Optional[str] = None,
) -> dict[str, Any]:
    """Materialize aligned independence groups without reporting raw IDs."""

    groups = dataset.groups() if callable(getattr(dataset, "groups", None)) else None
    if groups is None:
        raise ValueError("Dataset does not define independence groups.")
    output_key = key or groups_artifact_key(dataset)
    stored_groups, group_encoding = _group_artifact_values(groups)
    manifest = {
        "artifact_type": "groups",
        "vertebrae_version": __version__,
        "output_key": output_key,
        "dataset_identity_key": dataset.identity_key(),
        "aligned_embedding_identity_key": aligned_embedding_identity_key,
        "content_digest": _groups_content_digest(dataset),
        "n_samples": int(len(groups)),
        "n_groups": int(len({semantic_label_key(value) for value in groups})),
        "group_name": dataset.metadata.get("group_name", "group"),
        "group_value_encoding": group_encoding,
    }
    store.put_labels_artifact(
        output_key,
        stored_groups,
        manifest,
        target_type="single_label",
    )
    return store.get_json(output_key)


def score_embedding_artifact(
    job: ScoringJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Score a persisted embedding artifact against persisted labels.

    Args:
        job: Scoring job.
        store: Artifact store containing inputs and receiving the score.

    Returns:
        JSON-compatible scoring artifact.
    """

    if job.scoring_config is None:
        job = replace(job, scoring_config=OverlapScoringConfig())

    embeddings, labels, embedding_metadata, label_metadata = (
        _load_validated_embedding_label_artifacts(
            store,
            embedding_key=job.embedding_key,
            labels_key=job.labels_key,
        )
    )
    if int(embeddings.shape[0]) != len(labels):
        raise ValueError("Embedding and label arrays have different row counts.")
    groups, group_metadata = _load_validated_groups(
        store,
        groups_key=job.groups_key,
        embedding_metadata=embedding_metadata,
        label_metadata=label_metadata,
    )
    configured = _configured_metrics(job.scoring_config, job.metrics, job.metric)
    names = [metric.name for metric in configured]
    if job.primary_metric not in names:
        raise ValueError("primary_metric must name one configured metric.")
    canonical_labels = canonical_metric_targets(
        labels,
        label_names=label_metadata.get("label_names"),
        target_type=label_metadata.get("target_type", "auto"),
        target_names=label_metadata.get("target_names"),
    )
    canonical_groups = None if groups is None else canonical_semantic_array(groups)
    metric_results = {}
    for metric in configured:
        result = metric.score(
            embeddings,
            canonical_labels,
            target_metadata=label_metadata,
            groups=canonical_groups,
            seed=job.seed,
        )
        result.metadata = {**label_metadata, **result.metadata}
        metric_results[metric.name] = result.to_dict()
    metric_recipes = [make_json_safe(metric.recipe()) for metric in configured]
    metric_identity_safe = _metric_recipes_cache_safe(metric_recipes)
    cache_eligible, cache_status = derived_cache_reuse_decision(
        embedding_metadata,
        identity_safe=metric_identity_safe,
    )
    if not cache_eligible and not job.output_key.startswith("runs/"):
        raise ValueError(
            "Cache-ineligible scoring inputs require a run-scoped score output key under "
            "'runs/'. Use plan_scoring_jobs(), enable source embedding caching, or give the "
            "callable metric a portable identity."
        )
    protocol = _labeled_scoring_protocol(
        job,
        metric_recipes=metric_recipes,
        embedding_metadata=embedding_metadata,
        label_metadata=label_metadata,
        group_metadata=group_metadata,
    )
    protocol_fingerprint = hash_json_exact(protocol)
    collection_protocol = {key: value for key, value in protocol.items() if key != "seed"}
    collection_protocol_fingerprint = hash_json_exact(collection_protocol)
    artifact = {
        "artifact_type": "metric_evaluation",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "embedding_key": job.embedding_key,
        "labels_key": job.labels_key,
        "groups_key": job.groups_key,
        "group_metadata": group_metadata,
        "seed": job.seed,
        "metrics": metric_results,
        "primary_metric": job.primary_metric,
        "metric_recipes": metric_recipes,
        "scoring_config": overlap_scoring_config_recipe(job.scoring_config),
        "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint,
        "collection_protocol_fingerprint": collection_protocol_fingerprint,
        "cache_eligible": cache_eligible,
        "cache_status": cache_status,
        "embedding_metadata": embedding_metadata,
        "label_metadata": label_metadata,
        "resources": asdict(job.resources),
    }
    store.put_json(job.output_key, artifact)
    return artifact


def _labeled_scoring_protocol(
    job: ScoringJob,
    *,
    metric_recipes: list[dict[str, Any]],
    embedding_metadata: dict[str, Any],
    label_metadata: dict[str, Any],
    group_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build the complete schema-versioned identity of one labeled evaluation."""

    return {
        "schema_version": 2,
        "kind": "labeled_embedding_scoring",
        "embedding": {
            "key": job.embedding_key,
            "dataset_identity_key": embedding_metadata.get("dataset_identity_key"),
            "recipe_hash": embedding_metadata.get("recipe_hash"),
            "source_embedding_key": embedding_metadata.get("source_embedding_key"),
            "compression": embedding_metadata.get("compression"),
            "output_name": embedding_metadata.get("output_name"),
            "shape": embedding_metadata.get("shape"),
            "dtype": embedding_metadata.get("dtype"),
        },
        "labels": {
            "key": job.labels_key,
            "content_digest": label_metadata.get("content_digest"),
            "dataset_identity_key": label_metadata.get("dataset_identity_key"),
            "aligned_embedding_identity_key": label_metadata.get("aligned_embedding_identity_key"),
            "n_samples": label_metadata.get("n_samples"),
            "target_type": label_metadata.get("target_type"),
            "target_view": label_metadata.get("target_view"),
            "label_view": label_metadata.get("label_view"),
        },
        "groups": (
            None
            if group_metadata is None
            else {
                "key": job.groups_key,
                "content_digest": group_metadata.get("content_digest"),
                "dataset_identity_key": group_metadata.get("dataset_identity_key"),
                "aligned_embedding_identity_key": group_metadata.get(
                    "aligned_embedding_identity_key"
                ),
                "n_samples": group_metadata.get("n_samples"),
                "group_name": group_metadata.get("group_name"),
            }
        ),
        "scoring_config": overlap_scoring_config_recipe(job.scoring_config),
        "metric_recipes": metric_recipes,
        "primary_metric": job.primary_metric,
        "seed": job.seed,
    }


def score_retrieval_artifact(job: RetrievalScoringJob, store: ArtifactStore) -> dict[str, Any]:
    """Score persisted query/gallery embeddings using a canonical relevance JSON artifact.

    The relevance artifact contains ``relevance`` as sparse row-index mappings and optional
    ``query_ids``, ``gallery_ids``, and ``exclusions`` fields. This keeps retrieval scoring
    independent of raw model objects and compatible with object stores.
    """
    from vertebrae.config import RetrievalConfig
    from vertebrae.scoring.retrieval import RetrievalScorer

    queries, query_metadata = store.get_artifact(job.query_embedding_key)
    gallery, gallery_metadata = store.get_artifact(job.gallery_embedding_key)
    relevance_data = store.get_json(job.relevance_key)
    _validate_retrieval_pair_metadata(query_metadata, gallery_metadata, relevance_data)
    raw_relevance = relevance_data.get("relevance", relevance_data)
    n_queries, n_gallery = int(queries.shape[0]), int(gallery.shape[0])
    if int(query_metadata.get("n_samples", n_queries)) != n_queries:
        raise ValueError("Query embedding metadata has a different row count than its array.")
    if int(gallery_metadata.get("n_samples", n_gallery)) != n_gallery:
        raise ValueError("Gallery embedding metadata has a different row count than its array.")
    if relevance_data.get("n_queries", n_queries) != n_queries:
        raise ValueError("Relevance artifact does not align with query embeddings.")
    if relevance_data.get("n_gallery", n_gallery) != n_gallery:
        raise ValueError("Relevance artifact does not align with gallery embeddings.")
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("Query and gallery artifacts have incompatible embedding dimensions.")
    relevance = {
        int(query): {int(candidate): float(grade) for candidate, grade in values.items()}
        for query, values in raw_relevance.items()
    }
    exclusions_data = store.get_json(job.exclusions_key) if job.exclusions_key else relevance_data
    exclusions: set[Tuple[int, int]] = {
        (int(pair[0]), int(pair[1])) for pair in exclusions_data.get("exclusions", [])
    }
    query_ids = relevance_data.get("query_ids")
    gallery_ids = relevance_data.get("gallery_ids")
    if query_ids is not None and len(query_ids) != n_queries:
        raise ValueError("Relevance artifact query IDs do not align with query embeddings.")
    if gallery_ids is not None and len(gallery_ids) != n_gallery:
        raise ValueError("Relevance artifact gallery IDs do not align with gallery embeddings.")
    retrieval_config = job.retrieval_config or RetrievalConfig()
    scorer = RetrievalScorer(retrieval_config)
    forward = scorer.score(
        queries,
        gallery,
        relevance,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        exclusions=exclusions,
    )
    reverse = None
    if retrieval_config.bidirectional:
        reverse_relevance, reverse_exclusions = _transpose_retrieval_relations(
            relevance,
            exclusions,
            n_gallery,
        )
        missing_reverse = [
            gallery_index
            for gallery_index, values in reverse_relevance.items()
            if not any(
                (gallery_index, query_index) not in reverse_exclusions for query_index in values
            )
        ]
        if missing_reverse:
            raise ValueError(
                "bidirectional retrieval requires every gallery item to have an "
                "eligible reverse relevance relation; missing gallery rows "
                f"{missing_reverse[:10]}."
            )
        reverse = scorer.score(
            gallery,
            queries,
            reverse_relevance,
            query_ids=gallery_ids,
            gallery_ids=query_ids,
            exclusions=reverse_exclusions,
        )
    primary_score = (
        forward.score if reverse is None else float((forward.score + reverse.score) / 2.0)
    )
    relevance_protocol_fingerprint = relevance_data.get("protocol_fingerprint")
    if not isinstance(relevance_protocol_fingerprint, str) or not relevance_protocol_fingerprint:
        raise ValueError("Relevance artifact must declare a non-empty protocol_fingerprint.")
    protocol_fingerprint = hash_json_exact(
        {
            "identity_schema": 2,
            "relevance_protocol_fingerprint": relevance_protocol_fingerprint,
            "exclusions": sorted(exclusions),
            "retrieval_config": retrieval_config,
        }
    )
    evaluation_fingerprint = hash_json_exact(
        {
            "identity_schema": 2,
            "query_embedding_key": job.query_embedding_key,
            "gallery_embedding_key": job.gallery_embedding_key,
            "relevance_key": job.relevance_key,
            "exclusions_key": job.exclusions_key,
            "retrieval_config": retrieval_config,
        }
    )
    cache_eligible, cache_status = derived_cache_reuse_decision(
        query_metadata,
        gallery_metadata,
    )
    artifact = {
        "artifact_type": "retrieval_evaluation",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "query_embedding_key": job.query_embedding_key,
        "gallery_embedding_key": job.gallery_embedding_key,
        "relevance_key": job.relevance_key,
        "exclusions_key": job.exclusions_key,
        "forward": forward.to_dict(),
        "reverse": reverse.to_dict() if reverse is not None else None,
        "primary_score": primary_score,
        "query_endpoint": query_metadata,
        "gallery_endpoint": gallery_metadata,
        "retrieval_config": asdict(retrieval_config),
        "protocol_fingerprint": protocol_fingerprint,
        "evaluation_fingerprint": evaluation_fingerprint,
        "cache_eligible": cache_eligible,
        "cache_status": cache_status,
        "resources": asdict(job.resources),
    }
    store.put_json(job.output_key, artifact)
    return artifact


def score_retrieval_artifacts(
    jobs: Iterable[RetrievalScoringJob], store: ArtifactStore, execution: Any
) -> list[dict[str, Any]]:
    """Run independent retrieval scoring jobs through a local, Ray, or Dask backend."""

    return execution.map(
        partial(_score_retrieval_artifact_job, store_config=store.config()),
        jobs,
    )


def _score_retrieval_artifact_job(
    job: RetrievalScoringJob, store_config: ArtifactStoreConfig
) -> dict[str, Any]:
    return score_retrieval_artifact(job, create_artifact_store_from_config(store_config))


def diagnose_embedding_artifact(
    job: SeparatixJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Run Separatix over a persisted embedding artifact and labels."""

    embeddings, labels, embedding_metadata, label_metadata = (
        _load_validated_embedding_label_artifacts(
            store,
            embedding_key=job.embedding_key,
            labels_key=job.labels_key,
        )
    )
    score_artifact = store.get_json(job.score_key)
    if score_artifact.get("artifact_type") != "metric_evaluation":
        raise ValueError("Separatix score_key must reference a metric_evaluation artifact.")
    if (
        score_artifact.get("embedding_key") != job.embedding_key
        or score_artifact.get("labels_key") != job.labels_key
        or score_artifact.get("groups_key") != job.groups_key
    ):
        raise ValueError("Separatix inputs do not match the referenced scoring protocol.")
    score_data = score_artifact.get("metrics", {}).get("overlap", {})
    overlap_score = float(score_data.get("score", score_data.get("macro_score")))

    from vertebrae.config import OverlapScoringConfig, SeparatixConfig

    overlap_config = OverlapScoringConfig()
    overlap_metadata = score_data.get("metadata", {})
    if "normalize_embeddings" in overlap_metadata:
        overlap_config.normalize_embeddings = bool(overlap_metadata["normalize_embeddings"])
    separatix_config = job.separatix_config or SeparatixConfig()
    scorer = SeparatixScorer(config=separatix_config, overlap_config=overlap_config)
    groups, _ = _load_validated_groups(
        store,
        groups_key=job.groups_key,
        embedding_metadata=embedding_metadata,
        label_metadata=label_metadata,
    )

    target_type = label_metadata.get("target_type", "single_label")
    threshold = (
        separatix_config.regression_overlap_threshold
        if target_type == REGRESSION_TARGET
        else separatix_config.overlap_threshold
    )
    if overlap_score < threshold:
        diagnostic = scorer.skipped_result(
            reason=(
                "Skipped Separatix because overlap score "
                f"{overlap_score:.4f} is below the configured threshold "
                f"{threshold:.4f}."
            ),
            overlap_score=overlap_score,
            threshold=threshold,
        )
    else:
        excluded = overlap_metadata.get("exclude_classes", [])
        if excluded and target_type != REGRESSION_TARGET:
            catalog_keys = {
                item.get("key")
                for item in overlap_metadata.get("label_catalog", [])
                if isinstance(item, dict) and isinstance(item.get("key"), str)
            }
            excluded_keys = {
                item
                if isinstance(item, str) and item in catalog_keys
                else semantic_label_key(item.item() if hasattr(item, "item") else item)
                for item in excluded
            }
            if target_type == MULTI_LABEL_TARGET:
                labels, target_metadata = metric_labels(
                    labels,
                    label_names=label_metadata.get("label_names"),
                    target_type=MULTI_LABEL_TARGET,
                )
                names = tuple(target_metadata.get("label_names") or ())
                keep = [
                    index
                    for index, name in enumerate(names)
                    if semantic_label_key(name) not in excluded_keys
                ]
                labels = labels[:, keep]
                active_rows = np.asarray(labels.sum(axis=1)).reshape(-1) > 0
                embeddings = embeddings[active_rows]
                labels = labels[active_rows]
                groups = None if groups is None else np.asarray(groups)[active_rows]
                label_metadata = dict(label_metadata)
                label_metadata["label_names"] = [names[index] for index in keep]
            else:
                mask = np.asarray(
                    [semantic_label_key(label) not in excluded_keys for label in labels],
                    dtype=bool,
                )
                embeddings = embeddings[mask]
                labels = np.asarray(labels)[mask]
                groups = None if groups is None else np.asarray(groups)[mask]
        if (
            target_type == MULTI_LABEL_TARGET
            and int(getattr(labels, "ndim", 1)) == 2
            and int(labels.shape[1]) == 0
        ):
            diagnostic = scorer.skipped_result(
                reason="Skipped Separatix because all classes were excluded from diagnostics.",
                overlap_score=overlap_score,
                threshold=threshold,
            )
        else:
            try:
                diagnostic = scorer.score(
                    embeddings,
                    labels,
                    label_names=label_metadata.get("label_names"),
                    target_type=target_type,
                    target_names=label_metadata.get("target_names"),
                    groups=groups,
                )
            except ValueError as exc:
                if groups is None:
                    raise
                diagnostic = scorer.skipped_result(
                    reason=f"Skipped grouped Separatix diagnostic: {exc}",
                    overlap_score=overlap_score,
                    threshold=threshold,
                )

    cache_eligible, cache_status = derived_cache_reuse_decision(
        embedding_metadata,
        score_artifact,
    )
    artifact = {
        "artifact_type": "separatix_diagnostic",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "embedding_key": job.embedding_key,
        "labels_key": job.labels_key,
        "score_key": job.score_key,
        "groups_key": job.groups_key,
        "diagnostic": diagnostic.to_dict(),
        "cache_eligible": cache_eligible,
        "cache_status": cache_status,
        "embedding_metadata": embedding_metadata,
        "label_metadata": label_metadata,
        "resources": asdict(job.resources),
    }
    store.put_json(job.output_key, artifact)
    return artifact


def compress_embedding_artifact(
    job: CompressionJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Compress a persisted embedding artifact."""

    embeddings, embedding_metadata = store.get_artifact(job.embedding_key)
    compression_result = compress_embeddings(embeddings, config=job.compression_config)
    compressed_embeddings = compression_result.embeddings
    sparse_output = is_sparse_matrix(compressed_embeddings)
    compressed_metadata = dict(embedding_metadata)
    cache_eligible, cache_status = derived_cache_reuse_decision(embedding_metadata)
    compressed_metadata.update(
        {
            "artifact_type": "compressed_embedding",
            "output_key": job.output_key,
            "cache_key": job.output_key,
            "source_embedding_key": job.embedding_key,
            "n_samples": int(compressed_embeddings.shape[0]),
            "embedding_dim": int(compressed_embeddings.shape[1]),
            "shape": list(compressed_embeddings.shape),
            "dtype": str(compressed_embeddings.dtype),
            "sparse": sparse_output,
            "nnz": int(compressed_embeddings.nnz) if sparse_output else None,
            "storage_format": (
                sparse_storage_format(compressed_embeddings) if sparse_output else "dense"
            ),
            "compression": compression_result.metadata,
            "cache_eligible": cache_eligible,
            "cache_status": cache_status,
        }
    )

    def finalize_compressed_metadata(
        metadata: dict[str, Any], _array_manifest: Any, artifact_stat: Any
    ) -> dict[str, Any]:
        persisted_storage = bool(
            embedding_metadata.get("resource_profiling_config", {}).get("persisted_storage", True)
        )
        serialized_profile = embedding_metadata.get("distributed_resource_profile")
        if serialized_profile is not None:
            distributed_profile = with_distributed_embedding_footprint(
                distributed_resource_profile_from_dict(dict(serialized_profile)),
                embeddings,
                compression_result.embeddings,
                store=store,
                raw_key=job.embedding_key,
                evaluated_stat=artifact_stat,
                persisted_storage=persisted_storage,
            )
            metadata["distributed_resource_profile"] = make_json_safe(distributed_profile)
        elif embedding_metadata.get("resource_profile") is not None:
            local_profile = with_embedding_footprint(
                resource_profile_from_dict(dict(embedding_metadata["resource_profile"])),
                embeddings,
                compression_result.embeddings,
                store=store,
                raw_key=job.embedding_key,
                evaluated_stat=artifact_stat,
                persisted_storage=persisted_storage,
            )
            metadata["resource_profile"] = make_json_safe(local_profile)
        return metadata

    store.put_artifact(
        job.output_key,
        compressed_embeddings,
        compressed_metadata,
        metadata_finalizer=finalize_compressed_metadata,
    )
    compressed_metadata = store.get_json(job.output_key)
    return {
        "artifact_type": "compressed_embedding",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "source_embedding_key": job.embedding_key,
        "embedding_metadata": compressed_metadata,
        "compression_metadata": compression_result.metadata,
        "resources": asdict(job.resources),
    }


def score_embedding_artifacts(
    jobs: Iterable[ScoringJob],
    store: ArtifactStore,
    execution: Any,
) -> list[dict[str, Any]]:
    """Score persisted embeddings with an execution backend.

    Args:
        jobs: Scoring jobs.
        store: Artifact store containing inputs and receiving outputs.
        execution: Backend with a `map()` method.

    Returns:
        Scoring artifacts in job order.
    """

    return execution.map(
        partial(_score_embedding_artifact_job, store_config=store.config()),
        jobs,
    )


def compress_embedding_artifacts(
    jobs: Iterable[CompressionJob], store: ArtifactStore, execution: Any
) -> list[dict[str, Any]]:
    """Run independent compression jobs through an execution backend."""

    return execution.map(
        partial(_compress_embedding_artifact_job, store_config=store.config()),
        jobs,
    )


def diagnose_embedding_artifacts(
    jobs: Iterable[SeparatixJob], store: ArtifactStore, execution: Any
) -> list[dict[str, Any]]:
    """Run independent Separatix jobs through an execution backend."""

    return execution.map(
        partial(_diagnose_embedding_artifact_job, store_config=store.config()),
        jobs,
    )


def run_stability_artifact(job: StabilityJob, store: ArtifactStore) -> dict[str, Any]:
    """Run stability analysis over persisted embeddings and targets."""

    embeddings, labels, embedding_metadata, label_metadata = (
        _load_validated_embedding_label_artifacts(
            store,
            embedding_key=job.embedding_key,
            labels_key=job.labels_key,
        )
    )
    from vertebrae.scoring.stability import run_stability_analysis

    payload = run_stability_analysis(
        embeddings,
        labels,
        job.scoring_config,
        job.stability_config,
        label_names=label_metadata.get("label_names"),
        target_type=label_metadata.get("target_type", "auto"),
        target_names=label_metadata.get("target_names"),
    )
    cache_eligible, cache_status = derived_cache_reuse_decision(embedding_metadata)
    artifact = {
        "artifact_type": "stability_diagnostic",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "embedding_key": job.embedding_key,
        "labels_key": job.labels_key,
        "stability": payload,
        "cache_eligible": cache_eligible,
        "cache_status": cache_status,
        "embedding_metadata": embedding_metadata,
        "label_metadata": label_metadata,
        "resources": asdict(job.resources),
    }
    store.put_json(job.output_key, artifact)
    return artifact


def run_stability_artifacts(
    jobs: Iterable[StabilityJob], store: ArtifactStore, execution: Any
) -> list[dict[str, Any]]:
    """Run independent stability jobs through an execution backend."""

    return execution.map(
        partial(_run_stability_artifact_job, store_config=store.config()),
        jobs,
    )


def plan_compression_job(
    embedding_key: str,
    compression_config: Any,
) -> CompressionJob:
    """Create a compression job for a persisted embedding artifact."""

    return CompressionJob(
        embedding_key=embedding_key,
        output_key=compress_embedding_artifact_key(embedding_key, compression_config),
        compression_config=compression_config,
    )


def validate_embedding_label_artifacts(
    store: ArtifactStore,
    embedding_key: str,
    labels_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate that embedding and label artifacts can be scored together.

    Args:
        store: Artifact store containing the artifacts.
        embedding_key: Complete embedding artifact key.
        labels_key: Label artifact key.

    Returns:
        Embedding and label manifests.

    Raises:
        ValueError: If metadata is incompatible.
    """

    _embeddings, _labels, embedding_metadata, label_metadata = (
        _load_validated_embedding_label_artifacts(
            store,
            embedding_key=embedding_key,
            labels_key=labels_key,
        )
    )
    return embedding_metadata, label_metadata


def _load_validated_embedding_label_artifacts(
    store: ArtifactStore,
    *,
    embedding_key: str,
    labels_key: str,
) -> tuple[Any, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load each data/metadata generation atomically and validate cross-artifact alignment."""

    embeddings, embedding_metadata = store.get_artifact(embedding_key)
    labels, label_metadata = store.get_labels_artifact(labels_key)
    embedding_rows = int(embedding_metadata.get("n_samples", -1))
    label_rows = int(label_metadata.get("n_samples", -2))
    if embedding_rows != label_rows:
        raise ValueError(
            "Embedding and label artifacts have different row counts; "
            f"got {embedding_rows} and {label_rows}."
        )
    embedding_identity_key = embedding_metadata.get("dataset_identity_key")
    label_identity_key = label_metadata.get("dataset_identity_key")
    aligned_identity_key = label_metadata.get("aligned_embedding_identity_key")
    if (
        embedding_identity_key
        and label_identity_key
        and (
            embedding_identity_key != label_identity_key
            and embedding_identity_key != aligned_identity_key
        )
    ):
        raise ValueError("Embedding and label artifacts have different dataset identities.")
    if int(embeddings.shape[0]) != embedding_rows or len(labels) != label_rows:
        raise ValueError("Embedding or label metadata does not match its committed data.")
    return embeddings, np.asarray(labels), embedding_metadata, label_metadata


def _load_validated_groups(
    store: ArtifactStore,
    *,
    groups_key: Optional[str],
    embedding_metadata: dict[str, Any],
    label_metadata: dict[str, Any],
) -> tuple[Optional[np.ndarray], Optional[dict[str, Any]]]:
    if groups_key is None:
        return None, None
    groups, group_metadata = store.get_labels_artifact(groups_key)
    if group_metadata.get("artifact_type") != "groups":
        raise ValueError("Group artifact metadata must declare artifact_type='groups'.")
    expected_rows = int(label_metadata.get("n_samples", -1))
    if int(group_metadata.get("n_samples", -2)) != expected_rows:
        raise ValueError("Group and label artifacts have different row counts.")
    embedding_identity = embedding_metadata.get("dataset_identity_key")
    compatible_identities = {
        label_metadata.get("aligned_embedding_identity_key")
        or label_metadata.get("dataset_identity_key"),
        group_metadata.get("aligned_embedding_identity_key")
        or group_metadata.get("dataset_identity_key"),
    }
    compatible_identities.discard(None)
    if embedding_identity and any(
        identity != embedding_identity for identity in compatible_identities
    ):
        raise ValueError("Embedding, label, and group artifacts have different dataset identities.")
    if len(groups) != expected_rows:
        raise ValueError(
            "Group artifact metadata does not match its labels; "
            f"expected {expected_rows} rows, loaded {len(groups)}."
        )
    if group_metadata.get("group_value_encoding") == "semantic_label_key/v1":
        groups = np.asarray(
            [SemanticLabelKey(str(value)) for value in groups],
            dtype=object,
        )
    return np.asarray(groups), group_metadata


def _group_artifact_values(groups: Any) -> tuple[np.ndarray, str]:
    """Choose a reversible primitive representation or canonical semantic keys."""

    values = np.asarray(groups, dtype=object)
    normalized = [value.item() if hasattr(value, "item") else value for value in values]
    primitive = all(
        isinstance(value, (str, bool, int, float)) and not isinstance(value, complex)
        for value in normalized
    )
    if primitive:
        return np.asarray(normalized, dtype=object), "primitive/v1"
    return (
        np.asarray(
            [SemanticLabelKey(semantic_label_key(value)) for value in normalized],
            dtype=object,
        ),
        "semantic_label_key/v1",
    )


def plan_scoring_jobs(
    embedding_key: str,
    labels_key: str,
    seeds: Iterable[Optional[int]],
    scoring_config: Any = None,
    metrics: Any = None,
    primary_metric: str = "overlap",
    metric: Any = None,
    groups_key: Optional[str] = None,
) -> list[ScoringJob]:
    """Create scoring jobs for one embedding and label artifact pair.

    Args:
        embedding_key: Complete embedding artifact key.
        labels_key: Label artifact key.
        seeds: Seeds for scoring jobs. Use `None` for the default single score.
        scoring_config: Optional scoring configuration shared by all jobs.
        metrics: Optional generic metrics shared by all jobs.
        primary_metric: Aggregate metric selected for score collection.
        groups_key: Optional aligned independence-group artifact key.

    Returns:
        Scoring jobs with canonical output keys.
    """

    resolved_scoring_config = scoring_config or OverlapScoringConfig()
    resolved_metrics = tuple(metrics or ())
    metric_recipes = _configured_metric_recipes(
        resolved_scoring_config,
        resolved_metrics,
        metric,
    )
    run_prefix = None if _metric_recipes_cache_safe(metric_recipes) else f"runs/{uuid4().hex}"
    return [
        ScoringJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            output_key=scoring_artifact_key(
                embedding_key,
                seed=seed,
                labels_key=labels_key,
                groups_key=groups_key,
                scoring_config=resolved_scoring_config,
                metrics=resolved_metrics,
                metric=metric,
                primary_metric=primary_metric,
                run_prefix=run_prefix,
            ),
            groups_key=groups_key,
            scoring_config=resolved_scoring_config,
            metrics=resolved_metrics,
            primary_metric=primary_metric,
            metric=metric,
            seed=seed,
        )
        for seed in seeds
    ]


def collect_score_artifacts(
    score_keys: Iterable[str],
    store: ArtifactStore,
    output_key: str,
    interval_level: float = 0.95,
    metric_name: Optional[str] = None,
) -> dict[str, Any]:
    """Collect scoring artifacts into a stability-style summary.

    Args:
        score_keys: Score artifact keys to aggregate.
        store: Artifact store containing score artifacts.
        output_key: Artifact key for the collection summary.
        interval_level: Percentile interval level for summary statistics.

    Returns:
        JSON-compatible score collection artifact.
    """

    keys = list(score_keys)
    artifacts = [store.get_json(key) for key in keys]
    if not artifacts:
        raise ValueError("At least one score artifact is required.")
    groups_keys = {artifact.get("groups_key") for artifact in artifacts}
    if len(groups_keys) != 1:
        raise ValueError("Score artifacts must share one groups protocol.")
    groups_key = next(iter(groups_keys))
    first_group_metadata = artifacts[0].get("group_metadata")
    if any(artifact.get("group_metadata") != first_group_metadata for artifact in artifacts[1:]):
        raise ValueError("Score artifacts must share identical group metadata.")
    if any(artifact.get("artifact_type") != "metric_evaluation" for artifact in artifacts):
        raise ValueError("Score collections require metric_evaluation artifacts.")
    for artifact in artifacts:
        _validate_labeled_scoring_artifact_protocol(artifact)
    collection_protocol_fingerprints = {
        artifact.get("collection_protocol_fingerprint") for artifact in artifacts
    }
    if len(collection_protocol_fingerprints) != 1 or None in collection_protocol_fingerprints:
        raise ValueError(
            "Score artifacts must share one complete embedding, target, and metric protocol."
        )
    metric_name = metric_name or artifacts[0].get("primary_metric", "overlap")
    if any(metric_name not in artifact.get("metrics", {}) for artifact in artifacts):
        raise ValueError(f"Every score artifact must contain metric {metric_name!r}.")
    directions = {
        bool(artifact["metrics"][metric_name].get("higher_is_better", True))
        for artifact in artifacts
    }
    if len(directions) != 1:
        raise ValueError("Collected metric artifacts must share one optimization direction.")
    scores = [float(artifact["metrics"][metric_name]["score"]) for artifact in artifacts]
    warnings = sorted(
        {
            warning
            for artifact in artifacts
            for metric in artifact.get("metrics", {}).values()
            for warning in metric.get("warnings", [])
        }
    )
    seeds = [artifact.get("seed") for artifact in artifacts]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Score collection seeds must be unique.")
    cache_eligible, cache_status = derived_cache_reuse_decision(*artifacts)
    collection = {
        "artifact_type": "score_collection",
        "vertebrae_version": __version__,
        "output_key": output_key,
        "score_keys": keys,
        "scores": scores,
        "seeds": seeds,
        "summary": _score_summary(scores, interval_level),
        "interval_level": interval_level,
        "warnings": warnings,
        "embedding_key": artifacts[0].get("embedding_key"),
        "labels_key": artifacts[0].get("labels_key"),
        "groups_key": groups_key,
        "group_metadata": first_group_metadata,
        "metric_name": metric_name,
        "collection_protocol_fingerprint": next(iter(collection_protocol_fingerprints)),
        "protocol_fingerprints": [artifact["protocol_fingerprint"] for artifact in artifacts],
        "cache_eligible": cache_eligible,
        "cache_status": cache_status,
    }
    store.put_json(output_key, collection)
    return collection


def _validate_labeled_scoring_artifact_protocol(artifact: dict[str, Any]) -> None:
    """Reject incomplete, stale, or internally inconsistent scoring identities."""

    protocol = artifact.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(
            "Score artifact is missing its complete embedding, target, and metric protocol."
        )
    if protocol.get("schema_version") != 2 or protocol.get("kind") != ("labeled_embedding_scoring"):
        raise ValueError("Score artifact uses an unsupported scoring protocol schema.")
    if hash_json_exact(protocol) != artifact.get("protocol_fingerprint"):
        raise ValueError("Score artifact protocol_fingerprint is inconsistent.")
    collection_protocol = {key: value for key, value in protocol.items() if key != "seed"}
    if hash_json_exact(collection_protocol) != artifact.get("collection_protocol_fingerprint"):
        raise ValueError("Score artifact collection protocol fingerprint is inconsistent.")
    expected = {
        "embedding_key": (protocol.get("embedding") or {}).get("key"),
        "labels_key": (protocol.get("labels") or {}).get("key"),
        "groups_key": (
            None if protocol.get("groups") is None else (protocol.get("groups") or {}).get("key")
        ),
        "metric_recipes": protocol.get("metric_recipes"),
        "primary_metric": protocol.get("primary_metric"),
        "seed": protocol.get("seed"),
        "scoring_config": protocol.get("scoring_config"),
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise ValueError(f"Score artifact field {field!r} conflicts with its protocol record.")


def benchmark_result_from_artifacts(
    score_key: str,
    store: ArtifactStore,
    output_key: Optional[str] = None,
    stability_key: Optional[str] = None,
    separatix_key: Optional[str] = None,
) -> dict[str, Any]:
    """Build and optionally persist a benchmark-style result from artifacts.

    Args:
        score_key: Score artifact key.
        store: Artifact store containing score and source artifacts.
        output_key: Optional result artifact key.
        stability_key: Optional score collection key to use as stability metadata.

    Returns:
        JSON-compatible benchmark result.
    """

    from vertebrae.profiling import resource_profile_like_from_dict
    from vertebrae.reports.recommendations import (
        recommendation_for_extractor,
        recommendations_for_benchmark,
    )
    from vertebrae.results import BenchmarkResult, ExtractorResult
    from vertebrae.scoring.metrics import MetricResult

    score_artifact = store.get_json(score_key)
    _validate_labeled_scoring_artifact_protocol(score_artifact)
    metrics_data = score_artifact["metrics"]
    embedding_metadata = score_artifact.get("embedding_metadata", {})
    label_metadata = score_artifact.get("label_metadata", {})
    group_metadata = score_artifact.get("group_metadata") or {}
    stability_artifact = store.get_json(stability_key) if stability_key else None
    if stability_artifact is not None:
        if stability_artifact.get("embedding_key") != score_artifact.get("embedding_key"):
            raise ValueError("Stability artifact belongs to a different embedding artifact.")
        if stability_artifact.get("labels_key") != score_artifact.get("labels_key"):
            raise ValueError("Stability artifact belongs to a different label artifact.")
        if stability_artifact.get("artifact_type") == "score_collection" and stability_artifact.get(
            "collection_protocol_fingerprint"
        ) != score_artifact.get("collection_protocol_fingerprint"):
            raise ValueError("Score collection belongs to a different evaluation protocol.")
    stability = (
        stability_artifact.get("stability", stability_artifact)
        if stability_artifact is not None
        else None
    )
    separatix_artifact = store.get_json(separatix_key) if separatix_key else None
    if separatix_artifact is not None:
        if separatix_artifact.get("embedding_key") != score_artifact.get("embedding_key"):
            raise ValueError("Separatix artifact belongs to a different embedding artifact.")
        if separatix_artifact.get("labels_key") != score_artifact.get("labels_key"):
            raise ValueError("Separatix artifact belongs to a different label artifact.")
        if separatix_artifact.get("score_key") != score_key:
            raise ValueError("Separatix artifact references a different score artifact.")
    separatix = None
    if separatix_artifact:
        separatix = SeparatixResult(**separatix_artifact["diagnostic"])
    metrics = {name: MetricResult(**data) for name, data in metrics_data.items()}
    overlap = metrics["overlap"]
    score_metadata = overlap.metadata
    weakest_class, weakest_score = _weakest_class(
        overlap.per_class_scores,
        excluded_classes=score_metadata.get("exclude_classes"),
        label_catalog=score_metadata.get("label_catalog"),
    )
    primary_metric_name = score_artifact.get("primary_metric", "overlap")
    recommendation = (
        recommendation_for_extractor(
            overlap.score,
            stability,
            weakest_score,
            target_type=overlap.metadata.get("target_type", "single_label"),
        )
        if primary_metric_name == "overlap" and overlap.metadata.get("aggregate_valid", True)
        else "aggregate_unavailable"
        if primary_metric_name == "overlap"
        else f"ranked_by_{primary_metric_name}"
    )
    compression_metadata = embedding_metadata.get("compression", {"method": "none"})
    base_name = embedding_metadata.get("extractor_recipe", {}).get(
        "name",
        embedding_metadata.get("extractor_name", "artifact"),
    )
    output_name = embedding_metadata.get("output_name")
    if output_name and not str(base_name).endswith(f":{output_name}"):
        base_name = f"{base_name}:{output_name}"
    target_view = label_metadata.get("target_view", embedding_metadata.get("target_view"))
    label_view = label_metadata.get("label_view", embedding_metadata.get("label_view"))
    extractor_result = ExtractorResult(
        name=_variant_extractor_name(
            f"{base_name}{target_view_suffix(target_view)}{label_view_suffix(label_view)}",
            compression_metadata,
        ),
        extractor_type=embedding_metadata.get("extractor_recipe", {}).get(
            "extractor_type",
            embedding_metadata.get("extractor_type", "artifact"),
        ),
        stability=stability,
        separatix=separatix,
        embedding_metadata=embedding_metadata,
        compression_metadata=compression_metadata,
        runtime={},
        warnings=sorted({warning for metric in metrics.values() for warning in metric.warnings}),
        label_view=label_view,
        target_view=target_view,
        weakest_class=weakest_class,
        weakest_class_score=weakest_score,
        recommendation=recommendation,
        metrics=metrics,
        primary_metric_name=primary_metric_name,
        resource_profile=(
            resource_profile_like_from_dict(
                dict(
                    embedding_metadata.get("distributed_resource_profile")
                    or embedding_metadata["resource_profile"]
                )
            )
            if (
                embedding_metadata.get("distributed_resource_profile") is not None
                or embedding_metadata.get("resource_profile") is not None
            )
            else None
        ),
    )
    result = BenchmarkResult(
        dataset_summary={
            "n_samples": label_metadata.get("n_samples", embedding_metadata.get("n_samples")),
            "n_classes": label_metadata.get(
                "n_classes",
                len(label_metadata.get("class_counts", {})),
            ),
            "class_counts": label_metadata.get("class_counts", {}),
            "target_type": label_metadata.get("target_type", "single_label"),
            "label_names": label_metadata.get("label_names"),
            "labelset_counts": label_metadata.get("labelset_counts"),
            "mean_label_cardinality": label_metadata.get("mean_label_cardinality"),
            "label_density": label_metadata.get("label_density"),
            "n_targets": label_metadata.get("n_targets"),
            "target_names": label_metadata.get("target_names"),
            "target_means": label_metadata.get("target_means"),
            "target_variances": label_metadata.get("target_variances"),
            "constant_targets": label_metadata.get("constant_targets"),
            "nonconstant_targets": label_metadata.get("nonconstant_targets"),
            "modality": embedding_metadata.get("modality", "artifact"),
            "target_view": label_metadata.get("target_view"),
            "label_view": label_metadata.get("label_view"),
            "grouped": bool(group_metadata),
            "group_name": group_metadata.get("group_name"),
            "n_groups": group_metadata.get("n_groups"),
        },
        extractor_results=[extractor_result],
        recommendations=recommendations_for_benchmark([extractor_result]),
        metadata={
            "vertebrae_version": __version__,
            "source_score_key": score_key,
            "source_stability_key": stability_key,
            "source_separatix_key": separatix_key,
            "source_groups_key": score_artifact.get("groups_key"),
            "distributed_artifacts": True,
            "resource_profiling_config": embedding_metadata.get("resource_profiling_config", {}),
        },
    )
    payload = result.to_dict()
    if output_key:
        store.put_json(output_key, payload)
    return payload


def retrieval_benchmark_result_from_artifacts(
    score_keys: Iterable[str],
    store: ArtifactStore,
    output_key: Optional[str] = None,
) -> Any:
    """Reconstruct a rankable retrieval result from persisted score artifacts."""

    from vertebrae.retrieval import RetrievalBenchmarkResult, RetrievalExtractorResult
    from vertebrae.scoring.retrieval import RetrievalScoreResult

    keys = list(score_keys)
    artifacts = [store.get_json(key) for key in keys]
    if not artifacts:
        raise ValueError("At least one retrieval score artifact is required.")
    if any(item.get("artifact_type") != "retrieval_evaluation" for item in artifacts):
        raise ValueError("All score keys must reference retrieval evaluation artifacts.")
    protocol_fingerprints = {item.get("protocol_fingerprint") for item in artifacts}
    if len(protocol_fingerprints) != 1 or None in protocol_fingerprints:
        raise ValueError(
            "Retrieval score artifacts must share one non-empty retrieval protocol fingerprint."
        )
    relevance_keys = {item.get("relevance_key") for item in artifacts}
    if len(relevance_keys) != 1 or None in relevance_keys:
        raise ValueError("Retrieval score artifacts must share one relevance artifact.")

    relevance = store.get_json(str(next(iter(relevance_keys))))
    relevance_protocol_fingerprint = relevance.get("protocol_fingerprint")
    if not isinstance(relevance_protocol_fingerprint, str) or not relevance_protocol_fingerprint:
        raise ValueError("Retrieval relevance artifact must declare a protocol fingerprint.")
    retrieval_configs = []
    for artifact in artifacts:
        try:
            retrieval_config = RetrievalConfig(**dict(artifact.get("retrieval_config") or {}))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Retrieval score artifact contains an invalid retrieval_config."
            ) from exc
        exclusions_key = artifact.get("exclusions_key")
        exclusions_data = store.get_json(str(exclusions_key)) if exclusions_key else relevance
        exclusions = sorted(
            (int(pair[0]), int(pair[1])) for pair in exclusions_data.get("exclusions", [])
        )
        expected_protocol_fingerprint = hash_json_exact(
            {
                "identity_schema": 2,
                "relevance_protocol_fingerprint": relevance_protocol_fingerprint,
                "exclusions": exclusions,
                "retrieval_config": retrieval_config,
            }
        )
        if artifact.get("protocol_fingerprint") != expected_protocol_fingerprint:
            raise ValueError(
                "Retrieval score artifact protocol fingerprint is inconsistent with its "
                "relevance, exclusions, or retrieval_config."
            )
        retrieval_configs.append(retrieval_config)
    if any(config != retrieval_configs[0] for config in retrieval_configs[1:]):
        raise ValueError("Retrieval score artifacts must share one retrieval configuration.")

    results = []
    for artifact in artifacts:
        query_endpoint = dict(artifact["query_endpoint"])
        gallery_endpoint = dict(artifact["gallery_endpoint"])
        query_recipe = dict(query_endpoint.get("extractor_recipe") or {})
        gallery_recipe = dict(gallery_endpoint.get("extractor_recipe") or {})
        if query_recipe != gallery_recipe:
            raise ValueError("Retrieval endpoint artifacts must share an extractor recipe.")
        compression = dict(
            query_endpoint.get("compression")
            or gallery_endpoint.get("compression")
            or {"method": "none"}
        )
        forward = RetrievalScoreResult(**dict(artifact["forward"]))
        reverse_payload = artifact.get("reverse")
        reverse = (
            RetrievalScoreResult(**dict(reverse_payload)) if reverse_payload is not None else None
        )
        if bool(artifact.get("retrieval_config", {}).get("bidirectional")) != (reverse is not None):
            raise ValueError(
                "Retrieval artifact reverse result is inconsistent with bidirectional config."
            )
        expected_primary = (
            forward.score if reverse is None else float((forward.score + reverse.score) / 2.0)
        )
        declared_primary = artifact.get("primary_score")
        if not isinstance(
            declared_primary, (int, float, np.integer, np.floating)
        ) or not np.isclose(float(declared_primary), expected_primary):
            raise ValueError(
                "Retrieval artifact primary_score is inconsistent with its directions."
            )
        resource_profiles = {}
        for side, endpoint in (("query", query_endpoint), ("gallery", gallery_endpoint)):
            serialized = endpoint.get("distributed_resource_profile") or endpoint.get(
                "resource_profile"
            )
            if serialized is not None:
                from vertebrae.profiling import resource_profile_like_from_dict

                profile = resource_profile_like_from_dict(dict(serialized))
                if profile is not None:
                    resource_profiles[side] = profile
        base_name = str(
            query_recipe.get("name") or query_endpoint.get("extractor_name") or "artifact"
        )
        results.append(
            RetrievalExtractorResult(
                name=compression_variant_name(base_name, compression),
                extractor_type=query_recipe.get("extractor_type", "artifact"),
                forward=forward,
                reverse=reverse,
                primary_score=expected_primary,
                compression_metadata=compression,
                runtime={},
                warnings=sorted(
                    set(
                        forward.warnings
                        + (reverse.warnings if reverse is not None else [])
                        + list(compression.get("warnings", []))
                    )
                ),
                recipe=query_recipe,
                resource_profiles=resource_profiles,
            )
        )

    first = artifacts[0]
    result = RetrievalBenchmarkResult(
        dataset_summary={
            "modality": "retrieval",
            "n_queries": relevance.get("n_queries", first["query_endpoint"].get("n_samples")),
            "n_gallery": relevance.get("n_gallery", first["gallery_endpoint"].get("n_samples")),
            "query_modality": first["query_endpoint"].get("modality"),
            "gallery_modality": first["gallery_endpoint"].get("modality"),
        },
        extractor_results=results,
        metadata={
            "artifact_backed": True,
            "score_keys": keys,
            "relevance_key": next(iter(relevance_keys)),
            "protocol_fingerprint": next(iter(protocol_fingerprints)),
            "retrieval_config": first.get("retrieval_config", {}),
            "resource_profiling_config": first["query_endpoint"].get(
                "resource_profiling_config", {}
            ),
        },
    )
    if output_key:
        store.put_json(output_key, result.to_dict())
    return result


def _score_summary(scores: list[float], interval_level: float) -> dict[str, float]:
    if not np.isfinite(float(interval_level)) or not 0.0 < float(interval_level) <= 1.0:
        raise ValueError("interval_level must be finite and in (0, 1].")
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("Collected scores must be non-empty finite values.")
    alpha = 1.0 - interval_level
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "lower": float(np.percentile(arr, lower_q)),
        "upper": float(np.percentile(arr, upper_q)),
        "width": float(np.percentile(arr, upper_q) - np.percentile(arr, lower_q)),
    }


def _weakest_class(
    per_class_scores: dict[str, Any],
    excluded_classes: Optional[Any] = None,
    label_catalog: Optional[Any] = None,
) -> tuple[Optional[str], Optional[float]]:
    excluded = (
        []
        if excluded_classes is None
        else [excluded_classes]
        if isinstance(excluded_classes, (str, bytes))
        else list(excluded_classes)
    )
    catalog = label_catalog or []
    catalog_keys = {
        item.get("key")
        for item in catalog
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }
    excluded_keys = {
        item
        if isinstance(item, str) and item in catalog_keys
        else semantic_label_key(item.item() if hasattr(item, "item") else item)
        for item in excluded
    }
    numeric = {}
    for label, score in per_class_scores.items():
        if not isinstance(score, (int, float, np.number)):
            continue
        label_value = label.item() if hasattr(label, "item") else label
        label_key = (
            label_value
            if isinstance(label_value, str) and label_value in catalog_keys
            else semantic_label_key(label_value)
        )
        if label_key not in excluded_keys:
            numeric[label] = float(score)
    if not numeric:
        return None, None
    label, score = min(numeric.items(), key=lambda item: item[1])
    return label_display(label, catalog), score


def _variant_extractor_name(name: str, compression_metadata: dict[str, Any]) -> str:
    return compression_variant_name(name, compression_metadata)


def _materialize_embedding_shard_job(
    job: EmbeddingShardJob,
    store_config: ArtifactStoreConfig,
) -> dict[str, Any]:
    return materialize_embedding_shard(job, create_artifact_store_from_config(store_config))


def _score_embedding_artifact_job(
    job: ScoringJob,
    store_config: ArtifactStoreConfig,
) -> dict[str, Any]:
    return score_embedding_artifact(job, create_artifact_store_from_config(store_config))


def _diagnose_embedding_artifact_job(
    job: SeparatixJob,
    store_config: ArtifactStoreConfig,
) -> dict[str, Any]:
    return diagnose_embedding_artifact(job, create_artifact_store_from_config(store_config))


def _compress_embedding_artifact_job(
    job: CompressionJob,
    store_config: ArtifactStoreConfig,
) -> dict[str, Any]:
    return compress_embedding_artifact(job, create_artifact_store_from_config(store_config))


def _run_stability_artifact_job(
    job: StabilityJob,
    store_config: ArtifactStoreConfig,
) -> dict[str, Any]:
    return run_stability_artifact(job, create_artifact_store_from_config(store_config))


def materialize_embedding_shard(
    job: EmbeddingShardJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Materialize one embedding shard artifact.

    Args:
        job: Embedding shard job.
        store: Artifact store receiving the shard.

    Returns:
        JSON-compatible shard manifest.

    Raises:
        ValueError: If the extractor emits invalid embeddings.
    """

    dataset = job.dataset
    extractor = job.extractor
    sample_indices = job.shard.indices(len(dataset.y))
    if len(sample_indices) == 0:
        raise ValueError("Embedding shard contains no samples.")
    profiler = ResourceProfiler(
        job.resource_profiling_config,
        extractor,
        streaming=True,
        context={
            "measurement_scope": "worker_shard",
            "shard": asdict(job.shard),
            "configured_batch_size": job.batch_size,
        },
    )
    local_positions = {
        int(sample_index): position for position, sample_index in enumerate(sample_indices)
    }
    if _is_multi_output_extractor(extractor):
        return _materialize_multi_output_embedding_shard(
            job=job,
            store=store,
            sample_indices=sample_indices,
            local_positions=local_positions,
            profiler=profiler,
        )
    batches = _local_embedding_batches(dataset, extractor, job, local_positions, profiler)
    manifest = {
        "artifact_type": "embedding_shard",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "dataset_identity_key": dataset.identity_key(),
        "extractor_recipe": extractor.recipe(),
        "recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "shard": asdict(job.shard),
        "sample_indices": sample_indices.tolist(),
        "cache_eligible": job.cache_eligible,
        "cache_status": job.cache_status,
        "batch_size": job.batch_size,
        "resources": asdict(job.resources),
        "resource_profiling_config": asdict(job.resource_profiling_config),
    }

    def finalize_shard_metadata(
        metadata: dict[str, Any], array_manifest: Any, artifact_stat: Any
    ) -> dict[str, Any]:
        profile = profiler.finish() if job.resource_profiling_config.enabled else None
        contract = _embedding_metadata_from_array_manifest(array_manifest)
        profile = with_embedding_footprint(
            profile,
            contract,
            contract,
            raw_stat=artifact_stat,
            evaluated_stat=artifact_stat,
            persisted_storage=job.resource_profiling_config.persisted_storage,
        )
        metadata["resource_profile"] = make_json_safe(profile) if profile is not None else None
        return metadata

    store.put_artifact_batches(
        job.output_key,
        batches,
        n_samples=len(sample_indices),
        metadata=manifest,
        require_complete=True,
        metadata_finalizer=finalize_shard_metadata,
    )
    return store.get_json(job.output_key)


def materialize_retrieval_embedding_shard(
    job: RetrievalEmbeddingShardJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Materialize one deterministic query or gallery embedding shard."""
    dataset = job.dataset
    dataset.validated()
    values = dataset.query_values() if job.side == "query" else dataset.gallery_values()
    query_modality, gallery_modality = dataset.protocol_modalities()
    modality = query_modality if job.side == "query" else gallery_modality
    indices = job.shard.indices(len(values))
    if not len(indices):
        raise ValueError("Retrieval embedding shard contains no samples.")
    profiler = ResourceProfiler(
        job.resource_profiling_config,
        job.extractor,
        streaming=True,
        context={
            "measurement_scope": "worker_shard",
            "endpoint": job.side,
            "branch": job.branch,
            "modality": modality,
            "shard": asdict(job.shard),
            "configured_batch_size": job.batch_size,
        },
    )
    selected = take_endpoint_rows(values, indices)
    if job.branch is None:
        extractor = job.extractor
        if not job.fitted_bundle and getattr(extractor, "already_fitted", True) is False:
            raise ValueError(
                "Distributed retrieval shards require a frozen or already-fitted extractor; "
                "fit it before serializing the extractor pickle."
            )

        def encode_standard(batch: Any) -> Any:
            return extractor.transform(batch)

        encode = encode_standard
    else:
        encoder = getattr(job.extractor, "encode_retrieval", None)
        if not callable(encoder):
            raise TypeError("Retrieval branch materialization requires encode_retrieval().")

        def encode_retrieval(batch: Any) -> Any:
            return encoder(batch, branch=job.branch, modality=modality)

        encode = encode_retrieval
    effective_batch_size = job.batch_size if job.streaming_enabled else len(indices)
    embeddings = encode_endpoint_batches(
        selected,
        batch_size=effective_batch_size,
        encode=encode,
        owner=f"Retriever '{job.extractor.name}' {job.side} shard embeddings",
        profiler=profiler if job.resource_profiling_config.enabled else None,
        call_type=f"encode_retrieval_{job.side}",
    )
    if embeddings.shape[0] != len(indices):
        raise ValueError("Retrieval extractor output does not align with its endpoint shard.")
    sparse = is_sparse_matrix(embeddings)
    manifest = {
        "artifact_type": "retrieval_embedding_shard",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "dataset_identity_key": dataset.identity_key(),
        "extractor_recipe": job.extractor.recipe(),
        "recipe_hash": fingerprint_extractor_recipe(job.extractor.recipe()),
        "side": job.side,
        "branch": job.branch,
        "modality": modality,
        "shard": asdict(job.shard),
        "sample_indices": indices.tolist(),
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "sparse": sparse,
        "nnz": int(embeddings.nnz) if sparse else None,
        "storage_format": sparse_storage_format(embeddings) if sparse else "dense",
        "cache_eligible": job.cache_eligible,
        "cache_status": job.cache_status,
        "batch_size": job.batch_size,
        "resource_profiling_config": asdict(job.resource_profiling_config),
    }

    def finalize_retrieval_shard_metadata(
        metadata: dict[str, Any], _array_manifest: Any, artifact_stat: Any
    ) -> dict[str, Any]:
        profile = profiler.finish() if job.resource_profiling_config.enabled else None
        profile = with_embedding_footprint(
            profile,
            embeddings,
            embeddings,
            raw_stat=artifact_stat,
            evaluated_stat=artifact_stat,
            persisted_storage=job.resource_profiling_config.persisted_storage,
        )
        metadata["resource_profile"] = make_json_safe(profile) if profile is not None else None
        return metadata

    store.put_artifact(
        job.output_key,
        embeddings,
        manifest,
        metadata_finalizer=finalize_retrieval_shard_metadata,
    )
    return store.get_json(job.output_key)


def compress_retrieval_embedding_artifacts(
    job: RetrievalCompressionJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Fit one compressor on gallery embeddings and transform both retrieval endpoints."""
    query, query_metadata = store.get_artifact(job.query_embedding_key)
    gallery, gallery_metadata = store.get_artifact(job.gallery_embedding_key)
    _validate_retrieval_pair_metadata(query_metadata, gallery_metadata)
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("Paired retrieval compression requires matching endpoint dimensions.")
    config = job.compression_config
    gallery_result, query_result, metadata = compress_embedding_pair(
        gallery,
        query,
        config,
        fit_name="gallery embeddings",
        paired_name="query embeddings",
    )
    metadata["fit_side"] = "gallery"
    cache_eligible, cache_status = derived_cache_reuse_decision(
        query_metadata,
        gallery_metadata,
    )
    manifests: list[dict[str, Any]] = []
    for key, values, source_metadata, side in (
        (job.query_output_key, query_result, query_metadata, "query"),
        (job.gallery_output_key, gallery_result, gallery_metadata, "gallery"),
    ):
        manifest = dict(source_metadata)
        manifest.update(
            {
                "artifact_type": "retrieval_compressed_embedding",
                "output_key": key,
                "side": side,
                "n_samples": int(values.shape[0]),
                "embedding_dim": int(values.shape[1]),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "sparse": is_sparse_matrix(values),
                "nnz": int(values.nnz) if is_sparse_matrix(values) else None,
                "storage_format": (
                    sparse_storage_format(values) if is_sparse_matrix(values) else "dense"
                ),
                "compression": metadata,
                "cache_eligible": cache_eligible,
                "cache_status": cache_status,
            }
        )
        serialized_profile = source_metadata.get("distributed_resource_profile")
        source_values = query if side == "query" else gallery
        source_key = job.query_embedding_key if side == "query" else job.gallery_embedding_key

        def finalize_retrieval_compression_metadata(
            candidate: dict[str, Any],
            _array_manifest: Any,
            artifact_stat: Any,
            serialized: Any = serialized_profile,
            raw_values: Any = source_values,
            evaluated_values: Any = values,
            raw_key: str = source_key,
            source_manifest: dict[str, Any] = source_metadata,
        ) -> dict[str, Any]:
            if serialized is not None:
                distributed_profile = with_distributed_embedding_footprint(
                    distributed_resource_profile_from_dict(dict(serialized)),
                    raw_values,
                    evaluated_values,
                    store=store,
                    raw_key=raw_key,
                    evaluated_stat=artifact_stat,
                    persisted_storage=bool(
                        source_manifest.get("resource_profiling_config", {}).get(
                            "persisted_storage", True
                        )
                    ),
                )
                candidate["distributed_resource_profile"] = make_json_safe(distributed_profile)
            return candidate

        store.put_artifact(
            key,
            values,
            manifest,
            metadata_finalizer=finalize_retrieval_compression_metadata,
        )
        manifests.append(store.get_json(key))
    artifact = {
        "artifact_type": "retrieval_compression",
        "query_output_key": job.query_output_key,
        "gallery_output_key": job.gallery_output_key,
        "query_embedding_key": job.query_embedding_key,
        "gallery_embedding_key": job.gallery_embedding_key,
        "compression_metadata": metadata,
        "cache_eligible": cache_eligible,
        "cache_status": cache_status,
    }
    prefix = _shared_artifact_prefix(job.query_output_key, job.gallery_output_key)
    if prefix:
        store.put_json(prefix, artifact)
    return artifact


def merge_embedding_shards(
    job: EmbeddingMergeJob,
    store: ArtifactStore,
    *,
    metadata_updates: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Merge embedding shard artifacts into a complete embedding artifact.

    Args:
        job: Embedding merge job.
        store: Artifact store containing shards and receiving the merge.

    Returns:
        JSON-compatible merged embedding manifest.

    Raises:
        ValueError: If shard metadata is inconsistent, duplicated, or incomplete.
    """

    manifests = [store.get_json(key) for key in job.shard_keys]
    if manifests and manifests[0].get("artifact_type") == "multi_output_embedding_shard":
        if metadata_updates:
            raise ValueError("Multi-output merges do not accept top-level metadata updates.")
        return _merge_multi_output_embedding_shards(job, store, manifests)
    _validate_shard_manifests(manifests, expected_n_samples=job.n_samples)

    def committed_batches() -> Iterator[tuple[np.ndarray, Any]]:
        for expected in manifests:
            values, committed = store.get_artifact(expected["output_key"])
            if committed != expected:
                raise ValueError(
                    "Embedding shard changed after merge validation; retry from a stable "
                    "set of committed shard generations."
                )
            yield np.asarray(committed["sample_indices"], dtype=int), values

    batches = committed_batches()
    manifest = {
        "artifact_type": "embedding",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "shard_keys": list(job.shard_keys),
        "n_shards": len(job.shard_keys),
        "resources": asdict(job.resources),
    }
    first = manifests[0]
    for key in (
        "dataset_identity_key",
        "extractor_recipe",
        "recipe_hash",
        "output_name",
        "output_recipe",
        "output_metadata",
        "cache_eligible",
        "cache_status",
    ):
        manifest[key] = first.get(key)
    if metadata_updates:
        manifest.update(metadata_updates)
    serialized_profiles = [
        (item["output_key"], item.get("resource_profile"))
        for item in manifests
        if item.get("resource_profile") is not None
    ]
    config = dict(first.get("resource_profiling_config") or {})
    if serialized_profiles:
        manifest["resource_profiling_config"] = config

    def finalize_merged_metadata(
        metadata: dict[str, Any], array_manifest: Any, artifact_stat: Any
    ) -> dict[str, Any]:
        if serialized_profiles:
            contract = _embedding_metadata_from_array_manifest(array_manifest)
            distributed_profile = aggregate_distributed_resource_profiles(
                [
                    (key, resource_profile_from_dict(dict(profile or {})))
                    for key, profile in serialized_profiles
                ],
                merged_embeddings=contract,
                all_shard_keys=[item["output_key"] for item in manifests],
                merged_stat=artifact_stat,
                persisted_storage=bool(config.get("persisted_storage", True)),
            )
            metadata["distributed_resource_profile"] = make_json_safe(distributed_profile)
        return metadata

    store.put_artifact_batches(
        job.output_key,
        batches,
        n_samples=job.n_samples,
        metadata=manifest,
        require_complete=True,
        metadata_finalizer=finalize_merged_metadata,
    )
    return store.get_json(job.output_key)


def merge_retrieval_embedding_shards(
    job: EmbeddingMergeJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Merge retrieval endpoint shards and preserve their endpoint identity."""
    shards = [store.get_json(key) for key in job.shard_keys]
    _validate_retrieval_shard_manifests(shards, expected_n_samples=job.n_samples)
    sides = {shard.get("side") for shard in shards}
    branches = {shard.get("branch") for shard in shards}
    modalities = {shard.get("modality") for shard in shards}
    if len(sides) != 1 or len(branches) != 1 or len(modalities) != 1:
        raise ValueError("Retrieval endpoint shards must share one side, branch, and modality.")
    return merge_embedding_shards(
        job,
        store,
        metadata_updates={
            "artifact_type": "retrieval_embedding",
            "side": sides.pop(),
            "branch": branches.pop(),
            "modality": modalities.pop(),
            "cache_eligible": shards[0].get("cache_eligible", True),
            "cache_status": shards[0].get("cache_status", "miss"),
        },
    )


def plan_retrieval_embedding_shard_jobs(
    dataset: Any,
    extractor: Any,
    total_shards: int,
    *,
    side: str,
    branch: Optional[str] = None,
    batch_size: int = 128,
    resource_profiling_config: Optional[ResourceProfilingConfig] = None,
    output_key: Optional[str] = None,
    run_prefix: Optional[str] = None,
) -> list[RetrievalEmbeddingShardJob]:
    """Plan deterministic embedding jobs for one retrieval endpoint."""
    if isinstance(total_shards, bool) or not isinstance(total_shards, (int, np.integer)):
        raise ValueError("total_shards must be an integer >= 1.")
    if int(total_shards) < 1:
        raise ValueError("total_shards must be >= 1.")
    if side not in {"query", "gallery"}:
        raise ValueError("side must be 'query' or 'gallery'.")
    values = dataset.query_values() if side == "query" else dataset.gallery_values()
    n_samples = int(len(values))
    if n_samples < 1:
        raise ValueError(f"Cannot plan retrieval shards for an empty {side} endpoint.")
    streaming_enabled = bool(getattr(extractor, "streaming_safe", False))
    planned_shards = min(int(total_shards), n_samples) if streaming_enabled else 1
    base_key, cache_eligible, cache_status = _execution_artifact_identity(
        retrieval_embedding_artifact_key(dataset, extractor, side, branch),
        extractor.recipe(),
        output_key=output_key,
        run_prefix=run_prefix,
    )
    return [
        RetrievalEmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            side=side,
            branch=branch,
            shard=ShardSpec(total_shards=planned_shards, shard_index=index),
            output_key=retrieval_embedding_shard_key(
                base_key, ShardSpec(total_shards=planned_shards, shard_index=index)
            ),
            batch_size=batch_size,
            streaming_enabled=streaming_enabled,
            cache_eligible=cache_eligible,
            cache_status=cache_status,
            resource_profiling_config=(resource_profiling_config or ResourceProfilingConfig()),
        )
        for index in range(planned_shards)
    ]


def _local_embedding_batches(
    dataset: Any,
    extractor: Any,
    job: EmbeddingShardJob,
    local_positions: dict[int, int],
    profiler: ResourceProfiler,
) -> Iterator[Tuple[np.ndarray, Any]]:
    batch_size = job.batch_size if job.streaming_enabled else len(dataset.y)
    for batch in dataset.iter_batches(batch_size=batch_size, shard=job.shard):

        def call(batch: Any = batch) -> Any:
            return extractor.transform(batch.X)

        raw_embeddings = (
            profiler.measure_call(
                call,
                samples=len(batch.indices),
                call_type="transform",
            )
            if job.resource_profiling_config.enabled
            else call()
        )
        embeddings = ensure_numeric_matrix(
            raw_embeddings,
            f"Extractor '{extractor.name}' shard embeddings",
            allow_sparse=True,
        )
        if embeddings.shape[0] != len(batch.indices):
            raise ValueError(
                f"Extractor '{extractor.name}' returned {embeddings.shape[0]} embeddings "
                f"for a shard batch with {len(batch.indices)} samples."
            )
        indices = np.asarray([local_positions[int(index)] for index in batch.indices], dtype=int)
        yield indices, embeddings


def _materialize_multi_output_embedding_shard(
    job: EmbeddingShardJob,
    store: ArtifactStore,
    sample_indices: np.ndarray,
    local_positions: dict[int, int],
    profiler: ResourceProfiler,
) -> dict[str, Any]:
    dataset = job.dataset
    extractor = job.extractor
    output_specs = _multi_output_specs(extractor)
    output_recipes: Dict[str, dict[str, Any]] = {}
    output_metadata: Dict[str, dict[str, Any]] = {}
    batch_size = job.batch_size if job.streaming_enabled else len(dataset.y)
    manifests = []
    with (
        IncrementalMatrixStager(
            job.memory_config,
            purpose=f"Extractor '{extractor.name}' distributed multi-output staging",
        ) as stager,
        IncrementalMatrixReferenceStager(
            job.memory_config,
            purpose=f"Extractor '{extractor.name}' distributed multi-output staging",
            matrix_stager=stager,
        ) as reference_stager,
    ):
        for batch in dataset.iter_batches(batch_size=batch_size, shard=job.shard):

            def call(batch: Any = batch) -> Any:
                return _validated_multi_outputs(
                    extractor,
                    batch.X,
                    len(batch.indices),
                )

            outputs = (
                profiler.measure_call(
                    call,
                    samples=len(batch.indices),
                    call_type="transform_many",
                )
                if job.resource_profiling_config.enabled
                else call()
            )
            positions = [local_positions[int(index)] for index in batch.indices]
            for output in outputs:
                recipe = dict(output.recipe)
                metadata = dict(output.metadata)
                if output.name in output_recipes and output_recipes[output.name] != recipe:
                    raise ValueError(
                        f"Extractor output {output.name!r} changed its recipe between batches."
                    )
                if output.name in output_metadata and output_metadata[output.name] != metadata:
                    raise ValueError(
                        f"Extractor output {output.name!r} changed its metadata between batches."
                    )
                output_recipes[output.name] = recipe
                output_metadata[output.name] = metadata
                for row_index, position in enumerate(positions):
                    row = output.embeddings[row_index : row_index + 1]
                    reference_stager.append(
                        output.name,
                        position,
                        stager.append(output.name, row),
                    )

        base_profile = profiler.finish() if job.resource_profiling_config.enabled else None
        output_keys = named_output_artifact_keys(
            job.output_key, (spec["name"] for spec in output_specs)
        )
        for spec in output_specs:
            output_name = spec["name"]
            assembly = reference_stager.assemble(
                output_name,
                expected_rows=len(sample_indices),
                purpose=f"Extractor '{extractor.name}' output '{output_name}' shard",
            )
            embeddings = assembly.matrix
            output_key = output_keys[output_name]
            sparse_embeddings = is_sparse_matrix(embeddings)
            output_manifest = {
                "artifact_type": "embedding_shard",
                "vertebrae_version": __version__,
                "output_key": output_key,
                "dataset_identity_key": dataset.identity_key(),
                "extractor_recipe": extractor.recipe(),
                "recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
                "output_name": output_name,
                "output_recipe": output_recipes.get(output_name, {}),
                "output_metadata": output_metadata.get(output_name, {}),
                "shard": asdict(job.shard),
                "sample_indices": sample_indices.tolist(),
                "n_samples": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "sparse": sparse_embeddings,
                "nnz": int(embeddings.nnz) if sparse_embeddings else None,
                "storage_format": (
                    sparse_storage_format(embeddings) if sparse_embeddings else "dense"
                ),
                "cache_eligible": job.cache_eligible,
                "cache_status": job.cache_status,
                "memory_staging": {
                    "strategy": assembly.strategy,
                    "required_bytes": assembly.required_bytes,
                    "budget_bytes": assembly.budget_bytes,
                    "staging_strategy": assembly.staging_strategy,
                },
                "batch_size": job.batch_size,
                "resources": asdict(job.resources),
                "resource_profiling_config": asdict(job.resource_profiling_config),
            }

            def finalize_multi_output_metadata(
                metadata: dict[str, Any],
                _array_manifest: Any,
                artifact_stat: Any,
                embedded_values: Any = embeddings,
            ) -> dict[str, Any]:
                output_profile = with_embedding_footprint(
                    base_profile,
                    embedded_values,
                    embedded_values,
                    raw_stat=artifact_stat,
                    evaluated_stat=artifact_stat,
                    persisted_storage=job.resource_profiling_config.persisted_storage,
                )
                metadata["resource_profile"] = (
                    make_json_safe(output_profile) if output_profile is not None else None
                )
                return metadata

            store.put_artifact(
                output_key,
                embeddings,
                output_manifest,
                metadata_finalizer=finalize_multi_output_metadata,
            )
            manifests.append(store.get_json(output_key))
            del embeddings, assembly

    bundle_manifest = {
        "artifact_type": "multi_output_embedding_shard",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "dataset_identity_key": dataset.identity_key(),
        "extractor_recipe": extractor.recipe(),
        "recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "shard": asdict(job.shard),
        "sample_indices": sample_indices.tolist(),
        "n_samples": int(len(sample_indices)),
        "batch_size": job.batch_size,
        "resources": asdict(job.resources),
        "resource_profiling_config": asdict(job.resource_profiling_config),
        "resource_profile": (make_json_safe(base_profile) if base_profile is not None else None),
        "cache_eligible": job.cache_eligible,
        "cache_status": job.cache_status,
        "outputs": [
            {
                "output_name": manifest["output_name"],
                "output_key": manifest["output_key"],
                "embedding_dim": manifest["embedding_dim"],
                "shape": manifest["shape"],
                "dtype": manifest["dtype"],
                "sparse": manifest["sparse"],
                "nnz": manifest["nnz"],
                "storage_format": manifest["storage_format"],
                "output_recipe": manifest.get("output_recipe", {}),
                "output_metadata": manifest.get("output_metadata", {}),
            }
            for manifest in manifests
        ],
    }
    store.put_json(job.output_key, bundle_manifest)
    return bundle_manifest


def _merge_multi_output_embedding_shards(
    job: EmbeddingMergeJob,
    store: ArtifactStore,
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    output_names = _validate_multi_output_shard_manifests(manifests)
    output_keys = named_output_artifact_keys(job.output_key, output_names)
    output_manifests = []
    for output_name in output_names:
        shard_keys = []
        for manifest in manifests:
            output = _find_output_manifest_entry(manifest, output_name)
            shard_keys.append(output["output_key"])
        output_key = output_keys[output_name]
        merged = merge_embedding_shards(
            EmbeddingMergeJob(
                shard_keys=tuple(shard_keys),
                output_key=output_key,
                n_samples=job.n_samples,
                resources=job.resources,
            ),
            store,
        )
        output_manifests.append(
            {
                "output_name": output_name,
                "output_key": merged["output_key"],
                "artifact_path": merged["artifact_path"],
                "shape": merged["shape"],
                "n_samples": merged["n_samples"],
                "embedding_dim": merged["embedding_dim"],
                "dtype": merged["dtype"],
                "sparse": merged["sparse"],
                "nnz": merged["nnz"],
                "storage_format": merged["storage_format"],
                "output_recipe": merged.get("output_recipe", {}),
                "output_metadata": merged.get("output_metadata", {}),
                "dataset_identity_key": merged.get("dataset_identity_key"),
                "distributed_resource_profile": merged.get("distributed_resource_profile"),
            }
        )
    first = manifests[0]
    bundle_manifest = {
        "artifact_type": "multi_output_embedding",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "shard_keys": list(job.shard_keys),
        "n_shards": len(job.shard_keys),
        "n_samples": int(job.n_samples),
        "dataset_identity_key": first.get("dataset_identity_key"),
        "extractor_recipe": first.get("extractor_recipe"),
        "recipe_hash": first.get("recipe_hash"),
        "cache_eligible": first.get("cache_eligible"),
        "cache_status": first.get("cache_status"),
        "resources": asdict(job.resources),
        "resource_profiling_config": first.get("resource_profiling_config", {}),
        "outputs": output_manifests,
    }
    store.put_json(job.output_key, bundle_manifest)
    return bundle_manifest


def _is_multi_output_extractor(extractor: Any) -> bool:
    if not callable(getattr(extractor, "transform_many", None)):
        return False
    if not callable(getattr(extractor, "output_specs", None)):
        return False
    return len(list(extractor.output_specs())) > 1


def _multi_output_specs(extractor: Any) -> list[dict[str, Any]]:
    specs = []
    for spec in extractor.output_specs():
        specs.append(
            {
                "name": spec.name,
                "pooling": getattr(spec, "pooling", None),
                "hidden_layer": getattr(spec, "hidden_layer", None),
                "metadata": dict(getattr(spec, "metadata", {}) or {}),
            }
        )
    return specs


def _validated_multi_outputs(
    extractor: Any,
    X: Any,
    expected_rows: int,
) -> list[EmbeddingOutput]:
    outputs = list(extractor.transform_many(X))
    expected_names = [spec["name"] for spec in _multi_output_specs(extractor)]
    actual_names = [output.name for output in outputs]
    if set(actual_names) != set(expected_names):
        raise ValueError(
            f"Extractor '{extractor.name}' returned outputs {sorted(actual_names)}, expected "
            f"{sorted(expected_names)}."
        )
    validated = []
    for output in outputs:
        embeddings = ensure_numeric_matrix(
            output.embeddings,
            f"Extractor '{extractor.name}' output '{output.name}' shard embeddings",
            allow_sparse=True,
        )
        if embeddings.shape[0] != expected_rows:
            raise ValueError(
                f"Extractor '{extractor.name}' output '{output.name}' returned "
                f"{embeddings.shape[0]} embeddings for a shard batch with {expected_rows} samples."
            )
        validated.append(
            EmbeddingOutput(
                name=output.name,
                embeddings=embeddings,
                recipe=dict(output.recipe),
                metadata=dict(output.metadata),
            )
        )
    validated.sort(key=lambda item: expected_names.index(item.name))
    return validated


def _validate_multi_output_shard_manifests(manifests: list[dict[str, Any]]) -> list[str]:
    if not manifests:
        raise ValueError("At least one shard manifest is required.")
    output_names: Optional[list[str]] = None
    for manifest in manifests:
        names = [str(output["output_name"]) for output in manifest.get("outputs", [])]
        if output_names is None:
            output_names = names
        elif names != output_names:
            raise ValueError("Multi-output embedding shards have inconsistent output names.")
    return output_names or []


def _find_output_manifest_entry(manifest: dict[str, Any], output_name: str) -> dict[str, Any]:
    for output in manifest.get("outputs", []):
        if output.get("output_name") == output_name:
            return output
    raise ValueError(f"Output '{output_name}' was not found in shard manifest.")


def _validate_shard_manifests(
    manifests: Iterable[dict[str, Any]],
    expected_n_samples: int,
) -> None:
    manifest_list = list(manifests)
    if not manifest_list:
        raise ValueError("At least one shard manifest is required.")
    recipe_hashes = {manifest.get("recipe_hash") for manifest in manifest_list}
    dataset_identity_keys = {manifest.get("dataset_identity_key") for manifest in manifest_list}
    dtypes = {manifest.get("dtype") for manifest in manifest_list}
    sparse_values = {manifest.get("sparse") for manifest in manifest_list}
    dims = {manifest.get("embedding_dim") for manifest in manifest_list}
    cache_protocols = {
        (manifest.get("cache_eligible"), manifest.get("cache_status")) for manifest in manifest_list
    }
    if len(recipe_hashes) != 1:
        raise ValueError("Embedding shards have inconsistent extractor recipes.")
    if len(dataset_identity_keys) != 1:
        raise ValueError("Embedding shards have inconsistent dataset identities.")
    if len(dtypes) != 1 or len(sparse_values) != 1 or len(dims) != 1:
        raise ValueError("Embedding shards have inconsistent embedding formats.")
    if len(cache_protocols) != 1:
        raise ValueError("Embedding shards have inconsistent cache identity policies.")
    output_names = {manifest.get("output_name") for manifest in manifest_list}
    output_recipe_hashes = {
        hash_json_exact(manifest.get("output_recipe", {})) for manifest in manifest_list
    }
    output_metadata_hashes = {
        hash_json_exact(manifest.get("output_metadata", {})) for manifest in manifest_list
    }
    if len(output_names) != 1 or len(output_recipe_hashes) != 1 or len(output_metadata_hashes) != 1:
        raise ValueError("Embedding shards have inconsistent output-specific metadata.")

    written = np.zeros(expected_n_samples, dtype=bool)
    for manifest in manifest_list:
        indices = np.asarray(manifest.get("sample_indices", []), dtype=int)
        if len(indices) != int(manifest.get("n_samples", -1)):
            raise ValueError("Shard sample index count does not match shard row count.")
        if np.any(indices < 0) or np.any(indices >= expected_n_samples):
            raise ValueError("Shard sample indices are outside the expected dataset bounds.")
        if np.any(written[indices]):
            duplicates = indices[written[indices]]
            raise ValueError(f"Duplicate embedding rows for sample indices {duplicates[:10]}.")
        written[indices] = True
    if not bool(np.all(written)):
        missing = np.flatnonzero(~written)
        raise ValueError(f"Embedding shards did not cover all samples; missing {missing[:10]}.")


def _validate_retrieval_shard_manifests(
    manifests: Iterable[dict[str, Any]], expected_n_samples: int
) -> None:
    """Validate endpoint identity before a retrieval merge writes any output."""
    manifest_list = list(manifests)
    if any(
        manifest.get("artifact_type") != "retrieval_embedding_shard" for manifest in manifest_list
    ):
        raise ValueError("Retrieval merges require retrieval_embedding_shard artifacts.")
    _validate_shard_manifests(manifest_list, expected_n_samples)
    sides = {manifest.get("side") for manifest in manifest_list}
    branches = {manifest.get("branch") for manifest in manifest_list}
    modalities = {manifest.get("modality") for manifest in manifest_list}
    if len(sides) != 1 or sides == {None}:
        raise ValueError("Retrieval endpoint shards must share one valid side.")
    if len(branches) != 1:
        raise ValueError("Retrieval endpoint shards must share one branch.")
    if len(modalities) != 1 or modalities == {None}:
        raise ValueError("Retrieval endpoint shards must share one valid modality.")


def _validate_retrieval_pair_metadata(
    query_metadata: dict[str, Any],
    gallery_metadata: dict[str, Any],
    relevance_metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Ensure paired artifacts belong to one retrieval dataset and extractor recipe."""
    if query_metadata.get("side") != "query" or gallery_metadata.get("side") != "gallery":
        raise ValueError("Retrieval artifacts must declare query and gallery endpoint sides.")
    query_identity_key = query_metadata.get("dataset_identity_key")
    gallery_identity_key = gallery_metadata.get("dataset_identity_key")
    if not query_identity_key or query_identity_key != gallery_identity_key:
        raise ValueError("Query and gallery artifacts have incompatible dataset identities.")
    query_recipe = query_metadata.get("recipe_hash")
    gallery_recipe = gallery_metadata.get("recipe_hash")
    if not query_recipe or query_recipe != gallery_recipe:
        raise ValueError("Query and gallery artifacts have incompatible extractor recipes.")
    if relevance_metadata is not None:
        relevance_identity_key = relevance_metadata.get("dataset_identity_key")
        if not relevance_identity_key or relevance_identity_key != query_identity_key:
            raise ValueError("Relevance artifact has an incompatible dataset identity.")
        if relevance_metadata.get("protocol_fingerprint") != relevance_identity_key:
            raise ValueError(
                "Relevance artifact protocol_fingerprint does not match its dataset identity."
            )


def _transpose_retrieval_relations(
    relevance: dict[int, dict[int, float]],
    exclusions: set[Tuple[int, int]],
    n_gallery: int,
) -> tuple[dict[int, dict[int, float]], set[Tuple[int, int]]]:
    """Transpose a complete query-gallery protocol for reverse evaluation."""

    transposed: dict[int, dict[int, float]] = {index: {} for index in range(n_gallery)}
    for query_index, values in relevance.items():
        for gallery_index, grade in values.items():
            transposed[gallery_index][query_index] = grade
    reverse_exclusions = {(gallery_index, query_index) for query_index, gallery_index in exclusions}
    return transposed, reverse_exclusions


def _shared_artifact_prefix(query_key: str, gallery_key: str) -> Optional[str]:
    """Return a paired endpoint prefix when output keys use the standard layout."""
    if query_key.endswith("/query") and gallery_key == f"{query_key[:-6]}/gallery":
        return query_key[:-6]
    return None
