"""Distributed embedding primitives."""

from dataclasses import asdict
from functools import partial
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np

from vertebrae import __version__
from vertebrae.cache import (
    ArtifactStore,
    ArtifactStoreConfig,
    create_artifact_store_from_config,
)
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe, hash_json
from vertebrae.compression import (
    compress_embedding_artifact_key,
    compress_embeddings,
    compression_variant_name,
)
from vertebrae.compression.paired import compress_embedding_pair
from vertebrae.config import ResourceProfilingConfig
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
    REGRESSION_TARGET,
    label_view_suffix,
    target_summary,
    target_view_suffix,
)
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


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
    if recipe.get("cache_safe") is False:
        raise ValueError(
            "Cannot plan canonical retrieval artifacts for a callable extractor without "
            "portable callable paths or cache_identity."
        )
    recipe_hash = fingerprint_extractor_recipe(recipe)
    branch_key = "default" if branch is None else hash_json({"branch": branch})
    return f"retrieval/embeddings/{dataset.identity_key()}/{recipe_hash}/{side}/{branch_key}"


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

    safe_name = str(output_name).replace("/", "_")
    return f"{base_key}/outputs/{safe_name}"


def embedding_output_shard_key(shard_key: str, output_name: str) -> str:
    """Build an artifact key for one named embedding shard output."""

    safe_name = str(output_name).replace("/", "_")
    return f"{shard_key}/outputs/{safe_name}"


def labels_artifact_key(dataset: Any) -> str:
    """Build the canonical label artifact key.

    Args:
        dataset: Dataset object with an `identity_key()` method.

    Returns:
        Artifact key for labels.
    """

    return f"labels/{dataset.identity_key()}"


def groups_artifact_key(dataset: Any) -> str:
    """Build the canonical independence-group artifact key."""

    return f"groups/{dataset.identity_key()}"


def scoring_artifact_key(embedding_key: str, seed: Any = None) -> str:
    """Build a scoring artifact key.

    Args:
        embedding_key: Complete embedding artifact key.
        seed: Optional scoring seed.

    Returns:
        Artifact key for scoring output.
    """

    suffix = "default" if seed is None else f"seed-{seed}"
    return f"{embedding_key}/scores/{suffix}"


def retrieval_scoring_artifact_key(query_embedding_key: str, gallery_embedding_key: str) -> str:
    """Build a stable key for a query--gallery retrieval score artifact."""
    from vertebrae.cache.fingerprint import hash_json

    identity = {"query": query_embedding_key, "gallery": gallery_embedding_key}
    return f"retrieval/scores/{hash_json(identity)}"


def retrieval_compression_artifact_key(
    query_embedding_key: str, gallery_embedding_key: str, config: Any
) -> str:
    """Build a paired compression artifact prefix."""
    from vertebrae.compression import compression_recipe_hash

    identity = hash_json({"query": query_embedding_key, "gallery": gallery_embedding_key})
    return f"retrieval/compressions/{identity}/{compression_recipe_hash(config)}"


def separatix_artifact_key(embedding_key: str) -> str:
    """Build a Separatix diagnostic artifact key."""

    return f"{embedding_key}/diagnostics/separatix"


def materialize_segmentation_artifacts(
    dataset: Any,
    extractor: Any,
    store: ArtifactStore,
    segmentation_config: Any = None,
    batch_size: int = 16,
    resource_profiling_config: Optional[ResourceProfilingConfig] = None,
) -> dict[str, Any]:
    """Materialize spatial segmentation outputs into standard artifact boundaries."""

    from vertebrae.segmentation import materialize_segmentation_outputs

    recipe = extractor.recipe()
    resource_config = resource_profiling_config or ResourceProfilingConfig()
    profiler = ResourceProfiler(
        resource_config,
        extractor,
        streaming=True,
        context={"measurement_scope": "artifact_materialization", "modality": "segmentation"},
    )
    base_key = f"segmentation/{dataset.identity_key()}/" f"{fingerprint_extractor_recipe(recipe)}"
    outputs = []
    materializations = list(
        materialize_segmentation_outputs(
            dataset,
            extractor,
            config=segmentation_config,
            batch_size=batch_size,
            resource_profiler=profiler if resource_config.enabled else None,
        )
    )
    shared_profile = profiler.finish() if resource_config.enabled else None
    for materialization in materializations:
        safe_name = materialization.name.replace("/", "_")
        output_key = f"{base_key}/outputs/{safe_name}"
        labels_key = f"{output_key}/labels"
        groups_key = f"{output_key}/groups"
        provenance_key = f"{output_key}/provenance"
        embeddings = materialization.dataset.X
        labels = materialization.dataset.y
        groups = materialization.dataset.groups()
        if groups is None:
            raise ValueError("Segmentation materialization must define image groups.")
        artifact_path = store.put_array(output_key, embeddings)
        embedding_manifest = {
            "artifact_type": "segmentation_embedding",
            "vertebrae_version": __version__,
            "output_key": output_key,
            "artifact_path": artifact_path,
            "dataset_identity_key": materialization.dataset.identity_key(),
            "source_dataset_identity_key": dataset.identity_key(),
            "extractor_recipe": recipe,
            "recipe_hash": fingerprint_extractor_recipe(recipe),
            "output_name": materialization.name,
            "n_samples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "sparse": False,
            "storage_format": "dense",
            "modality": "segmentation",
            "segmentation": materialization.metadata,
            "labels_key": labels_key,
            "groups_key": groups_key,
            "provenance_key": provenance_key,
            "resource_profiling_config": asdict(resource_config),
        }
        profile = with_embedding_footprint(
            shared_profile,
            embeddings,
            embeddings,
            store=store,
            raw_key=output_key,
            evaluated_key=output_key,
            persisted_storage=resource_config.persisted_storage,
        )
        if profile is not None:
            embedding_manifest["resource_profile"] = make_json_safe(profile)
        store.put_json(output_key, embedding_manifest)
        label_path = store.put_labels(labels_key, labels)
        label_summary = target_summary(
            labels,
            target_type=materialization.dataset.metadata.get("target_type", "auto"),
            target_names=materialization.dataset.metadata.get("target_names"),
        )
        store.put_json(
            labels_key,
            {
                "artifact_type": "labels",
                "vertebrae_version": __version__,
                "output_key": labels_key,
                "artifact_path": label_path,
                "dataset_identity_key": materialization.dataset.identity_key(),
                "n_samples": int(len(labels)),
                "target_type": label_summary["target_type"],
                "class_counts": make_json_safe(label_summary["class_counts"]),
                "n_classes": label_summary["n_classes"],
                "target_view": materialization.dataset.active_target_view(),
                "label_view": materialization.dataset.active_label_view(),
            },
        )
        group_path = store.put_labels(groups_key, groups)
        store.put_json(
            groups_key,
            {
                "artifact_type": "groups",
                "vertebrae_version": __version__,
                "output_key": groups_key,
                "artifact_path": group_path,
                "dataset_identity_key": materialization.dataset.identity_key(),
                "n_samples": int(len(groups)),
                "n_groups": int(len(np.unique(groups))),
                "group_name": "image_id",
            },
        )
        store.put_json(provenance_key, {"rows": materialization.provenance})
        outputs.append(embedding_manifest)
    bundle = {
        "artifact_type": "segmentation_embedding_bundle",
        "vertebrae_version": __version__,
        "output_key": base_key,
        "dataset_identity_key": dataset.identity_key(),
        "extractor_recipe": recipe,
        "resource_profiling_config": asdict(resource_config),
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
) -> dict[str, Any]:
    """Materialize structured unit outputs into standard artifact boundaries."""

    from vertebrae.structured import materialize_structured_outputs

    recipe = extractor.recipe()
    resource_config = resource_profiling_config or ResourceProfilingConfig()
    profiler = ResourceProfiler(
        resource_config,
        extractor,
        streaming=True,
        context={"measurement_scope": "artifact_materialization", "modality": "structured"},
    )
    base_key = f"structured/{dataset.identity_key()}/{fingerprint_extractor_recipe(recipe)}"
    outputs = []
    materializations = list(
        materialize_structured_outputs(
            dataset,
            extractor,
            batch_size=batch_size,
            aligners=aligners,
            resource_profiler=profiler if resource_config.enabled else None,
        )
    )
    shared_profile = profiler.finish() if resource_config.enabled else None
    for materialization in materializations:
        safe_name = materialization.name.replace("/", "_")
        output_key = f"{base_key}/outputs/{safe_name}"
        labels_key = f"{output_key}/labels"
        groups_key = f"{output_key}/groups"
        provenance_key = f"{output_key}/provenance"
        embeddings = materialization.dataset.X
        labels = materialization.dataset.y
        groups = materialization.dataset.groups()
        if groups is None:
            raise ValueError("Structured materialization must define parent groups.")
        artifact_path = store.put_array(output_key, embeddings)
        embedding_manifest = {
            "artifact_type": "structured_embedding",
            "vertebrae_version": __version__,
            "output_key": output_key,
            "artifact_path": artifact_path,
            "dataset_identity_key": materialization.dataset.identity_key(),
            "source_dataset_identity_key": dataset.identity_key(),
            "extractor_recipe": recipe,
            "recipe_hash": fingerprint_extractor_recipe(recipe),
            "output_name": materialization.name,
            "n_samples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "sparse": False,
            "storage_format": "dense",
            "modality": materialization.dataset.modality,
            "structured": materialization.metadata,
            "unit_type": materialization.metadata.get("unit_type"),
            "task_family": materialization.metadata.get("task_family"),
            "alignment_mode": materialization.metadata.get("alignment_mode"),
            "alignment_recipe": materialization.metadata.get("alignment_recipe"),
            "labels_key": labels_key,
            "groups_key": groups_key,
            "provenance_key": provenance_key,
            "resource_profiling_config": asdict(resource_config),
        }
        profile = with_embedding_footprint(
            shared_profile,
            embeddings,
            embeddings,
            store=store,
            raw_key=output_key,
            evaluated_key=output_key,
            persisted_storage=resource_config.persisted_storage,
        )
        if profile is not None:
            embedding_manifest["resource_profile"] = make_json_safe(profile)
        store.put_json(output_key, embedding_manifest)
        label_path = store.put_labels(labels_key, labels)
        label_summary = target_summary(
            labels,
            target_type=materialization.dataset.metadata.get("target_type", "auto"),
            target_names=materialization.dataset.metadata.get("target_names"),
        )
        store.put_json(
            labels_key,
            {
                "artifact_type": "labels",
                "vertebrae_version": __version__,
                "output_key": labels_key,
                "artifact_path": label_path,
                "dataset_identity_key": materialization.dataset.identity_key(),
                "n_samples": int(len(labels)),
                "target_type": label_summary["target_type"],
                "class_counts": make_json_safe(label_summary["class_counts"]),
                "n_classes": label_summary["n_classes"],
                "target_view": materialization.dataset.active_target_view(),
                "label_view": materialization.dataset.active_label_view(),
            },
        )
        group_path = store.put_labels(groups_key, groups)
        store.put_json(
            groups_key,
            {
                "artifact_type": "groups",
                "vertebrae_version": __version__,
                "output_key": groups_key,
                "artifact_path": group_path,
                "dataset_identity_key": materialization.dataset.identity_key(),
                "n_samples": int(len(groups)),
                "n_groups": int(len(np.unique(groups))),
                "group_name": "parent_id",
            },
        )
        store.put_json(provenance_key, {"rows": materialization.provenance})
        outputs.append(embedding_manifest)
    bundle = {
        "artifact_type": "structured_embedding_bundle",
        "vertebrae_version": __version__,
        "output_key": base_key,
        "dataset_identity_key": dataset.identity_key(),
        "extractor_recipe": recipe,
        "resource_profiling_config": asdict(resource_config),
        "resource_profile": (
            make_json_safe(shared_profile) if shared_profile is not None else None
        ),
        "outputs": outputs,
    }
    store.put_json(base_key, bundle)
    return bundle


def plan_embedding_shard_jobs(
    dataset: Any,
    extractor: Any,
    total_shards: int,
    batch_size: int = 128,
    resource_profiling_config: Optional[ResourceProfilingConfig] = None,
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

    base_key = embedding_artifact_key(dataset, extractor)
    return [
        EmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            shard=shard,
            output_key=embedding_shard_key(base_key, shard),
            batch_size=batch_size,
            resource_profiling_config=(resource_profiling_config or ResourceProfilingConfig()),
        )
        for shard in (
            ShardSpec(total_shards=total_shards, shard_index=i) for i in range(total_shards)
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

    jobs = plan_embedding_shard_jobs(
        dataset=dataset,
        extractor=extractor,
        total_shards=total_shards,
        batch_size=batch_size,
    )
    shard_manifests = materialize_embedding_shards(jobs, store=store, execution=execution)
    return merge_embedding_shards(
        EmbeddingMergeJob(
            shard_keys=tuple(manifest["output_key"] for manifest in shard_manifests),
            output_key=embedding_artifact_key(dataset, extractor),
            n_samples=len(dataset.y),
        ),
        store=store,
    )


def materialize_label_artifact(
    dataset: Any,
    store: ArtifactStore,
    key: Any = None,
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
    artifact_path = store.put_labels(output_key, dataset.y)
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
        "artifact_path": artifact_path,
        "dataset_identity_key": dataset.identity_key(),
        "n_samples": int(len(dataset.y)),
        "dtype": str(np.asarray(dataset.y).dtype),
        "target_type": labels["target_type"],
        "class_counts": make_json_safe(labels["class_counts"]),
        "n_classes": labels["n_classes"],
        "target_view": make_json_safe(dataset.active_target_view()),
        "label_view": make_json_safe(dataset.active_label_view()),
    }
    for label_key in (
        "label_names",
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
    store.put_json(output_key, manifest)
    return manifest


def materialize_group_artifact(
    dataset: Any,
    store: ArtifactStore,
    key: Any = None,
) -> dict[str, Any]:
    """Materialize aligned independence groups without reporting raw IDs."""

    groups = dataset.groups() if callable(getattr(dataset, "groups", None)) else None
    if groups is None:
        raise ValueError("Dataset does not define independence groups.")
    output_key = key or groups_artifact_key(dataset)
    artifact_path = store.put_labels(output_key, groups)
    manifest = {
        "artifact_type": "groups",
        "vertebrae_version": __version__,
        "output_key": output_key,
        "artifact_path": artifact_path,
        "dataset_identity_key": dataset.identity_key(),
        "n_samples": int(len(groups)),
        "n_groups": int(len(np.unique(groups))),
        "group_name": dataset.metadata.get("group_name", "group"),
    }
    store.put_json(output_key, manifest)
    return manifest


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

    embedding_metadata, label_metadata = validate_embedding_label_artifacts(
        store,
        embedding_key=job.embedding_key,
        labels_key=job.labels_key,
    )
    embeddings = store.get_array(job.embedding_key)
    labels = store.get_labels(job.labels_key)
    groups, group_metadata = _load_validated_groups(
        store,
        groups_key=job.groups_key,
        embedding_metadata=embedding_metadata,
        label_metadata=label_metadata,
    )
    from vertebrae.scoring.metrics import OverlapMetric, as_embedding_metric

    configured = [as_embedding_metric(metric) for metric in (job.metrics or [])]
    if job.metric is not None:
        configured.append(as_embedding_metric(job.metric))
    if not any(metric.name == "overlap" for metric in configured):
        configured.insert(0, OverlapMetric(config=job.scoring_config))
    names = [metric.name for metric in configured]
    if len(names) != len(set(names)):
        raise ValueError("Metric names must be unique within a scoring job.")
    if job.primary_metric not in names:
        raise ValueError("primary_metric must name one configured metric.")
    metric_results = {}
    for metric in configured:
        result = metric.score(
            embeddings,
            labels,
            target_metadata=label_metadata,
            groups=groups,
            seed=job.seed,
        )
        result.metadata = {**label_metadata, **result.metadata}
        metric_results[metric.name] = result.to_dict()
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
        "metric_recipes": [metric.recipe() for metric in configured],
        "embedding_metadata": embedding_metadata,
        "label_metadata": label_metadata,
        "resources": asdict(job.resources),
    }
    store.put_json(job.output_key, artifact)
    return artifact


def score_retrieval_artifact(job: RetrievalScoringJob, store: ArtifactStore) -> dict[str, Any]:
    """Score persisted query/gallery embeddings using a canonical relevance JSON artifact.

    The relevance artifact contains ``relevance`` as sparse row-index mappings and optional
    ``query_ids``, ``gallery_ids``, and ``exclusions`` fields. This keeps retrieval scoring
    independent of raw model objects and compatible with object stores.
    """
    from vertebrae.config import RetrievalConfig
    from vertebrae.scoring.retrieval import RetrievalScorer

    query_metadata = store.get_json(job.query_embedding_key)
    gallery_metadata = store.get_json(job.gallery_embedding_key)
    relevance_data = store.get_json(job.relevance_key)
    _validate_retrieval_pair_metadata(query_metadata, gallery_metadata, relevance_data)
    queries = store.get_array(job.query_embedding_key)
    gallery = store.get_array(job.gallery_embedding_key)
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
    result = RetrievalScorer(job.retrieval_config or RetrievalConfig()).score(
        queries,
        gallery,
        relevance,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        exclusions=exclusions,
    )
    artifact = {
        "artifact_type": "retrieval_evaluation",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "query_embedding_key": job.query_embedding_key,
        "gallery_embedding_key": job.gallery_embedding_key,
        "relevance_key": job.relevance_key,
        "exclusions_key": job.exclusions_key,
        "result": result.to_dict(),
        "query_endpoint": query_metadata,
        "gallery_endpoint": gallery_metadata,
        "retrieval_config": asdict(job.retrieval_config or RetrievalConfig()),
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

    embedding_metadata, label_metadata = validate_embedding_label_artifacts(
        store,
        embedding_key=job.embedding_key,
        labels_key=job.labels_key,
    )
    score_artifact = store.get_json(job.score_key)
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
        embeddings = store.get_array(job.embedding_key)
        labels = store.get_labels(job.labels_key)
        excluded = overlap_metadata.get("exclude_classes", [])
        if excluded and target_type != REGRESSION_TARGET:
            mask = np.asarray(
                [not any(label == item for item in excluded) for label in labels],
                dtype=bool,
            )
            embeddings = embeddings[mask]
            labels = np.asarray(labels)[mask]
            groups = None if groups is None else np.asarray(groups)[mask]
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

    artifact = {
        "artifact_type": "separatix_diagnostic",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "embedding_key": job.embedding_key,
        "labels_key": job.labels_key,
        "score_key": job.score_key,
        "groups_key": job.groups_key,
        "diagnostic": diagnostic.to_dict(),
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

    embedding_metadata = store.get_json(job.embedding_key)
    embeddings = store.get_array(job.embedding_key)
    compression_result = compress_embeddings(embeddings, config=job.compression_config)
    store.put_array(job.output_key, compression_result.embeddings)
    compressed_metadata = dict(embedding_metadata)
    compressed_metadata["cache_key"] = job.output_key
    compressed_metadata["embedding_dim"] = compression_result.metadata.get(
        "compressed_dim",
        embedding_metadata.get("embedding_dim"),
    )
    compressed_metadata["shape"] = [
        embedding_metadata.get("n_samples"),
        compressed_metadata["embedding_dim"],
    ]
    compressed_metadata["sparse"] = compression_result.metadata.get(
        "output_sparse",
        embedding_metadata.get("sparse"),
    )
    compressed_metadata["storage_format"] = (
        compression_result.embeddings.getformat()
        if is_sparse_matrix(compression_result.embeddings)
        else "dense"
    )
    compressed_metadata["compression"] = compression_result.metadata
    serialized_profile = embedding_metadata.get("distributed_resource_profile")
    if serialized_profile is not None:
        distributed_profile = with_distributed_embedding_footprint(
            distributed_resource_profile_from_dict(dict(serialized_profile)),
            embeddings,
            compression_result.embeddings,
            store=store,
            raw_key=job.embedding_key,
            evaluated_key=job.output_key,
            persisted_storage=bool(
                embedding_metadata.get("resource_profiling_config", {}).get(
                    "persisted_storage", True
                )
            ),
        )
        compressed_metadata["distributed_resource_profile"] = make_json_safe(distributed_profile)
    elif embedding_metadata.get("resource_profile") is not None:
        local_profile = with_embedding_footprint(
            resource_profile_from_dict(dict(embedding_metadata["resource_profile"])),
            embeddings,
            compression_result.embeddings,
            store=store,
            raw_key=job.embedding_key,
            evaluated_key=job.output_key,
            persisted_storage=bool(
                embedding_metadata.get("resource_profiling_config", {}).get(
                    "persisted_storage", True
                )
            ),
        )
        compressed_metadata["resource_profile"] = make_json_safe(local_profile)
    store.put_json(job.output_key, compressed_metadata)
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

    embedding_metadata = store.get_json(embedding_key)
    label_metadata = store.get_json(labels_key)
    embedding_rows = int(embedding_metadata.get("n_samples", -1))
    label_rows = int(label_metadata.get("n_samples", -2))
    if embedding_rows != label_rows:
        raise ValueError(
            "Embedding and label artifacts have different row counts; "
            f"got {embedding_rows} and {label_rows}."
        )
    embedding_identity_key = embedding_metadata.get("dataset_identity_key")
    label_identity_key = label_metadata.get("dataset_identity_key")
    if (
        embedding_identity_key
        and label_identity_key
        and embedding_identity_key != label_identity_key
    ):
        raise ValueError("Embedding and label artifacts have different dataset identities.")
    return embedding_metadata, label_metadata


def _load_validated_groups(
    store: ArtifactStore,
    *,
    groups_key: Optional[str],
    embedding_metadata: dict[str, Any],
    label_metadata: dict[str, Any],
) -> tuple[Optional[np.ndarray], Optional[dict[str, Any]]]:
    if groups_key is None:
        return None, None
    group_metadata = store.get_json(groups_key)
    if group_metadata.get("artifact_type") != "groups":
        raise ValueError("Group artifact metadata must declare artifact_type='groups'.")
    expected_rows = int(label_metadata.get("n_samples", -1))
    if int(group_metadata.get("n_samples", -2)) != expected_rows:
        raise ValueError("Group and label artifacts have different row counts.")
    identities = {
        identity
        for identity in (
            embedding_metadata.get("dataset_identity_key"),
            label_metadata.get("dataset_identity_key"),
            group_metadata.get("dataset_identity_key"),
        )
        if identity
    }
    if len(identities) > 1:
        raise ValueError("Embedding, label, and group artifacts have different dataset identities.")
    groups = store.get_labels(groups_key)
    if len(groups) != expected_rows:
        raise ValueError(
            "Group artifact metadata does not match its labels; "
            f"expected {expected_rows} rows, loaded {len(groups)}."
        )
    return np.asarray(groups), group_metadata


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

    return [
        ScoringJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            output_key=scoring_artifact_key(embedding_key, seed=seed),
            groups_key=groups_key,
            scoring_config=scoring_config,
            metrics=metrics,
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
    metric_name = metric_name or artifacts[0].get("primary_metric", "overlap")
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
    }
    store.put_json(output_key, collection)
    return collection


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
    metrics_data = score_artifact["metrics"]
    embedding_metadata = score_artifact.get("embedding_metadata", {})
    label_metadata = score_artifact.get("label_metadata", {})
    group_metadata = score_artifact.get("group_metadata") or {}
    stability = store.get_json(stability_key) if stability_key else None
    separatix_artifact = store.get_json(separatix_key) if separatix_key else None
    separatix = None
    if separatix_artifact:
        separatix = SeparatixResult(**separatix_artifact["diagnostic"])
    metrics = {name: MetricResult(**data) for name, data in metrics_data.items()}
    overlap = metrics["overlap"]
    score_metadata = overlap.metadata
    weakest_class, weakest_score = _weakest_class(
        overlap.per_class_scores,
        excluded_classes=score_metadata.get("exclude_classes"),
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
    relevance_keys = {item.get("relevance_key") for item in artifacts}
    if len(relevance_keys) != 1:
        raise ValueError("Retrieval score artifacts must share one relevance protocol.")

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
        score = RetrievalScoreResult(**dict(artifact["result"]))
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
                forward=score,
                reverse=None,
                primary_score=score.score,
                compression_metadata=compression,
                runtime={},
                warnings=sorted(set(score.warnings + list(compression.get("warnings", [])))),
                recipe=query_recipe,
                resource_profiles=resource_profiles,
            )
        )

    relevance = store.get_json(str(next(iter(relevance_keys))))
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
    arr = np.asarray(scores, dtype=float)
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
) -> tuple[Optional[str], Optional[float]]:
    excluded = (
        []
        if excluded_classes is None
        else [excluded_classes]
        if isinstance(excluded_classes, (str, bytes))
        else list(excluded_classes)
    )
    numeric = {
        str(label): float(score)
        for label, score in per_class_scores.items()
        if isinstance(score, (int, float, np.number))
        and not any(label == item or str(label) == str(item) for item in excluded)
    }
    if not numeric:
        return None, None
    label, score = min(numeric.items(), key=lambda item: item[1])
    return label, score


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
    extractor.fit(dataset.X, dataset.y)
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
    artifact_path = store.put_array_batches(
        job.output_key,
        batches,
        n_samples=len(sample_indices),
        require_complete=True,
    )
    embeddings = store.get_array(job.output_key)
    profile = profiler.finish() if job.resource_profiling_config.enabled else None
    profile = with_embedding_footprint(
        profile,
        embeddings,
        embeddings,
        store=store,
        raw_key=job.output_key,
        evaluated_key=job.output_key,
        persisted_storage=job.resource_profiling_config.persisted_storage,
    )
    sparse_embeddings = is_sparse_matrix(embeddings)
    manifest = {
        "artifact_type": "embedding_shard",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "artifact_path": artifact_path,
        "dataset_identity_key": dataset.identity_key(),
        "extractor_recipe": extractor.recipe(),
        "recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "shard": asdict(job.shard),
        "sample_indices": sample_indices.tolist(),
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "sparse": sparse_embeddings,
        "nnz": int(embeddings.nnz) if sparse_embeddings else None,
        "storage_format": embeddings.getformat() if sparse_embeddings else "dense",
        "batch_size": job.batch_size,
        "resources": asdict(job.resources),
        "resource_profiling_config": asdict(job.resource_profiling_config),
        "resource_profile": make_json_safe(profile) if profile is not None else None,
    }
    store.put_json(job.output_key, manifest)
    return manifest


def materialize_retrieval_embedding_shard(
    job: RetrievalEmbeddingShardJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Materialize one deterministic query or gallery embedding shard."""
    dataset = job.dataset
    dataset.validated()
    values = dataset.queries if job.side == "query" else dataset.gallery
    modality = dataset.query_modality if job.side == "query" else dataset.gallery_modality
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
        if getattr(extractor, "already_fitted", True) is False:
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
    embeddings = encode_endpoint_batches(
        selected,
        batch_size=job.batch_size,
        encode=encode,
        owner=f"Retriever '{job.extractor.name}' {job.side} shard embeddings",
        profiler=profiler if job.resource_profiling_config.enabled else None,
        call_type=f"encode_retrieval_{job.side}",
    )
    if embeddings.shape[0] != len(indices):
        raise ValueError("Retrieval extractor output does not align with its endpoint shard.")
    path = store.put_array(job.output_key, embeddings)
    profile = profiler.finish() if job.resource_profiling_config.enabled else None
    profile = with_embedding_footprint(
        profile,
        embeddings,
        embeddings,
        store=store,
        raw_key=job.output_key,
        evaluated_key=job.output_key,
        persisted_storage=job.resource_profiling_config.persisted_storage,
    )
    sparse = is_sparse_matrix(embeddings)
    manifest = {
        "artifact_type": "retrieval_embedding_shard",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "artifact_path": path,
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
        "storage_format": embeddings.getformat() if sparse else "dense",
        "batch_size": job.batch_size,
        "resource_profiling_config": asdict(job.resource_profiling_config),
        "resource_profile": make_json_safe(profile) if profile is not None else None,
    }
    store.put_json(job.output_key, manifest)
    return manifest


def compress_retrieval_embedding_artifacts(
    job: RetrievalCompressionJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Fit one compressor on gallery embeddings and transform both retrieval endpoints."""
    query_metadata = store.get_json(job.query_embedding_key)
    gallery_metadata = store.get_json(job.gallery_embedding_key)
    _validate_retrieval_pair_metadata(query_metadata, gallery_metadata)
    query = store.get_array(job.query_embedding_key)
    gallery = store.get_array(job.gallery_embedding_key)
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
    manifests: list[dict[str, Any]] = []
    for key, values, source_metadata, side in (
        (job.query_output_key, query_result, query_metadata, "query"),
        (job.gallery_output_key, gallery_result, gallery_metadata, "gallery"),
    ):
        path = store.put_array(key, values)
        manifest = dict(source_metadata)
        manifest.update(
            {
                "artifact_type": "retrieval_compressed_embedding",
                "output_key": key,
                "artifact_path": path,
                "side": side,
                "n_samples": int(values.shape[0]),
                "embedding_dim": int(values.shape[1]),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "sparse": is_sparse_matrix(values),
                "storage_format": values.getformat() if is_sparse_matrix(values) else "dense",
                "compression": metadata,
            }
        )
        serialized_profile = source_metadata.get("distributed_resource_profile")
        if serialized_profile is not None:
            source_values = query if side == "query" else gallery
            distributed_profile = with_distributed_embedding_footprint(
                distributed_resource_profile_from_dict(dict(serialized_profile)),
                source_values,
                values,
                store=store,
                raw_key=(job.query_embedding_key if side == "query" else job.gallery_embedding_key),
                evaluated_key=key,
                persisted_storage=bool(
                    source_metadata.get("resource_profiling_config", {}).get(
                        "persisted_storage", True
                    )
                ),
            )
            manifest["distributed_resource_profile"] = make_json_safe(distributed_profile)
        manifests.append(manifest)
    for manifest in manifests:
        store.put_json(manifest["output_key"], manifest)
    artifact = {
        "artifact_type": "retrieval_compression",
        "query_output_key": job.query_output_key,
        "gallery_output_key": job.gallery_output_key,
        "query_embedding_key": job.query_embedding_key,
        "gallery_embedding_key": job.gallery_embedding_key,
        "compression_metadata": metadata,
    }
    prefix = _shared_artifact_prefix(job.query_output_key, job.gallery_output_key)
    if prefix:
        store.put_json(prefix, artifact)
    return artifact


def merge_embedding_shards(
    job: EmbeddingMergeJob,
    store: ArtifactStore,
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
        return _merge_multi_output_embedding_shards(job, store, manifests)
    _validate_shard_manifests(manifests, expected_n_samples=job.n_samples)
    batches = [
        (np.asarray(manifest["sample_indices"], dtype=int), store.get_array(manifest["output_key"]))
        for manifest in manifests
    ]
    artifact_path = store.put_array_batches(
        job.output_key,
        batches,
        n_samples=job.n_samples,
        require_complete=True,
    )
    embeddings = store.get_array(job.output_key)
    sparse_embeddings = is_sparse_matrix(embeddings)
    manifest = {
        "artifact_type": "embedding",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "artifact_path": artifact_path,
        "shard_keys": list(job.shard_keys),
        "n_shards": len(job.shard_keys),
        "shape": list(embeddings.shape),
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "sparse": sparse_embeddings,
        "nnz": int(embeddings.nnz) if sparse_embeddings else None,
        "storage_format": embeddings.getformat() if sparse_embeddings else "dense",
        "resources": asdict(job.resources),
    }
    first = manifests[0]
    for key in ("dataset_identity_key", "extractor_recipe", "recipe_hash"):
        manifest[key] = first.get(key)
    serialized_profiles = [
        (item["output_key"], item.get("resource_profile"))
        for item in manifests
        if item.get("resource_profile") is not None
    ]
    if serialized_profiles:
        config = dict(first.get("resource_profiling_config") or {})
        distributed_profile = aggregate_distributed_resource_profiles(
            [
                (key, resource_profile_from_dict(dict(profile or {})))
                for key, profile in serialized_profiles
            ],
            merged_embeddings=embeddings,
            all_shard_keys=[item["output_key"] for item in manifests],
            store=store,
            merged_key=job.output_key,
            persisted_storage=bool(config.get("persisted_storage", True)),
        )
        manifest["distributed_resource_profile"] = make_json_safe(distributed_profile)
        manifest["resource_profiling_config"] = config
    store.put_json(job.output_key, manifest)
    return manifest


def merge_retrieval_embedding_shards(
    job: EmbeddingMergeJob,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Merge retrieval endpoint shards and preserve their endpoint identity."""
    shards = [store.get_json(key) for key in job.shard_keys]
    _validate_retrieval_shard_manifests(shards, expected_n_samples=job.n_samples)
    manifest = merge_embedding_shards(job, store)
    sides = {shard.get("side") for shard in shards}
    branches = {shard.get("branch") for shard in shards}
    modalities = {shard.get("modality") for shard in shards}
    if len(sides) != 1 or len(branches) != 1 or len(modalities) != 1:
        raise ValueError("Retrieval endpoint shards must share one side, branch, and modality.")
    manifest.update(
        {
            "artifact_type": "retrieval_embedding",
            "side": sides.pop(),
            "branch": branches.pop(),
            "modality": modalities.pop(),
        }
    )
    store.put_json(job.output_key, manifest)
    return manifest


def plan_retrieval_embedding_shard_jobs(
    dataset: Any,
    extractor: Any,
    total_shards: int,
    *,
    side: str,
    branch: Optional[str] = None,
    batch_size: int = 128,
    resource_profiling_config: Optional[ResourceProfilingConfig] = None,
) -> list[RetrievalEmbeddingShardJob]:
    """Plan deterministic embedding jobs for one retrieval endpoint."""
    base_key = retrieval_embedding_artifact_key(dataset, extractor, side, branch)
    return [
        RetrievalEmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            side=side,
            branch=branch,
            shard=ShardSpec(total_shards=total_shards, shard_index=index),
            output_key=retrieval_embedding_shard_key(
                base_key, ShardSpec(total_shards=total_shards, shard_index=index)
            ),
            batch_size=batch_size,
            resource_profiling_config=(resource_profiling_config or ResourceProfilingConfig()),
        )
        for index in range(total_shards)
    ]


def _local_embedding_batches(
    dataset: Any,
    extractor: Any,
    job: EmbeddingShardJob,
    local_positions: dict[int, int],
    profiler: ResourceProfiler,
) -> Iterator[Tuple[np.ndarray, Any]]:
    for batch in dataset.iter_batches(batch_size=job.batch_size, shard=job.shard):

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
    output_batches: Dict[str, list[Tuple[np.ndarray, Any]]] = {
        spec["name"]: [] for spec in output_specs
    }
    output_recipes: Dict[str, dict[str, Any]] = {}
    output_metadata: Dict[str, dict[str, Any]] = {}
    for batch in dataset.iter_batches(batch_size=job.batch_size, shard=job.shard):

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
        indices = np.asarray([local_positions[int(index)] for index in batch.indices], dtype=int)
        for output in outputs:
            output_batches[output.name].append((indices, output.embeddings))
            output_recipes[output.name] = dict(output.recipe)
            output_metadata[output.name] = dict(output.metadata)

    manifests = []
    base_profile = profiler.finish() if job.resource_profiling_config.enabled else None
    for spec in output_specs:
        output_name = spec["name"]
        output_key = embedding_output_shard_key(job.output_key, output_name)
        artifact_path = store.put_array_batches(
            output_key,
            output_batches[output_name],
            n_samples=len(sample_indices),
            require_complete=True,
        )
        embeddings = store.get_array(output_key)
        output_profile = with_embedding_footprint(
            base_profile,
            embeddings,
            embeddings,
            store=store,
            raw_key=output_key,
            evaluated_key=output_key,
            persisted_storage=job.resource_profiling_config.persisted_storage,
        )
        sparse_embeddings = is_sparse_matrix(embeddings)
        output_manifest = {
            "artifact_type": "embedding_shard",
            "vertebrae_version": __version__,
            "output_key": output_key,
            "artifact_path": artifact_path,
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
            "storage_format": embeddings.getformat() if sparse_embeddings else "dense",
            "batch_size": job.batch_size,
            "resources": asdict(job.resources),
            "resource_profiling_config": asdict(job.resource_profiling_config),
            "resource_profile": (
                make_json_safe(output_profile) if output_profile is not None else None
            ),
        }
        store.put_json(output_key, output_manifest)
        manifests.append(output_manifest)

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
    output_manifests = []
    for output_name in output_names:
        shard_keys = []
        for manifest in manifests:
            output = _find_output_manifest_entry(manifest, output_name)
            shard_keys.append(output["output_key"])
        output_key = embedding_output_key(job.output_key, output_name)
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
    if len(recipe_hashes) != 1:
        raise ValueError("Embedding shards have inconsistent extractor recipes.")
    if len(dataset_identity_keys) != 1:
        raise ValueError("Embedding shards have inconsistent dataset identities.")
    if len(dtypes) != 1 or len(sparse_values) != 1 or len(dims) != 1:
        raise ValueError("Embedding shards have inconsistent embedding formats.")

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


def _shared_artifact_prefix(query_key: str, gallery_key: str) -> Optional[str]:
    """Return a paired endpoint prefix when output keys use the standard layout."""
    if query_key.endswith("/query") and gallery_key == f"{query_key[:-6]}/gallery":
        return query_key[:-6]
    return None
