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
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.compression import compress_embedding_artifact_key, compress_embeddings
from vertebrae.execution.jobs import (
    CompressionJob,
    EmbeddingMergeJob,
    EmbeddingShardJob,
    ScoringJob,
    SeparatixJob,
    ShardSpec,
)
from vertebrae.extractors.base import EmbeddingOutput
from vertebrae.scoring.separatix import SeparatixResult, SeparatixScorer
from vertebrae.utils.labels import label_view_suffix, target_summary
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


def embedding_artifact_key(dataset: Any, extractor: Any) -> str:
    """Build the canonical embedding artifact key.

    Args:
        dataset: Dataset object with a `fingerprint()` method.
        extractor: Feature extractor with a serializable `recipe()`.

    Returns:
        Artifact key for the complete embedding matrix.
    """

    dataset_key = dataset.fingerprint()
    extractor_key = fingerprint_extractor_recipe(extractor.recipe())
    return f"embeddings/{dataset_key}/{extractor_key}"


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
        dataset: Dataset object with a `fingerprint()` method.

    Returns:
        Artifact key for labels.
    """

    return f"labels/{dataset.fingerprint()}"


def groups_artifact_key(dataset: Any) -> str:
    """Build the canonical independence-group artifact key."""

    return f"groups/{dataset.fingerprint()}"


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


def separatix_artifact_key(embedding_key: str) -> str:
    """Build a Separatix diagnostic artifact key."""

    return f"{embedding_key}/diagnostics/separatix"


def materialize_segmentation_artifacts(
    dataset: Any,
    extractor: Any,
    store: ArtifactStore,
    segmentation_config: Any = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Materialize spatial segmentation outputs into standard artifact boundaries."""

    from vertebrae.segmentation import materialize_segmentation_outputs

    recipe = extractor.recipe()
    base_key = f"segmentation/{dataset.fingerprint()}/" f"{fingerprint_extractor_recipe(recipe)}"
    outputs = []
    for materialization in materialize_segmentation_outputs(
        dataset,
        extractor,
        config=segmentation_config,
        batch_size=batch_size,
    ):
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
            "dataset_fingerprint": materialization.dataset.fingerprint(),
            "source_dataset_fingerprint": dataset.fingerprint(),
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
        }
        store.put_json(output_key, embedding_manifest)
        label_path = store.put_labels(labels_key, labels)
        label_summary = target_summary(labels)
        store.put_json(
            labels_key,
            {
                "artifact_type": "labels",
                "vertebrae_version": __version__,
                "output_key": labels_key,
                "artifact_path": label_path,
                "dataset_fingerprint": materialization.dataset.fingerprint(),
                "n_samples": int(len(labels)),
                "target_type": label_summary["target_type"],
                "class_counts": make_json_safe(label_summary["class_counts"]),
                "n_classes": label_summary["n_classes"],
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
                "dataset_fingerprint": materialization.dataset.fingerprint(),
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
        "dataset_fingerprint": dataset.fingerprint(),
        "extractor_recipe": recipe,
        "outputs": outputs,
    }
    store.put_json(base_key, bundle)
    return bundle


def plan_embedding_shard_jobs(
    dataset: Any,
    extractor: Any,
    total_shards: int,
    batch_size: int = 128,
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
    labels = target_summary(dataset.y, label_names=dataset.metadata.get("label_names"))
    manifest = {
        "artifact_type": "labels",
        "vertebrae_version": __version__,
        "output_key": output_key,
        "artifact_path": artifact_path,
        "dataset_fingerprint": dataset.fingerprint(),
        "n_samples": int(len(dataset.y)),
        "dtype": str(np.asarray(dataset.y).dtype),
        "target_type": labels["target_type"],
        "class_counts": make_json_safe(labels["class_counts"]),
        "n_classes": labels["n_classes"],
        "label_view": make_json_safe(dataset.active_label_view()),
    }
    for label_key in (
        "label_names",
        "labelset_counts",
        "mean_label_cardinality",
        "label_density",
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
        "dataset_fingerprint": dataset.fingerprint(),
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
    from vertebrae.config import OverlapScoringConfig
    from vertebrae.scoring.overlap import OverlapIndexScorer

    config = job.scoring_config or OverlapScoringConfig()
    score = OverlapIndexScorer(config).score(
        embeddings,
        labels,
        seed=job.seed,
        label_names=label_metadata.get("label_names"),
    )
    artifact = {
        "artifact_type": "overlap_score",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "embedding_key": job.embedding_key,
        "labels_key": job.labels_key,
        "seed": job.seed,
        "score": score.to_dict(),
        "embedding_metadata": embedding_metadata,
        "label_metadata": label_metadata,
        "resources": asdict(job.resources),
    }
    store.put_json(job.output_key, artifact)
    return artifact


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
    score_data = score_artifact.get("score", {})
    macro_score = float(score_data.get("macro_score"))

    from vertebrae.config import OverlapScoringConfig, SeparatixConfig

    overlap_config = OverlapScoringConfig()
    overlap_metadata = score_data.get("metadata", {})
    if "normalize_embeddings" in overlap_metadata:
        overlap_config.normalize_embeddings = bool(overlap_metadata["normalize_embeddings"])
    separatix_config = job.separatix_config or SeparatixConfig()
    scorer = SeparatixScorer(config=separatix_config, overlap_config=overlap_config)

    if macro_score < separatix_config.overlap_threshold:
        diagnostic = scorer.skipped_result(
            reason=(
                "Skipped Separatix because overlap macro "
                f"{macro_score:.4f} is below the configured threshold "
                f"{separatix_config.overlap_threshold:.4f}."
            ),
            macro_score=macro_score,
        )
    else:
        embeddings = store.get_array(job.embedding_key)
        labels = store.get_labels(job.labels_key)
        groups = None
        if job.groups_key:
            group_metadata = store.get_json(job.groups_key)
            if int(group_metadata.get("n_samples", -1)) != len(labels):
                raise ValueError("Group and label artifacts have different row counts.")
            group_fingerprint = group_metadata.get("dataset_fingerprint")
            label_fingerprint = label_metadata.get("dataset_fingerprint")
            if group_fingerprint and label_fingerprint and group_fingerprint != label_fingerprint:
                raise ValueError("Group and label artifacts have different dataset fingerprints.")
            groups = store.get_labels(job.groups_key)
        excluded = overlap_metadata.get("exclude_classes", [])
        if excluded:
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
                groups=groups,
            )
        except ValueError as exc:
            if groups is None:
                raise
            diagnostic = scorer.skipped_result(
                reason=f"Skipped grouped Separatix diagnostic: {exc}",
                macro_score=macro_score,
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
    embedding_fingerprint = embedding_metadata.get("dataset_fingerprint")
    label_fingerprint = label_metadata.get("dataset_fingerprint")
    if embedding_fingerprint and label_fingerprint and embedding_fingerprint != label_fingerprint:
        raise ValueError("Embedding and label artifacts have different dataset fingerprints.")
    return embedding_metadata, label_metadata


def plan_scoring_jobs(
    embedding_key: str,
    labels_key: str,
    seeds: Iterable[Optional[int]],
    scoring_config: Any = None,
) -> list[ScoringJob]:
    """Create scoring jobs for one embedding and label artifact pair.

    Args:
        embedding_key: Complete embedding artifact key.
        labels_key: Label artifact key.
        seeds: Seeds for scoring jobs. Use `None` for the default single score.
        scoring_config: Optional scoring configuration shared by all jobs.

    Returns:
        Scoring jobs with canonical output keys.
    """

    return [
        ScoringJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            output_key=scoring_artifact_key(embedding_key, seed=seed),
            scoring_config=scoring_config,
            seed=seed,
        )
        for seed in seeds
    ]


def collect_score_artifacts(
    score_keys: Iterable[str],
    store: ArtifactStore,
    output_key: str,
    interval_level: float = 0.95,
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

    artifacts = [store.get_json(key) for key in score_keys]
    if not artifacts:
        raise ValueError("At least one score artifact is required.")
    scores = [float(artifact["score"]["macro_score"]) for artifact in artifacts]
    warnings = sorted(
        {
            warning
            for artifact in artifacts
            for warning in artifact.get("score", {}).get("warnings", [])
        }
    )
    seeds = [artifact.get("seed") for artifact in artifacts]
    collection = {
        "artifact_type": "score_collection",
        "vertebrae_version": __version__,
        "output_key": output_key,
        "score_keys": list(score_keys),
        "scores": scores,
        "seeds": seeds,
        "summary": _score_summary(scores, interval_level),
        "interval_level": interval_level,
        "warnings": warnings,
        "embedding_key": artifacts[0].get("embedding_key"),
        "labels_key": artifacts[0].get("labels_key"),
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

    from vertebrae.reports.recommendations import (
        recommendation_for_extractor,
        recommendations_for_benchmark,
    )
    from vertebrae.results import BenchmarkResult, ExtractorResult
    from vertebrae.scoring.overlap import OverlapScoreResult

    score_artifact = store.get_json(score_key)
    score_data = score_artifact["score"]
    embedding_metadata = score_artifact.get("embedding_metadata", {})
    label_metadata = score_artifact.get("label_metadata", {})
    stability = store.get_json(stability_key) if stability_key else None
    separatix_artifact = store.get_json(separatix_key) if separatix_key else None
    separatix = None
    if separatix_artifact:
        separatix = SeparatixResult(**separatix_artifact["diagnostic"])
    score_metadata = score_data.get("metadata", {})
    weakest_class, weakest_score = _weakest_class(
        score_data.get("per_class_scores", {}),
        excluded_classes=score_metadata.get("exclude_classes"),
    )
    overlap = OverlapScoreResult(
        macro_score=float(score_data["macro_score"]),
        weighted_score=score_data.get("weighted_score"),
        per_class_scores=score_data.get("per_class_scores", {}),
        pairwise_scores=score_data.get("pairwise_scores", {}),
        sparse_adjacency=score_data.get("sparse_adjacency"),
        class_counts=score_data.get("class_counts", {}),
        k_per_class=score_data.get("k_per_class", {}),
        warnings=score_data.get("warnings", []),
        metadata=score_metadata,
    )
    recommendation = (
        recommendation_for_extractor(overlap.macro_score, stability, weakest_score)
        if overlap.metadata.get("aggregate_valid", True)
        else "aggregate_unavailable"
    )
    compression_metadata = embedding_metadata.get("compression", {"method": "none"})
    base_name = embedding_metadata.get("extractor_recipe", {}).get(
        "name",
        embedding_metadata.get("extractor_name", "artifact"),
    )
    label_view = label_metadata.get("label_view", embedding_metadata.get("label_view"))
    extractor_result = ExtractorResult(
        name=_variant_extractor_name(
            f"{base_name}{label_view_suffix(label_view)}",
            compression_metadata,
        ),
        extractor_type=embedding_metadata.get("extractor_recipe", {}).get(
            "extractor_type",
            embedding_metadata.get("extractor_type", "artifact"),
        ),
        overlap=overlap,
        stability=stability,
        probes=None,
        separatix=separatix,
        embedding_metadata=embedding_metadata,
        compression_metadata=compression_metadata,
        runtime={},
        warnings=sorted(set(score_data.get("warnings", []))),
        label_view=label_view,
        weakest_class=weakest_class,
        weakest_class_score=weakest_score,
        recommendation=recommendation,
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
            "modality": embedding_metadata.get("modality", "artifact"),
            "label_view": label_metadata.get("label_view"),
        },
        extractor_results=[extractor_result],
        recommendations=recommendations_for_benchmark([extractor_result]),
        metadata={
            "vertebrae_version": __version__,
            "source_score_key": score_key,
            "source_stability_key": stability_key,
            "source_separatix_key": separatix_key,
            "distributed_artifacts": True,
        },
    )
    payload = result.to_dict()
    if output_key:
        store.put_json(output_key, payload)
    return payload


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
    method = compression_metadata.get("method", "none")
    if method == "none":
        return name
    precision = compression_metadata.get("precision")
    if precision:
        return f"{name}[{method}_{precision}]"
    compressed_dim = compression_metadata.get("compressed_dim")
    if compressed_dim is None:
        return f"{name}[{method}]"
    return f"{name}[{method}_{compressed_dim}]"


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
    local_positions = {
        int(sample_index): position for position, sample_index in enumerate(sample_indices)
    }
    if _is_multi_output_extractor(extractor):
        return _materialize_multi_output_embedding_shard(
            job=job,
            store=store,
            sample_indices=sample_indices,
            local_positions=local_positions,
        )
    batches = _local_embedding_batches(dataset, extractor, job, local_positions)
    artifact_path = store.put_array_batches(
        job.output_key,
        batches,
        n_samples=len(sample_indices),
        require_complete=True,
    )
    embeddings = store.get_array(job.output_key)
    sparse_embeddings = is_sparse_matrix(embeddings)
    manifest = {
        "artifact_type": "embedding_shard",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "artifact_path": artifact_path,
        "dataset_fingerprint": dataset.fingerprint(),
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
    }
    store.put_json(job.output_key, manifest)
    return manifest


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
    for key in ("dataset_fingerprint", "extractor_recipe", "recipe_hash"):
        manifest[key] = first.get(key)
    store.put_json(job.output_key, manifest)
    return manifest


def _local_embedding_batches(
    dataset: Any,
    extractor: Any,
    job: EmbeddingShardJob,
    local_positions: dict[int, int],
) -> Iterator[Tuple[np.ndarray, Any]]:
    for batch in dataset.iter_batches(batch_size=job.batch_size, shard=job.shard):
        embeddings = ensure_numeric_matrix(
            extractor.transform(batch.X),
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
        outputs = _validated_multi_outputs(extractor, batch.X, len(batch.indices))
        indices = np.asarray([local_positions[int(index)] for index in batch.indices], dtype=int)
        for output in outputs:
            output_batches[output.name].append((indices, output.embeddings))
            output_recipes[output.name] = dict(output.recipe)
            output_metadata[output.name] = dict(output.metadata)

    manifests = []
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
        sparse_embeddings = is_sparse_matrix(embeddings)
        output_manifest = {
            "artifact_type": "embedding_shard",
            "vertebrae_version": __version__,
            "output_key": output_key,
            "artifact_path": artifact_path,
            "dataset_fingerprint": dataset.fingerprint(),
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
        }
        store.put_json(output_key, output_manifest)
        manifests.append(output_manifest)

    bundle_manifest = {
        "artifact_type": "multi_output_embedding_shard",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "dataset_fingerprint": dataset.fingerprint(),
        "extractor_recipe": extractor.recipe(),
        "recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "shard": asdict(job.shard),
        "sample_indices": sample_indices.tolist(),
        "n_samples": int(len(sample_indices)),
        "batch_size": job.batch_size,
        "resources": asdict(job.resources),
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
        "dataset_fingerprint": first.get("dataset_fingerprint"),
        "extractor_recipe": first.get("extractor_recipe"),
        "recipe_hash": first.get("recipe_hash"),
        "resources": asdict(job.resources),
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
    dataset_fingerprints = {manifest.get("dataset_fingerprint") for manifest in manifest_list}
    dtypes = {manifest.get("dtype") for manifest in manifest_list}
    sparse_values = {manifest.get("sparse") for manifest in manifest_list}
    dims = {manifest.get("embedding_dim") for manifest in manifest_list}
    if len(recipe_hashes) != 1:
        raise ValueError("Embedding shards have inconsistent extractor recipes.")
    if len(dataset_fingerprints) != 1:
        raise ValueError("Embedding shards have inconsistent dataset fingerprints.")
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
