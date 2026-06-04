"""Distributed embedding primitives."""

from dataclasses import asdict
from functools import partial
from typing import Any, Iterable, Iterator, Tuple

import numpy as np

from vertebrae import __version__
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.execution.jobs import EmbeddingMergeJob, EmbeddingShardJob, ShardSpec
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
    store: LocalArtifactStore,
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

    return execution.map(partial(materialize_embedding_shard, store=store), jobs)


def materialize_and_merge_embeddings(
    dataset: Any,
    extractor: Any,
    store: LocalArtifactStore,
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


def materialize_embedding_shard(
    job: EmbeddingShardJob,
    store: LocalArtifactStore,
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
    store: LocalArtifactStore,
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
