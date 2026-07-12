"""Artifact-backed primitives for frozen zero-shot evaluation."""

from collections import Counter
from dataclasses import asdict
from functools import partial
from typing import Any, Iterable, Optional

import numpy as np

from vertebrae import __version__
from vertebrae.cache import ArtifactStore, ArtifactStoreConfig, create_artifact_store_from_config
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe, hash_json, hash_json_exact
from vertebrae.compression.paired import compress_embedding_pair
from vertebrae.config import EmbeddingCompressionConfig, OverlapScoringConfig, ZeroShotConfig
from vertebrae.execution.jobs import (
    EmbeddingMergeJob,
    ShardSpec,
    ZeroShotCompressionJob,
    ZeroShotEmbeddingShardJob,
    ZeroShotScoringJob,
)
from vertebrae.scoring.metrics import MetricResult, OverlapMetric
from vertebrae.scoring.zero_shot import ZeroShotScorer, ZeroShotScoreResult
from vertebrae.utils.semantic_labels import LABEL_ENCODING, portable_json, validate_label_catalog
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


def zero_shot_embedding_artifact_key(dataset: Any, extractor: Any, side: str, branch: str) -> str:
    """Build a stable endpoint key while allowing sample artifacts to be prompt-reused."""

    if side not in {"samples", "prompts"}:
        raise ValueError("side must be 'samples' or 'prompts'.")
    identity = dataset.dataset.fingerprint() if side == "samples" else dataset.fingerprint()
    recipe = extractor.recipe()
    if recipe.get("cache_safe") is False:
        raise ValueError(
            "Cannot plan canonical zero-shot artifacts for a callable extractor without "
            "portable callable paths or cache_identity. Supply cache_identity explicitly."
        )
    recipe_hash = fingerprint_extractor_recipe(recipe)
    return f"zero_shot/embeddings/{identity}/{recipe_hash}/{side}/{hash_json(branch)}"


def zero_shot_protocol_artifact_key(dataset: Any) -> str:
    """Build the canonical serialized prompt-protocol artifact key."""

    return f"zero_shot/protocols/{dataset.fingerprint()}"


def zero_shot_compression_artifact_key(
    sample_embedding_key: str, prompt_embedding_key: str, config: Any
) -> str:
    """Build a paired compression prefix for zero-shot endpoints."""

    identity = hash_json({"samples": sample_embedding_key, "prompts": prompt_embedding_key})
    return f"zero_shot/compressions/{identity}/{hash_json_exact(config)}"


def zero_shot_scoring_artifact_key(
    sample_embedding_key: str,
    prompt_embedding_key: str,
    protocol_key: str,
    zero_shot_config: Optional[ZeroShotConfig] = None,
    scoring_config: Optional[OverlapScoringConfig] = None,
) -> str:
    """Build a stable zero-shot scoring artifact key for resolved evaluation settings."""

    identity = hash_json_exact(
        {
            "samples": sample_embedding_key,
            "prompts": prompt_embedding_key,
            "protocol": protocol_key,
            "evaluation": _evaluation_recipe(zero_shot_config, scoring_config),
        }
    )
    return f"zero_shot/scores/{identity}"


def _evaluation_recipe(
    zero_shot_config: Optional[ZeroShotConfig], scoring_config: Optional[OverlapScoringConfig]
) -> dict:
    zero_config = zero_shot_config or ZeroShotConfig()
    overlap_config = scoring_config or OverlapScoringConfig()
    if not isinstance(zero_config, ZeroShotConfig):
        raise TypeError("zero_shot_config must be a ZeroShotConfig.")
    if not isinstance(overlap_config, OverlapScoringConfig):
        raise TypeError("scoring_config must be an OverlapScoringConfig.")
    return {
        "zero_shot_config": asdict(zero_config),
        "overlap_scoring_config": asdict(overlap_config),
    }


def materialize_zero_shot_protocol(
    dataset: Any, store: ArtifactStore, key: Optional[str] = None
) -> dict:
    """Persist labels and fixed prompt declarations independently of model outputs."""

    dataset.validated()
    protocol_recipe = dataset.protocol_recipe()
    output_key = key or zero_shot_protocol_artifact_key(dataset)
    payload = {
        "artifact_type": "zero_shot_protocol",
        "vertebrae_version": __version__,
        "output_key": output_key,
        "dataset_fingerprint": dataset.fingerprint(),
        "protocol_fingerprint": dataset.fingerprint(),
        "source_dataset_fingerprint": dataset.dataset.fingerprint(),
        "sample_modality": dataset.modality,
        "label_encoding": LABEL_ENCODING,
        "label_catalog": protocol_recipe["label_catalog"],
        "labels": protocol_recipe["sample_labels"],
        "sample_ids": protocol_recipe["sample_ids"],
        "class_labels": protocol_recipe["ordered_labels"],
        "prompts": protocol_recipe["prompts"],
        "prompt_labels": protocol_recipe["prompt_labels"],
        "template_ids": protocol_recipe["template_ids"],
        "n_samples": protocol_recipe["n_samples"],
        "n_prompts": protocol_recipe["n_prompts"],
        "protocol": protocol_recipe,
        "dataset_summary": dataset.summary(),
    }
    store.put_json(output_key, payload)
    return payload


def materialize_zero_shot_embedding_shard(
    job: ZeroShotEmbeddingShardJob, store: ArtifactStore
) -> dict:
    """Materialize one deterministic zero-shot endpoint shard."""

    dataset = job.dataset
    dataset.validated()
    if job.side == "samples":
        values, modality = dataset.samples, dataset.modality
        identity = dataset.dataset.fingerprint()
    else:
        values, _prompt_labels, _template_ids = dataset.prompt_rows()
        modality = "text"
        identity = dataset.fingerprint()
    indices = job.shard.indices(len(values))
    if not len(indices):
        raise ValueError("Zero-shot embedding shard contains no rows.")
    encode = getattr(job.extractor, "encode_retrieval", None)
    if not callable(encode):
        raise TypeError("Zero-shot branch materialization requires encode_retrieval().")
    embeddings = ensure_numeric_matrix(
        encode(_take_rows(values, indices), branch=job.branch, modality=modality),
        f"Zero-shot {job.side} shard embeddings",
        allow_sparse=True,
    )
    if embeddings.shape[0] != len(indices):
        raise ValueError("Zero-shot extractor output does not align with its endpoint shard.")
    path = store.put_array(job.output_key, embeddings)
    sparse = is_sparse_matrix(embeddings)
    manifest = {
        "artifact_type": "zero_shot_embedding_shard",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "artifact_path": path,
        "source_dataset_fingerprint": dataset.dataset.fingerprint(),
        "protocol_fingerprint": dataset.fingerprint() if job.side == "prompts" else None,
        "endpoint_identity": identity,
        "extractor_recipe": job.extractor.recipe(),
        "recipe_hash": fingerprint_extractor_recipe(job.extractor.recipe()),
        "side": job.side,
        "branch": job.branch,
        "shard": asdict(job.shard),
        "sample_indices": indices.tolist(),
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "sparse": sparse,
        "storage_format": embeddings.getformat() if sparse else "dense",
        "compression_pair_id": None,
        "compression_source_key": None,
    }
    store.put_json(job.output_key, manifest)
    return manifest


def merge_zero_shot_embedding_shards(job: EmbeddingMergeJob, store: ArtifactStore) -> dict:
    """Merge one endpoint after validating its side, branch, and provenance."""

    manifests = [store.get_json(key) for key in job.shard_keys]
    if not manifests:
        raise ValueError("At least one zero-shot shard is required for merge.")
    _validate_shards(manifests, job.n_samples)
    artifact_path = store.put_array_batches(
        job.output_key,
        [
            (np.asarray(item["sample_indices"], dtype=int), store.get_array(item["output_key"]))
            for item in manifests
        ],
        n_samples=job.n_samples,
    )
    embeddings = store.get_array(job.output_key)
    first = manifests[0]
    manifest = {
        "artifact_type": "zero_shot_embedding",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "artifact_path": artifact_path,
        "shard_keys": list(job.shard_keys),
        "n_shards": len(job.shard_keys),
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "sparse": is_sparse_matrix(embeddings),
        "source_dataset_fingerprint": first["source_dataset_fingerprint"],
        "protocol_fingerprint": first.get("protocol_fingerprint"),
        "endpoint_identity": first["endpoint_identity"],
        "extractor_recipe": first["extractor_recipe"],
        "recipe_hash": first["recipe_hash"],
        "side": first["side"],
        "branch": first["branch"],
        "compression_pair_id": None,
        "compression_source_key": None,
    }
    store.put_json(job.output_key, manifest)
    return manifest


def compress_zero_shot_embedding_artifacts(
    job: ZeroShotCompressionJob, store: ArtifactStore
) -> dict:
    """Fit compression on sample embeddings and transform prompt embeddings."""

    sample_metadata = store.get_json(job.sample_embedding_key)
    prompt_metadata = store.get_json(job.prompt_embedding_key)
    _validate_pair(sample_metadata, prompt_metadata)
    samples = store.get_array(job.sample_embedding_key)
    prompts = store.get_array(job.prompt_embedding_key)
    _validate_endpoint_array(samples, sample_metadata)
    _validate_endpoint_array(prompts, prompt_metadata)
    if samples.shape[1] != prompts.shape[1]:
        raise ValueError("Zero-shot endpoint artifacts have incompatible embedding dimensions.")
    config = job.compression_config
    if not isinstance(config, EmbeddingCompressionConfig):
        raise TypeError("compression_config must be an EmbeddingCompressionConfig.")
    compressed_samples, compressed_prompts, metadata = compress_embedding_pair(
        samples, prompts, config
    )
    metadata["fit_side"] = "samples"
    compression_pair_id = hash_json_exact(
        {
            "sample_embedding_key": job.sample_embedding_key,
            "prompt_embedding_key": job.prompt_embedding_key,
            "sample_output_key": job.sample_output_key,
            "prompt_output_key": job.prompt_output_key,
            "compression_config": asdict(config),
        }
    )
    for key, values, source, side in (
        (job.sample_output_key, compressed_samples, sample_metadata, "samples"),
        (job.prompt_output_key, compressed_prompts, prompt_metadata, "prompts"),
    ):
        path = store.put_array(key, values)
        manifest = {
            **source,
            "artifact_type": "zero_shot_compressed_embedding",
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
            "compression_pair_id": compression_pair_id,
            "compression_source_key": (
                job.sample_embedding_key if side == "samples" else job.prompt_embedding_key
            ),
        }
        store.put_json(key, manifest)
    summary_key = job.output_key or zero_shot_compression_artifact_key(
        job.sample_embedding_key,
        job.prompt_embedding_key,
        config,
    )
    output = {
        "artifact_type": "zero_shot_compression",
        "output_key": summary_key,
        "sample_output_key": job.sample_output_key,
        "prompt_output_key": job.prompt_output_key,
        "compression_metadata": metadata,
        "source_dataset_fingerprint": sample_metadata["source_dataset_fingerprint"],
        "protocol_fingerprint": prompt_metadata.get("protocol_fingerprint"),
        "sample_endpoint": job.sample_embedding_key,
        "prompt_endpoint": job.prompt_embedding_key,
        "compression_pair_id": compression_pair_id,
    }
    store.put_json(summary_key, output)
    return output


def score_zero_shot_artifact(job: ZeroShotScoringJob, store: ArtifactStore) -> dict:
    """Score persisted frozen endpoints and contextual sample overlap."""

    sample_metadata = store.get_json(job.sample_embedding_key)
    prompt_metadata = store.get_json(job.prompt_embedding_key)
    protocol = store.get_json(job.protocol_key)
    _validate_protocol_artifact(protocol)
    _validate_pair(sample_metadata, prompt_metadata)
    if protocol.get("source_dataset_fingerprint") != sample_metadata.get(
        "source_dataset_fingerprint"
    ):
        raise ValueError("Zero-shot protocol and sample artifact have different source datasets.")
    protocol_fingerprint = protocol["protocol_fingerprint"]
    if protocol_fingerprint != prompt_metadata.get("protocol_fingerprint"):
        raise ValueError("Zero-shot protocol and prompt artifact have different prompt protocols.")
    samples = store.get_array(job.sample_embedding_key)
    prompts = store.get_array(job.prompt_embedding_key)
    _validate_endpoint_array(samples, sample_metadata)
    _validate_endpoint_array(prompts, prompt_metadata)
    if sample_metadata.get("n_samples") != len(protocol.get("labels", [])):
        raise ValueError("Zero-shot sample artifact row count does not match protocol labels.")
    if prompt_metadata.get("n_samples") != len(protocol.get("prompts", [])):
        raise ValueError("Zero-shot prompt artifact row count does not match protocol prompts.")
    evaluation_recipe = _evaluation_recipe(job.zero_shot_config, job.scoring_config)
    zero_config = ZeroShotConfig(**evaluation_recipe["zero_shot_config"])
    overlap_config = OverlapScoringConfig(**evaluation_recipe["overlap_scoring_config"])
    result = ZeroShotScorer(zero_config).score(
        samples,
        prompts,
        protocol["labels"],
        class_labels=protocol["class_labels"],
        prompt_labels=protocol["prompt_labels"],
        template_ids=protocol.get("template_ids"),
        sample_ids=protocol.get("sample_ids", list(range(len(protocol["labels"])))),
    )
    result.metadata["label_encoding"] = LABEL_ENCODING
    result.metadata["label_catalog"] = list(protocol["label_catalog"])
    overlap = OverlapMetric(config=overlap_config).score(
        samples,
        protocol["labels"],
        target_metadata={"target_type": "single_label"},
    )
    artifact = {
        "artifact_type": "zero_shot_evaluation",
        "vertebrae_version": __version__,
        "output_key": job.output_key,
        "sample_embedding_key": job.sample_embedding_key,
        "prompt_embedding_key": job.prompt_embedding_key,
        "protocol_key": job.protocol_key,
        "source_dataset_fingerprint": sample_metadata["source_dataset_fingerprint"],
        "protocol_fingerprint": protocol_fingerprint,
        "label_encoding": protocol["label_encoding"],
        "sample_endpoint": sample_metadata,
        "prompt_endpoint": prompt_metadata,
        "compression_metadata": sample_metadata.get("compression", {"method": "none"}),
        "evaluation_recipe": evaluation_recipe,
        "evaluation_fingerprint": hash_json_exact(evaluation_recipe),
        "zero_shot": result.to_dict(),
        "overlap": portable_json(
            {
                "name": overlap.name,
                "score": overlap.score,
                "higher_is_better": overlap.higher_is_better,
                "kind": overlap.kind,
                "diagnostics": overlap.diagnostics,
                "warnings": overlap.warnings,
                "metadata": overlap.metadata,
            }
        ),
        "resources": asdict(job.resources),
    }
    store.put_json(job.output_key, artifact)
    return artifact


def score_zero_shot_artifacts(
    jobs: Iterable[ZeroShotScoringJob], store: ArtifactStore, execution: Any
) -> list[dict]:
    """Submit independent zero-shot scoring jobs through local, Ray, or Dask execution."""

    return execution.map(partial(_score_zero_shot_artifact_job, store_config=store.config()), jobs)


def _score_zero_shot_artifact_job(
    job: ZeroShotScoringJob, store_config: ArtifactStoreConfig
) -> dict:
    return score_zero_shot_artifact(job, create_artifact_store_from_config(store_config))


def plan_zero_shot_embedding_shard_jobs(
    dataset: Any,
    extractor: Any,
    total_shards: int,
    *,
    side: str,
    branch: str,
) -> list[ZeroShotEmbeddingShardJob]:
    """Plan deterministic endpoint jobs for one zero-shot side."""

    if total_shards < 1:
        raise ValueError("total_shards must be >= 1.")
    n_rows = len(dataset.samples) if side == "samples" else len(dataset.prompt_rows()[0])
    planned_shards = min(total_shards, n_rows)
    base_key = zero_shot_embedding_artifact_key(dataset, extractor, side, branch)
    return [
        ZeroShotEmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            side=side,
            branch=branch,
            shard=ShardSpec(total_shards=planned_shards, shard_index=index),
            output_key=f"{base_key}/shards/{index:05d}-of-{planned_shards:05d}",
        )
        for index in range(planned_shards)
    ]


def _validate_pair(samples: dict, prompts: dict) -> None:
    expected_types = {"zero_shot_embedding", "zero_shot_compressed_embedding"}
    if (
        samples.get("artifact_type") not in expected_types
        or prompts.get("artifact_type") not in expected_types
    ):
        raise ValueError("Zero-shot endpoint artifacts must be merged endpoint embeddings.")
    if samples.get("side") != "samples" or prompts.get("side") != "prompts":
        raise ValueError("Zero-shot artifacts must pair samples with prompts.")
    if samples.get("recipe_hash") != prompts.get("recipe_hash"):
        raise ValueError("Zero-shot endpoint artifacts have different extractor recipes.")
    if samples.get("source_dataset_fingerprint") != prompts.get("source_dataset_fingerprint"):
        raise ValueError("Zero-shot endpoint artifacts have different source datasets.")
    if samples.get("embedding_dim") != prompts.get("embedding_dim"):
        raise ValueError("Zero-shot endpoint artifacts have incompatible embedding dimensions.")
    if samples.get("protocol_fingerprint") is not None:
        raise ValueError(
            "Reusable zero-shot sample endpoint artifacts must not bind a prompt protocol."
        )
    if not prompts.get("protocol_fingerprint"):
        raise ValueError("Zero-shot prompt endpoint artifacts must declare a prompt protocol.")
    sample_compressed = samples.get("artifact_type") == "zero_shot_compressed_embedding"
    prompt_compressed = prompts.get("artifact_type") == "zero_shot_compressed_embedding"
    if sample_compressed != prompt_compressed:
        raise ValueError("Zero-shot endpoints must both be raw or from the same compression pair.")
    if sample_compressed:
        sample_pair = samples.get("compression_pair_id")
        prompt_pair = prompts.get("compression_pair_id")
        if not sample_pair or sample_pair != prompt_pair:
            raise ValueError(
                "Compressed zero-shot endpoints must share a verified compression_pair_id; "
                "regenerate the compressed endpoint pair."
            )


def _validate_endpoint_array(values: Any, metadata: dict) -> None:
    if values.ndim != 2:
        raise ValueError("Zero-shot endpoint artifacts must contain two-dimensional embeddings.")
    if values.shape[0] != metadata.get("n_samples") or values.shape[1] != metadata.get(
        "embedding_dim"
    ):
        raise ValueError("Zero-shot endpoint artifact metadata does not match its embedding array.")


def _validate_shards(manifests: list[dict], n_samples: int) -> None:
    required = {
        "artifact_type",
        "source_dataset_fingerprint",
        "recipe_hash",
        "side",
        "branch",
    }
    if any(item.get("artifact_type") != "zero_shot_embedding_shard" for item in manifests):
        raise ValueError("All merged artifacts must be zero-shot embedding shards.")
    for field in required:
        if len({item.get(field) for item in manifests}) != 1:
            raise ValueError(f"Zero-shot shard manifests disagree on {field}.")
    side = manifests[0]["side"]
    protocol_values = {item.get("protocol_fingerprint") for item in manifests}
    if side == "samples" and protocol_values != {None}:
        raise ValueError("Sample endpoint shards must not be tied to a prompt protocol.")
    if side == "prompts" and (len(protocol_values) != 1 or None in protocol_values):
        raise ValueError("Prompt endpoint shards must agree on a prompt protocol.")
    covered = np.concatenate([np.asarray(item["sample_indices"], dtype=int) for item in manifests])
    if len(covered) != n_samples or set(covered.tolist()) != set(range(n_samples)):
        raise ValueError("Zero-shot shards do not cover each endpoint row exactly once.")


def _take_rows(values: Any, indices: np.ndarray) -> Any:
    if isinstance(values, dict):
        return {key: _take_rows(value, indices) for key, value in values.items()}
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    if is_sparse_matrix(values) or isinstance(values, np.ndarray):
        return values[indices]
    sequence = list(values)
    return [sequence[int(index)] for index in indices]


def zero_shot_benchmark_result_from_artifacts(
    score_keys: Iterable[str], store: ArtifactStore, output_key: Optional[str] = None
) -> Any:
    """Reconstruct a rankable zero-shot report from persisted score artifacts.

    Alignment and overlap results remain separate fields; the ranking always follows
    the configured zero-shot primary metric recorded in each score artifact.
    """

    from vertebrae.zero_shot import ZeroShotBenchmarkResult, ZeroShotExtractorResult, _variant_name

    keys = list(score_keys)
    artifacts = [store.get_json(key) for key in keys]
    if not artifacts:
        raise ValueError("At least one zero-shot score artifact is required.")
    if any(item.get("artifact_type") != "zero_shot_evaluation" for item in artifacts):
        raise ValueError("All score keys must reference zero-shot evaluation artifacts.")
    protocol_fingerprints = {item.get("protocol_fingerprint") for item in artifacts}
    if len(protocol_fingerprints) != 1:
        raise ValueError(
            "Zero-shot score artifacts must share one prompt protocol to report together."
        )
    evaluation_fingerprints = {item.get("evaluation_fingerprint") for item in artifacts}
    if None in evaluation_fingerprints or len(evaluation_fingerprints) != 1:
        raise ValueError(
            "Zero-shot score artifacts must share one verified evaluation configuration; "
            "regenerate invalid score artifacts before combining them."
        )
    protocols = [store.get_json(item["protocol_key"]) for item in artifacts]
    for item in protocols:
        _validate_protocol_artifact(item)
    protocol = protocols[0]
    if any(
        artifact.get("protocol_fingerprint") != item.get("protocol_fingerprint")
        or artifact.get("label_encoding") != item.get("label_encoding")
        for artifact, item in zip(artifacts, protocols)
    ):
        raise ValueError("Zero-shot score artifacts do not match their referenced protocols.")
    results = []
    for artifact in artifacts:
        sample_endpoint = artifact["sample_endpoint"]
        recipe = dict(sample_endpoint["extractor_recipe"])
        zero_shot = _zero_shot_result_from_dict(artifact["zero_shot"])
        overlap = _metric_result_from_dict(artifact["overlap"])
        compression = dict(artifact.get("compression_metadata") or {"method": "none"})
        results.append(
            ZeroShotExtractorResult(
                name=_variant_name(recipe.get("name", "extractor"), compression),
                extractor_type=recipe.get("extractor_type", "unknown"),
                zero_shot=zero_shot,
                overlap=overlap,
                primary_score=zero_shot.score,
                compression_metadata=compression,
                runtime={},
                embedding_metadata={
                    "sample_branch": sample_endpoint["branch"],
                    "text_branch": artifact["prompt_endpoint"]["branch"],
                    "source_dataset_fingerprint": artifact["source_dataset_fingerprint"],
                    "protocol_fingerprint": artifact["protocol_fingerprint"],
                    "sample_embedding_dim": sample_endpoint["embedding_dim"],
                    "prompt_embedding_dim": artifact["prompt_endpoint"]["embedding_dim"],
                },
                cache_metadata={"artifact_backed": True},
                warnings=sorted(
                    set(
                        list(zero_shot.warnings)
                        + list(overlap.warnings)
                        + list(compression.get("warnings", []))
                    )
                ),
                recipe={
                    **recipe,
                    "zero_shot_sample_branch": sample_endpoint["branch"],
                    "zero_shot_text_branch": artifact["prompt_endpoint"]["branch"],
                },
            )
        )
    dataset_summary = dict(
        protocol.get(
            "dataset_summary",
            {
                "modality": "zero_shot",
                "sample_modality": protocol.get("sample_modality"),
                "n_samples": protocol["n_samples"],
                "n_classes": len(protocol["class_labels"]),
                "n_prompts": protocol["n_prompts"],
                "protocol": protocol.get("protocol"),
                "source_dataset_fingerprint": protocol["source_dataset_fingerprint"],
            },
        )
    )
    evaluation_recipe = artifacts[0].get("evaluation_recipe")
    result = ZeroShotBenchmarkResult(
        dataset_summary=dataset_summary,
        extractor_results=results,
        metadata={
            "artifact_backed": True,
            "score_keys": keys,
            "protocol_key": artifacts[0]["protocol_key"],
            "protocol": protocol.get("protocol"),
            "evaluation_recipe": evaluation_recipe,
            "evaluation_fingerprint": artifacts[0].get("evaluation_fingerprint"),
            "interpretation": (
                "Zero-shot scores measure frozen semantic text alignment. Overlap is "
                "reported as separate contextual evidence and is not combined with alignment."
            ),
        },
    )
    if output_key:
        store.put_json(output_key, result.to_dict())
    return result


def _zero_shot_result_from_dict(value: dict) -> ZeroShotScoreResult:
    return ZeroShotScoreResult(
        score=float(value["score"]),
        primary_metric=value["primary_metric"],
        metrics=dict(value["metrics"]),
        per_class=dict(value["per_class"]),
        confusion_matrix=list(value["confusion_matrix"]),
        diagnostics=dict(value.get("diagnostics", {})),
        warnings=list(value.get("warnings", [])),
        metadata=dict(value.get("metadata", {})),
    )


def _metric_result_from_dict(value: dict) -> MetricResult:
    return MetricResult(
        name=value["name"],
        score=float(value["score"]),
        higher_is_better=bool(value.get("higher_is_better", True)),
        kind=value.get("kind", "custom"),
        diagnostics=dict(value.get("diagnostics", {})),
        warnings=list(value.get("warnings", [])),
        metadata=dict(value.get("metadata", {})),
    )


def _validate_protocol_artifact(protocol: dict) -> None:
    """Verify protocol schema, duplicated fields, and content-derived identity."""

    if protocol.get("artifact_type") != "zero_shot_protocol":
        raise ValueError("protocol_key does not reference a zero-shot protocol artifact.")
    encoding = protocol.get("label_encoding")
    if encoding != LABEL_ENCODING:
        raise ValueError(
            "Zero-shot protocol artifacts must use label encoding "
            f"{LABEL_ENCODING!r}; found {encoding!r}."
        )
    catalog = validate_label_catalog(protocol.get("label_catalog"))
    recipe = protocol.get("protocol")
    if not isinstance(recipe, dict):
        raise ValueError("Zero-shot protocol artifact is missing its protocol recipe.")
    identity_recipe = dict(recipe)
    declared_fingerprint = identity_recipe.pop("protocol_fingerprint", None)
    computed_fingerprint = hash_json_exact(identity_recipe)
    if not declared_fingerprint or computed_fingerprint != declared_fingerprint:
        raise ValueError("Zero-shot protocol recipe fingerprint does not match its content.")
    if (
        protocol.get("protocol_fingerprint") != computed_fingerprint
        or protocol.get("dataset_fingerprint") != computed_fingerprint
    ):
        raise ValueError("Zero-shot protocol manifest fingerprint does not match its recipe.")
    expected = {
        "label_encoding": recipe.get("label_encoding"),
        "label_catalog": recipe.get("label_catalog"),
        "labels": recipe.get("sample_labels"),
        "sample_ids": recipe.get("sample_ids"),
        "class_labels": recipe.get("ordered_labels"),
        "prompts": recipe.get("prompts"),
        "prompt_labels": recipe.get("prompt_labels"),
        "template_ids": recipe.get("template_ids"),
        "n_samples": recipe.get("n_samples"),
        "n_prompts": recipe.get("n_prompts"),
        "source_dataset_fingerprint": recipe.get("source_dataset_fingerprint"),
    }
    for field, value in expected.items():
        if protocol.get(field) != value:
            raise ValueError(f"Zero-shot protocol manifest {field} does not match its recipe.")
    catalog_keys = {item["key"] for item in catalog}
    classes = protocol.get("class_labels")
    labels = protocol.get("labels")
    prompt_labels = protocol.get("prompt_labels")
    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Zero-shot protocol must declare at least two unique class labels.")
    if set(classes) != catalog_keys:
        raise ValueError("Zero-shot protocol classes do not match its label catalog.")
    if classes != [item["key"] for item in catalog]:
        raise ValueError("Zero-shot protocol class order does not match its label catalog.")
    if not isinstance(labels, list) or set(labels) != set(classes):
        raise ValueError("Zero-shot protocol sample labels do not cover its declared classes.")
    if not isinstance(prompt_labels, list) or set(prompt_labels) != set(classes):
        raise ValueError("Zero-shot protocol prompt labels do not cover its declared classes.")
    if len(labels) != protocol.get("n_samples") or len(protocol.get("sample_ids", [])) != len(
        labels
    ):
        raise ValueError("Zero-shot protocol sample rows are inconsistent.")
    if len(protocol.get("prompts", [])) != protocol.get("n_prompts") or len(
        prompt_labels
    ) != protocol.get("n_prompts"):
        raise ValueError("Zero-shot protocol prompt rows are inconsistent.")
    template_ids = protocol.get("template_ids")
    if template_ids is not None and len(template_ids) != protocol.get("n_prompts"):
        raise ValueError("Zero-shot protocol template IDs do not align with prompts.")
    if any(count < 2 for count in Counter(labels).values()):
        raise ValueError("Zero-shot protocol classes must each contain at least two samples.")
    _validate_protocol_class_specs(recipe, classes)


def _validate_protocol_class_specs(recipe: dict, classes: list[str]) -> None:
    specs = recipe.get("class_specs")
    if not isinstance(specs, list) or len(specs) != len(classes):
        raise ValueError("Zero-shot class specifications do not align with declared classes.")
    if [item.get("label") for item in specs if isinstance(item, dict)] != classes:
        raise ValueError("Zero-shot class specifications do not preserve declared class order.")
    prompts = []
    prompt_labels = []
    template_ids = []
    has_templates = all(item.get("template_ids") is not None for item in specs)
    for item in specs:
        declared_prompts = item.get("prompts")
        if not isinstance(declared_prompts, list) or not declared_prompts:
            raise ValueError("Every zero-shot class specification must contain prompts.")
        prompts.extend(declared_prompts)
        prompt_labels.extend([item["label"]] * len(declared_prompts))
        if has_templates:
            declared_templates = item.get("template_ids")
            if not isinstance(declared_templates, list) or len(declared_templates) != len(
                declared_prompts
            ):
                raise ValueError("Zero-shot class template IDs do not align with prompts.")
            template_ids.extend(declared_templates)
    if prompts != recipe.get("prompts") or prompt_labels != recipe.get("prompt_labels"):
        raise ValueError("Zero-shot class specifications disagree with flattened prompt rows.")
    expected_templates = template_ids if has_templates else None
    if expected_templates != recipe.get("template_ids"):
        raise ValueError("Zero-shot class specifications disagree with flattened template IDs.")
