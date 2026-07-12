"""Command line interface for distributed vertebrae workflows."""

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from vertebrae.cache import create_artifact_store
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.config import (
    EmbeddingCompressionConfig,
    OverlapScoringConfig,
    RetrievalConfig,
    ZeroShotConfig,
)
from vertebrae.execution import (
    EmbeddingMergeJob,
    RetrievalCompressionJob,
    RetrievalEmbeddingShardJob,
    RetrievalScoringJob,
    ScoringJob,
    SeparatixJob,
    ShardSpec,
    ZeroShotCompressionJob,
    ZeroShotEmbeddingShardJob,
    ZeroShotScoringJob,
    benchmark_result_from_artifacts,
    collect_score_artifacts,
    compress_embedding_artifact,
    compress_retrieval_embedding_artifacts,
    compress_zero_shot_embedding_artifacts,
    create_execution_backend,
    diagnose_embedding_artifact,
    embedding_artifact_key,
    embedding_output_key,
    embedding_output_shard_key,
    embedding_shard_key,
    groups_artifact_key,
    labels_artifact_key,
    materialize_embedding_shard,
    materialize_group_artifact,
    materialize_label_artifact,
    materialize_retrieval_embedding_shard,
    materialize_segmentation_artifacts,
    materialize_structured_artifacts,
    materialize_zero_shot_embedding_shard,
    materialize_zero_shot_protocol,
    merge_embedding_shards,
    merge_retrieval_embedding_shards,
    merge_zero_shot_embedding_shards,
    plan_compression_job,
    plan_embedding_shard_jobs,
    plan_retrieval_embedding_shard_jobs,
    plan_scoring_jobs,
    plan_zero_shot_embedding_shard_jobs,
    retrieval_compression_artifact_key,
    retrieval_embedding_artifact_key,
    retrieval_embedding_shard_key,
    retrieval_scoring_artifact_key,
    score_embedding_artifact,
    score_embedding_artifacts,
    score_retrieval_artifact,
    score_zero_shot_artifact,
    scoring_artifact_key,
    separatix_artifact_key,
    zero_shot_benchmark_result_from_artifacts,
    zero_shot_compression_artifact_key,
    zero_shot_embedding_artifact_key,
    zero_shot_protocol_artifact_key,
    zero_shot_scoring_artifact_key,
)
from vertebrae.execution.jobs import EmbeddingShardJob
from vertebrae.scoring.metrics import CallableMetric
from vertebrae.structured import (
    drop_special_rows,
    keep_row_indices,
    select_frame_rows,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the vertebrae CLI.

    Args:
        argv: Optional argument vector. Uses `sys.argv` when omitted.

    Returns:
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    payload = args.func(args)
    if payload is not None:
        _write_json_payload(payload, getattr(args, "output_json", None))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(prog="vertebrae")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Plan deterministic embedding shard jobs.")
    _add_object_args(plan)
    _add_cache_arg(plan)
    plan.add_argument("--total-shards", type=int, required=True)
    plan.add_argument("--batch-size", type=int, default=128)
    _add_backend_args(plan, include_local_parallel=True)
    plan.add_argument("--output-json")
    plan.set_defaults(func=_cmd_plan)

    embed = subparsers.add_parser("embed-shard", help="Materialize one embedding shard.")
    _add_object_args(embed)
    _add_cache_arg(embed)
    embed.add_argument("--total-shards", type=int, required=True)
    embed.add_argument("--shard-index", type=int, required=True)
    embed.add_argument("--batch-size", type=int, default=128)
    embed.add_argument("--output-key")
    embed.add_argument("--output-json")
    embed.set_defaults(func=_cmd_embed_shard)

    merge = subparsers.add_parser("merge-embeddings", help="Merge embedding shard artifacts.")
    _add_cache_arg(merge)
    merge.add_argument("--plan-json")
    merge.add_argument("--shard-key", action="append", default=[])
    merge.add_argument("--output-key")
    merge.add_argument("--n-samples", type=int)
    merge.add_argument("--output-json")
    merge.set_defaults(func=_cmd_merge_embeddings)

    retrieval_plan = subparsers.add_parser(
        "plan-retrieval", help="Plan deterministic query and gallery embedding shards."
    )
    _add_object_args(retrieval_plan)
    _add_cache_arg(retrieval_plan)
    retrieval_plan.add_argument("--total-shards", type=int, required=True)
    retrieval_plan.add_argument("--query-branch")
    retrieval_plan.add_argument("--gallery-branch")
    retrieval_plan.add_argument("--batch-size", type=int, default=128)
    retrieval_plan.add_argument("--output-json")
    retrieval_plan.set_defaults(func=_cmd_plan_retrieval)

    retrieval_embed = subparsers.add_parser(
        "embed-retrieval-shard", help="Materialize one query or gallery retrieval endpoint shard."
    )
    _add_object_args(retrieval_embed)
    _add_cache_arg(retrieval_embed)
    retrieval_embed.add_argument("--side", choices=["query", "gallery"], required=True)
    retrieval_embed.add_argument("--branch")
    retrieval_embed.add_argument("--total-shards", type=int, required=True)
    retrieval_embed.add_argument("--shard-index", type=int, required=True)
    retrieval_embed.add_argument("--batch-size", type=int, default=128)
    retrieval_embed.add_argument("--output-key")
    retrieval_embed.add_argument("--output-json")
    retrieval_embed.set_defaults(func=_cmd_embed_retrieval_shard)

    retrieval_merge = subparsers.add_parser(
        "merge-retrieval-embeddings", help="Merge one retrieval endpoint's shard artifacts."
    )
    _add_cache_arg(retrieval_merge)
    retrieval_merge.add_argument("--plan-json")
    retrieval_merge.add_argument("--side", choices=["query", "gallery"])
    retrieval_merge.add_argument("--shard-key", action="append", default=[])
    retrieval_merge.add_argument("--output-key")
    retrieval_merge.add_argument("--n-samples", type=int)
    retrieval_merge.add_argument("--output-json")
    retrieval_merge.set_defaults(func=_cmd_merge_retrieval_embeddings)

    zero_plan = subparsers.add_parser(
        "plan-zero-shot", help="Plan deterministic sample and prompt embedding shards."
    )
    _add_object_args(zero_plan)
    _add_cache_arg(zero_plan)
    zero_plan.add_argument("--total-shards", type=int, required=True)
    zero_plan.add_argument("--sample-branch", required=True)
    zero_plan.add_argument("--text-branch", required=True)
    zero_plan.add_argument("--output-json")
    zero_plan.set_defaults(func=_cmd_plan_zero_shot)

    zero_embed = subparsers.add_parser(
        "embed-zero-shot-shard", help="Materialize one zero-shot sample or prompt shard."
    )
    _add_object_args(zero_embed)
    _add_cache_arg(zero_embed)
    zero_embed.add_argument("--side", choices=["samples", "prompts"], required=True)
    zero_embed.add_argument("--plan-json")
    zero_embed.add_argument("--branch")
    zero_embed.add_argument("--total-shards", type=int)
    zero_embed.add_argument("--shard-index", type=int, required=True)
    zero_embed.add_argument("--output-key")
    zero_embed.add_argument("--output-json")
    zero_embed.set_defaults(func=_cmd_embed_zero_shot_shard)

    zero_merge = subparsers.add_parser(
        "merge-zero-shot-embeddings", help="Merge one zero-shot endpoint's shard artifacts."
    )
    _add_cache_arg(zero_merge)
    zero_merge.add_argument("--plan-json")
    zero_merge.add_argument("--side", choices=["samples", "prompts"])
    zero_merge.add_argument("--shard-key", action="append", default=[])
    zero_merge.add_argument("--output-key")
    zero_merge.add_argument("--n-samples", type=int)
    zero_merge.add_argument("--output-json")
    zero_merge.set_defaults(func=_cmd_merge_zero_shot_embeddings)

    zero_protocol = subparsers.add_parser(
        "write-zero-shot-protocol", help="Materialize a ZeroShotDataset prompt protocol."
    )
    zero_protocol.add_argument("--dataset-pickle", required=True)
    _add_cache_arg(zero_protocol)
    zero_protocol.add_argument("--output-key")
    zero_protocol.add_argument("--output-json")
    zero_protocol.set_defaults(func=_cmd_write_zero_shot_protocol)

    labels = subparsers.add_parser("write-labels", help="Materialize dataset labels.")
    labels.add_argument("--dataset-pickle", required=True)
    _add_cache_arg(labels)
    labels.add_argument("--output-key")
    labels.add_argument("--output-json")
    labels.set_defaults(func=_cmd_write_labels)

    groups = subparsers.add_parser("write-groups", help="Materialize dataset groups.")
    groups.add_argument("--dataset-pickle", required=True)
    _add_cache_arg(groups)
    groups.add_argument("--output-key")
    groups.add_argument("--output-json")
    groups.set_defaults(func=_cmd_write_groups)

    segmentation = subparsers.add_parser(
        "materialize-segmentation",
        help="Materialize spatial segmentation embeddings, labels, groups, and provenance.",
    )
    _add_object_args(segmentation)
    _add_cache_arg(segmentation)
    segmentation.add_argument("--segmentation-config-pickle")
    segmentation.add_argument("--batch-size", type=int, default=16)
    segmentation.add_argument("--output-json")
    segmentation.set_defaults(func=_cmd_materialize_segmentation)

    structured = subparsers.add_parser(
        "materialize-structured",
        help="Materialize structured unit embeddings, labels, groups, and provenance.",
    )
    _add_object_args(structured)
    _add_cache_arg(structured)
    structured.add_argument("--batch-size", type=int, default=16)
    structured.add_argument(
        "--aligner",
        action="append",
        default=[],
        metavar="OUTPUT=HELPER[:JSON]",
        help=(
            "Attach a standard structured aligner recipe to one output, for example "
            'tokens=drop_special_rows:{"leading":1,"trailing":1} or '
            'frames=select_frame_rows:{"indices_metadata_key":"sampled_frames"}.'
        ),
    )
    structured.add_argument("--output-json")
    structured.set_defaults(func=_cmd_materialize_structured)

    score = subparsers.add_parser("score", help="Score persisted embeddings and labels.")
    _add_cache_arg(score)
    score.add_argument("--embedding-key")
    score.add_argument("--labels-key")
    score.add_argument("--output-key")
    score.add_argument("--plan-json")
    score.add_argument("--scoring-config-pickle")
    score.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Importable custom metric in module:callable form.",
    )
    score.add_argument("--metric-name", action="append", default=[])
    score.add_argument(
        "--metric-config-json",
        action="append",
        default=[],
        help="JSON object forwarded to the corresponding custom metric as config.",
    )
    score.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Rank lower custom metric scores ahead of higher scores.",
    )
    score.add_argument("--seed", type=int)
    score.add_argument("--primary-metric", default="overlap")
    score.add_argument("--output-json")
    score.set_defaults(func=_cmd_score)

    retrieval_score = subparsers.add_parser(
        "score-retrieval", help="Score persisted query/gallery embeddings and relevance."
    )
    _add_cache_arg(retrieval_score)
    retrieval_score.add_argument("--query-embedding-key", required=True)
    retrieval_score.add_argument("--gallery-embedding-key", required=True)
    retrieval_score.add_argument("--relevance-key", required=True)
    retrieval_score.add_argument("--exclusions-key")
    retrieval_score.add_argument("--retrieval-config-pickle")
    retrieval_score.add_argument("--output-key")
    retrieval_score.add_argument("--output-json")
    retrieval_score.set_defaults(func=_cmd_score_retrieval)

    retrieval_compress = subparsers.add_parser(
        "compress-retrieval", help="Fit gallery compression and transform paired query embeddings."
    )
    _add_cache_arg(retrieval_compress)
    retrieval_compress.add_argument("--query-embedding-key", required=True)
    retrieval_compress.add_argument("--gallery-embedding-key", required=True)
    retrieval_compress.add_argument("--compression-config-pickle", required=True)
    retrieval_compress.add_argument("--output-prefix")
    retrieval_compress.add_argument("--output-json")
    retrieval_compress.set_defaults(func=_cmd_compress_retrieval)

    relevance = subparsers.add_parser(
        "write-retrieval-relevance",
        help="Materialize a RetrievalDataset relevance artifact.",
    )
    relevance.add_argument("--dataset-pickle", required=True)
    _add_cache_arg(relevance)
    relevance.add_argument("--output-key")
    relevance.add_argument("--output-json")
    relevance.set_defaults(func=_cmd_write_retrieval_relevance)

    zero_score = subparsers.add_parser(
        "score-zero-shot", help="Score persisted frozen sample and prompt embeddings."
    )
    _add_cache_arg(zero_score)
    zero_score.add_argument("--sample-embedding-key", required=True)
    zero_score.add_argument("--prompt-embedding-key", required=True)
    zero_score.add_argument("--protocol-key", required=True)
    zero_score.add_argument("--zero-shot-config-pickle")
    zero_score.add_argument("--scoring-config-pickle")
    zero_score.add_argument("--output-key")
    zero_score.add_argument("--output-json")
    zero_score.set_defaults(func=_cmd_score_zero_shot)

    zero_compress = subparsers.add_parser(
        "compress-zero-shot", help="Fit sample compression and transform paired prompts."
    )
    _add_cache_arg(zero_compress)
    zero_compress.add_argument("--sample-embedding-key", required=True)
    zero_compress.add_argument("--prompt-embedding-key", required=True)
    zero_compress.add_argument("--compression-config-pickle", required=True)
    zero_compress.add_argument("--output-prefix")
    zero_compress.add_argument("--output-json")
    zero_compress.set_defaults(func=_cmd_compress_zero_shot)

    zero_collect = subparsers.add_parser(
        "zero-shot-from-artifacts",
        help="Reconstruct ranked zero-shot JSON and Markdown reports from score artifacts.",
    )
    _add_cache_arg(zero_collect)
    zero_collect.add_argument("--score-key", action="append", required=True)
    zero_collect.add_argument("--output-key")
    zero_collect.add_argument("--json-output")
    zero_collect.add_argument("--markdown-output")
    zero_collect.add_argument("--output-json")
    zero_collect.set_defaults(func=_cmd_zero_shot_from_artifacts)

    diagnose = subparsers.add_parser(
        "diagnose-complexity",
        help="Run Separatix on persisted embeddings and labels.",
    )
    _add_cache_arg(diagnose)
    diagnose.add_argument("--embedding-key")
    diagnose.add_argument("--labels-key")
    diagnose.add_argument("--score-key")
    diagnose.add_argument("--plan-json")
    diagnose.add_argument("--separatix-config-pickle")
    diagnose.add_argument("--groups-key")
    diagnose.add_argument("--output-key")
    diagnose.add_argument("--output-json")
    diagnose.set_defaults(func=_cmd_diagnose_complexity)

    compress = subparsers.add_parser("compress", help="Compress a persisted embedding artifact.")
    _add_cache_arg(compress)
    compress.add_argument("--embedding-key", required=True)
    compress.add_argument(
        "--method",
        choices=[
            "none",
            "pca",
            "incremental_pca",
            "truncated_svd",
            "gaussian_random_projection",
            "sparse_random_projection",
            "prefix_truncate",
            "quantize",
        ],
        required=True,
    )
    compress.add_argument("--n-components", type=int)
    compress.add_argument("--preserve-variance", type=float)
    compress.add_argument("--precision")
    compress.add_argument("--assume-matryoshka", action="store_true")
    compress.add_argument("--random-state", type=int, default=42)
    compress.add_argument("--whiten", action="store_true")
    compress.add_argument("--dtype")
    compress.add_argument("--output-key")
    compress.add_argument("--output-json")
    compress.set_defaults(func=_cmd_compress)

    repeats = subparsers.add_parser("score-repeats", help="Run repeated scoring jobs.")
    _add_cache_arg(repeats)
    repeats.add_argument("--embedding-key")
    repeats.add_argument("--labels-key")
    repeats.add_argument("--plan-json")
    repeats.add_argument("--seed", action="append", type=int, default=[])
    repeats.add_argument("--repeats", type=int)
    repeats.add_argument("--random-state", type=int, default=42)
    repeats.add_argument("--scoring-config-pickle")
    repeats.add_argument("--metric", action="append", default=[])
    repeats.add_argument("--metric-name", action="append", default=[])
    repeats.add_argument("--metric-config-json", action="append", default=[])
    repeats.add_argument("--lower-is-better", action="store_true")
    repeats.add_argument("--primary-metric", default="overlap")
    _add_backend_args(repeats, include_local_parallel=True)
    repeats.add_argument("--output-json")
    repeats.set_defaults(func=_cmd_score_repeats)

    collect = subparsers.add_parser("collect-scores", help="Collect score artifacts.")
    _add_cache_arg(collect)
    collect.add_argument("--score-key", action="append", default=[])
    collect.add_argument("--score-plan-json")
    collect.add_argument("--output-key", required=True)
    collect.add_argument("--interval-level", type=float, default=0.95)
    collect.add_argument("--metric-name")
    collect.add_argument("--output-json")
    collect.set_defaults(func=_cmd_collect_scores)

    artifacts = subparsers.add_parser(
        "benchmark-from-artifacts",
        help="Build a benchmark-style result from score artifacts.",
    )
    _add_cache_arg(artifacts)
    artifacts.add_argument("--score-key", required=True)
    artifacts.add_argument("--stability-key")
    artifacts.add_argument("--separatix-key")
    artifacts.add_argument("--output-key")
    artifacts.add_argument("--json-output")
    artifacts.add_argument("--markdown-output")
    artifacts.add_argument("--output-json")
    artifacts.set_defaults(func=_cmd_benchmark_from_artifacts)

    slurm = subparsers.add_parser("slurm-array", help="Generate a SLURM array script.")
    _add_object_args(slurm)
    _add_cache_arg(slurm)
    slurm.add_argument("--total-shards", type=int, required=True)
    slurm.add_argument("--batch-size", type=int, default=128)
    slurm.add_argument("--script-output", required=True)
    slurm.add_argument("--job-name", default="vertebrae-embed")
    slurm.add_argument("--time", default="04:00:00")
    slurm.add_argument("--mem", default="16G")
    slurm.add_argument("--cpus-per-task", type=int, default=1)
    slurm.add_argument("--partition")
    slurm.add_argument("--python-executable", default=sys.executable)
    slurm.add_argument("--output-json")
    slurm.set_defaults(func=_cmd_slurm_array)

    slurm_score = subparsers.add_parser(
        "slurm-score-array",
        help="Generate a SLURM array script for repeated scoring.",
    )
    _add_cache_arg(slurm_score)
    slurm_score.add_argument("--embedding-key")
    slurm_score.add_argument("--labels-key")
    slurm_score.add_argument("--plan-json")
    slurm_score.add_argument("--repeats", type=int, required=True)
    slurm_score.add_argument("--random-state", type=int, default=42)
    slurm_score.add_argument("--script-output", required=True)
    slurm_score.add_argument("--job-name", default="vertebrae-score")
    slurm_score.add_argument("--time", default="04:00:00")
    slurm_score.add_argument("--mem", default="16G")
    slurm_score.add_argument("--cpus-per-task", type=int, default=1)
    slurm_score.add_argument("--partition")
    slurm_score.add_argument("--python-executable", default=sys.executable)
    slurm_score.add_argument("--output-json")
    slurm_score.set_defaults(func=_cmd_slurm_score_array)

    run_embed = subparsers.add_parser(
        "run-embedding-shards",
        help="Run embedding shards with the selected execution backend.",
    )
    _add_object_args(run_embed)
    _add_cache_arg(run_embed)
    run_embed.add_argument("--total-shards", type=int, required=True)
    run_embed.add_argument("--batch-size", type=int, default=128)
    _add_backend_args(run_embed, include_local_parallel=True)
    run_embed.add_argument("--output-json")
    run_embed.set_defaults(func=_cmd_run_embedding_shards)

    return parser


def _cmd_plan(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    jobs = plan_embedding_shard_jobs(
        dataset=dataset,
        extractor=extractor,
        total_shards=args.total_shards,
        batch_size=args.batch_size,
    )
    base_key = embedding_artifact_key(dataset, extractor)
    plan = {
        "dataset_pickle": str(Path(args.dataset_pickle)),
        "extractor_pickle": str(Path(args.extractor_pickle)),
        "cache_dir": args.cache_dir,
        "storage_options": _artifact_store_options_from_args(args),
        "base_key": base_key,
        "output_key": base_key,
        "labels_key": labels_artifact_key(dataset),
        "groups_key": (
            groups_artifact_key(dataset)
            if callable(getattr(dataset, "groups", None)) and dataset.groups() is not None
            else None
        ),
        "score_key": scoring_artifact_key(base_key),
        "n_samples": int(len(dataset.y)),
        "total_shards": args.total_shards,
        "batch_size": args.batch_size,
        "backend": args.backend,
        "shard_jobs": [
            {
                "total_shards": job.shard.total_shards,
                "shard_index": job.shard.shard_index,
                "output_key": job.output_key,
            }
            for job in jobs
        ],
    }
    output_names = _multi_output_plan_names(extractor)
    if output_names:
        plan["outputs"] = [
            {
                "name": output_name,
                "output_key": embedding_output_key(base_key, output_name),
                "score_key": scoring_artifact_key(embedding_output_key(base_key, output_name)),
                "shard_keys": [
                    embedding_output_shard_key(job["output_key"], output_name)
                    for job in plan["shard_jobs"]
                ],
            }
            for output_name in output_names
        ]
    return plan


def _cmd_embed_shard(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    shard = ShardSpec(total_shards=args.total_shards, shard_index=args.shard_index)
    output_key = args.output_key or embedding_shard_key(
        embedding_artifact_key(dataset, extractor),
        shard,
    )
    job = EmbeddingShardJob(
        dataset=dataset,
        extractor=extractor,
        shard=shard,
        output_key=output_key,
        batch_size=args.batch_size,
    )
    return materialize_embedding_shard(job, _store_from_args(args))


def _cmd_merge_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json) if args.plan_json else {}
    shard_keys = tuple(args.shard_key or [job["output_key"] for job in plan.get("shard_jobs", [])])
    output_key = args.output_key or plan.get("output_key")
    n_samples = args.n_samples or plan.get("n_samples")
    if not shard_keys:
        raise ValueError("merge-embeddings requires --shard-key or --plan-json.")
    if output_key is None:
        raise ValueError("merge-embeddings requires --output-key or --plan-json.")
    if n_samples is None:
        raise ValueError("merge-embeddings requires --n-samples or --plan-json.")
    return merge_embedding_shards(
        EmbeddingMergeJob(
            shard_keys=shard_keys,
            output_key=output_key,
            n_samples=int(n_samples),
        ),
        _store_from_args(args),
    )


def _cmd_plan_retrieval(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    query_jobs = plan_retrieval_embedding_shard_jobs(
        dataset,
        extractor,
        args.total_shards,
        side="query",
        branch=args.query_branch,
        batch_size=args.batch_size,
    )
    gallery_jobs = plan_retrieval_embedding_shard_jobs(
        dataset,
        extractor,
        args.total_shards,
        side="gallery",
        branch=args.gallery_branch,
        batch_size=args.batch_size,
    )
    return {
        "artifact_type": "retrieval_embedding_plan",
        "dataset_fingerprint": dataset.fingerprint(),
        "extractor_recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "endpoints": {
            "query": _retrieval_endpoint_plan(query_jobs),
            "gallery": _retrieval_endpoint_plan(gallery_jobs),
        },
    }


def _retrieval_endpoint_plan(jobs: Sequence[RetrievalEmbeddingShardJob]) -> dict[str, Any]:
    """Serialize endpoint jobs without embedding live dataset or extractor objects."""
    if not jobs:
        raise ValueError("Retrieval endpoint planning requires at least one shard job.")
    first = jobs[0]
    values = first.dataset.queries if first.side == "query" else first.dataset.gallery
    base_key = retrieval_embedding_artifact_key(
        first.dataset, first.extractor, first.side, first.branch
    )
    return {
        "side": first.side,
        "branch": first.branch,
        "n_samples": len(values),
        "output_key": base_key,
        "shards": [
            {
                "side": job.side,
                "branch": job.branch,
                "shard": asdict(job.shard),
                "output_key": job.output_key,
                "batch_size": job.batch_size,
            }
            for job in jobs
        ],
    }


def _cmd_embed_retrieval_shard(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    shard = ShardSpec(total_shards=args.total_shards, shard_index=args.shard_index)
    base_key = retrieval_embedding_artifact_key(dataset, extractor, args.side, args.branch)
    return materialize_retrieval_embedding_shard(
        RetrievalEmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            side=args.side,
            branch=args.branch,
            shard=shard,
            output_key=args.output_key or retrieval_embedding_shard_key(base_key, shard),
            batch_size=args.batch_size,
        ),
        _store_from_args(args),
    )


def _cmd_merge_retrieval_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json) if args.plan_json else {}
    endpoint = plan.get("endpoints", {}).get(args.side) if args.side else None
    shard_keys = tuple(
        args.shard_key or (job["output_key"] for job in (endpoint or {}).get("shards", []))
    )
    output_key = args.output_key or (endpoint or {}).get("output_key")
    n_samples = args.n_samples or (endpoint or {}).get("n_samples")
    if not shard_keys or output_key is None or n_samples is None:
        raise ValueError(
            "merge-retrieval-embeddings requires explicit shard keys, output key, and sample count "
            "or --plan-json with --side."
        )
    return merge_retrieval_embedding_shards(
        EmbeddingMergeJob(
            shard_keys=shard_keys,
            output_key=output_key,
            n_samples=int(n_samples),
        ),
        _store_from_args(args),
    )


def _cmd_plan_zero_shot(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    sample_jobs = plan_zero_shot_embedding_shard_jobs(
        dataset,
        extractor,
        args.total_shards,
        side="samples",
        branch=args.sample_branch,
    )
    prompt_jobs = plan_zero_shot_embedding_shard_jobs(
        dataset,
        extractor,
        args.total_shards,
        side="prompts",
        branch=args.text_branch,
    )
    return {
        "artifact_type": "zero_shot_embedding_plan",
        "dataset_fingerprint": dataset.fingerprint(),
        "extractor_recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "protocol_key": zero_shot_protocol_artifact_key(dataset),
        "endpoints": {
            "samples": _zero_shot_endpoint_plan(sample_jobs),
            "prompts": _zero_shot_endpoint_plan(prompt_jobs),
        },
    }


def _zero_shot_endpoint_plan(jobs: Sequence[ZeroShotEmbeddingShardJob]) -> dict[str, Any]:
    if not jobs:
        raise ValueError("Zero-shot endpoint planning requires at least one shard job.")
    first = jobs[0]
    values = first.dataset.samples if first.side == "samples" else first.dataset.prompt_rows()[0]
    base_key = zero_shot_embedding_artifact_key(
        first.dataset, first.extractor, first.side, first.branch
    )
    return {
        "side": first.side,
        "branch": first.branch,
        "n_samples": len(values),
        "output_key": base_key,
        "shards": [
            {
                "side": job.side,
                "branch": job.branch,
                "shard": asdict(job.shard),
                "output_key": job.output_key,
            }
            for job in jobs
        ],
    }


def _cmd_embed_zero_shot_shard(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    plan = _load_json(args.plan_json) if args.plan_json else None
    planned = None
    if plan is not None:
        endpoint = plan.get("endpoints", {}).get(args.side)
        if endpoint is None:
            raise ValueError("--plan-json does not contain the requested zero-shot endpoint.")
        planned = next(
            (
                item
                for item in endpoint.get("shards", [])
                if item.get("shard", {}).get("shard_index") == args.shard_index
            ),
            None,
        )
        if planned is None:
            raise ValueError("--plan-json does not contain the requested zero-shot shard index.")
        if args.branch is not None and args.branch != planned["branch"]:
            raise ValueError("--branch conflicts with the selected zero-shot plan entry.")
        if args.total_shards is not None and args.total_shards != planned["shard"]["total_shards"]:
            raise ValueError("--total-shards conflicts with the selected zero-shot plan entry.")
        branch = planned["branch"]
        shard = ShardSpec(**planned["shard"])
        output_key = args.output_key or planned["output_key"]
    else:
        if not args.branch or args.total_shards is None:
            raise ValueError(
                "embed-zero-shot-shard requires --branch and --total-shards without --plan-json."
            )
        branch = args.branch
        shard = ShardSpec(total_shards=args.total_shards, shard_index=args.shard_index)
        base_key = zero_shot_embedding_artifact_key(dataset, extractor, args.side, branch)
        output_key = args.output_key or (
            f"{base_key}/shards/{args.shard_index:05d}-of-{args.total_shards:05d}"
        )
    return materialize_zero_shot_embedding_shard(
        ZeroShotEmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            side=args.side,
            branch=branch,
            shard=shard,
            output_key=output_key,
        ),
        _store_from_args(args),
    )


def _cmd_merge_zero_shot_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json) if args.plan_json else {}
    endpoint = plan.get("endpoints", {}).get(args.side) if args.side else None
    shard_keys = tuple(
        args.shard_key or (item["output_key"] for item in (endpoint or {}).get("shards", []))
    )
    output_key = args.output_key or (endpoint or {}).get("output_key")
    n_samples = args.n_samples or (endpoint or {}).get("n_samples")
    if not shard_keys or output_key is None or n_samples is None:
        raise ValueError(
            "merge-zero-shot-embeddings requires explicit shard keys, output key, and sample "
            "count or --plan-json with --side."
        )
    return merge_zero_shot_embedding_shards(
        EmbeddingMergeJob(shard_keys=shard_keys, output_key=output_key, n_samples=int(n_samples)),
        _store_from_args(args),
    )


def _cmd_write_zero_shot_protocol(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    if not hasattr(dataset, "prompt_rows"):
        raise TypeError("--dataset-pickle must contain a ZeroShotDataset.")
    return materialize_zero_shot_protocol(
        dataset,
        _store_from_args(args),
        key=args.output_key,
    )


def _cmd_write_labels(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    return materialize_label_artifact(
        dataset,
        _store_from_args(args),
        key=args.output_key,
    )


def _cmd_write_groups(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    return materialize_group_artifact(
        dataset,
        _store_from_args(args),
        key=args.output_key,
    )


def _cmd_materialize_segmentation(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    config = (
        _load_pickle(args.segmentation_config_pickle) if args.segmentation_config_pickle else None
    )
    return materialize_segmentation_artifacts(
        dataset,
        extractor,
        _store_from_args(args),
        segmentation_config=config,
        batch_size=args.batch_size,
    )


def _cmd_materialize_structured(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    return materialize_structured_artifacts(
        dataset,
        extractor,
        _store_from_args(args),
        batch_size=args.batch_size,
        aligners=_structured_aligners_from_specs(args.aligner),
    )


def _cmd_score(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json) if args.plan_json else {}
    embedding_key = args.embedding_key or _resolve_embedding_key_from_plan(plan)
    labels_key = args.labels_key or _resolve_related_key_from_plan(
        plan,
        embedding_key,
        "labels_key",
    )
    if embedding_key is None:
        raise ValueError("score requires --embedding-key or --plan-json.")
    if labels_key is None:
        raise ValueError("score requires --labels-key or --plan-json.")
    output_key = args.output_key or scoring_artifact_key(embedding_key, seed=args.seed)
    scoring_config = (
        _load_pickle(args.scoring_config_pickle) if args.scoring_config_pickle else None
    )
    metrics = _metrics_from_args(args)
    return score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            output_key=output_key,
            scoring_config=scoring_config,
            metrics=metrics,
            primary_metric=args.primary_metric,
            seed=args.seed,
        ),
        _store_from_args(args),
    )


def _cmd_score_retrieval(args: argparse.Namespace) -> dict[str, Any]:
    config = (
        _load_pickle(args.retrieval_config_pickle)
        if args.retrieval_config_pickle
        else RetrievalConfig()
    )
    if not isinstance(config, RetrievalConfig):
        raise TypeError("--retrieval-config-pickle must contain a RetrievalConfig.")
    output_key = args.output_key or retrieval_scoring_artifact_key(
        args.query_embedding_key, args.gallery_embedding_key
    )
    return score_retrieval_artifact(
        RetrievalScoringJob(
            query_embedding_key=args.query_embedding_key,
            gallery_embedding_key=args.gallery_embedding_key,
            relevance_key=args.relevance_key,
            exclusions_key=args.exclusions_key,
            output_key=output_key,
            retrieval_config=config,
        ),
        _store_from_args(args),
    )


def _cmd_compress_retrieval(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_pickle(args.compression_config_pickle)
    if not isinstance(config, EmbeddingCompressionConfig):
        raise TypeError("--compression-config-pickle must contain an EmbeddingCompressionConfig.")
    prefix = args.output_prefix or retrieval_compression_artifact_key(
        args.query_embedding_key, args.gallery_embedding_key, config
    )
    return compress_retrieval_embedding_artifacts(
        RetrievalCompressionJob(
            query_embedding_key=args.query_embedding_key,
            gallery_embedding_key=args.gallery_embedding_key,
            query_output_key=f"{prefix}/query",
            gallery_output_key=f"{prefix}/gallery",
            compression_config=config,
        ),
        _store_from_args(args),
    )


def _cmd_write_retrieval_relevance(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    if not all(
        hasattr(dataset, attribute) for attribute in ("relevance", "query_ids", "gallery_ids")
    ):
        raise TypeError("--dataset-pickle must contain a RetrievalDataset.")
    output_key = args.output_key or f"retrieval/relevance/{dataset.fingerprint()}"
    payload = {
        "artifact_type": "retrieval_relevance",
        "query_ids": list(dataset.query_ids),
        "gallery_ids": list(dataset.gallery_ids),
        "n_queries": len(dataset.query_ids),
        "n_gallery": len(dataset.gallery_ids),
        "relevance": dataset.relevance,
        "exclusions": sorted(dataset.exclusions or ()),
        "dataset_fingerprint": dataset.fingerprint(),
    }
    _store_from_args(args).put_json(output_key, payload)
    return {"output_key": output_key, **payload}


def _cmd_score_zero_shot(args: argparse.Namespace) -> dict[str, Any]:
    zero_config = (
        _load_pickle(args.zero_shot_config_pickle)
        if args.zero_shot_config_pickle
        else ZeroShotConfig()
    )
    if not isinstance(zero_config, ZeroShotConfig):
        raise TypeError("--zero-shot-config-pickle must contain a ZeroShotConfig.")
    scoring_config = (
        _load_pickle(args.scoring_config_pickle)
        if args.scoring_config_pickle
        else OverlapScoringConfig()
    )
    if not isinstance(scoring_config, OverlapScoringConfig):
        raise TypeError("--scoring-config-pickle must contain an OverlapScoringConfig.")
    output_key = args.output_key or zero_shot_scoring_artifact_key(
        args.sample_embedding_key,
        args.prompt_embedding_key,
        args.protocol_key,
        zero_config,
        scoring_config,
    )
    return score_zero_shot_artifact(
        ZeroShotScoringJob(
            sample_embedding_key=args.sample_embedding_key,
            prompt_embedding_key=args.prompt_embedding_key,
            protocol_key=args.protocol_key,
            output_key=output_key,
            zero_shot_config=zero_config,
            scoring_config=scoring_config,
        ),
        _store_from_args(args),
    )


def _cmd_compress_zero_shot(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_pickle(args.compression_config_pickle)
    if not isinstance(config, EmbeddingCompressionConfig):
        raise TypeError("--compression-config-pickle must contain an EmbeddingCompressionConfig.")
    prefix = args.output_prefix or zero_shot_compression_artifact_key(
        args.sample_embedding_key,
        args.prompt_embedding_key,
        config,
    )
    return compress_zero_shot_embedding_artifacts(
        ZeroShotCompressionJob(
            sample_embedding_key=args.sample_embedding_key,
            prompt_embedding_key=args.prompt_embedding_key,
            sample_output_key=f"{prefix}/samples",
            prompt_output_key=f"{prefix}/prompts",
            compression_config=config,
            output_key=prefix,
        ),
        _store_from_args(args),
    )


def _cmd_zero_shot_from_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    result = zero_shot_benchmark_result_from_artifacts(
        args.score_key,
        _store_from_args(args),
        output_key=args.output_key,
    )
    if args.json_output:
        result.save_json(args.json_output)
    if args.markdown_output:
        result.save_markdown(args.markdown_output)
    return result.to_dict()


def _metrics_from_args(args: argparse.Namespace) -> list[CallableMetric]:
    if len(args.metric_name) > len(args.metric) or len(args.metric_config_json) > len(args.metric):
        raise ValueError("--metric-name and --metric-config-json must correspond to --metric.")
    metrics = []
    for index, path in enumerate(args.metric):
        raw_config = (
            args.metric_config_json[index] if index < len(args.metric_config_json) else None
        )
        config = json.loads(raw_config) if raw_config else {}
        if not isinstance(config, dict):
            raise ValueError("--metric-config-json must decode to a JSON object.")
        metrics.append(
            CallableMetric.from_import_path(
                path,
                name=args.metric_name[index] if index < len(args.metric_name) else None,
                config=config,
                higher_is_better=not args.lower_is_better,
            )
        )
    return metrics


def _cmd_diagnose_complexity(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json) if args.plan_json else {}
    embedding_key = args.embedding_key or _resolve_embedding_key_from_plan(plan)
    labels_key = args.labels_key or _resolve_related_key_from_plan(
        plan,
        embedding_key,
        "labels_key",
    )
    groups_key = args.groups_key or _resolve_related_key_from_plan(
        plan,
        embedding_key,
        "groups_key",
    )
    if embedding_key is None:
        raise ValueError("diagnose-complexity requires --embedding-key or --plan-json.")
    if labels_key is None:
        raise ValueError("diagnose-complexity requires --labels-key or --plan-json.")
    score_key = args.score_key or _resolve_score_key_from_plan(plan, embedding_key)
    separatix_config = (
        _load_pickle(args.separatix_config_pickle) if args.separatix_config_pickle else None
    )
    output_key = args.output_key or separatix_artifact_key(embedding_key)
    return diagnose_embedding_artifact(
        SeparatixJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            score_key=score_key,
            output_key=output_key,
            separatix_config=separatix_config,
            groups_key=groups_key,
        ),
        _store_from_args(args),
    )


def _cmd_compress(args: argparse.Namespace) -> dict[str, Any]:
    compression_config = EmbeddingCompressionConfig(
        enabled=args.method != "none",
        method=args.method,
        n_components=args.n_components,
        preserve_variance=args.preserve_variance,
        precision=args.precision,
        assume_matryoshka=args.assume_matryoshka,
        random_state=args.random_state,
        whiten=args.whiten,
        dtype=args.dtype,
    )
    job = plan_compression_job(args.embedding_key, compression_config)
    if args.output_key:
        from dataclasses import replace

        job = replace(job, output_key=args.output_key)
    return compress_embedding_artifact(job, _store_from_args(args))


def _cmd_score_repeats(args: argparse.Namespace) -> dict[str, Any]:
    embedding_key, labels_key = _embedding_and_labels_from_args(args)
    seeds = args.seed or _repeat_seeds(args.repeats, args.random_state)
    scoring_config = (
        _load_pickle(args.scoring_config_pickle) if args.scoring_config_pickle else None
    )
    jobs = plan_scoring_jobs(
        embedding_key=embedding_key,
        labels_key=labels_key,
        seeds=seeds,
        scoring_config=scoring_config,
        metrics=_metrics_from_args(args),
        primary_metric=args.primary_metric,
    )
    backend = _create_backend_from_args(args)
    artifacts = score_embedding_artifacts(jobs, _store_from_args(args), backend)
    return {
        "artifact_type": "score_plan",
        "backend": args.backend,
        "cache_dir": args.cache_dir,
        "storage_options": _artifact_store_options_from_args(args),
        "embedding_key": embedding_key,
        "labels_key": labels_key,
        "score_keys": [artifact["output_key"] for artifact in artifacts],
        "seeds": seeds,
        "primary_metric": args.primary_metric,
    }


def _cmd_collect_scores(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.score_plan_json) if args.score_plan_json else {}
    score_keys = args.score_key or plan.get("score_keys", [])
    if not score_keys:
        raise ValueError("collect-scores requires --score-key or --score-plan-json.")
    return collect_score_artifacts(
        score_keys=score_keys,
        store=_store_from_args(args),
        output_key=args.output_key,
        interval_level=args.interval_level,
        metric_name=args.metric_name,
    )


def _cmd_benchmark_from_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    result = benchmark_result_from_artifacts(
        score_key=args.score_key,
        store=_store_from_args(args),
        output_key=args.output_key,
        stability_key=args.stability_key,
        separatix_key=args.separatix_key,
    )
    if args.json_output:
        _write_json_file(result, args.json_output)
    if args.markdown_output:
        from vertebrae.reports.markdown_report import render_markdown_report
        from vertebrae.results import BenchmarkResult, ExtractorResult
        from vertebrae.scoring.metrics import MetricResult
        from vertebrae.scoring.overlap import OverlapScoreResult
        from vertebrae.scoring.separatix import SeparatixResult

        markdown = render_markdown_report(
            _benchmark_result_from_dict(
                result,
                BenchmarkResult,
                ExtractorResult,
                OverlapScoreResult,
                SeparatixResult,
                MetricResult,
            )
        )
        target = Path(args.markdown_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
    return result


def _cmd_slurm_array(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    base_key = embedding_artifact_key(dataset, extractor)
    script = _render_slurm_array_script(
        args=args,
        output_key=base_key,
        labels_key=labels_artifact_key(dataset),
        n_samples=len(dataset.y),
    )
    target = Path(args.script_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")
    return {
        "script_path": str(target),
        "output_key": base_key,
        "n_samples": int(len(dataset.y)),
        "total_shards": args.total_shards,
        "batch_size": args.batch_size,
    }


def _cmd_run_embedding_shards(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor = _load_pickle(args.extractor_pickle)
    jobs = plan_embedding_shard_jobs(
        dataset=dataset,
        extractor=extractor,
        total_shards=args.total_shards,
        batch_size=args.batch_size,
    )
    backend = _create_backend_from_args(args)
    from vertebrae.execution import materialize_embedding_shards

    manifests = materialize_embedding_shards(
        jobs=jobs,
        store=_store_from_args(args),
        execution=backend,
    )
    return {
        "artifact_type": "embedding_shard_plan",
        "backend": args.backend,
        "shard_keys": [manifest["output_key"] for manifest in manifests],
        "n_shards": len(manifests),
    }


def _cmd_slurm_score_array(args: argparse.Namespace) -> dict[str, Any]:
    embedding_key, labels_key = _embedding_and_labels_from_args(args)
    seeds = _repeat_seeds(args.repeats, args.random_state)
    script = _render_slurm_score_array_script(args=args, seeds=seeds)
    target = Path(args.script_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")
    return {
        "script_path": str(target),
        "embedding_key": embedding_key,
        "labels_key": labels_key,
        "score_keys": [scoring_artifact_key(embedding_key, seed=seed) for seed in seeds],
        "seeds": seeds,
    }


def _render_slurm_array_script(
    args: argparse.Namespace,
    output_key: str,
    labels_key: str,
    n_samples: int,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --array=0-{args.total_shards - 1}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
    ]
    if args.partition:
        lines.append(f"#SBATCH --partition={args.partition}")
    lines.extend(
        [
            "set -euo pipefail",
            "",
            f"{args.python_executable} -m vertebrae.cli embed-shard \\",
            f"  --dataset-pickle {args.dataset_pickle} \\",
            f"  --extractor-pickle {args.extractor_pickle} \\",
            f"  --cache-dir {args.cache_dir} \\",
            *_cache_flag_lines(args),
            f"  --total-shards {args.total_shards} \\",
            "  --shard-index ${SLURM_ARRAY_TASK_ID} \\",
            f"  --batch-size {args.batch_size}",
            "",
            "# After the array completes, merge the shards with:",
            f"# {args.python_executable} -m vertebrae.cli merge-embeddings \\",
            f"#   --cache-dir {args.cache_dir} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"#   --output-key {output_key} \\",
            f"#   --n-samples {n_samples} \\",
        ]
    )
    for shard_index in range(args.total_shards):
        shard = ShardSpec(total_shards=args.total_shards, shard_index=shard_index)
        suffix = " \\" if shard_index < args.total_shards - 1 else ""
        lines.append(f"#   --shard-key {embedding_shard_key(output_key, shard)}{suffix}")
    lines.extend(
        [
            "#",
            "# Then materialize labels and score:",
            f"# {args.python_executable} -m vertebrae.cli write-labels \\",
            f"#   --dataset-pickle {args.dataset_pickle} \\",
            f"#   --cache-dir {args.cache_dir} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"# {args.python_executable} -m vertebrae.cli score \\",
            f"#   --cache-dir {args.cache_dir} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"#   --embedding-key {output_key} \\",
            f"#   --labels-key {labels_key}",
            "#",
            "# For distributed stability scoring, generate a scoring array with:",
            f"# {args.python_executable} -m vertebrae.cli slurm-score-array \\",
            f"#   --cache-dir {args.cache_dir} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"#   --embedding-key {output_key} \\",
            f"#   --labels-key {labels_key} \\",
            "#   --repeats 20 \\",
            "#   --script-output vertebrae_score.sbatch",
            "",
        ]
    )
    return "\n".join(lines)


def _render_slurm_score_array_script(args: argparse.Namespace, seeds: list[int]) -> str:
    embedding_key, labels_key = _embedding_and_labels_from_args(args)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --array=0-{len(seeds) - 1}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
    ]
    if args.partition:
        lines.append(f"#SBATCH --partition={args.partition}")
    seed_values = " ".join(str(seed) for seed in seeds)
    lines.extend(
        [
            "set -euo pipefail",
            f"SEEDS=({seed_values})",
            "SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}",
            "",
            f"{args.python_executable} -m vertebrae.cli score \\",
            f"  --cache-dir {args.cache_dir} \\",
            *_cache_flag_lines(args),
            f"  --embedding-key {embedding_key} \\",
            f"  --labels-key {labels_key} \\",
            "  --seed ${SEED}",
            "",
            "# After the array completes, collect scores with:",
            f"# {args.python_executable} -m vertebrae.cli collect-scores \\",
            f"#   --cache-dir {args.cache_dir} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"#   --output-key {embedding_key}/scores/stability \\",
        ]
    )
    for index, seed in enumerate(seeds):
        suffix = " \\" if index < len(seeds) - 1 else ""
        lines.append(f"#   --score-key {scoring_artifact_key(embedding_key, seed=seed)}{suffix}")
    lines.append("")
    return "\n".join(lines)


def _add_object_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-pickle", required=True)
    parser.add_argument("--extractor-pickle", required=True)


def _add_cache_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", default=".vertebrae_cache")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--s3-profile")
    parser.add_argument("--s3-region")
    parser.add_argument("--gcs-project")


def _add_backend_args(
    parser: argparse.ArgumentParser,
    include_local_parallel: bool = False,
) -> None:
    parser.add_argument("--backend", choices=["local", "ray", "dask"], default="local")
    parser.add_argument("--ray-address")
    parser.add_argument("--dask-address")
    if include_local_parallel:
        parser.add_argument("--n-jobs", type=int, default=1)
        parser.add_argument("--joblib-backend", default="loky")


def _load_pickle(path: str) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _create_backend_from_args(args: argparse.Namespace) -> Any:
    return create_execution_backend(
        args.backend,
        n_jobs=getattr(args, "n_jobs", 1),
        joblib_backend=getattr(args, "joblib_backend", "loky"),
        ray_address=getattr(args, "ray_address", None),
        dask_address=getattr(args, "dask_address", None),
    )


def _artifact_store_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    options = {}
    for plan_path_attr in ("plan_json", "score_plan_json"):
        plan_path = getattr(args, plan_path_attr, None)
        if plan_path:
            options.update(_load_json(plan_path).get("storage_options", {}))
    options = {
        **options,
        "endpoint_url": getattr(args, "s3_endpoint_url", None),
        "profile_name": getattr(args, "s3_profile", None),
        "region_name": getattr(args, "s3_region", None),
        "project": getattr(args, "gcs_project", None),
    }
    return {key: value for key, value in options.items() if value is not None}


def _store_from_args(args: argparse.Namespace) -> Any:
    return create_artifact_store(args.cache_dir, **_artifact_store_options_from_args(args))


def _cache_flag_lines(args: argparse.Namespace, indent: bool = True) -> list[str]:
    prefix = "  " if indent else ""
    flags = []
    if getattr(args, "s3_endpoint_url", None):
        flags.append(f"{prefix}--s3-endpoint-url {args.s3_endpoint_url} \\")
    if getattr(args, "s3_profile", None):
        flags.append(f"{prefix}--s3-profile {args.s3_profile} \\")
    if getattr(args, "s3_region", None):
        flags.append(f"{prefix}--s3-region {args.s3_region} \\")
    if getattr(args, "gcs_project", None):
        flags.append(f"{prefix}--gcs-project {args.gcs_project} \\")
    return flags


def _embedding_and_labels_from_args(args: argparse.Namespace) -> tuple[str, str]:
    plan = _load_json(args.plan_json) if getattr(args, "plan_json", None) else {}
    embedding_key = getattr(args, "embedding_key", None) or _resolve_embedding_key_from_plan(plan)
    labels_key = getattr(args, "labels_key", None) or _resolve_related_key_from_plan(
        plan,
        embedding_key,
        "labels_key",
    )
    if embedding_key is None:
        raise ValueError("An embedding key or plan JSON is required.")
    if labels_key is None:
        raise ValueError("A labels key or plan JSON is required.")
    return embedding_key, labels_key


def _resolve_embedding_key_from_plan(plan: dict[str, Any]) -> Optional[str]:
    output = _resolve_plan_output(plan)
    if output is None:
        return plan.get("output_key")
    return output.get("output_key")


def _multi_output_plan_names(extractor: Any) -> list[str]:
    if not callable(getattr(extractor, "output_specs", None)):
        return []
    if not callable(getattr(extractor, "transform_many", None)):
        return []
    names = [spec.name for spec in extractor.output_specs()]
    return names if len(names) > 1 else []


def _repeat_seeds(repeats: Any, random_state: int) -> list[int]:
    if repeats is None or int(repeats) < 1:
        raise ValueError("repeats must be >= 1.")
    import numpy as np

    rng = np.random.default_rng(random_state)
    return [int(seed) for seed in rng.integers(0, np.iinfo(np.int32).max, size=int(repeats))]


def _structured_aligners_from_specs(specs: Sequence[str]) -> Optional[dict[str, Any]]:
    if not specs:
        return None
    aligners: dict[str, Any] = {}
    for spec in specs:
        output_name, aligner = _structured_aligner_from_spec(spec)
        if output_name in aligners:
            raise ValueError(f"Duplicate structured aligner spec for output {output_name!r}.")
        aligners[output_name] = aligner
    return aligners


def _structured_aligner_from_spec(spec: str) -> tuple[str, Any]:
    output_name, separator, helper_spec = str(spec).partition("=")
    output_name = output_name.strip()
    if separator != "=" or not output_name or not helper_spec.strip():
        raise ValueError(
            "Structured aligner specs must look like "
            "'output_name=helper_name:{...json params...}'."
        )
    helper_name, has_params, raw_params = helper_spec.partition(":")
    helper_name = helper_name.strip()
    if not helper_name:
        raise ValueError("Structured aligner specs must include a helper name after '='.")
    params: dict[str, Any] = {}
    if has_params:
        raw_params = raw_params.strip()
        if not raw_params:
            raise ValueError(
                f"Structured aligner spec for output {output_name!r} included ':' but no JSON "
                "parameters."
            )
        try:
            loaded = json.loads(raw_params)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Structured aligner spec for output {output_name!r} has invalid JSON "
                f"parameters: {exc.msg}."
            ) from exc
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Structured aligner spec for output {output_name!r} must decode to a JSON "
                "object of helper parameters."
            )
        params = loaded
    return output_name, _build_structured_aligner(helper_name, params)


def _build_structured_aligner(helper_name: str, params: dict[str, Any]) -> Any:
    helper_factories: dict[str, Callable[..., Any]] = {
        "drop_special_rows": drop_special_rows,
        "keep_row_indices": keep_row_indices,
        "select_frame_rows": select_frame_rows,
    }
    if helper_name not in helper_factories:
        supported = ", ".join(sorted(helper_factories))
        raise ValueError(
            f"Unknown structured aligner helper {helper_name!r}. Supported helpers: {supported}."
        )
    try:
        return helper_factories[helper_name](**params)
    except TypeError as exc:
        raise ValueError(
            f"Invalid parameters for structured aligner helper {helper_name!r}: {exc}."
        ) from exc


def _write_json_file(payload: dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _benchmark_result_from_dict(
    data: dict[str, Any],
    benchmark_cls: Any,
    extractor_cls: Any,
    overlap_cls: Any,
    separatix_cls: Any,
    metric_cls: Any,
) -> Any:
    extractor_results = []
    for item in data.get("extractor_results", []):
        separatix = None
        if item.get("separatix") is not None:
            separatix = separatix_cls(**item["separatix"])
        extractor_results.append(
            extractor_cls(
                name=item["name"],
                extractor_type=item["extractor_type"],
                stability=item.get("stability"),
                separatix=separatix,
                embedding_metadata=item.get("embedding_metadata", {}),
                compression_metadata=item.get("compression_metadata", {"method": "none"}),
                runtime=item.get("runtime", {}),
                warnings=item.get("warnings", []),
                recommendation=item.get("recommendation", ""),
                metrics={
                    name: metric_cls(**metric) for name, metric in item.get("metrics", {}).items()
                },
                primary_metric_name=item.get("primary_metric_name", "overlap"),
                label_view=item.get("label_view"),
                target_view=item.get("target_view"),
                weakest_class=item.get("weakest_class"),
                weakest_class_score=item.get("weakest_class_score"),
            )
        )
    return benchmark_cls(
        dataset_summary=data.get("dataset_summary", {}),
        extractor_results=extractor_results,
        recommendations=data.get("recommendations", []),
        metadata=data.get("metadata", {}),
    )


def _resolve_score_key_from_plan(plan: dict[str, Any], embedding_key: str) -> str:
    return _resolve_related_key_from_plan(plan, embedding_key, "score_key") or scoring_artifact_key(
        embedding_key
    )


def _resolve_related_key_from_plan(
    plan: dict[str, Any],
    embedding_key: Optional[str],
    field: str,
) -> Optional[str]:
    outputs = plan.get("outputs", [])
    if not outputs:
        return plan.get(field)
    output = _resolve_plan_output(plan, embedding_key, require_match=False)
    if output is None:
        return None
    if output is not None and output.get(field) is not None:
        return output.get(field)
    return plan.get(field)


def _resolve_plan_output(
    plan: dict[str, Any],
    embedding_key: Optional[str] = None,
    require_match: bool = True,
) -> Optional[dict[str, Any]]:
    outputs = plan.get("outputs", [])
    if not outputs:
        return None
    if embedding_key is None:
        if len(outputs) == 1:
            return outputs[0]
        raise ValueError(
            "This plan contains multiple embedding outputs. Pass --embedding-key with one of "
            "the output keys listed in the plan JSON."
        )
    for output in outputs:
        if output.get("output_key") == embedding_key:
            return output
    if require_match:
        raise ValueError("The requested --embedding-key was not found in the plan JSON outputs.")
    return None


def _write_json_payload(payload: dict[str, Any], output_json: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if output_json:
        target = Path(output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    raise SystemExit(main())
