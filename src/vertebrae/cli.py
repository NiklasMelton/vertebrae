"""Command line interface for distributed vertebrae workflows."""

import argparse
import hashlib
import json
import os
import pickle
import re
import shlex
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from uuid import uuid4

from vertebrae.cache import create_artifact_store
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.config import (
    EmbeddingCompressionConfig,
    OverlapScoringConfig,
    ResourceProfilingConfig,
    RetrievalConfig,
    SeparatixConfig,
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
    retrieval_benchmark_result_from_artifacts,
    retrieval_compression_artifact_key,
    retrieval_scoring_artifact_key,
    score_embedding_artifact,
    score_embedding_artifacts,
    score_retrieval_artifact,
    score_zero_shot_artifact,
    scoring_artifact_key,
    separatix_artifact_key,
    zero_shot_benchmark_result_from_artifacts,
    zero_shot_compression_artifact_key,
    zero_shot_protocol_artifact_key,
    zero_shot_scoring_artifact_key,
)
from vertebrae.execution.jobs import EmbeddingShardJob
from vertebrae.extractors._identity import extractor_cache_reuse_decision
from vertebrae.scoring.metrics import CallableMetric
from vertebrae.structured import (
    drop_special_rows,
    keep_row_indices,
    select_frame_rows,
)
from vertebrae.utils.serialization import json_dumps_strict

_TRUSTED_PICKLE_WARNING = (
    "TRUSTED INPUT ONLY: loading a Python pickle can execute arbitrary code; "
    "never use a file from an untrusted source."
)
_TRUSTED_PICKLE_OUTPUT_WARNING = (
    "The generated pickle is TRUSTED INPUT ONLY when loaded; protect it from "
    "untrusted modification."
)
_EMBEDDING_PLAN_TYPE = "embedding_shard_plan"
_EMBEDDING_PLAN_SCHEMA_VERSION = 2


def _trusted_pickle_help(description: str) -> str:
    return f"{description} {_TRUSTED_PICKLE_WARNING}"


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
    _add_resource_profiling_arg(plan)
    _add_backend_args(plan, include_local_parallel=True)
    plan.add_argument("--output-json")
    plan.set_defaults(func=_cmd_plan)

    fit = subparsers.add_parser(
        "fit-extractor",
        help="Fit an extractor once and write a dataset-bound bundle for shard workers.",
    )
    _add_object_args(
        fit,
        extractor_description="Pickled unfitted source extractor object.",
    )
    fit.add_argument(
        "--output-pickle",
        required=True,
        help=(
            "Output path for the fitted, dataset-bound extractor bundle. "
            f"{_TRUSTED_PICKLE_OUTPUT_WARNING}"
        ),
    )
    fit.add_argument("--force", action="store_true")
    fit.add_argument("--output-json")
    fit.set_defaults(func=_cmd_fit_extractor)

    embed = subparsers.add_parser("embed-shard", help="Materialize one embedding shard.")
    _add_object_args(
        embed,
        extractor_description=(
            "Pickled fitted-extractor bundle created by `vertebrae fit-extractor`."
        ),
    )
    _add_cache_arg(embed)
    embed.add_argument("--plan-json")
    embed.add_argument("--total-shards", type=int)
    embed.add_argument("--shard-index", type=int, required=True)
    embed.add_argument("--batch-size", type=int, default=128)
    _add_resource_profiling_arg(embed)
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
    _add_resource_profiling_arg(retrieval_plan)
    retrieval_plan.add_argument("--output-json")
    retrieval_plan.set_defaults(func=_cmd_plan_retrieval)

    retrieval_embed = subparsers.add_parser(
        "embed-retrieval-shard", help="Materialize one query or gallery retrieval endpoint shard."
    )
    _add_object_args(retrieval_embed)
    _add_cache_arg(retrieval_embed)
    retrieval_embed.add_argument("--side", choices=["query", "gallery"], required=True)
    retrieval_embed.add_argument("--plan-json")
    retrieval_embed.add_argument("--branch")
    retrieval_embed.add_argument("--total-shards", type=int)
    retrieval_embed.add_argument("--shard-index", type=int, required=True)
    retrieval_embed.add_argument("--batch-size", type=int, default=128)
    _add_resource_profiling_arg(retrieval_embed)
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
    zero_plan.add_argument("--batch-size", type=int, default=128)
    _add_resource_profiling_arg(zero_plan)
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
    zero_embed.add_argument("--batch-size", type=int, default=128)
    _add_resource_profiling_arg(zero_embed)
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
    zero_protocol.add_argument(
        "--dataset-pickle",
        required=True,
        help=_trusted_pickle_help("Pickled ZeroShotDataset."),
    )
    _add_cache_arg(zero_protocol)
    zero_protocol.add_argument("--output-key")
    zero_protocol.add_argument("--output-json")
    zero_protocol.set_defaults(func=_cmd_write_zero_shot_protocol)

    labels = subparsers.add_parser("write-labels", help="Materialize dataset labels.")
    labels.add_argument(
        "--dataset-pickle",
        required=True,
        help=_trusted_pickle_help("Pickled BenchmarkDataset."),
    )
    _add_cache_arg(labels)
    labels.add_argument("--output-key")
    labels.add_argument("--output-json")
    labels.set_defaults(func=_cmd_write_labels)

    groups = subparsers.add_parser("write-groups", help="Materialize dataset groups.")
    groups.add_argument(
        "--dataset-pickle",
        required=True,
        help=_trusted_pickle_help("Pickled BenchmarkDataset."),
    )
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
    segmentation.add_argument(
        "--segmentation-config-pickle",
        help=_trusted_pickle_help("Optional pickled SegmentationConfig."),
    )
    segmentation.add_argument("--batch-size", type=int, default=16)
    _add_resource_profiling_arg(segmentation)
    segmentation.add_argument("--output-json")
    segmentation.set_defaults(func=_cmd_materialize_segmentation)

    structured = subparsers.add_parser(
        "materialize-structured",
        help="Materialize structured unit embeddings, labels, groups, and provenance.",
    )
    _add_object_args(structured)
    _add_cache_arg(structured)
    structured.add_argument("--batch-size", type=int, default=16)
    _add_resource_profiling_arg(structured)
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
    score.add_argument("--groups-key")
    score.add_argument("--output-key")
    score.add_argument("--plan-json")
    score.add_argument(
        "--scoring-config-pickle",
        help=_trusted_pickle_help("Optional pickled overlap scoring configuration."),
    )
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
    retrieval_score.add_argument(
        "--retrieval-config-pickle",
        help=_trusted_pickle_help("Optional pickled RetrievalConfig."),
    )
    retrieval_score.add_argument("--output-key")
    retrieval_score.add_argument("--output-json")
    retrieval_score.set_defaults(func=_cmd_score_retrieval)

    retrieval_compress = subparsers.add_parser(
        "compress-retrieval", help="Fit gallery compression and transform paired query embeddings."
    )
    _add_cache_arg(retrieval_compress)
    retrieval_compress.add_argument("--query-embedding-key", required=True)
    retrieval_compress.add_argument("--gallery-embedding-key", required=True)
    retrieval_compress.add_argument(
        "--compression-config-pickle",
        required=True,
        help=_trusted_pickle_help("Pickled EmbeddingCompressionConfig."),
    )
    retrieval_compress.add_argument("--output-prefix")
    retrieval_compress.add_argument("--output-json")
    retrieval_compress.set_defaults(func=_cmd_compress_retrieval)

    relevance = subparsers.add_parser(
        "write-retrieval-relevance",
        help="Materialize a RetrievalDataset relevance artifact.",
    )
    relevance.add_argument(
        "--dataset-pickle",
        required=True,
        help=_trusted_pickle_help("Pickled RetrievalDataset."),
    )
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
    zero_score.add_argument(
        "--zero-shot-config-pickle",
        help=_trusted_pickle_help("Optional pickled ZeroShotConfig."),
    )
    zero_score.add_argument(
        "--scoring-config-pickle",
        help=_trusted_pickle_help("Optional pickled overlap scoring configuration."),
    )
    zero_score.add_argument("--output-key")
    zero_score.add_argument("--output-json")
    zero_score.set_defaults(func=_cmd_score_zero_shot)

    zero_compress = subparsers.add_parser(
        "compress-zero-shot", help="Fit sample compression and transform paired prompts."
    )
    _add_cache_arg(zero_compress)
    zero_compress.add_argument("--sample-embedding-key", required=True)
    zero_compress.add_argument("--prompt-embedding-key", required=True)
    zero_compress.add_argument(
        "--compression-config-pickle",
        required=True,
        help=_trusted_pickle_help("Pickled EmbeddingCompressionConfig."),
    )
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

    retrieval_collect = subparsers.add_parser(
        "retrieval-from-artifacts",
        help="Reconstruct ranked retrieval JSON and Markdown reports from score artifacts.",
    )
    _add_cache_arg(retrieval_collect)
    retrieval_collect.add_argument("--score-key", action="append", required=True)
    retrieval_collect.add_argument("--output-key")
    retrieval_collect.add_argument("--json-output")
    retrieval_collect.add_argument("--markdown-output")
    retrieval_collect.add_argument("--output-json")
    retrieval_collect.set_defaults(func=_cmd_retrieval_from_artifacts)

    diagnose = subparsers.add_parser(
        "diagnose-complexity",
        help="Run Separatix on persisted embeddings and labels.",
    )
    _add_cache_arg(diagnose)
    diagnose.add_argument("--embedding-key")
    diagnose.add_argument("--labels-key")
    diagnose.add_argument("--score-key")
    diagnose.add_argument("--plan-json")
    diagnose.add_argument(
        "--separatix-config-pickle",
        help=_trusted_pickle_help("Optional pickled SeparatixConfig."),
    )
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
    pca_dimension = compress.add_mutually_exclusive_group()
    pca_dimension.add_argument("--n-components", type=int)
    pca_dimension.add_argument("--preserve-variance", type=float)
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
    repeats.add_argument("--groups-key")
    repeats.add_argument("--plan-json")
    repeats.add_argument("--seed", action="append", type=int, default=[])
    repeats.add_argument("--repeats", type=int)
    repeats.add_argument("--random-state", type=int, default=42)
    repeats.add_argument(
        "--scoring-config-pickle",
        help=_trusted_pickle_help("Optional pickled overlap scoring configuration."),
    )
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

    slurm = subparsers.add_parser(
        "slurm-array",
        help="Generate the SLURM fit/plan/array submission workflow files.",
    )
    _add_object_args(slurm)
    _add_cache_arg(slurm)
    slurm.add_argument("--total-shards", type=int, required=True)
    slurm.add_argument("--batch-size", type=int, default=128)
    _add_resource_profiling_arg(slurm)
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
    slurm_score.add_argument("--groups-key")
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
    _add_resource_profiling_arg(run_embed)
    _add_backend_args(run_embed, include_local_parallel=True)
    run_embed.add_argument("--output-json")
    run_embed.set_defaults(func=_cmd_run_embedding_shards)

    return parser


def _cmd_plan(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor, fitted_extractor_path = _load_or_create_fitted_extractor(
        args.extractor_pickle, dataset
    )
    fitted_extractor_path = str(Path(fitted_extractor_path).resolve())
    resource_config = _resource_profiling_config_from_args(args)
    jobs = plan_embedding_shard_jobs(
        dataset=dataset,
        extractor=extractor,
        total_shards=args.total_shards,
        batch_size=args.batch_size,
        resource_profiling_config=resource_config,
    )
    base_key = jobs[0].output_key.rsplit("/shards/", 1)[0]
    labels_key = labels_artifact_key(dataset)
    groups_key = (
        groups_artifact_key(dataset)
        if callable(getattr(dataset, "groups", None)) and dataset.groups() is not None
        else None
    )
    default_scoring_config = OverlapScoringConfig()
    plan = {
        "artifact_type": _EMBEDDING_PLAN_TYPE,
        "schema_version": _EMBEDDING_PLAN_SCHEMA_VERSION,
        "dataset_identity_key": dataset.identity_key(),
        "fitted_recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "fitted_bundle_sha256": _sha256_file(Path(fitted_extractor_path)),
        "dataset_pickle": str(Path(args.dataset_pickle).resolve()),
        "extractor_pickle": fitted_extractor_path,
        "cache_dir": args.cache_dir,
        "storage_options": _artifact_store_options_from_args(args),
        "base_key": base_key,
        "output_key": base_key,
        "labels_key": labels_key,
        "groups_key": groups_key,
        "score_key": scoring_artifact_key(
            base_key,
            labels_key=labels_key,
            groups_key=groups_key,
            scoring_config=default_scoring_config,
            metrics=(),
            primary_metric="overlap",
        ),
        "n_samples": int(len(dataset.y)),
        "requested_total_shards": args.total_shards,
        "total_shards": len(jobs),
        "batch_size": args.batch_size,
        "backend": args.backend,
        "cache_eligible": jobs[0].cache_eligible,
        "cache_status": jobs[0].cache_status,
        "resource_profiling_config": asdict(resource_config),
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
                "score_key": scoring_artifact_key(
                    embedding_output_key(base_key, output_name),
                    labels_key=labels_key,
                    groups_key=groups_key,
                    scoring_config=default_scoring_config,
                    metrics=(),
                    primary_metric="overlap",
                ),
                "shard_keys": [
                    embedding_output_shard_key(job["output_key"], output_name)
                    for job in plan["shard_jobs"]
                ],
            }
            for output_name in output_names
        ]
    return plan


def _validated_embedding_plan_entry(
    plan: Any,
    *,
    dataset: Any,
    extractor: Any,
    extractor_pickle: str,
    shard_index: int,
    total_shards: Optional[int],
    output_key: Optional[str],
) -> dict[str, Any]:
    """Validate a schema-v2 embedding plan before a shard can publish data."""

    if not isinstance(plan, dict):
        raise TypeError("--plan-json must contain a JSON object.")
    if (
        plan.get("artifact_type") != _EMBEDDING_PLAN_TYPE
        or plan.get("schema_version") != _EMBEDDING_PLAN_SCHEMA_VERSION
    ):
        raise ValueError(
            "--plan-json is stale or invalid; embed-shard requires an embedding "
            "shard plan with schema_version=2."
        )

    dataset_identity_key = dataset.identity_key()
    if plan.get("dataset_identity_key") != dataset_identity_key:
        raise ValueError("--plan-json belongs to a different dataset identity.")
    fitted_recipe_hash = fingerprint_extractor_recipe(extractor.recipe())
    if plan.get("fitted_recipe_hash") != fitted_recipe_hash:
        raise ValueError("--plan-json belongs to a different fitted extractor recipe.")

    planned_bundle = plan.get("extractor_pickle")
    if not isinstance(planned_bundle, str) or not planned_bundle:
        raise ValueError("--plan-json is missing its fitted extractor bundle path.")
    actual_bundle_path = Path(extractor_pickle).resolve()
    if actual_bundle_path != Path(planned_bundle).resolve():
        raise ValueError(
            "--extractor-pickle differs from the fitted extractor bundle bound to " "--plan-json."
        )
    expected_cache_eligible, expected_cache_status = extractor_cache_reuse_decision(
        extractor.recipe()
    )
    if type(plan.get("cache_eligible")) is not bool or (
        plan["cache_eligible"] is not expected_cache_eligible
    ):
        raise ValueError("--plan-json has an invalid cache-eligibility contract.")
    if plan.get("cache_status") != expected_cache_status:
        raise ValueError("--plan-json has an invalid cache-status contract.")

    n_samples = _validated_positive_int(plan.get("n_samples"), "plan n_samples")
    if n_samples != int(len(dataset.y)):
        raise ValueError("--plan-json has a sample count inconsistent with the dataset.")
    requested_total_shards = _validated_positive_int(
        plan.get("requested_total_shards"), "plan requested_total_shards"
    )
    planned_total_shards = _validated_positive_int(plan.get("total_shards"), "plan total_shards")
    _validated_positive_int(plan.get("batch_size"), "plan batch_size")
    expected_total_shards = (
        min(requested_total_shards, n_samples)
        if bool(getattr(extractor, "streaming_safe", False))
        else 1
    )
    if planned_total_shards != expected_total_shards:
        raise ValueError("--plan-json has an invalid effective shard count.")

    canonical_base_key = embedding_artifact_key(dataset, extractor)
    base_key = plan.get("base_key")
    if not isinstance(base_key, str) or not base_key:
        raise ValueError("--plan-json is missing its embedding base key.")
    if expected_cache_eligible:
        if base_key != canonical_base_key:
            raise ValueError("--plan-json has a noncanonical embedding base key.")
    else:
        suffix = f"/{canonical_base_key}"
        run_prefix = base_key[: -len(suffix)] if base_key.endswith(suffix) else ""
        if re.fullmatch(r"runs/[0-9a-f]{32}", run_prefix) is None:
            raise ValueError("--plan-json has an invalid run-scoped embedding base key.")
    if plan.get("output_key") != base_key:
        raise ValueError("--plan-json output_key must equal its canonical base_key.")

    shard_jobs = plan.get("shard_jobs")
    if not isinstance(shard_jobs, list) or len(shard_jobs) != planned_total_shards:
        raise ValueError("--plan-json has an incomplete shard job list.")
    validated_entries: dict[int, dict[str, Any]] = {}
    canonical_shard_keys: list[str] = []
    for item in shard_jobs:
        if not isinstance(item, dict):
            raise ValueError("--plan-json shard jobs must be JSON objects.")
        item_total = _validated_positive_int(item.get("total_shards"), "plan shard total_shards")
        item_index = item.get("shard_index")
        if (
            isinstance(item_index, bool)
            or not isinstance(item_index, int)
            or item_index < 0
            or item_index >= planned_total_shards
        ):
            raise ValueError("--plan-json contains an invalid shard index.")
        if item_total != planned_total_shards or item_index in validated_entries:
            raise ValueError("--plan-json contains inconsistent or duplicate shard jobs.")
        shard = ShardSpec(total_shards=item_total, shard_index=item_index)
        expected_shard_key = embedding_shard_key(base_key, shard)
        if item.get("output_key") != expected_shard_key:
            raise ValueError("--plan-json contains a noncanonical shard output key.")
        validated_entries[item_index] = item
        canonical_shard_keys.append(expected_shard_key)
    if set(validated_entries) != set(range(planned_total_shards)):
        raise ValueError("--plan-json does not cover every planned shard exactly once.")

    output_names = _multi_output_plan_names(extractor)
    outputs = plan.get("outputs")
    if output_names:
        if not isinstance(outputs, list) or len(outputs) != len(output_names):
            raise ValueError("--plan-json has an incomplete multi-output contract.")
        for index, name in enumerate(output_names):
            output = outputs[index]
            if not isinstance(output, dict) or output.get("name") != name:
                raise ValueError("--plan-json has inconsistent multi-output names.")
            if output.get("output_key") != embedding_output_key(base_key, name):
                raise ValueError("--plan-json has a noncanonical named-output key.")
            expected_output_shards = [
                embedding_output_shard_key(key, name) for key in canonical_shard_keys
            ]
            if output.get("shard_keys") != expected_output_shards:
                raise ValueError("--plan-json has noncanonical named-output shard keys.")
    elif outputs not in (None, []):
        raise ValueError("--plan-json declares outputs for a single-output extractor.")

    if output_key is not None:
        raise ValueError("--output-key cannot be combined with --plan-json.")
    selected = validated_entries.get(shard_index)
    if selected is None:
        raise ValueError("--plan-json does not contain the requested shard index.")
    if total_shards is not None and total_shards != selected["total_shards"]:
        raise ValueError("--total-shards conflicts with the selected plan entry.")
    return selected


def _cmd_fit_extractor(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor, bundle_path = _fit_extractor_bundle(
        source_path=Path(args.extractor_pickle),
        dataset=dataset,
        bundle_path=Path(args.output_pickle),
        overwrite=bool(args.force),
    )
    bundle = _load_pickle(bundle_path)
    _validated_fitted_extractor_bundle(bundle, dataset)
    return {
        "artifact_type": _FITTED_EXTRACTOR_BUNDLE_TYPE,
        "dataset_identity_key": dataset.identity_key(),
        "source_recipe_hash": bundle["source_recipe_hash"],
        "fitted_recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "output_pickle": bundle_path,
    }


def _cmd_embed_shard(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    plan = _load_json(args.plan_json) if args.plan_json else None
    if plan is not None:
        if not isinstance(plan, dict):
            raise TypeError("--plan-json must contain a JSON object.")
        if (
            plan.get("artifact_type") != _EMBEDDING_PLAN_TYPE
            or plan.get("schema_version") != _EMBEDDING_PLAN_SCHEMA_VERSION
        ):
            raise ValueError(
                "--plan-json is stale or invalid; embed-shard requires an embedding "
                "shard plan with schema_version=2."
            )
        planned_bundle = plan.get("extractor_pickle")
        if not isinstance(planned_bundle, str) or not planned_bundle:
            raise ValueError("--plan-json is missing its fitted extractor bundle path.")
        if Path(args.extractor_pickle).resolve() != Path(planned_bundle).resolve():
            raise ValueError(
                "--extractor-pickle differs from the fitted extractor bundle bound to "
                "--plan-json."
            )
        planned_bundle_sha256 = plan.get("fitted_bundle_sha256")
        if (
            not isinstance(planned_bundle_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", planned_bundle_sha256) is None
        ):
            raise ValueError("--plan-json has an invalid fitted extractor bundle digest.")
        extractor = _load_planned_fitted_extractor_bundle(
            args.extractor_pickle,
            dataset,
            expected_sha256=planned_bundle_sha256,
        )
    else:
        extractor = _load_fitted_extractor_bundle(args.extractor_pickle, dataset)
    if plan is not None:
        planned = _validated_embedding_plan_entry(
            plan,
            dataset=dataset,
            extractor=extractor,
            extractor_pickle=args.extractor_pickle,
            shard_index=args.shard_index,
            total_shards=args.total_shards,
            output_key=args.output_key,
        )
        shard = ShardSpec(
            total_shards=planned["total_shards"],
            shard_index=planned["shard_index"],
        )
        batch_size = int(plan.get("batch_size", args.batch_size))
        job = EmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            shard=shard,
            output_key=planned["output_key"],
            batch_size=batch_size,
            streaming_enabled=bool(getattr(extractor, "streaming_safe", False)),
            cache_eligible=bool(plan.get("cache_eligible", True)),
            cache_status=str(plan.get("cache_status", "miss")),
            resource_profiling_config=_resource_profiling_config_from_args(args, plan),
        )
    else:
        if args.total_shards is None:
            raise ValueError("embed-shard requires --total-shards without --plan-json.")
        jobs = plan_embedding_shard_jobs(
            dataset,
            extractor,
            total_shards=args.total_shards,
            batch_size=args.batch_size,
            resource_profiling_config=_resource_profiling_config_from_args(args),
        )
        try:
            job = jobs[args.shard_index]
        except IndexError as exc:
            raise ValueError(
                f"shard-index {args.shard_index} is outside the effective {len(jobs)} shards."
            ) from exc
        if args.output_key is not None:
            if not job.cache_eligible and not args.output_key.startswith("runs/"):
                raise ValueError(
                    "Unsafe extractor identities require --output-key beneath 'runs/'."
                )
            job = replace(job, output_key=args.output_key)
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
    if args.query_branch is None or args.gallery_branch is None:
        extractor, extractor_pickle = _load_or_create_fitted_retrieval_extractor(
            args.extractor_pickle,
            dataset,
        )
    else:
        extractor = _load_pickle(args.extractor_pickle)
        extractor_pickle = str(Path(args.extractor_pickle))
    resource_config = _resource_profiling_config_from_args(args)
    run_prefix = f"runs/{uuid4().hex}" if extractor.recipe().get("cache_safe") is False else None
    query_jobs = plan_retrieval_embedding_shard_jobs(
        dataset,
        extractor,
        args.total_shards,
        side="query",
        branch=args.query_branch,
        batch_size=args.batch_size,
        resource_profiling_config=resource_config,
        run_prefix=run_prefix,
    )
    gallery_jobs = plan_retrieval_embedding_shard_jobs(
        dataset,
        extractor,
        args.total_shards,
        side="gallery",
        branch=args.gallery_branch,
        batch_size=args.batch_size,
        resource_profiling_config=resource_config,
        run_prefix=run_prefix,
    )
    return {
        "artifact_type": "retrieval_embedding_plan",
        "dataset_identity_key": dataset.identity_key(),
        "extractor_recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "extractor_pickle": extractor_pickle,
        "resource_profiling_config": asdict(resource_config),
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
    values = (
        first.dataset.query_values() if first.side == "query" else first.dataset.gallery_values()
    )
    base_key = first.output_key.rsplit("/shards/", 1)[0]
    return {
        "side": first.side,
        "branch": first.branch,
        "n_samples": len(values),
        "output_key": base_key,
        "cache_eligible": first.cache_eligible,
        "cache_status": first.cache_status,
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
    fitted_bundle = False
    if isinstance(extractor, dict) and extractor.get("artifact_type") == (
        _FITTED_RETRIEVAL_EXTRACTOR_BUNDLE_TYPE
    ):
        extractor = _validated_fitted_retrieval_extractor_bundle(extractor, dataset)
        fitted_bundle = True
    elif args.branch is None and getattr(extractor, "already_fitted", True) is False:
        raise TypeError(
            "Standard retrieval shard workers require a fitted retrieval bundle. "
            "Run plan-retrieval first and use its extractor_pickle."
        )
    plan = _load_json(args.plan_json) if args.plan_json else None
    if plan is not None:
        endpoint = plan.get("endpoints", {}).get(args.side)
        if endpoint is None:
            raise ValueError("--plan-json does not contain the requested retrieval endpoint.")
        planned = next(
            (
                item
                for item in endpoint.get("shards", [])
                if item.get("shard", {}).get("shard_index") == args.shard_index
            ),
            None,
        )
        if planned is None:
            raise ValueError("--plan-json does not contain the requested retrieval shard.")
        if args.branch is not None and args.branch != planned.get("branch"):
            raise ValueError("--branch conflicts with the selected retrieval plan entry.")
        if args.total_shards is not None and args.total_shards != planned["shard"]["total_shards"]:
            raise ValueError("--total-shards conflicts with the retrieval plan entry.")
        job = RetrievalEmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            side=args.side,
            branch=planned.get("branch"),
            shard=ShardSpec(**planned["shard"]),
            output_key=args.output_key or planned["output_key"],
            batch_size=int(planned.get("batch_size", args.batch_size)),
            streaming_enabled=bool(getattr(extractor, "streaming_safe", False)),
            cache_eligible=bool(endpoint.get("cache_eligible", True)),
            cache_status=str(endpoint.get("cache_status", "miss")),
            fitted_bundle=fitted_bundle,
            resource_profiling_config=_resource_profiling_config_from_args(args, plan),
        )
    else:
        if args.total_shards is None:
            raise ValueError("embed-retrieval-shard requires --total-shards without --plan-json.")
        jobs = plan_retrieval_embedding_shard_jobs(
            dataset,
            extractor,
            args.total_shards,
            side=args.side,
            branch=args.branch,
            batch_size=args.batch_size,
            resource_profiling_config=_resource_profiling_config_from_args(args),
        )
        try:
            job = jobs[args.shard_index]
        except IndexError as exc:
            raise ValueError(
                f"shard-index {args.shard_index} is outside the effective {len(jobs)} shards."
            ) from exc
        if args.output_key is not None:
            if not job.cache_eligible and not args.output_key.startswith("runs/"):
                raise ValueError(
                    "Unsafe extractor identities require --output-key beneath 'runs/'."
                )
            job = replace(job, output_key=args.output_key)
        if fitted_bundle:
            job = replace(job, fitted_bundle=True)
    return materialize_retrieval_embedding_shard(job, _store_from_args(args))


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
    resource_config = _resource_profiling_config_from_args(args)
    run_prefix = f"runs/{uuid4().hex}" if extractor.recipe().get("cache_safe") is False else None
    sample_jobs = plan_zero_shot_embedding_shard_jobs(
        dataset,
        extractor,
        args.total_shards,
        side="samples",
        branch=args.sample_branch,
        batch_size=args.batch_size,
        resource_profiling_config=resource_config,
        run_prefix=run_prefix,
    )
    prompt_jobs = plan_zero_shot_embedding_shard_jobs(
        dataset,
        extractor,
        args.total_shards,
        side="prompts",
        branch=args.text_branch,
        batch_size=args.batch_size,
        resource_profiling_config=resource_config,
        run_prefix=run_prefix,
    )
    return {
        "artifact_type": "zero_shot_embedding_plan",
        "dataset_identity_key": dataset.dataset.identity_key(),
        "extractor_recipe_hash": fingerprint_extractor_recipe(extractor.recipe()),
        "protocol_key": zero_shot_protocol_artifact_key(dataset),
        "resource_profiling_config": asdict(resource_config),
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
    base_key = first.output_key.rsplit("/shards/", 1)[0]
    return {
        "side": first.side,
        "branch": first.branch,
        "n_samples": len(values),
        "output_key": base_key,
        "cache_eligible": first.cache_eligible,
        "cache_status": first.cache_status,
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
        batch_size = int(planned.get("batch_size", args.batch_size))
        cache_eligible = bool(endpoint.get("cache_eligible", True))
        cache_status = str(endpoint.get("cache_status", "miss"))
    else:
        if not args.branch or args.total_shards is None:
            raise ValueError(
                "embed-zero-shot-shard requires --branch and --total-shards without --plan-json."
            )
        jobs = plan_zero_shot_embedding_shard_jobs(
            dataset,
            extractor,
            args.total_shards,
            side=args.side,
            branch=args.branch,
            batch_size=args.batch_size,
            resource_profiling_config=_resource_profiling_config_from_args(args),
        )
        try:
            selected_job = jobs[args.shard_index]
        except IndexError as exc:
            raise ValueError(
                f"shard-index {args.shard_index} is outside the effective {len(jobs)} shards."
            ) from exc
        if args.output_key is not None:
            if not selected_job.cache_eligible and not args.output_key.startswith("runs/"):
                raise ValueError(
                    "Unsafe extractor identities require --output-key beneath 'runs/'."
                )
            selected_job = replace(selected_job, output_key=args.output_key)
        branch = selected_job.branch
        shard = selected_job.shard
        output_key = selected_job.output_key
        batch_size = selected_job.batch_size
        cache_eligible = selected_job.cache_eligible
        cache_status = selected_job.cache_status
    return materialize_zero_shot_embedding_shard(
        ZeroShotEmbeddingShardJob(
            dataset=dataset,
            extractor=extractor,
            side=args.side,
            branch=branch,
            shard=shard,
            output_key=output_key,
            batch_size=batch_size,
            streaming_enabled=bool(getattr(extractor, "streaming_safe", False)),
            cache_eligible=cache_eligible,
            cache_status=cache_status,
            resource_profiling_config=_resource_profiling_config_from_args(args, plan),
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
        resource_profiling_config=_resource_profiling_config_from_args(args),
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
        resource_profiling_config=_resource_profiling_config_from_args(args),
    )


def _cmd_score(args: argparse.Namespace) -> dict[str, Any]:
    embedding_key, labels_key, groups_key = _scoring_inputs_from_args(args)
    scoring_config = (
        _load_pickle(args.scoring_config_pickle)
        if args.scoring_config_pickle
        else OverlapScoringConfig()
    )
    metrics = _metrics_from_args(args)
    output_key = args.output_key or scoring_artifact_key(
        embedding_key,
        seed=args.seed,
        labels_key=labels_key,
        groups_key=groups_key,
        scoring_config=scoring_config,
        metrics=metrics,
        primary_metric=args.primary_metric,
    )
    return score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            output_key=output_key,
            groups_key=groups_key,
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
        args.query_embedding_key,
        args.gallery_embedding_key,
        relevance_key=args.relevance_key,
        exclusions_key=args.exclusions_key,
        retrieval_config=config,
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
        callable(getattr(dataset, attribute, None))
        for attribute in (
            "query_id_values",
            "gallery_id_values",
            "normalized_relevance",
            "normalized_exclusions",
        )
    ):
        raise TypeError("--dataset-pickle must contain a RetrievalDataset.")
    query_ids = dataset.query_id_values()
    gallery_ids = dataset.gallery_id_values()
    output_key = args.output_key or f"retrieval/relevance/{dataset.identity_key()}"
    payload = {
        "artifact_type": "retrieval_relevance",
        "query_ids": list(query_ids),
        "gallery_ids": list(gallery_ids),
        "n_queries": len(query_ids),
        "n_gallery": len(gallery_ids),
        "relevance": dataset.normalized_relevance(),
        "exclusions": sorted(dataset.normalized_exclusions()),
        "dataset_identity_key": dataset.identity_key(),
        "protocol_fingerprint": dataset.identity_key(),
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


def _cmd_retrieval_from_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    result = retrieval_benchmark_result_from_artifacts(
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
    score_key = args.score_key or _resolve_score_key_from_plan(
        plan,
        embedding_key,
        labels_key=labels_key,
        groups_key=groups_key,
    )
    separatix_config = (
        _load_pickle(args.separatix_config_pickle)
        if args.separatix_config_pickle
        else SeparatixConfig()
    )
    output_key = args.output_key or separatix_artifact_key(
        embedding_key,
        labels_key=labels_key,
        groups_key=groups_key,
        score_key=score_key,
        separatix_config=separatix_config,
    )
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
    embedding_key, labels_key, groups_key = _scoring_inputs_from_args(args)
    seeds = args.seed or _repeat_seeds(args.repeats, args.random_state)
    scoring_config = (
        _load_pickle(args.scoring_config_pickle)
        if args.scoring_config_pickle
        else OverlapScoringConfig()
    )
    jobs = plan_scoring_jobs(
        embedding_key=embedding_key,
        labels_key=labels_key,
        groups_key=groups_key,
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
        "groups_key": groups_key,
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
    extractor, fitted_extractor_path, needs_fit = _planned_fitted_extractor(
        args.extractor_pickle, dataset
    )
    requested_shards = _validated_positive_int(args.total_shards, "total_shards")
    batch_size = _validated_positive_int(args.batch_size, "batch_size")
    planned_shards = (
        min(requested_shards, len(dataset.y))
        if bool(getattr(extractor, "streaming_safe", False))
        else 1
    )
    target = Path(args.script_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    plan_target = target.with_name(f"{target.stem}.plan.json")
    rendered_args = argparse.Namespace(
        **{
            **vars(args),
            "extractor_pickle": fitted_extractor_path,
            "total_shards": planned_shards,
            "batch_size": batch_size,
        }
    )
    script = _render_slurm_array_script(
        args=rendered_args,
        plan_json=str(plan_target),
    )
    target.write_text(script, encoding="utf-8")
    fit_target = target.with_name(f"{target.stem}.fit{target.suffix}")
    submit_target = target.with_name(f"{target.stem}.submit.sh")
    if needs_fit:
        fit_target.write_text(
            _render_slurm_fit_script(
                args,
                fitted_extractor_path,
                plan_json=str(plan_target),
                planned_shards=planned_shards,
            ),
            encoding="utf-8",
        )
        submit_target.write_text(
            _render_slurm_dependency_submission(fit_target, target),
            encoding="utf-8",
        )
    else:
        plan_args = argparse.Namespace(
            **{
                **vars(rendered_args),
                "backend": "local",
            }
        )
        _write_json_file(_cmd_plan(plan_args), str(plan_target))
        submit_target.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" f"sbatch {_shell_quote(target)}\n",
            encoding="utf-8",
        )
    return {
        "script_path": str(target),
        "fit_script_path": str(fit_target) if needs_fit else None,
        "submit_script_path": str(submit_target),
        "fitted_extractor_pickle": fitted_extractor_path,
        "plan_json": str(plan_target),
        "output_key": None,
        "n_samples": int(len(dataset.y)),
        "requested_total_shards": args.total_shards,
        "total_shards": planned_shards,
        "batch_size": batch_size,
    }


def _cmd_run_embedding_shards(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    extractor, _ = _load_or_create_fitted_extractor(args.extractor_pickle, dataset)
    jobs = plan_embedding_shard_jobs(
        dataset=dataset,
        extractor=extractor,
        total_shards=args.total_shards,
        batch_size=args.batch_size,
        resource_profiling_config=_resource_profiling_config_from_args(args),
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
    embedding_key, labels_key, groups_key = _scoring_inputs_from_args(args)
    seeds = _repeat_seeds(args.repeats, args.random_state)
    script = _render_slurm_score_array_script(args=args, seeds=seeds)
    target = Path(args.script_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")
    return {
        "script_path": str(target),
        "embedding_key": embedding_key,
        "labels_key": labels_key,
        "groups_key": groups_key,
        "score_keys": [
            scoring_artifact_key(
                embedding_key,
                seed=seed,
                labels_key=labels_key,
                groups_key=groups_key,
                scoring_config=OverlapScoringConfig(),
                metrics=(),
                primary_metric="overlap",
            )
            for seed in seeds
        ],
        "seeds": seeds,
    }


def _render_slurm_array_script(
    args: argparse.Namespace,
    plan_json: str,
) -> str:
    total_shards = _validated_positive_int(args.total_shards, "total_shards")
    batch_size = _validated_positive_int(args.batch_size, "batch_size")
    job_name = _validated_slurm_value(args.job_name, "job_name")
    time_limit = _validated_slurm_value(args.time, "time")
    memory = _validated_slurm_value(args.mem, "mem")
    cpus = _validated_positive_int(args.cpus_per_task, "cpus_per_task")
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{total_shards - 1}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --cpus-per-task={cpus}",
    ]
    if args.partition is not None:
        lines.append(f"#SBATCH --partition={_validated_slurm_value(args.partition, 'partition')}")
    lines.extend(
        [
            "set -euo pipefail",
            "",
            f"{_shell_quote(args.python_executable)} -m vertebrae.cli embed-shard \\",
            f"  --dataset-pickle {_shell_quote(args.dataset_pickle)} \\",
            f"  --extractor-pickle {_shell_quote(args.extractor_pickle)} \\",
            f"  --cache-dir {_shell_quote(args.cache_dir)} \\",
            *_cache_flag_lines(args),
            f"  --plan-json {_shell_quote(plan_json)} \\",
            '  --shard-index "${SLURM_ARRAY_TASK_ID}" \\',
            *(
                [
                    "  --resource-profiling-config-pickle "
                    f"{_shell_quote(args.resource_profiling_config_pickle)} \\",
                ]
                if args.resource_profiling_config_pickle
                else []
            ),
            f"  --batch-size {batch_size}",
            "",
            "# After the array completes, merge the shards with:",
            f"# {_shell_quote(args.python_executable)} -m vertebrae.cli merge-embeddings \\",
            f"#   --cache-dir {_shell_quote(args.cache_dir)} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"#   --plan-json {_shell_quote(plan_json)}",
        ]
    )
    lines.extend(
        [
            "#",
            "# Then materialize labels and score:",
            f"# {_shell_quote(args.python_executable)} -m vertebrae.cli write-labels \\",
            f"#   --dataset-pickle {_shell_quote(args.dataset_pickle)} \\",
            f"#   --cache-dir {_shell_quote(args.cache_dir)} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"# {_shell_quote(args.python_executable)} -m vertebrae.cli score \\",
            f"#   --cache-dir {_shell_quote(args.cache_dir)} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"#   --plan-json {_shell_quote(plan_json)}",
            "#",
            "# For distributed stability scoring, generate a scoring array with:",
            f"# {_shell_quote(args.python_executable)} -m vertebrae.cli slurm-score-array \\",
            f"#   --cache-dir {_shell_quote(args.cache_dir)} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"#   --plan-json {_shell_quote(plan_json)} \\",
            "#   --repeats 20 \\",
            "#   --script-output vertebrae_score.sbatch",
            "",
        ]
    )
    return "\n".join(lines)


def _render_slurm_fit_script(
    args: argparse.Namespace,
    output_pickle: str,
    *,
    plan_json: str,
    planned_shards: int,
) -> str:
    planned_shards = _validated_positive_int(planned_shards, "total_shards")
    batch_size = _validated_positive_int(args.batch_size, "batch_size")
    job_name = _validated_slurm_value(f"{args.job_name}-fit", "job_name")
    time_limit = _validated_slurm_value(args.time, "time")
    memory = _validated_slurm_value(args.mem, "mem")
    cpus = _validated_positive_int(args.cpus_per_task, "cpus_per_task")
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --cpus-per-task={cpus}",
    ]
    if args.partition is not None:
        lines.append(f"#SBATCH --partition={_validated_slurm_value(args.partition, 'partition')}")
    lines.extend(
        [
            "set -euo pipefail",
            "",
            f"{_shell_quote(args.python_executable)} -m vertebrae.cli fit-extractor \\",
            f"  --dataset-pickle {_shell_quote(args.dataset_pickle)} \\",
            f"  --extractor-pickle {_shell_quote(args.extractor_pickle)} \\",
            f"  --output-pickle {_shell_quote(output_pickle)}",
            "",
            f"{_shell_quote(args.python_executable)} -m vertebrae.cli plan \\",
            f"  --dataset-pickle {_shell_quote(args.dataset_pickle)} \\",
            f"  --extractor-pickle {_shell_quote(output_pickle)} \\",
            f"  --cache-dir {_shell_quote(args.cache_dir)} \\",
            *_cache_flag_lines(args),
            f"  --total-shards {planned_shards} \\",
            f"  --batch-size {batch_size} \\",
            *(
                [
                    "  --resource-profiling-config-pickle "
                    f"{_shell_quote(args.resource_profiling_config_pickle)} \\",
                ]
                if args.resource_profiling_config_pickle
                else []
            ),
            f"  --output-json {_shell_quote(plan_json)}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_slurm_dependency_submission(fit_script: Path, array_script: Path) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'FIT_JOB_ID="$(sbatch --parsable {_shell_quote(fit_script)})"',
            f'sbatch --dependency="afterok:${{FIT_JOB_ID}}" {_shell_quote(array_script)}',
            "",
        ]
    )


def _render_slurm_score_array_script(args: argparse.Namespace, seeds: list[int]) -> str:
    embedding_key, labels_key, groups_key = _scoring_inputs_from_args(args)
    job_name = _validated_slurm_value(args.job_name, "job_name")
    time_limit = _validated_slurm_value(args.time, "time")
    memory = _validated_slurm_value(args.mem, "mem")
    cpus = _validated_positive_int(args.cpus_per_task, "cpus_per_task")
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{len(seeds) - 1}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --cpus-per-task={cpus}",
    ]
    if args.partition is not None:
        lines.append(f"#SBATCH --partition={_validated_slurm_value(args.partition, 'partition')}")
    seed_values = " ".join(str(seed) for seed in seeds)
    lines.extend(
        [
            "set -euo pipefail",
            f"SEEDS=({seed_values})",
            'SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"',
            "",
            f"{_shell_quote(args.python_executable)} -m vertebrae.cli score \\",
            f"  --cache-dir {_shell_quote(args.cache_dir)} \\",
            *_cache_flag_lines(args),
            f"  --embedding-key {_shell_quote(embedding_key)} \\",
            f"  --labels-key {_shell_quote(labels_key)} \\",
        ]
    )
    if groups_key is not None:
        lines.append(f"  --groups-key {_shell_quote(groups_key)} \\")
    lines.extend(
        [
            '  --seed "${SEED}"',
            "",
            "# After the array completes, collect scores with:",
            f"# {_shell_quote(args.python_executable)} -m vertebrae.cli collect-scores \\",
            f"#   --cache-dir {_shell_quote(args.cache_dir)} \\",
            *[f"#   {line}" for line in _cache_flag_lines(args, indent=False)],
            f"#   --output-key {_shell_quote(f'{embedding_key}/scores/stability')} \\",
        ]
    )
    for index, seed in enumerate(seeds):
        suffix = " \\" if index < len(seeds) - 1 else ""
        score_key = scoring_artifact_key(
            embedding_key,
            seed=seed,
            labels_key=labels_key,
            groups_key=groups_key,
            scoring_config=OverlapScoringConfig(),
            metrics=(),
            primary_metric="overlap",
        )
        lines.append(f"#   --score-key {_shell_quote(score_key)}{suffix}")
    lines.append("")
    return "\n".join(lines)


def _add_object_args(
    parser: argparse.ArgumentParser,
    *,
    extractor_description: str = "Pickled extractor object.",
) -> None:
    parser.add_argument(
        "--dataset-pickle",
        required=True,
        help=_trusted_pickle_help("Pickled dataset object."),
    )
    parser.add_argument(
        "--extractor-pickle",
        required=True,
        help=_trusted_pickle_help(extractor_description),
    )


def _add_resource_profiling_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resource-profiling-config-pickle",
        help=_trusted_pickle_help(
            "Pickled ResourceProfilingConfig propagated to extraction workers."
        ),
    )


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


_FITTED_EXTRACTOR_BUNDLE_TYPE = "vertebrae_fitted_extractor_v1"
_FITTED_RETRIEVAL_EXTRACTOR_BUNDLE_TYPE = "vertebrae_fitted_retrieval_extractor_v1"


def _load_or_create_fitted_retrieval_extractor(
    path: str,
    dataset: Any,
) -> tuple[Any, str]:
    """Fit a standard retrieval extractor once on the gallery and publish a bundle."""

    source_path = Path(path)
    loaded = _load_pickle(path)
    if isinstance(loaded, dict) and loaded.get("artifact_type") == (
        _FITTED_RETRIEVAL_EXTRACTOR_BUNDLE_TYPE
    ):
        return _validated_fitted_retrieval_extractor_bundle(loaded, dataset), str(source_path)
    if not callable(getattr(loaded, "fit", None)) or not callable(
        getattr(loaded, "transform", None)
    ):
        raise TypeError(
            "Standard retrieval execution requires an extractor with fit() and transform()."
        )
    recipe = loaded.recipe()
    source_recipe_hash = fingerprint_extractor_recipe(recipe)
    # Unsafe recipes cannot prove that their serialized source state is covered
    # by the recipe hash. Keep one fitted bundle shared by this plan's workers,
    # but never reuse it as a fitted-state cache in a later driver invocation.
    run_scope = f"-{uuid4().hex}" if recipe.get("cache_safe") is False else ""
    bundle_path = source_path.with_name(
        f"{source_path.name}.vertebrae-retrieval-fitted-"
        f"{dataset.identity_key()[:16]}-{source_recipe_hash[:16]}{run_scope}.pkl"
    )
    lock_path = bundle_path.with_suffix(bundle_path.suffix + ".lock")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        if bundle_path.exists():
            bundle = _load_pickle(str(bundle_path))
            return (
                _validated_fitted_retrieval_extractor_bundle(
                    bundle,
                    dataset,
                    expected_source_recipe_hash=source_recipe_hash,
                ),
                str(bundle_path),
            )
        gallery = (
            dataset.gallery_values()
            if callable(getattr(dataset, "gallery_values", None))
            else dataset.gallery
        )
        loaded.fit(gallery, y=None)
        bundle = {
            "artifact_type": _FITTED_RETRIEVAL_EXTRACTOR_BUNDLE_TYPE,
            "dataset_identity_key": dataset.identity_key(),
            "fit_side": "gallery",
            "source_recipe_hash": source_recipe_hash,
            "fitted_recipe_hash": fingerprint_extractor_recipe(loaded.recipe()),
            "extractor": loaded,
        }
        _write_pickle_atomic(bundle_path, bundle)
        return loaded, str(bundle_path)


def _validated_fitted_retrieval_extractor_bundle(
    bundle: Any,
    dataset: Any,
    *,
    expected_source_recipe_hash: Optional[str] = None,
) -> Any:
    """Validate dataset binding and fitted recipe identity for retrieval workers."""

    if not isinstance(bundle, dict) or bundle.get("artifact_type") != (
        _FITTED_RETRIEVAL_EXTRACTOR_BUNDLE_TYPE
    ):
        raise TypeError("Extractor pickle is not a fitted retrieval-extractor bundle.")
    if bundle.get("dataset_identity_key") != dataset.identity_key():
        raise ValueError("Fitted retrieval extractor belongs to a different dataset identity.")
    if bundle.get("fit_side") != "gallery":
        raise ValueError("Fitted retrieval extractor bundle has an invalid fit-side contract.")
    if (
        expected_source_recipe_hash is not None
        and bundle.get("source_recipe_hash") != expected_source_recipe_hash
    ):
        raise ValueError("Fitted retrieval extractor belongs to a different source recipe.")
    extractor = bundle.get("extractor")
    if extractor is None or not callable(getattr(extractor, "transform", None)):
        raise TypeError("Fitted retrieval bundle does not contain a valid extractor.")
    if bundle.get("fitted_recipe_hash") != fingerprint_extractor_recipe(extractor.recipe()):
        raise ValueError("Fitted retrieval bundle recipe fingerprint is inconsistent.")
    return extractor


def _planned_fitted_extractor(path: str, dataset: Any) -> tuple[Any, str, bool]:
    """Resolve the bundle path for a scheduler without fitting during script generation."""

    source_path = Path(path)
    loaded = _load_pickle(path)
    if isinstance(loaded, dict) and loaded.get("artifact_type") == _FITTED_EXTRACTOR_BUNDLE_TYPE:
        return _validated_fitted_extractor_bundle(loaded, dataset), str(source_path), False
    if (
        not callable(getattr(loaded, "fit", None))
        or not callable(getattr(loaded, "transform", None))
        or not callable(getattr(loaded, "recipe", None))
    ):
        raise TypeError(
            "Source pickle must contain an extractor with fit(), transform(), and recipe()."
        )
    return loaded, str(_default_fitted_bundle_path(source_path, dataset, loaded)), True


def _load_or_create_fitted_extractor(path: str, dataset: Any) -> tuple[Any, str]:
    """Load a fitted bundle, or fit once under a shared filesystem lock and create it."""

    source_path = Path(path)
    loaded = _load_pickle(path)
    if isinstance(loaded, dict) and loaded.get("artifact_type") == _FITTED_EXTRACTOR_BUNDLE_TYPE:
        return _validated_fitted_extractor_bundle(loaded, dataset), str(source_path)

    bundle_path = _default_fitted_bundle_path(source_path, dataset, loaded)
    return _fit_extractor_bundle(
        source_path=source_path,
        dataset=dataset,
        bundle_path=bundle_path,
        overwrite=False,
    )


def _default_fitted_bundle_path(source_path: Path, dataset: Any, extractor: Any) -> Path:
    recipe = extractor.recipe()
    source_recipe_hash = fingerprint_extractor_recipe(recipe)
    dataset_identity_key = dataset.identity_key()
    # An unsafe recipe cannot prove that this hash covers fitted/live state. Give
    # each driver invocation its own bundle so workers share one fit without a
    # later invocation reusing it as a cache entry.
    run_scope = f"-{uuid4().hex}" if recipe.get("cache_safe") is False else ""
    return source_path.with_name(
        f"{source_path.name}.vertebrae-fitted-"
        f"{dataset_identity_key[:16]}-{source_recipe_hash[:16]}{run_scope}.pkl"
    )


def _fit_extractor_bundle(
    *,
    source_path: Path,
    dataset: Any,
    bundle_path: Path,
    overwrite: bool,
) -> tuple[Any, str]:
    """Fit one source extractor and atomically publish a validated bundle."""

    if source_path.resolve(strict=False) == bundle_path.resolve(strict=False):
        raise ValueError("The fitted bundle output must differ from the source extractor pickle.")
    loaded = _load_pickle(str(source_path))
    if isinstance(loaded, dict) and loaded.get("artifact_type") == _FITTED_EXTRACTOR_BUNDLE_TYPE:
        raise TypeError("fit-extractor requires an unfitted source extractor pickle.")
    if not callable(getattr(loaded, "fit", None)) or not callable(
        getattr(loaded, "transform", None)
    ):
        raise TypeError("Source pickle must contain an extractor with fit() and transform().")
    source_recipe_hash = fingerprint_extractor_recipe(loaded.recipe())
    dataset_identity_key = dataset.identity_key()
    lock_path = bundle_path.with_suffix(bundle_path.suffix + ".lock")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        if bundle_path.exists() and not overwrite:
            bundle = _load_pickle(str(bundle_path))
            return (
                _validated_fitted_extractor_bundle(
                    bundle,
                    dataset,
                    expected_source_recipe_hash=source_recipe_hash,
                ),
                str(bundle_path),
            )
        loaded.fit(dataset.X, dataset.y)
        bundle = {
            "artifact_type": _FITTED_EXTRACTOR_BUNDLE_TYPE,
            "dataset_identity_key": dataset_identity_key,
            "source_recipe_hash": source_recipe_hash,
            "fitted_recipe_hash": fingerprint_extractor_recipe(loaded.recipe()),
            "extractor": loaded,
        }
        _write_pickle_atomic(bundle_path, bundle)
        return loaded, str(bundle_path)


def _load_fitted_extractor_bundle(path: str, dataset: Any) -> Any:
    """Require and validate a fitted bundle at a shard-worker boundary."""

    bundle = _load_pickle(path)
    if not isinstance(bundle, dict) or bundle.get("artifact_type") != _FITTED_EXTRACTOR_BUNDLE_TYPE:
        raise TypeError(
            "embed-shard requires a fitted extractor bundle; create one with "
            "`vertebrae fit-extractor`."
        )
    return _validated_fitted_extractor_bundle(bundle, dataset)


def _load_planned_fitted_extractor_bundle(
    path: str,
    dataset: Any,
    *,
    expected_sha256: str,
) -> Any:
    """Hash and load one planned bundle from the same open file description."""

    with Path(path).open("rb") as file:
        actual_sha256 = _sha256_stream(file)
        if actual_sha256 != expected_sha256:
            raise ValueError("The fitted extractor bundle content does not match --plan-json.")
        file.seek(0)
        bundle = pickle.load(file)
    if not isinstance(bundle, dict) or bundle.get("artifact_type") != (
        _FITTED_EXTRACTOR_BUNDLE_TYPE
    ):
        raise TypeError(
            "embed-shard requires a fitted extractor bundle; create one with "
            "`vertebrae fit-extractor`."
        )
    return _validated_fitted_extractor_bundle(bundle, dataset)


def _validated_fitted_extractor_bundle(
    bundle: Any,
    dataset: Any,
    *,
    expected_source_recipe_hash: Optional[str] = None,
) -> Any:
    if not isinstance(bundle, dict) or bundle.get("artifact_type") != _FITTED_EXTRACTOR_BUNDLE_TYPE:
        raise TypeError("Extractor pickle is not a valid vertebrae fitted-extractor bundle.")
    if bundle.get("dataset_identity_key") != dataset.identity_key():
        raise ValueError("Fitted extractor bundle belongs to a different dataset identity.")
    if (
        expected_source_recipe_hash is not None
        and bundle.get("source_recipe_hash") != expected_source_recipe_hash
    ):
        raise ValueError("Fitted extractor bundle belongs to a different source recipe.")
    extractor = bundle.get("extractor")
    if extractor is None or not callable(getattr(extractor, "transform", None)):
        raise TypeError("Fitted extractor bundle does not contain a valid extractor.")
    if bundle.get("fitted_recipe_hash") != fingerprint_extractor_recipe(extractor.recipe()):
        raise ValueError("Fitted extractor bundle recipe fingerprint is inconsistent.")
    return extractor


def _write_pickle_atomic(target: Path, value: Any) -> None:
    descriptor = None
    temporary_name = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        with os.fdopen(descriptor, "wb") as file:
            descriptor = None
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _sha256_stream(file: Any, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Hash all bytes from the current position of an open binary stream."""

    digest = hashlib.sha256()
    while chunk := file.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Hash complete file bytes without materializing a fitted bundle in memory."""

    with path.open("rb") as file:
        return _sha256_stream(file, chunk_size)


def _validated_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return value


def _shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def _validated_slurm_value(value: Any, name: str) -> str:
    text = str(value)
    if not text or any(not (character.isalnum() or character in "._:/+-") for character in text):
        raise ValueError(
            f"SLURM {name} must contain only letters, digits, '.', '_', ':', '/', '+', or '-'."
        )
    if name == "time":
        match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", text)
        if match is None:
            raise ValueError(
                "SLURM time must use [days-]hours:minutes:seconds with two-digit "
                "minutes and seconds."
            )
        days, hours, minutes, seconds = (
            int(match.group(1) or 0),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        if match.group(1) is not None and hours > 23:
            raise ValueError("SLURM time hours must be <= 23 when days are present.")
        if days == hours == minutes == seconds == 0:
            raise ValueError("SLURM time must be greater than zero.")
    elif name == "mem":
        if re.fullmatch(r"[1-9]\d*(?:[KMGTP](?:i?B)?)?", text, re.IGNORECASE) is None:
            raise ValueError(
                "SLURM mem must be a positive integer with an optional K, M, G, T, or P suffix."
            )
    return text


def _resource_profiling_config_from_args(
    args: argparse.Namespace,
    plan: Optional[dict[str, Any]] = None,
) -> ResourceProfilingConfig:
    path = getattr(args, "resource_profiling_config_pickle", None)
    if path:
        config = _load_pickle(path)
        if not isinstance(config, ResourceProfilingConfig):
            raise TypeError(
                "--resource-profiling-config-pickle must contain a " "ResourceProfilingConfig."
            )
        return config
    serialized = (plan or {}).get("resource_profiling_config")
    if serialized is not None:
        if not isinstance(serialized, dict):
            raise TypeError("Plan resource_profiling_config must be a JSON object.")
        return ResourceProfilingConfig(**serialized)
    return ResourceProfilingConfig()


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
        flags.append(f"{prefix}--s3-endpoint-url {_shell_quote(args.s3_endpoint_url)} \\")
    if getattr(args, "s3_profile", None):
        flags.append(f"{prefix}--s3-profile {_shell_quote(args.s3_profile)} \\")
    if getattr(args, "s3_region", None):
        flags.append(f"{prefix}--s3-region {_shell_quote(args.s3_region)} \\")
    if getattr(args, "gcs_project", None):
        flags.append(f"{prefix}--gcs-project {_shell_quote(args.gcs_project)} \\")
    return flags


def _scoring_inputs_from_args(args: argparse.Namespace) -> tuple[str, str, Optional[str]]:
    plan = _load_json(args.plan_json) if getattr(args, "plan_json", None) else {}
    embedding_key = getattr(args, "embedding_key", None) or _resolve_embedding_key_from_plan(plan)
    labels_key = getattr(args, "labels_key", None) or _resolve_related_key_from_plan(
        plan,
        embedding_key,
        "labels_key",
    )
    groups_key = getattr(args, "groups_key", None) or _resolve_related_key_from_plan(
        plan,
        embedding_key,
        "groups_key",
    )
    if embedding_key is None:
        raise ValueError("An embedding key or plan JSON is required.")
    if labels_key is None:
        raise ValueError("A labels key or plan JSON is required.")
    return embedding_key, labels_key, groups_key


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
    target.write_text(json_dumps_strict(payload, indent=2, sort_keys=True) + "\n")


def _benchmark_result_from_dict(
    data: dict[str, Any],
    benchmark_cls: Any,
    extractor_cls: Any,
    overlap_cls: Any,
    separatix_cls: Any,
    metric_cls: Any,
) -> Any:
    from vertebrae.profiling import resource_profile_like_from_dict

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
                resource_profile=resource_profile_like_from_dict(item.get("resource_profile")),
            )
        )
    return benchmark_cls(
        dataset_summary=data.get("dataset_summary", {}),
        extractor_results=extractor_results,
        recommendations=data.get("recommendations", []),
        metadata=data.get("metadata", {}),
    )


def _resolve_score_key_from_plan(
    plan: dict[str, Any],
    embedding_key: str,
    *,
    labels_key: str,
    groups_key: Optional[str],
) -> str:
    return _resolve_related_key_from_plan(plan, embedding_key, "score_key") or scoring_artifact_key(
        embedding_key,
        labels_key=labels_key,
        groups_key=groups_key,
        scoring_config=OverlapScoringConfig(),
        metrics=(),
        primary_metric="overlap",
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
    text = json_dumps_strict(payload, indent=2, sort_keys=True)
    if output_json:
        target = Path(output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    raise SystemExit(main())
