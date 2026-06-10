"""Command line interface for distributed vertebrae workflows."""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from vertebrae.cache import create_artifact_store
from vertebrae.config import EmbeddingCompressionConfig
from vertebrae.execution import (
    EmbeddingMergeJob,
    ScoringJob,
    SeparatixJob,
    ShardSpec,
    benchmark_result_from_artifacts,
    collect_score_artifacts,
    compress_embedding_artifact,
    create_execution_backend,
    diagnose_embedding_artifact,
    embedding_artifact_key,
    embedding_output_key,
    embedding_output_shard_key,
    embedding_shard_key,
    labels_artifact_key,
    materialize_embedding_shard,
    materialize_label_artifact,
    merge_embedding_shards,
    plan_compression_job,
    plan_embedding_shard_jobs,
    plan_scoring_jobs,
    score_embedding_artifact,
    score_embedding_artifacts,
    scoring_artifact_key,
    separatix_artifact_key,
)
from vertebrae.execution.jobs import EmbeddingShardJob


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

    labels = subparsers.add_parser("write-labels", help="Materialize dataset labels.")
    labels.add_argument("--dataset-pickle", required=True)
    _add_cache_arg(labels)
    labels.add_argument("--output-key")
    labels.add_argument("--output-json")
    labels.set_defaults(func=_cmd_write_labels)

    score = subparsers.add_parser("score", help="Score persisted embeddings and labels.")
    _add_cache_arg(score)
    score.add_argument("--embedding-key")
    score.add_argument("--labels-key")
    score.add_argument("--output-key")
    score.add_argument("--plan-json")
    score.add_argument("--scoring-config-pickle")
    score.add_argument("--seed", type=int)
    score.add_argument("--output-json")
    score.set_defaults(func=_cmd_score)

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
    _add_backend_args(repeats, include_local_parallel=True)
    repeats.add_argument("--output-json")
    repeats.set_defaults(func=_cmd_score_repeats)

    collect = subparsers.add_parser("collect-scores", help="Collect score artifacts.")
    _add_cache_arg(collect)
    collect.add_argument("--score-key", action="append", default=[])
    collect.add_argument("--score-plan-json")
    collect.add_argument("--output-key", required=True)
    collect.add_argument("--interval-level", type=float, default=0.95)
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


def _cmd_write_labels(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    return materialize_label_artifact(
        dataset,
        _store_from_args(args),
        key=args.output_key,
    )


def _cmd_score(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json) if args.plan_json else {}
    embedding_key = args.embedding_key or _resolve_embedding_key_from_plan(plan)
    labels_key = args.labels_key or plan.get("labels_key")
    if embedding_key is None:
        raise ValueError("score requires --embedding-key or --plan-json.")
    if labels_key is None:
        raise ValueError("score requires --labels-key or --plan-json.")
    output_key = args.output_key or scoring_artifact_key(embedding_key, seed=args.seed)
    scoring_config = (
        _load_pickle(args.scoring_config_pickle) if args.scoring_config_pickle else None
    )
    return score_embedding_artifact(
        ScoringJob(
            embedding_key=embedding_key,
            labels_key=labels_key,
            output_key=output_key,
            scoring_config=scoring_config,
            seed=args.seed,
        ),
        _store_from_args(args),
    )


def _cmd_diagnose_complexity(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json) if args.plan_json else {}
    embedding_key = args.embedding_key or _resolve_embedding_key_from_plan(plan)
    labels_key = args.labels_key or plan.get("labels_key")
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
        from vertebrae.scoring.overlap import OverlapScoreResult
        from vertebrae.scoring.separatix import SeparatixResult

        markdown = render_markdown_report(
            _benchmark_result_from_dict(
                result,
                BenchmarkResult,
                ExtractorResult,
                OverlapScoreResult,
                SeparatixResult,
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
    labels_key = getattr(args, "labels_key", None) or plan.get("labels_key")
    if embedding_key is None:
        raise ValueError("An embedding key or plan JSON is required.")
    if labels_key is None:
        raise ValueError("A labels key or plan JSON is required.")
    return embedding_key, labels_key


def _resolve_embedding_key_from_plan(plan: dict[str, Any]) -> Optional[str]:
    outputs = plan.get("outputs", [])
    if not outputs:
        return plan.get("output_key")
    if len(outputs) == 1:
        return outputs[0].get("output_key")
    raise ValueError(
        "This plan contains multiple embedding outputs. Pass --embedding-key with one of the "
        "output keys listed in the plan JSON."
    )


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
) -> Any:
    extractor_results = []
    for item in data.get("extractor_results", []):
        overlap = overlap_cls(**item["overlap"])
        separatix = None
        if item.get("separatix") is not None:
            separatix = separatix_cls(**item["separatix"])
        extractor_results.append(
            extractor_cls(
                name=item["name"],
                extractor_type=item["extractor_type"],
                overlap=overlap,
                stability=item.get("stability"),
                probes=item.get("probes"),
                separatix=separatix,
                embedding_metadata=item.get("embedding_metadata", {}),
                compression_metadata=item.get("compression_metadata", {"method": "none"}),
                runtime=item.get("runtime", {}),
                warnings=item.get("warnings", []),
                recommendation=item.get("recommendation", ""),
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
    outputs = plan.get("outputs", [])
    if not outputs:
        return plan.get("score_key") or scoring_artifact_key(embedding_key)
    for output in outputs:
        if output.get("output_key") == embedding_key:
            return output.get("score_key") or scoring_artifact_key(embedding_key)
    return scoring_artifact_key(embedding_key)


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
