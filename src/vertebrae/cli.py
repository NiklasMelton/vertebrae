"""Command line interface for distributed vertebrae workflows."""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.execution import (
    EmbeddingMergeJob,
    ScoringJob,
    ShardSpec,
    embedding_artifact_key,
    embedding_shard_key,
    labels_artifact_key,
    materialize_embedding_shard,
    materialize_label_artifact,
    merge_embedding_shards,
    plan_embedding_shard_jobs,
    score_embedding_artifact,
    scoring_artifact_key,
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
    return {
        "dataset_pickle": str(Path(args.dataset_pickle)),
        "extractor_pickle": str(Path(args.extractor_pickle)),
        "cache_dir": args.cache_dir,
        "base_key": base_key,
        "output_key": base_key,
        "labels_key": labels_artifact_key(dataset),
        "score_key": scoring_artifact_key(base_key),
        "n_samples": int(len(dataset.y)),
        "total_shards": args.total_shards,
        "batch_size": args.batch_size,
        "shard_jobs": [
            {
                "total_shards": job.shard.total_shards,
                "shard_index": job.shard.shard_index,
                "output_key": job.output_key,
            }
            for job in jobs
        ],
    }


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
    return materialize_embedding_shard(job, LocalArtifactStore(args.cache_dir))


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
        LocalArtifactStore(args.cache_dir),
    )


def _cmd_write_labels(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_pickle(args.dataset_pickle)
    return materialize_label_artifact(
        dataset,
        LocalArtifactStore(args.cache_dir),
        key=args.output_key,
    )


def _cmd_score(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json) if args.plan_json else {}
    embedding_key = args.embedding_key or plan.get("output_key")
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
        LocalArtifactStore(args.cache_dir),
    )


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
            f"  --total-shards {args.total_shards} \\",
            "  --shard-index ${SLURM_ARRAY_TASK_ID} \\",
            f"  --batch-size {args.batch_size}",
            "",
            "# After the array completes, merge the shards with:",
            f"# {args.python_executable} -m vertebrae.cli merge-embeddings \\",
            f"#   --cache-dir {args.cache_dir} \\",
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
            f"#   --cache-dir {args.cache_dir}",
            f"# {args.python_executable} -m vertebrae.cli score \\",
            f"#   --cache-dir {args.cache_dir} \\",
            f"#   --embedding-key {output_key} \\",
            f"#   --labels-key {labels_key}",
            "",
        ]
    )
    return "\n".join(lines)


def _add_object_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-pickle", required=True)
    parser.add_argument("--extractor-pickle", required=True)


def _add_cache_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", default=".vertebrae_cache")


def _load_pickle(path: str) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


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
