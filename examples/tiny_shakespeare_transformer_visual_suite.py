"""Train and inspect a laptop-scale character GPT on Tiny Shakespeare.

The held-out probe asks a deliberately exact question: how well does each hidden
representation separate the *next character* after a 64-character validation
context? Language-model cross-entropy is measured separately on the naturally
distributed validation windows.

Install and run from the repository root:

    poetry install -E text-visuals
    poetry run python examples/tiny_shakespeare_transformer_visual_suite.py

Run the larger model with best-validation checkpoint selection:

    poetry run python examples/tiny_shakespeare_transformer_visual_suite.py \
        --profile quality
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import string
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from _common import ensure_output_dir

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    DatasetIdentity,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.config import CacheConfig
from vertebrae.extractors import PrecomputedExtractor, TorchExtractor

_CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/" "data/tinyshakespeare/input.txt"
)
_CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
_CORPUS_BYTES = 1_115_394
_OUTPUT_ORDER = ("token_position", "block_1", "block_2", "block_4_final")
_OUTPUT_COLORS = {
    "token_position": "#2563EB",
    "block_1": "#7C3AED",
    "block_2": "#DB2777",
    "block_4": "#D97706",
    "block_4_final": "#EA580C",
    "block_6_final": "#DC2626",
}


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    default_steps: int
    context_length: int
    train_batch_size: int
    n_layers: int
    n_heads: int
    width: int
    mlp_width: int
    dropout: float
    output_layers: Tuple[int, ...]

    @property
    def output_order(self) -> Tuple[str, ...]:
        names = ["token_position"]
        for layer in self.output_layers:
            suffix = "_final" if layer == self.n_layers else ""
            names.append(f"block_{layer}{suffix}")
        return tuple(names)


_PROFILES = {
    "fast": TrainingProfile(
        name="fast",
        default_steps=30_000,
        context_length=64,
        train_batch_size=12,
        n_layers=4,
        n_heads=4,
        width=128,
        mlp_width=512,
        dropout=0.0,
        output_layers=(1, 2, 4),
    ),
    "quality": TrainingProfile(
        name="quality",
        default_steps=10_000,
        context_length=256,
        train_batch_size=32,
        n_layers=6,
        n_heads=8,
        width=256,
        mlp_width=1_024,
        dropout=0.1,
        output_layers=(2, 4, 6),
    ),
}


@dataclass(frozen=True)
class CorpusSplits:
    train: str
    validation: str
    test: str
    boundaries: Tuple[int, int]
    checksum: str


@dataclass(frozen=True)
class ProbeData:
    contexts: np.ndarray
    labels: np.ndarray
    token_ids: np.ndarray
    source_offsets: np.ndarray
    class_support: Dict[str, int]
    excluded_singletons: Tuple[str, ...]
    low_support_classes: Tuple[str, ...]
    eligible_classes: Tuple[str, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILES),
        default="fast",
        help=(
            "Training/model profile; quality uses a larger model and restores "
            "its best checkpoint."
        ),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--monitor-every", type=int, default=1_000)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--probe-per-class-cap", type=int, default=256)
    parser.add_argument("--minimum-macro-support", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps", "xpu"),
        default="auto",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("examples/data/tiny_shakespeare"))
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require a checksum-valid cached corpus and never access the network.",
    )
    return parser


def _apply_profile_defaults(args: argparse.Namespace) -> TrainingProfile:
    profile = _PROFILES[args.profile]
    if args.steps is None:
        args.steps = profile.default_steps
    if args.context_length is None:
        args.context_length = profile.context_length
    if args.train_batch_size is None:
        args.train_batch_size = profile.train_batch_size
    return profile


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive = (
        "monitor_every",
        "context_length",
        "train_batch_size",
        "evaluation_batch_size",
        "probe_per_class_cap",
        "minimum_macro_support",
    )
    if args.steps < 0:
        parser.error("--steps must be >= 0")
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")


def _profile_metadata(profile: TrainingProfile, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "name": profile.name,
        "steps": int(args.steps),
        "context_length": int(args.context_length),
        "train_batch_size": int(args.train_batch_size),
        "tokens_per_step": int(args.context_length * args.train_batch_size),
        "sampled_training_tokens": int(args.steps * args.context_length * args.train_batch_size),
        "blocks": profile.n_layers,
        "heads": profile.n_heads,
        "width": profile.width,
        "mlp_width": profile.mlp_width,
        "dropout": profile.dropout,
        "output_layers": list(profile.output_layers),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    profile = _apply_profile_defaults(args)
    _validate_args(args, parser)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import psutil
        import torch
    except ImportError as exc:
        print(exc)
        print("Install the visual dependencies with: poetry install -E text-visuals")
        return

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = _resolve_device(torch, args.device)
    device_metadata = _device_metadata(torch, device)
    _reset_device_peak_memory(torch, device)
    process = psutil.Process(os.getpid())
    initial_rss = int(process.memory_info().rss)

    corpus_path = _ensure_corpus(args.data_dir, download=not args.no_download)
    text = corpus_path.read_text(encoding="utf-8")
    splits = _split_corpus(text, checksum=_CORPUS_SHA256)
    vocabulary = tuple(sorted(set(splits.train)))
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    _validate_split_vocabulary(splits, char_to_id)
    train_ids = _encode(splits.train, char_to_id)
    validation_ids = _encode(splits.validation, char_to_id)
    probe = _build_probe(
        splits.validation,
        validation_ids,
        context_length=args.context_length,
        per_class_cap=args.probe_per_class_cap,
        minimum_macro_support=args.minimum_macro_support,
        seed=args.seed,
        source_offset=splits.boundaries[0],
    )
    probe_dataset = _probe_dataset(probe, splits, args.context_length, args.seed)
    scoring_config = _scoring_config(probe.low_support_classes, args.seed)

    model = _build_model(
        torch,
        len(vocabulary),
        args.context_length,
        n_layers=profile.n_layers,
        n_heads=profile.n_heads,
        width=profile.width,
        mlp_width=profile.mlp_width,
        dropout=profile.dropout,
        output_layers=profile.output_layers,
        profile_name=profile.name,
    ).to(device)
    extractor = _multi_output_extractor(model, torch, device, args.context_length, args.seed)
    initial_generation = _generate(
        model,
        "ROMEO:\n",
        char_to_id,
        vocabulary,
        torch=torch,
        device=device,
        seed=args.seed,
    )

    history_rows = []
    initial_metrics = _validation_metrics(
        model,
        validation_ids,
        context_length=args.context_length,
        batch_size=args.evaluation_batch_size,
        torch=torch,
        device=device,
    )
    initial_result = _representation_benchmark(
        probe_dataset,
        extractor,
        scoring_config,
        batch_size=args.evaluation_batch_size,
    )
    history_rows.extend(_history_rows(initial_result, step=0, language_metrics=initial_metrics))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(args.seed)
    checkpoint_steps = set(range(args.monitor_every, args.steps + 1, args.monitor_every))
    checkpoint_steps.add(args.steps)
    final_result = initial_result
    last_metrics = initial_metrics
    best_result = initial_result
    best_metrics = dict(initial_metrics)
    best_step = 0
    best_model_state = _snapshot_model_state(model)
    for step in range(1, args.steps + 1):
        learning_rate = _learning_rate(step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        training_loss = _training_step(
            model,
            optimizer,
            train_ids,
            context_length=args.context_length,
            batch_size=args.train_batch_size,
            generator=cpu_generator,
            torch=torch,
            device=device,
        )
        if step not in checkpoint_steps:
            continue
        last_metrics = _validation_metrics(
            model,
            validation_ids,
            context_length=args.context_length,
            batch_size=args.evaluation_batch_size,
            torch=torch,
            device=device,
        )
        last_metrics["training_loss"] = training_loss
        last_metrics["learning_rate"] = learning_rate
        final_result = _representation_benchmark(
            probe_dataset,
            extractor,
            scoring_config,
            batch_size=args.evaluation_batch_size,
        )
        history_rows.extend(_history_rows(final_result, step=step, language_metrics=last_metrics))
        if last_metrics["cross_entropy"] < best_metrics["cross_entropy"]:
            best_result = final_result
            best_metrics = dict(last_metrics)
            best_step = step
            best_model_state = _snapshot_model_state(model)
        print(
            f"step {step:>5}/{args.steps}: validation loss "
            f"{last_metrics['cross_entropy']:.4f}, "
            f"accuracy {last_metrics['top1_accuracy']:.3f}"
        )

    last_checkpoint_metrics = dict(last_metrics)
    model.load_state_dict(best_model_state)
    final_result = best_result
    last_metrics = best_metrics
    print(
        f"restored best validation checkpoint at step {best_step}: "
        f"loss {best_metrics['cross_entropy']:.4f}"
    )

    trained_generation = _generate(
        model,
        "ROMEO:\n",
        char_to_id,
        vocabulary,
        torch=torch,
        device=device,
        seed=args.seed,
    )
    final_embeddings = _extract_output_in_batches(
        extractor,
        probe.contexts,
        output_name=model.final_output_name,
        batch_size=args.evaluation_batch_size,
    )
    compression_result = _compression_benchmark(
        final_embeddings,
        probe.labels,
        scoring_config,
        source_output=model.final_output_name,
    )

    output_dir = ensure_output_dir()
    figure_dir = args.figure_dir or output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "tiny_shakespeare_monitoring_history.csv", index=False)
    runtime_metadata = {
        **device_metadata,
        "wall_time_seconds": time.perf_counter() - started,
        "initial_host_rss_bytes": initial_rss,
        "final_host_rss_bytes": int(process.memory_info().rss),
        "peak_host_rss_bytes": _peak_host_rss_bytes(process),
        "peak_device_memory_bytes": _peak_device_memory(torch, device),
        "corpus_sha256": splits.checksum,
        "corpus_bytes": corpus_path.stat().st_size,
        "split_character_counts": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "vocabulary_size": len(vocabulary),
        "profile": _profile_metadata(profile, args),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_checkpoint_step": best_step,
        "probe_rows": len(probe.labels),
        "scored_token_count": len(probe.eligible_classes),
        "macro_token_count": len(probe.eligible_classes) - len(probe.low_support_classes),
        "low_support_tokens": list(probe.low_support_classes),
        "excluded_singletons": list(probe.excluded_singletons),
        "class_support": probe.class_support,
        "source_offsets": probe.source_offsets.tolist(),
        "validation_metrics": best_metrics,
        "last_checkpoint_validation_metrics": last_checkpoint_metrics,
    }
    initial_result.metadata["tiny_shakespeare_suite"] = runtime_metadata
    final_result.metadata["tiny_shakespeare_suite"] = runtime_metadata
    compression_result.metadata["tiny_shakespeare_suite"] = runtime_metadata
    initial_result.save_json(str(output_dir / "tiny_shakespeare_initial_benchmark.json"))
    final_result.save_json(str(output_dir / "tiny_shakespeare_final_benchmark.json"))
    compression_result.save_json(str(output_dir / "tiny_shakespeare_compression.json"))
    (output_dir / "tiny_shakespeare_initial_generation.txt").write_text(
        initial_generation,
        encoding="utf-8",
    )
    (output_dir / "tiny_shakespeare_trained_generation.txt").write_text(
        trained_generation,
        encoding="utf-8",
    )

    paths = []
    paths.extend(
        _plot_monitoring(
            history,
            figure_dir,
            plt,
            context_length=args.context_length,
            profile=profile,
        )
    )
    paths.extend(_plot_compression(compression_result, len(probe.labels), figure_dir, plt))
    paths.extend(
        _plot_token_heatmap(
            initial_result,
            final_result,
            probe,
            figure_dir,
            plt,
        )
    )
    for path in paths:
        print(f"Wrote {path}")
    print(f"Metrics and generations written to {output_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_corpus(data_dir: Path, *, download: bool) -> Path:
    path = data_dir / "input.txt"
    if path.is_file():
        actual = _sha256(path)
        if actual != _CORPUS_SHA256:
            raise ValueError(
                f"Tiny Shakespeare checksum mismatch for {path}: "
                f"expected {_CORPUS_SHA256}, got {actual}."
            )
        return path
    if not download:
        raise FileNotFoundError(
            f"Tiny Shakespeare is not cached at {path}; remove --no-download to fetch it."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    temporary = data_dir / "input.txt.download"
    try:
        with (
            urllib.request.urlopen(_CORPUS_URL, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        actual = _sha256(temporary)
        if actual != _CORPUS_SHA256 or temporary.stat().st_size != _CORPUS_BYTES:
            raise ValueError(
                "Downloaded Tiny Shakespeare failed integrity validation: "
                f"expected {_CORPUS_SHA256}/{_CORPUS_BYTES} bytes, "
                f"got {actual}/{temporary.stat().st_size}."
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _split_corpus(text: str, *, checksum: str) -> CorpusSplits:
    train_end = int(len(text) * 0.90)
    validation_end = int(len(text) * 0.95)
    return CorpusSplits(
        train=text[:train_end],
        validation=text[train_end:validation_end],
        test=text[validation_end:],
        boundaries=(train_end, validation_end),
        checksum=checksum,
    )


def _validate_split_vocabulary(splits: CorpusSplits, char_to_id: Dict[str, int]) -> None:
    unseen = sorted((set(splits.validation) | set(splits.test)) - set(char_to_id))
    if unseen:
        raise ValueError(
            "Validation/test contain characters absent from the training-only vocabulary: "
            f"{unseen!r}."
        )


def _encode(text: str, char_to_id: Dict[str, int]) -> np.ndarray:
    return np.fromiter((char_to_id[character] for character in text), dtype=np.int64)


def _build_probe(
    validation_text: str,
    validation_ids: np.ndarray,
    *,
    context_length: int,
    per_class_cap: int,
    minimum_macro_support: int,
    seed: int,
    source_offset: int = 0,
) -> ProbeData:
    if len(validation_ids) <= context_length:
        raise ValueError("Validation text must be longer than the probe context length.")
    positions_by_class: Dict[str, list[int]] = {}
    for position in range(context_length, len(validation_ids)):
        positions_by_class.setdefault(validation_text[position], []).append(position)
    support = {label: len(positions) for label, positions in positions_by_class.items()}
    singletons = tuple(sorted(label for label, count in support.items() if count < 2))
    eligible = tuple(sorted(label for label, count in support.items() if count >= 2))
    low_support = tuple(
        sorted(label for label in eligible if support[label] < minimum_macro_support)
    )
    rng = np.random.default_rng(seed)
    selected = []
    for label in eligible:
        candidates = np.asarray(positions_by_class[label], dtype=np.int64)
        count = min(per_class_cap, len(candidates))
        chosen = rng.choice(candidates, size=count, replace=False)
        selected.extend(int(value) for value in np.sort(chosen))
    selected_positions = np.asarray(sorted(selected), dtype=np.int64)
    contexts = np.stack(
        [validation_ids[position - context_length : position] for position in selected_positions]
    )
    labels = np.asarray([validation_text[position] for position in selected_positions])
    return ProbeData(
        contexts=contexts,
        labels=labels,
        token_ids=validation_ids[selected_positions],
        source_offsets=selected_positions + int(source_offset),
        class_support=support,
        excluded_singletons=singletons,
        low_support_classes=low_support,
        eligible_classes=eligible,
    )


def _probe_dataset(
    probe: ProbeData,
    splits: CorpusSplits,
    context_length: int,
    seed: int,
) -> BenchmarkDataset:
    return BenchmarkDataset.from_arrays(
        probe.contexts,
        probe.labels,
        modality="text",
        identity=DatasetIdentity.from_manifest(
            "tiny-shakespeare-fixed-next-character-probe",
            {
                "corpus_sha256": splits.checksum,
                "split": "validation",
                "context_length": context_length,
                "source_offsets": probe.source_offsets.tolist(),
                "seed": seed,
            },
        ),
        metadata={
            "example": "tiny_shakespeare_transformer_visual_suite",
            "split": "fixed_held_out_validation_probe",
            "label_semantics": "exact_next_character",
            "class_support": probe.class_support,
            "excluded_singletons": list(probe.excluded_singletons),
            "low_support_classes": list(probe.low_support_classes),
            "source_offsets": probe.source_offsets.tolist(),
            "corpus_sha256": splits.checksum,
        },
    )


def _scoring_config(low_support_classes: Iterable[str], seed: int) -> OverlapScoringConfig:
    return OverlapScoringConfig(
        k="auto",
        min_k=10,
        max_k=50,
        min_samples_per_cluster=5,
        kmeans_kwargs={"random_state": seed, "batch_size": 512, "n_init": 3},
        exclude_classes=list(low_support_classes),
    )


def _device_available(torch: Any, device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return bool(torch.cuda.is_available())
    backend = getattr(torch, device, None)
    return bool(backend is not None and backend.is_available())


def _device_smoke_test(torch: Any, device: str) -> None:
    layer = torch.nn.Linear(3, 2).to(device)
    values = torch.ones((2, 3), device=device, requires_grad=True)
    layer(values).square().mean().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu" and hasattr(torch.xpu, "synchronize"):
        torch.xpu.synchronize()


def _resolve_device(torch: Any, requested: str) -> str:
    candidates = ("cuda", "mps", "xpu", "cpu") if requested == "auto" else (requested,)
    failures = []
    for candidate in candidates:
        if not _device_available(torch, candidate):
            failures.append(f"{candidate}: unavailable")
            continue
        try:
            _device_smoke_test(torch, candidate)
        except Exception as exc:  # device runtimes fail through backend-specific exceptions
            failures.append(f"{candidate}: smoke test failed ({exc})")
            continue
        return candidate
    detail = "; ".join(failures)
    if requested == "auto":
        raise RuntimeError(f"No usable Torch device was found: {detail}")
    raise RuntimeError(f"Requested Torch device '{requested}' is not usable: {detail}")


def _device_metadata(torch: Any, device: str) -> Dict[str, Any]:
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    elif device == "xpu":
        name = (
            torch.xpu.get_device_name(0) if hasattr(torch.xpu, "get_device_name") else "Intel XPU"
        )
        backend = "xpu"
    elif device == "mps":
        name = "Apple Metal Performance Shaders"
        backend = "mps"
    else:
        name = platform.processor() or platform.machine() or "CPU"
        backend = "cpu"
    return {
        "device": device,
        "backend": backend,
        "device_name": name,
        "torch_version": str(torch.__version__),
    }


def _reset_device_peak_memory(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats()


def _peak_device_memory(torch: Any, device: str) -> Optional[int]:
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device == "xpu" and hasattr(torch.xpu, "max_memory_allocated"):
        return int(torch.xpu.max_memory_allocated())
    return None


def _peak_host_rss_bytes(process: Any) -> int:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except (ImportError, AttributeError):
        return int(process.memory_info().rss)


def _build_model(
    torch: Any,
    vocabulary_size: int,
    context_length: int,
    *,
    n_layers: int = 4,
    n_heads: int = 4,
    width: int = 128,
    mlp_width: int = 512,
    dropout: float = 0.0,
    output_layers: Tuple[int, ...] = (1, 2, 4),
    profile_name: str = "fast",
) -> Any:
    if width % n_heads != 0:
        raise ValueError("Model width must be divisible by the number of attention heads.")
    if not output_layers or output_layers[-1] != n_layers:
        raise ValueError("output_layers must end with the final model layer.")
    if any(layer < 1 or layer > n_layers for layer in output_layers):
        raise ValueError("output_layers entries must identify existing transformer blocks.")

    class CausalSelfAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(
                width,
                n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.output_dropout = torch.nn.Dropout(dropout)

        def forward(self, values: Any) -> Any:
            length = values.shape[1]
            mask = torch.triu(
                torch.ones((length, length), dtype=torch.bool, device=values.device),
                diagonal=1,
            )
            attended = self.attention(
                values,
                values,
                values,
                attn_mask=mask,
                need_weights=False,
            )[0]
            return self.output_dropout(attended)

    class TransformerBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = torch.nn.LayerNorm(width)
            self.attention = CausalSelfAttention()
            self.mlp_norm = torch.nn.LayerNorm(width)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(width, mlp_width),
                torch.nn.GELU(),
                torch.nn.Linear(mlp_width, width),
                torch.nn.Dropout(dropout),
            )

        def forward(self, values: Any) -> Any:
            values = values + self.attention(self.attention_norm(values))
            return values + self.mlp(self.mlp_norm(values))

    class TinyShakespeareGPT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_length = context_length
            self.profile_name = profile_name
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.width = width
            self.mlp_width = mlp_width
            self.dropout = dropout
            self.output_layers = output_layers
            self.final_output_name = f"block_{n_layers}_final"
            self.output_order = (
                "token_position",
                *(f"block_{layer}" for layer in output_layers[:-1]),
                self.final_output_name,
            )
            self.token_embedding = torch.nn.Embedding(vocabulary_size, width)
            self.position_embedding = torch.nn.Embedding(context_length, width)
            self.blocks = torch.nn.ModuleList([TransformerBlock() for _ in range(n_layers)])
            self.final_norm = torch.nn.LayerNorm(width)
            self.lm_head = torch.nn.Linear(width, vocabulary_size, bias=False)
            self.apply(self._initialize_module)
            self.lm_head.weight = self.token_embedding.weight

        @staticmethod
        def _initialize_module(module: Any) -> None:
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.MultiheadAttention):
                torch.nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
                if module.in_proj_bias is not None:
                    torch.nn.init.zeros_(module.in_proj_bias)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        def forward(self, token_ids: Any) -> Dict[str, Any]:
            if token_ids.shape[1] > self.context_length:
                raise ValueError("Input sequence exceeds the configured context length.")
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            values = self.token_embedding(token_ids) + self.position_embedding(positions)
            outputs = {"token_position": values}
            for index, block in enumerate(self.blocks, start=1):
                values = block(values)
                outputs[f"block_{index}"] = values
            final = self.final_norm(values)
            outputs[self.final_output_name] = final
            outputs["logits"] = self.lm_head(final)
            return outputs

    return TinyShakespeareGPT()


def _multi_output_extractor(
    model: Any,
    torch: Any,
    device: str,
    context_length: int,
    seed: int,
) -> TorchExtractor:
    output_order = tuple(model.output_order)

    def collate_fn(batch: Any) -> Any:
        return torch.as_tensor(np.asarray(batch), dtype=torch.long)

    def output_fn(raw: Dict[str, Any]) -> Dict[str, Any]:
        return {name: raw[name][:, -1, :] for name in output_order}

    outputs = []
    for name in output_order:
        layer = 0 if name == "token_position" else int(name.split("_")[1])
        outputs.append({"name": name, "hidden_layer": layer, "pooling": "last_token"})
    return TorchExtractor(
        name="tiny_shakespeare_gpt",
        model=model,
        collate_fn=collate_fn,
        output_fn=output_fn,
        outputs=outputs,
        device=device,
        modality="text",
        recipe_data={
            "example": "tiny_shakespeare_transformer_visual_suite",
            "profile": model.profile_name,
            "architecture": {
                "blocks": model.n_layers,
                "heads": model.n_heads,
                "width": model.width,
                "mlp_width": model.mlp_width,
                "context_length": context_length,
                "dropout": model.dropout,
                "pre_layer_norm": True,
                "tied_lm_head": True,
            },
            "training_seed": seed,
        },
        restore_model_mode=True,
    )


def _learning_rate(step: int, total_steps: int) -> float:
    if step <= 100:
        return 1e-3 * step / 100
    if total_steps <= 100:
        return 1e-3
    progress = min(1.0, (step - 100) / (total_steps - 100))
    return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + math.cos(math.pi * progress))


def _snapshot_model_state(model: Any) -> Dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _training_step(
    model: Any,
    optimizer: Any,
    train_ids: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    generator: Any,
    torch: Any,
    device: str,
) -> float:
    maximum_start = len(train_ids) - context_length - 1
    if maximum_start < 1:
        raise ValueError("Training split is too short for the configured context length.")
    starts = torch.randint(0, maximum_start + 1, (batch_size,), generator=generator).tolist()
    contexts = np.stack([train_ids[start : start + context_length] for start in starts])
    targets = np.stack([train_ids[start + 1 : start + context_length + 1] for start in starts])
    x = torch.as_tensor(contexts, dtype=torch.long, device=device)
    y = torch.as_tensor(targets, dtype=torch.long, device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)["logits"]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def _validation_metrics(
    model: Any,
    validation_ids: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    torch: Any,
    device: str,
) -> Dict[str, float]:
    starts = np.arange(0, len(validation_ids) - context_length, dtype=np.int64)
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.inference_mode():
        for begin in range(0, len(starts), batch_size):
            batch_starts = starts[begin : begin + batch_size]
            contexts = np.stack(
                [validation_ids[start : start + context_length] for start in batch_starts]
            )
            labels = validation_ids[batch_starts + context_length]
            x = torch.as_tensor(contexts, dtype=torch.long, device=device)
            y = torch.as_tensor(labels, dtype=torch.long, device=device)
            logits = model(x)["logits"][:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.cpu())
            correct += int((logits.argmax(dim=-1) == y).sum().cpu())
            seen += len(labels)
    model.train(previous_mode)
    cross_entropy = total_loss / seen
    return {
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(min(cross_entropy, 80.0)),
        "top1_accuracy": correct / seen,
        "validation_windows": seen,
    }


def _representation_benchmark(
    dataset: BenchmarkDataset,
    extractor: TorchExtractor,
    scoring_config: OverlapScoringConfig,
    *,
    batch_size: int,
) -> Any:
    return Benchmark(
        dataset,
        [extractor],
        scoring_config=scoring_config,
        embedding_config=EmbeddingConfig(batch_size=batch_size),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()


def _extract_output_in_batches(
    extractor: TorchExtractor,
    values: np.ndarray,
    *,
    output_name: str,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("Extraction batch_size must be >= 1.")
    chunks = []
    for start in range(0, len(values), batch_size):
        outputs = {
            item.name: item.embeddings
            for item in extractor.transform_many(values[start : start + batch_size])
        }
        if output_name not in outputs:
            raise ValueError(f"Extractor did not return requested output {output_name!r}.")
        chunks.append(outputs[output_name])
    if not chunks:
        raise ValueError("At least one row is required for batched extraction.")
    return np.concatenate(chunks, axis=0)


def _history_rows(
    result: Any, *, step: int, language_metrics: Dict[str, float]
) -> list[Dict[str, Any]]:
    rows = []
    for item in result.extractor_results:
        overlap = item.overlap
        output_name = item.embedding_metadata.get("output_name")
        rows.append(
            {
                "step": step,
                "output_name": output_name,
                "overlap_macro": overlap.macro_score,
                "overlap_weighted": overlap.weighted_score,
                "k_per_class": overlap.diagnostics.get("k_per_class", {}),
                "warnings": list(item.warnings),
                **language_metrics,
            }
        )
    return rows


def _generate(
    model: Any,
    prompt: str,
    char_to_id: Dict[str, int],
    vocabulary: Sequence[str],
    *,
    torch: Any,
    device: str,
    seed: int,
    generated_characters: int = 400,
    temperature: float = 0.8,
    top_k: int = 20,
) -> str:
    missing = sorted(set(prompt) - set(char_to_id))
    if missing:
        raise ValueError(f"Generation prompt contains unknown characters: {missing!r}.")
    generator_device = device if device in {"cuda", "cpu"} else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    ids = [char_to_id[character] for character in prompt]
    previous_mode = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(generated_characters):
            context = ids[-model.context_length :]
            values = torch.as_tensor([context], dtype=torch.long, device=device)
            logits = model(values)["logits"][0, -1] / temperature
            count = min(top_k, logits.numel())
            top_values, top_indices = torch.topk(logits, count)
            probabilities = torch.softmax(top_values, dim=-1)
            if generator_device == device:
                relative = torch.multinomial(probabilities, 1, generator=generator)
            else:
                relative = torch.multinomial(probabilities.cpu(), 1, generator=generator).to(device)
            ids.append(int(top_indices[relative].item()))
    model.train(previous_mode)
    return "".join(vocabulary[index] for index in ids)


def _compression_benchmark(
    embeddings: np.ndarray,
    labels: np.ndarray,
    scoring_config: OverlapScoringConfig,
    *,
    source_output: str = "block_4_final",
) -> Any:
    dataset = BenchmarkDataset.from_embeddings(
        embeddings,
        labels,
        identity=DatasetIdentity.from_content(),
        metadata={
            "example": "tiny_shakespeare_transformer_visual_suite_compression",
            "source_output": source_output,
        },
    )
    configurations = [EmbeddingCompressionConfig()]
    configurations.extend(
        EmbeddingCompressionConfig(
            enabled=True,
            method="pca",
            n_components=dimension,
            random_state=42,
        )
        for dimension in (2, 4, 8, 16, 32, 64)
    )
    configurations.extend(
        (
            EmbeddingCompressionConfig(enabled=True, method="quantize", precision="float16"),
            EmbeddingCompressionConfig(enabled=True, method="quantize", precision="int8"),
        )
    )
    return Benchmark(
        dataset,
        [PrecomputedExtractor("tiny_shakespeare_final_representation")],
        compression_configs=configurations,
        scoring_config=scoring_config,
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()


def _pareto_frontier_indices(
    bytes_per_sample: Sequence[float], scores: Sequence[float]
) -> list[int]:
    frontier = []
    for index, (size, score) in enumerate(zip(bytes_per_sample, scores)):
        dominated = any(
            other_size <= size
            and other_score >= score
            and (other_size < size or other_score > score)
            for other_index, (other_size, other_score) in enumerate(zip(bytes_per_sample, scores))
            if other_index != index
        )
        if not dominated:
            frontier.append(index)
    return frontier


def _plot_monitoring(
    history: Any,
    figure_dir: Path,
    plt: Any,
    *,
    context_length: int = 64,
    profile: TrainingProfile = _PROFILES["fast"],
) -> Tuple[Path, Path]:
    _apply_plot_style(plt)
    figure = plt.figure(figsize=(14.5, 7.4))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 2.5), hspace=0.42, wspace=0.24)
    architecture_axis = figure.add_subplot(grid[:, 0])
    overlap_axis = figure.add_subplot(grid[0, 1])
    loss_axis = figure.add_subplot(grid[1, 1])
    _draw_architecture(
        architecture_axis,
        plt,
        context_length=context_length,
        profile=profile,
    )
    output_order = list(dict.fromkeys(history["output_name"].tolist()))
    for index, name in enumerate(output_order):
        rows = history.loc[history["output_name"] == name].sort_values("step")
        overlap_axis.plot(
            rows["step"],
            rows["overlap_macro"],
            marker="o",
            linewidth=2.2,
            color=_output_color(name, index),
            label=_output_label(name),
        )
    metric_rows = history.drop_duplicates("step").sort_values("step")
    loss_axis.plot(
        metric_rows["step"],
        metric_rows["cross_entropy"],
        marker="o",
        linewidth=2.2,
        color="#334155",
    )
    overlap_axis.set_title("Exact next-character representation geometry", loc="left")
    overlap_axis.set_ylabel("OverlapIndex macro score")
    overlap_axis.legend(frameon=False, ncol=2, fontsize=9)
    loss_axis.set_title("Naturally distributed validation objective", loc="left")
    loss_axis.set_xlabel("Optimizer step (0 = initialization)")
    loss_axis.set_ylabel("Cross-entropy")
    for axis in (overlap_axis, loss_axis):
        axis.grid(axis="y", color="#CBD5E1", alpha=0.7)
    figure.suptitle(
        "Tiny Shakespeare: hidden geometry and language-model learning",
        x=0.055,
        ha="left",
        fontsize=18,
        fontweight="semibold",
    )
    figure.text(
        0.055,
        0.035,
        "Fixed held-out probe • exact next character • automatic per-class k • "
        "Separatix and stability disabled",
        color="#475569",
    )
    figure.subplots_adjust(left=0.055, right=0.97, top=0.88, bottom=0.12)
    return _save_figure(figure, figure_dir, "tiny-shakespeare-representation-monitoring", plt)


def _output_label(name: str) -> str:
    if name == "token_position":
        return "Token + position"
    layer = int(name.split("_")[1])
    return (
        f"Final normalized block {layer}"
        if name.endswith("_final")
        else f"Transformer block {layer}"
    )


def _output_color(name: str, index: int = 0) -> str:
    palette = ("#2563EB", "#7C3AED", "#DB2777", "#EA580C")
    return _OUTPUT_COLORS.get(name, palette[index % len(palette)])


def _draw_architecture(
    axis: Any,
    plt: Any,
    *,
    context_length: int,
    profile: TrainingProfile = _PROFILES["fast"],
) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.set_title(f"{profile.n_layers}-block causal GPT ({profile.name})", loc="left")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    blocks = (
        (f"{context_length}-character context", "learned token + position", "#E2E8F0"),
        (
            "Transformer block 1",
            f"{profile.n_heads} heads • width {profile.width}",
            "#7C3AED",
        ),
        (
            "Transformer block 2",
            f"pre-LN • MLP {profile.mlp_width}",
            "#DB2777",
        ),
        (
            f"Transformer blocks 3–{profile.n_layers}",
            f"causal • dropout {profile.dropout:g}",
            "#EA580C",
        ),
        ("Tied LM head", "next-character logits", "#E2E8F0"),
    )
    for index, (title, detail, color) in enumerate(blocks):
        y = 0.88 - index * 0.19
        dark = color != "#E2E8F0"
        patch = FancyBboxPatch(
            (0.08, y - 0.055),
            0.80,
            0.11,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=color,
            edgecolor="none",
        )
        axis.add_patch(patch)
        axis.text(0.13, y + 0.015, title, color="white" if dark else "#0F172A", weight="bold")
        axis.text(0.13, y - 0.025, detail, color="white" if dark else "#475569", fontsize=9)
        if index < len(blocks) - 1:
            axis.annotate(
                "",
                xy=(0.48, y - 0.12),
                xytext=(0.48, y - 0.065),
                arrowprops={"arrowstyle": "->", "color": "#94A3B8"},
            )


def _plot_compression(result: Any, n_samples: int, figure_dir: Path, plt: Any) -> Tuple[Path, Path]:
    _apply_plot_style(plt)
    rows = []
    for item in result.extractor_results:
        method = item.compression_metadata.get("method", "none")
        precision = item.compression_metadata.get("precision")
        embedding_metadata = getattr(item, "embedding_metadata", {})
        original_dimension = int(embedding_metadata.get("embedding_dim", 128))
        dimension = int(item.compression_metadata.get("compressed_dim", original_dimension))
        if method == "none":
            label = f"float32 {original_dimension}d"
            bytes_per_sample = original_dimension * 4
        elif method == "pca":
            label, bytes_per_sample = f"PCA {dimension}d", dimension * 4
        elif precision == "float16":
            label = f"float16 {original_dimension}d"
            bytes_per_sample = original_dimension * 2
        else:
            label = f"int8 {original_dimension}d"
            bytes_per_sample = original_dimension
        rows.append((label, bytes_per_sample, item.overlap.macro_score))
    sizes = [row[1] for row in rows]
    scores = [row[2] for row in rows]
    frontier = set(_pareto_frontier_indices(sizes, scores))
    figure, axis = plt.subplots(figsize=(12.5, 6.6))
    for index, (label, size, score) in enumerate(rows):
        axis.scatter(
            size,
            score,
            s=115 if index in frontier else 70,
            color="#EA580C" if index in frontier else "#64748B",
            zorder=3,
        )
        label_offsets = {"PCA 32d": (5, 10), "PCA 64d": (5, 10)}
        if label.startswith(("int8", "float16")):
            label_offsets[label] = (5, -17)
        axis.annotate(
            label,
            (size, score),
            xytext=label_offsets.get(label, (5, 7)),
            textcoords="offset points",
            fontsize=9,
        )
    frontier_rows = sorted((rows[index] for index in frontier), key=lambda row: row[1])
    axis.plot(
        [row[1] for row in frontier_rows],
        [row[2] for row in frontier_rows],
        color="#EA580C",
        linestyle="--",
        alpha=0.7,
    )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Stored bytes per probe representation")
    axis.set_ylabel("OverlapIndex macro score")
    axis.set_title(
        "Compression frontier for the trained final representation",
        loc="left",
        fontsize=17,
        weight="semibold",
    )
    axis.grid(color="#CBD5E1", alpha=0.65)
    figure.text(
        0.08,
        0.035,
        f"{n_samples:,} fixed validation contexts • shared labels and automatic k",
        color="#475569",
    )
    figure.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.14)
    return _save_figure(figure, figure_dir, "tiny-shakespeare-compression-frontier", plt)


def _character_sort_key(character: str) -> Tuple[int, str]:
    if character.isspace():
        category = 0
    elif character in string.punctuation:
        category = 1
    elif character.isdigit():
        category = 2
    elif character.isupper():
        category = 3
    elif character.islower():
        category = 4
    else:
        category = 5
    return category, character


def _display_character(character: str) -> str:
    return {"\n": "\\n", "\t": "\\t", " ": "space"}.get(character, character)


def _final_per_class(result: Any) -> Dict[str, float]:
    for item in result.extractor_results:
        output_name = item.embedding_metadata.get("output_name", "")
        if output_name.startswith("block_") and output_name.endswith("_final"):
            return {str(key): float(value) for key, value in item.overlap.per_class_scores.items()}
    raise ValueError("Final block output is missing from the benchmark result.")


def _plot_token_heatmap(
    initial: Any, trained: Any, probe: ProbeData, figure_dir: Path, plt: Any
) -> Tuple[Path, Path]:
    _apply_plot_style(plt)
    before = _final_per_class(initial)
    after = _final_per_class(trained)
    characters = sorted(set(before) & set(after), key=_character_sort_key)
    values = np.asarray([[before[c] for c in characters], [after[c] for c in characters]])
    width = max(14.0, len(characters) * 0.34)
    figure, axis = plt.subplots(figsize=(width, 5.8))
    image = axis.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="magma")
    labels = [f"{_display_character(c)}\n{probe.class_support[c]}" for c in characters]
    axis.set_xticks(range(len(characters)), labels, rotation=90, fontsize=8)
    axis.set_yticks((0, 1), ("Initialization", "Trained"))
    axis.set_title(
        "Every scored next-character token", loc="left", fontsize=17, weight="semibold", pad=14
    )
    axis.set_xlabel(
        "Character and natural validation support (hatched = diagnostic-only macro exclusion)"
    )
    low_support = set(probe.low_support_classes)
    for index, character in enumerate(characters):
        if character in low_support:
            axis.add_patch(
                plt.Rectangle(
                    (index - 0.5, -0.5),
                    1,
                    2,
                    fill=False,
                    hatch="///",
                    edgecolor="white",
                    linewidth=0.7,
                )
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Per-token OverlapIndex")
    figure.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.30)
    return _save_figure(figure, figure_dir, "tiny-shakespeare-next-token-heatmap", plt)


def _apply_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_figure(figure: Any, figure_dir: Path, stem: str, plt: Any) -> Tuple[Path, Path]:
    png = figure_dir / f"{stem}.png"
    svg = figure_dir / f"{stem}.svg"
    figure.savefig(png, dpi=180, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)
    return png, svg


if __name__ == "__main__":
    main()
