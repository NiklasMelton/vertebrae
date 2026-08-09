"""Animate a Fashion-MNIST classifier learning in a true 2D bottleneck.

Every point is one fixed held-out validation example. The two plotted coordinates
are the exact representation consumed by the linear classification head, not a
post-hoc projection. Frames are aligned with rigid Procrustes transforms for visual
continuity; the head is transformed with the points so its decision regions remain
exactly equivalent. OverlapIndex is computed on the unaligned 2D embeddings.

Install the optional dependencies and run from the repository root:

    poetry install -E visuals
    poetry run python examples/fashion_mnist_embedding_animation.py

The generated GIF loops forever. Fashion-MNIST is downloaded through torchvision
on the first run and reused from the local data directory afterward.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
from _common import ensure_output_dir

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    DatasetIdentity,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.config import CacheConfig
from vertebrae.extractors import PrecomputedExtractor

_FASHION_CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
_CLASS_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)
_NORMALIZATION_MEAN = 0.2860
_NORMALIZATION_STD = 0.3530


@dataclass(frozen=True)
class EmbeddingSnapshot:
    """One evaluation of the model on the fixed validation probe."""

    epoch: int
    global_step: int
    batch_in_epoch: int
    training_loss: float
    validation_accuracy: float
    overlap_index: float
    embeddings: np.ndarray
    classifier_weight: np.ndarray
    classifier_bias: np.ndarray


@dataclass(frozen=True)
class DisplaySnapshot:
    """A snapshot transformed into a temporally stable display coordinate system."""

    source: EmbeddingSnapshot
    embeddings: np.ndarray
    classifier_weight: np.ndarray
    classifier_bias: np.ndarray
    interpolation: Optional[InterpolationInfo] = None


@dataclass(frozen=True)
class InterpolationInfo:
    """Provenance for a synthetic display frame between real model checkpoints."""

    start_step: int
    end_step: int
    fraction: float


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.002,
        help="AdamW learning rate tuned for the compact animation run.",
    )
    parser.add_argument(
        "--snapshot-every-batches",
        type=int,
        default=8,
        help="Capture the fixed validation probe after this many optimizer steps.",
    )
    parser.add_argument(
        "--interpolation-frames",
        type=int,
        default=2,
        help=(
            "Display-only linear tween frames inserted between model checkpoints; "
            "set 0 to disable the extra scoring and rendering work."
        ),
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--final-hold-seconds",
        type=float,
        default=1.5,
        help="Pause on the trained representation before the GIF loops.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("examples/data"),
        help="Directory used by torchvision for the Fashion-MNIST download/cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "GIF destination; defaults to "
            "VERTABRAE_EXAMPLE_OUTPUT_DIR/fashion_mnist_embedding_evolution.gif."
        ),
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Show raw model coordinates instead of display-only rigid alignment.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require Fashion-MNIST to already exist under --data-dir.",
    )
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        from PIL import Image
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:
        print(exc)
        print("Install the animation dependencies with: poetry install -E visuals")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_x, train_y, validation_x, validation_y = _load_fashion_mnist(
        FashionMNIST,
        data_dir=args.data_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        seed=args.seed,
        download=not args.no_download,
    )
    model = _build_model(torch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    scoring_config = _scoring_config(args.seed)

    snapshots = [
        _capture_snapshot(
            model,
            validation_x,
            validation_y,
            training_loss=_mean_loss(
                model,
                train_x,
                train_y,
                loss_fn,
                batch_size=args.train_batch_size,
                torch=torch,
            ),
            epoch=0,
            global_step=0,
            batch_in_epoch=0,
            batch_size=args.embedding_batch_size,
            scoring_config=scoring_config,
            torch=torch,
        )
    ]

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for batch_number, total_batches, training_loss in _train_epoch_batches(
            model,
            optimizer,
            loss_fn,
            train_x,
            train_y,
            batch_size=args.train_batch_size,
            seed=args.seed + epoch,
            torch=torch,
        ):
            global_step += 1
            if (
                global_step % args.snapshot_every_batches != 0
                and batch_number != total_batches
            ):
                continue
            snapshots.append(
                _capture_snapshot(
                    model,
                    validation_x,
                    validation_y,
                    training_loss=training_loss,
                    epoch=epoch,
                    global_step=global_step,
                    batch_in_epoch=batch_number,
                    batch_size=args.embedding_batch_size,
                    scoring_config=scoring_config,
                    torch=torch,
                )
            )
            print(
                f"step={global_step:03d} epoch={epoch} "
                f"overlap={snapshots[-1].overlap_index:.3f} "
                f"accuracy={snapshots[-1].validation_accuracy:.3f}"
            )

    display_snapshots = _display_snapshots(snapshots, align=not args.no_align)
    display_frames = _interpolate_display_snapshots(
        display_snapshots,
        validation_y,
        frames_between=args.interpolation_frames,
        scoring_config=scoring_config,
    )
    output_dir = ensure_output_dir()
    output_path = args.output or output_dir / "fashion_mnist_embedding_evolution.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = output_path.with_suffix(".csv")
    _write_history(snapshots, history_path)

    frame_duration_ms = max(20, int(round(1_000.0 / args.fps)))
    final_hold_frames = max(0, int(round(args.final_hold_seconds * args.fps)))
    axis_limits = _shared_axis_limits(display_frames)
    frames = (
        _render_frame(
            snapshot,
            validation_y,
            axis_limits=axis_limits,
            aligned=not args.no_align,
            interpolation_enabled=args.interpolation_frames > 0,
            plt=plt,
            image_module=Image,
        )
        for snapshot in display_frames
    )
    _save_looping_gif(
        frames,
        output_path,
        frame_duration_ms=frame_duration_ms,
        final_hold_frames=final_hold_frames,
    )

    print(f"Wrote looping GIF {output_path}")
    print(f"Wrote snapshot metrics {history_path}")


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_options = (
        "epochs",
        "train_size",
        "validation_size",
        "train_batch_size",
        "embedding_batch_size",
        "snapshot_every_batches",
        "fps",
        "learning_rate",
    )
    for name in positive_options:
        value = getattr(args, name)
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if not math.isfinite(args.final_hold_seconds) or args.final_hold_seconds < 0:
        parser.error("--final-hold-seconds must be >= 0")
    if not 0.1 <= args.fps <= 50.0:
        parser.error("--fps must be between 0.1 and 50")
    if args.final_hold_seconds > 60.0:
        parser.error("--final-hold-seconds must be <= 60")
    if args.train_size < len(_FASHION_CLASS_NAMES):
        parser.error(f"--train-size must be >= {len(_FASHION_CLASS_NAMES)}")
    minimum_validation_size = 2 * len(_FASHION_CLASS_NAMES)
    if args.validation_size < minimum_validation_size:
        parser.error(f"--validation-size must be >= {minimum_validation_size}")
    if not 0 <= args.seed <= np.iinfo(np.uint32).max:
        parser.error(f"--seed must be between 0 and {np.iinfo(np.uint32).max}")
    if args.output is not None and args.output.suffix.lower() != ".gif":
        parser.error("--output must use a .gif suffix")
    if not 0 <= args.interpolation_frames <= 4:
        parser.error("--interpolation-frames must be between 0 and 4")


def _build_model(torch: Any) -> Any:
    class FashionMNIST2DClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
            )
            self.pre_bottleneck = torch.nn.Sequential(
                torch.nn.Linear(64 * 7 * 7, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
            )
            # Deliberately no activation: all of R^2 remains available to the model.
            self.bottleneck = torch.nn.Linear(128, 2)
            self.classifier = torch.nn.Linear(2, len(_FASHION_CLASS_NAMES))

        def forward(self, values: Any) -> dict[str, Any]:
            images = values.reshape(-1, 1, 28, 28)
            features = self.features(images).flatten(start_dim=1)
            embedding = self.bottleneck(self.pre_bottleneck(features))
            logits = self.classifier(embedding)
            return {"embedding": embedding, "logits": logits}

    return FashionMNIST2DClassifier()


def _scoring_config(seed: int) -> OverlapScoringConfig:
    return OverlapScoringConfig(
        k="auto",
        min_k=5,
        max_k=20,
        min_samples_per_cluster=5,
        kmeans_kwargs={"random_state": seed, "batch_size": 512, "n_init": 3},
        # The plot shows raw Euclidean geometry; scoring the same geometry keeps the
        # overlay interpretable and makes rigid display alignment score-invariant.
        normalize_embeddings=False,
    )


def _capture_snapshot(
    model: Any,
    values: np.ndarray,
    labels: np.ndarray,
    *,
    training_loss: float,
    epoch: int,
    global_step: int,
    batch_in_epoch: int,
    batch_size: int,
    scoring_config: OverlapScoringConfig,
    torch: Any,
) -> EmbeddingSnapshot:
    embeddings, logits = _model_outputs(
        model,
        values,
        batch_size=batch_size,
        torch=torch,
    )
    overlap_index = _score_embeddings(embeddings, labels, scoring_config)
    classifier_weight = model.classifier.weight.detach().cpu().numpy().copy()
    classifier_bias = model.classifier.bias.detach().cpu().numpy().copy()
    return EmbeddingSnapshot(
        epoch=epoch,
        global_step=global_step,
        batch_in_epoch=batch_in_epoch,
        training_loss=float(training_loss),
        validation_accuracy=float(np.mean(logits.argmax(axis=1) == labels)),
        overlap_index=overlap_index,
        embeddings=embeddings,
        classifier_weight=classifier_weight,
        classifier_bias=classifier_bias,
    )


def _model_outputs(
    model: Any,
    values: np.ndarray,
    *,
    batch_size: int,
    torch: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    previous_mode = model.training
    model.eval()
    embeddings = []
    logits = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            output = model(torch.as_tensor(values[start : start + batch_size], dtype=torch.float32))
            embeddings.append(output["embedding"].cpu().numpy())
            logits.append(output["logits"].cpu().numpy())
    model.train(previous_mode)
    return np.concatenate(embeddings), np.concatenate(logits)


def _score_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    scoring_config: OverlapScoringConfig,
) -> float:
    semantic_labels = np.asarray([_FASHION_CLASS_NAMES[int(label)] for label in labels])
    dataset = BenchmarkDataset.from_embeddings(
        embeddings,
        semantic_labels,
        identity=DatasetIdentity.ephemeral(),
        metadata={
            "example": "fashion_mnist_embedding_animation",
            "representation": "model_2d_bottleneck",
        },
    )
    result = Benchmark(
        dataset,
        [PrecomputedExtractor("fashion_mnist_2d_bottleneck")],
        scoring_config=scoring_config,
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()
    overlap = result.extractor_results[0].overlap
    if overlap is None:
        raise RuntimeError("The default overlap metric did not produce an OverlapIndex result.")
    return float(overlap.macro_score)


def _display_snapshots(
    snapshots: Sequence[EmbeddingSnapshot],
    *,
    align: bool,
) -> list[DisplaySnapshot]:
    if not snapshots:
        raise ValueError("At least one embedding snapshot is required.")
    if not align:
        return [
            DisplaySnapshot(
                source=snapshot,
                embeddings=snapshot.embeddings.copy(),
                classifier_weight=snapshot.classifier_weight.copy(),
                classifier_bias=snapshot.classifier_bias.copy(),
            )
            for snapshot in snapshots
        ]

    displayed = []
    reference: Optional[np.ndarray] = None
    for snapshot in snapshots:
        center = snapshot.embeddings.mean(axis=0)
        centered = snapshot.embeddings - center
        rotation = np.eye(2)
        if reference is not None:
            left, _, right_t = np.linalg.svd(centered.T @ reference, full_matrices=False)
            rotation = left @ right_t
        aligned_embeddings = centered @ rotation

        # z_display = (z - center) R, hence z = z_display R^T + center.
        # Transforming W and b this way preserves every original classifier logit.
        aligned_weight = snapshot.classifier_weight @ rotation
        aligned_bias = snapshot.classifier_bias + snapshot.classifier_weight @ center
        displayed.append(
            DisplaySnapshot(
                source=snapshot,
                embeddings=aligned_embeddings,
                classifier_weight=aligned_weight,
                classifier_bias=aligned_bias,
            )
        )
        reference = aligned_embeddings
    return displayed


def _interpolate_display_snapshots(
    snapshots: Sequence[DisplaySnapshot],
    labels: np.ndarray,
    *,
    frames_between: int,
    scoring_config: OverlapScoringConfig,
) -> list[DisplaySnapshot]:
    """Insert explicitly synthetic linear display frames between checkpoints."""

    if frames_between < 0:
        raise ValueError("frames_between must be >= 0.")
    if not snapshots or frames_between == 0:
        return list(snapshots)

    frames = []
    labels = np.asarray(labels)
    for left, right in zip(snapshots[:-1], snapshots[1:]):
        frames.append(left)
        for offset in range(1, frames_between + 1):
            fraction = offset / (frames_between + 1)
            embeddings = _linear_interpolate(left.embeddings, right.embeddings, fraction)
            classifier_weight = _linear_interpolate(
                left.classifier_weight,
                right.classifier_weight,
                fraction,
            )
            classifier_bias = _linear_interpolate(
                left.classifier_bias,
                right.classifier_bias,
                fraction,
            )
            logits = embeddings @ classifier_weight.T + classifier_bias
            source = EmbeddingSnapshot(
                epoch=-1,
                global_step=-1,
                batch_in_epoch=-1,
                training_loss=float("nan"),
                validation_accuracy=float(np.mean(logits.argmax(axis=1) == labels)),
                overlap_index=_score_embeddings(embeddings, labels, scoring_config),
                embeddings=embeddings,
                classifier_weight=classifier_weight,
                classifier_bias=classifier_bias,
            )
            frames.append(
                DisplaySnapshot(
                    source=source,
                    embeddings=embeddings,
                    classifier_weight=classifier_weight,
                    classifier_bias=classifier_bias,
                    interpolation=InterpolationInfo(
                        start_step=left.source.global_step,
                        end_step=right.source.global_step,
                        fraction=fraction,
                    ),
                )
            )
    frames.append(snapshots[-1])
    return frames


def _linear_interpolate(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    return (1.0 - fraction) * left + fraction * right


def _shared_axis_limits(snapshots: Sequence[DisplaySnapshot]) -> Tuple[float, float, float, float]:
    values = np.concatenate([snapshot.embeddings for snapshot in snapshots], axis=0)
    x_min, y_min = values.min(axis=0)
    x_max, y_max = values.max(axis=0)
    x_pad = max(0.25, 0.06 * max(x_max - x_min, 1.0))
    y_pad = max(0.25, 0.06 * max(y_max - y_min, 1.0))
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _frame_overlay_text(snapshot: DisplaySnapshot) -> str:
    if snapshot.interpolation is None:
        step = float(snapshot.source.global_step)
    else:
        interpolation = snapshot.interpolation
        step = (1.0 - interpolation.fraction) * interpolation.start_step + (
            interpolation.fraction * interpolation.end_step
        )
    return (
        f"STEP {step:06.1f}   "
        f"OVERLAPINDEX {snapshot.source.overlap_index:.3f}   "
        f"ACCURACY {snapshot.source.validation_accuracy:6.1%}"
    )


def _render_frame(
    snapshot: DisplaySnapshot,
    labels: np.ndarray,
    *,
    axis_limits: Tuple[float, float, float, float],
    aligned: bool,
    interpolation_enabled: bool,
    plt: Any,
    image_module: Any,
) -> Any:
    from matplotlib.colors import ListedColormap, to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    x_min, x_max, y_min, y_max = axis_limits
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#64748B",
            "ytick.color": "#64748B",
            "text.color": "#0F172A",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        figure, axis = plt.subplots(figsize=(8.4, 8.0), dpi=110)
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, 240),
            np.linspace(y_min, y_max, 240),
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        predicted = (
            grid @ snapshot.classifier_weight.T + snapshot.classifier_bias
        ).argmax(axis=1)
        background_colors = [(*to_rgb(color), 0.12) for color in _CLASS_COLORS]
        axis.contourf(
            grid_x,
            grid_y,
            predicted.reshape(grid_x.shape),
            levels=np.arange(len(_FASHION_CLASS_NAMES) + 1) - 0.5,
            cmap=ListedColormap(background_colors),
            antialiased=False,
        )

        for class_id, (class_name, color) in enumerate(
            zip(_FASHION_CLASS_NAMES, _CLASS_COLORS)
        ):
            mask = labels == class_id
            axis.scatter(
                snapshot.embeddings[mask, 0],
                snapshot.embeddings[mask, 1],
                s=18,
                c=color,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.25,
                rasterized=True,
                label=class_name,
            )

        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("2D bottleneck — dimension 1")
        axis.set_ylabel("2D bottleneck — dimension 2")
        axis.grid(color="#E2E8F0", linewidth=0.6, alpha=0.55)
        axis.set_title(
            "Fashion-MNIST learning in a 2D bottleneck",
            loc="left",
            fontsize=15,
            fontweight="semibold",
            pad=14,
        )
        panel = FancyBboxPatch(
            (0.012, 0.92),
            0.75,
            0.065,
            boxstyle="round,pad=0.01",
            transform=axis.transAxes,
            facecolor="white",
            edgecolor="#CBD5E1",
            linewidth=1.0,
            alpha=0.94,
            zorder=5,
        )
        axis.add_patch(panel)
        axis.text(
            0.027,
            0.952,
            _frame_overlay_text(snapshot),
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=10.0,
            family="DejaVu Sans Mono",
            zorder=6,
        )
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=7,
                label=class_name,
            )
            for class_name, color in zip(_FASHION_CLASS_NAMES, _CLASS_COLORS)
        ]
        axis.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.11),
            ncol=5,
            frameon=False,
            fontsize=8.5,
            handletextpad=0.25,
            columnspacing=0.9,
        )
        alignment_note = "rigid display alignment" if aligned else "raw-coordinate display"
        if interpolation_enabled:
            footer_text = (
                "Fixed held-out points  •  checkpoint frames/metrics are genuine  •  "
                "tween frames/metrics are display-only\n"
                f"Metrics match plotted geometry/head  •  {alignment_note}  •  "
                "background = displayed prediction"
            )
        else:
            footer_text = (
                "Fixed held-out points  •  checkpoint frames/metrics are genuine  •  "
                "checkpoint-only rendering\n"
                f"Metrics match plotted geometry/head  •  {alignment_note}  •  "
                "background = displayed prediction"
            )
        figure.text(
            0.5,
            0.018,
            footer_text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#64748B",
        )
        figure.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.20)
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba()).copy()
        plt.close(figure)
    return image_module.fromarray(pixels).convert(
        "P",
        palette=image_module.Palette.ADAPTIVE,
        colors=256,
    )


def _save_looping_gif(
    frames: Iterable[Any],
    output_path: Path,
    *,
    frame_duration_ms: int,
    final_hold_frames: int,
) -> None:
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("At least one animation frame is required.") from exc

    def remaining_frames() -> Iterator[Any]:
        last = first
        for frame in iterator:
            last = frame
            yield frame
        for _ in range(final_hold_frames):
            yield last.copy()

    first.save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=remaining_frames(),
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def _write_history(snapshots: Sequence[EmbeddingSnapshot], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "global_step",
                "batch_in_epoch",
                "training_loss",
                "validation_accuracy",
                "overlap_index",
            ),
        )
        writer.writeheader()
        for snapshot in snapshots:
            writer.writerow(
                {
                    "epoch": snapshot.epoch,
                    "global_step": snapshot.global_step,
                    "batch_in_epoch": snapshot.batch_in_epoch,
                    "training_loss": snapshot.training_loss,
                    "validation_accuracy": snapshot.validation_accuracy,
                    "overlap_index": snapshot.overlap_index,
                }
            )


def _load_fashion_mnist(
    dataset_class: Any,
    *,
    data_dir: Path,
    train_size: int,
    validation_size: int,
    seed: int,
    download: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    full_dataset = dataset_class(root=str(data_dir), train=True, download=download)
    all_values = np.asarray(full_dataset.data, dtype=np.float32)
    all_targets = np.asarray(full_dataset.targets, dtype=np.int64)
    if train_size + validation_size > len(all_targets):
        raise ValueError(
            "Requested Fashion-MNIST training and validation subsets exceed the available "
            f"training split; got train_size={train_size}, "
            f"validation_size={validation_size}."
        )
    validation_indices = _stratified_indices(all_targets, validation_size, seed + 1)
    available = np.ones(len(all_targets), dtype=bool)
    available[validation_indices] = False
    remaining_indices = np.flatnonzero(available)
    relative_train_indices = _stratified_indices(
        all_targets[remaining_indices],
        train_size,
        seed,
    )
    train_indices = remaining_indices[relative_train_indices]
    return (
        _normalize_fashion_mnist(all_values[train_indices]),
        all_targets[train_indices],
        _normalize_fashion_mnist(all_values[validation_indices]),
        all_targets[validation_indices],
    )


def _stratified_indices(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    base, remainder = divmod(size, len(classes))
    rng = np.random.default_rng(seed)
    selected = []
    for class_offset, label in enumerate(classes):
        class_indices = np.flatnonzero(labels == label)
        count = base + (1 if class_offset < remainder else 0)
        if count > len(class_indices):
            raise ValueError(f"Class {label!r} does not contain {count} requested samples.")
        selected.extend(rng.choice(class_indices, size=count, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)[rng.permutation(size)]


def _normalize_fashion_mnist(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(len(values), -1) / 255.0
    return ((flattened - _NORMALIZATION_MEAN) / _NORMALIZATION_STD).astype(
        np.float32,
        copy=False,
    )


def _train_epoch_batches(
    model: Any,
    optimizer: Any,
    loss_fn: Any,
    values: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    torch: Any,
) -> Iterable[Tuple[int, int, float]]:
    model.train()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    total_loss = 0.0
    seen = 0
    total_batches = int(np.ceil(len(order) / batch_size))
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch_x = torch.as_tensor(values[indices], dtype=torch.float32)
        batch_y = torch.as_tensor(labels[indices], dtype=torch.long)
        optimizer.zero_grad()
        logits = model(batch_x)["logits"]
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(indices)
        seen += len(indices)
        yield start // batch_size + 1, total_batches, total_loss / seen


def _mean_loss(
    model: Any,
    values: np.ndarray,
    labels: np.ndarray,
    loss_fn: Any,
    *,
    batch_size: int,
    torch: Any,
) -> float:
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch_x = torch.as_tensor(values[start : start + batch_size], dtype=torch.float32)
            batch_y = torch.as_tensor(labels[start : start + batch_size], dtype=torch.long)
            logits = model(batch_x)["logits"]
            total_loss += float(loss_fn(logits, batch_y)) * len(batch_y)
    model.train(previous_mode)
    return total_loss / len(labels)


if __name__ == "__main__":
    main()
