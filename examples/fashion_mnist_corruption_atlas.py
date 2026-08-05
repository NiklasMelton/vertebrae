"""Build a deployment-shift atlas for a trained Fashion-MNIST CNN.

The example evaluates the same official-test probe under blur, Gaussian noise,
occlusion, contrast reduction, and rotation. It combines layer-wise OverlapIndex
retention with accuracy, cross-entropy, per-class representation retention, and
the first class-pair confusions that emerge as severity rises.

The checkpoint written by ``fashion_mnist_visual_suite.py`` is reused when it is
available. Otherwise this script trains that compact CNN once and saves the same
checkpoint for subsequent runs.

Install the optional dependencies and run from the repository root:

    poetry install -E visuals
    poetry run python examples/fashion_mnist_corruption_atlas.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from _common import ensure_output_dir
from fashion_mnist_visual_suite import (
    _FASHION_CLASS_NAMES,
    _LAYER_COLORS,
    _LAYER_LABELS,
    _LAYER_MARKERS,
    _LAYER_ORDER,
    _NORMALIZATION_MEAN,
    _NORMALIZATION_STD,
    _build_model,
    _multi_output_extractor,
    _normalize_fashion_mnist,
    _stratified_indices,
    _train_epoch_batches,
)

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    DatasetIdentity,
    EmbeddingConfig,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.config import CacheConfig

_CORRUPTION_ORDER = ("blur", "noise", "occlusion", "contrast", "rotation")
_CORRUPTION_LABELS = {
    "blur": "Blur",
    "noise": "Noise",
    "occlusion": "Occlusion",
    "contrast": "Low contrast",
    "rotation": "Rotation",
}
_CORRUPTION_SHORT = {
    "blur": "B",
    "noise": "N",
    "occlusion": "O",
    "contrast": "C",
    "rotation": "R",
}
_CORRUPTION_COLORS = {
    "blur": "#2563EB",
    "noise": "#D97706",
    "occlusion": "#DC2626",
    "contrast": "#059669",
    "rotation": "#7C3AED",
}
_SEVERITY_LABELS = ("Clean", "Mild", "Medium", "Strong", "Severe")
_SEVERITY_VALUES: Mapping[str, Tuple[float, ...]] = {
    "blur": (0.0, 0.6, 1.0, 1.5, 2.2),
    "noise": (0.0, 0.08, 0.16, 0.24, 0.32),
    "occlusion": (0.0, 6.0, 10.0, 14.0, 18.0),
    "contrast": (1.0, 0.8, 0.6, 0.4, 0.2),
    "rotation": (0.0, 7.5, 15.0, 22.5, 30.0),
}
_SEVERITY_UNITS = {
    "blur": "sigma",
    "noise": "pixel std",
    "occlusion": "square px",
    "contrast": "factor",
    "rotation": "degrees",
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--evaluation-size", type=int, default=2_000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("examples/data"),
        help="Directory used by torchvision for the Fashion-MNIST download/cache.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Model checkpoint to reuse; defaults to "
            "VERTABRAE_EXAMPLE_OUTPUT_DIR/fashion_mnist_cnn.pt."
        ),
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore an existing checkpoint and retrain the compact CNN.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Figure destination; defaults to VERTABRAE_EXAMPLE_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--confusion-threshold",
        type=float,
        default=0.05,
        help="Symmetric class-confusion increase used to define pairwise onset.",
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
        import pandas as pd
        import torch
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:
        print(exc)
        print("Install the visual example dependencies with: poetry install -E visuals")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    output_dir = ensure_output_dir()
    figure_dir = args.figure_dir or output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or output_dir / "fashion_mnist_cnn.pt"

    model = _build_model(torch)
    checkpoint_status = _prepare_model(
        model,
        FashionMNIST,
        checkpoint_path=checkpoint_path,
        data_dir=args.data_dir,
        train_size=args.train_size,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        seed=args.seed,
        download=not args.no_download,
        force_retrain=args.force_retrain,
        torch=torch,
    )
    evaluation_x, evaluation_y = _load_evaluation_subset(
        FashionMNIST,
        data_dir=args.data_dir,
        size=args.evaluation_size,
        seed=args.seed,
        download=not args.no_download,
    )
    extractor = _multi_output_extractor(model, torch, args.seed)
    scoring_config = OverlapScoringConfig(
        k=5,
        kmeans_kwargs={"random_state": args.seed, "batch_size": 512, "n_init": 3},
    )

    summary_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    confusion_by_condition: Dict[Tuple[str, int], np.ndarray] = {}

    clean_rows, clean_class_rows, clean_confusion = _evaluate_condition(
        model,
        extractor,
        evaluation_x,
        evaluation_y,
        corruption="clean",
        severity_index=0,
        severity_value=0.0,
        scoring_config=scoring_config,
        batch_size=args.embedding_batch_size,
        torch=torch,
    )
    clean_oi = {row["layer"]: row["overlap_index"] for row in clean_rows}
    clean_class_oi = {
        (row["layer"], row["class_name"]): row["class_overlap_index"] for row in clean_class_rows
    }

    for corruption in _CORRUPTION_ORDER:
        for severity_index, severity_value in enumerate(_SEVERITY_VALUES[corruption]):
            if severity_index == 0:
                rows = [dict(row) for row in clean_rows]
                per_class = [dict(row) for row in clean_class_rows]
                confusion = clean_confusion.copy()
                for row in (*rows, *per_class):
                    row["corruption"] = corruption
                    row["corruption_label"] = _CORRUPTION_LABELS[corruption]
                    row["severity_value"] = severity_value
            else:
                shifted = _corrupt_images(
                    evaluation_x,
                    corruption,
                    severity_index,
                    seed=args.seed,
                    torch=torch,
                )
                rows, per_class, confusion = _evaluate_condition(
                    model,
                    extractor,
                    shifted,
                    evaluation_y,
                    corruption=corruption,
                    severity_index=severity_index,
                    severity_value=severity_value,
                    scoring_config=scoring_config,
                    batch_size=args.embedding_batch_size,
                    torch=torch,
                )
            for row in rows:
                row["oi_retention"] = _safe_retention(row["overlap_index"], clean_oi[row["layer"]])
            for row in per_class:
                row["class_oi_retention"] = _safe_retention(
                    row["class_overlap_index"],
                    clean_class_oi[(row["layer"], row["class_name"])],
                )
                row["class_oi_delta"] = (
                    row["class_overlap_index"] - clean_class_oi[(row["layer"], row["class_name"])]
                )
            summary_rows.extend(rows)
            class_rows.extend(per_class)
            confusion_by_condition[(corruption, severity_index)] = confusion
            print(
                f"Evaluated {_CORRUPTION_LABELS[corruption]:<12} "
                f"{_SEVERITY_LABELS[severity_index].lower():<6} "
                f"(accuracy={rows[0]['accuracy']:.3f})"
            )

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(class_rows)
    pairwise_onset = _pairwise_confusion_onset(
        confusion_by_condition,
        clean_confusion,
        threshold=args.confusion_threshold,
    )
    summary.to_csv(output_dir / "fashion_mnist_corruption_atlas_summary.csv", index=False)
    per_class.to_csv(output_dir / "fashion_mnist_corruption_atlas_per_class.csv", index=False)
    pairwise_onset.to_csv(
        output_dir / "fashion_mnist_corruption_atlas_pairwise_onset.csv", index=False
    )
    protocol = _protocol_payload(args, checkpoint_path, checkpoint_status)
    (output_dir / "fashion_mnist_corruption_atlas_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    figure_paths = _plot_corruption_atlas(
        summary,
        per_class,
        pairwise_onset,
        figure_dir,
        confusion_threshold=args.confusion_threshold,
        plt=plt,
    )
    print(_practitioner_summary(summary))
    for path in figure_paths:
        print(f"Wrote {path}")
    print(f"Metrics written to {output_dir}")


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in (
        "epochs",
        "train_size",
        "evaluation_size",
        "train_batch_size",
        "embedding_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if not 0.0 < args.confusion_threshold <= 1.0:
        parser.error("--confusion-threshold must be in (0, 1]")


def _prepare_model(
    model: Any,
    dataset_class: Any,
    *,
    checkpoint_path: Path,
    data_dir: Path,
    train_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
    download: bool,
    force_retrain: bool,
    torch: Any,
) -> str:
    if checkpoint_path.is_file() and not force_retrain:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict)
        print(f"Reused trained checkpoint {checkpoint_path}")
        return "reused"

    training_dataset = dataset_class(root=str(data_dir), train=True, download=download)
    all_targets = np.asarray(training_dataset.targets, dtype=np.int64)
    indices = _stratified_indices(all_targets, train_size, seed)
    train_x = _normalize_fashion_mnist(np.asarray(training_dataset.data)[indices])
    train_y = all_targets[indices]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        final_loss = float("nan")
        for _, _, batch_loss in _train_epoch_batches(
            model,
            optimizer,
            loss_fn,
            train_x,
            train_y,
            batch_size=batch_size,
            seed=seed + epoch,
            torch=torch,
        ):
            final_loss = batch_loss
        print(f"Trained epoch {epoch}/{epochs} (loss={final_loss:.4f})")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "fashion_mnist_visual_suite.compact_cnn.v1",
            "epochs": epochs,
            "train_size": train_size,
            "seed": seed,
            "normalization_mean": _NORMALIZATION_MEAN,
            "normalization_std": _NORMALIZATION_STD,
        },
        checkpoint_path,
    )
    print(f"Wrote trained checkpoint {checkpoint_path}")
    return "trained"


def _load_evaluation_subset(
    dataset_class: Any,
    *,
    data_dir: Path,
    size: int,
    seed: int,
    download: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = dataset_class(root=str(data_dir), train=False, download=download)
    targets = np.asarray(dataset.targets, dtype=np.int64)
    if size > len(targets):
        raise ValueError(
            f"evaluation_size={size} exceeds the Fashion-MNIST test split ({len(targets)})."
        )
    indices = _stratified_indices(targets, size, seed + 2)
    values = _normalize_fashion_mnist(np.asarray(dataset.data)[indices])
    return values, targets[indices]


def _corrupt_images(
    values: np.ndarray,
    corruption: str,
    severity_index: int,
    *,
    seed: int,
    torch: Any,
) -> np.ndarray:
    if corruption not in _SEVERITY_VALUES:
        raise ValueError(f"Unknown corruption {corruption!r}.")
    if not 0 <= severity_index < len(_SEVERITY_LABELS):
        raise ValueError("severity_index must be between 0 and 4.")
    if severity_index == 0:
        return np.asarray(values, dtype=np.float32).copy()

    raw = (
        torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1, 28, 28) * _NORMALIZATION_STD
        + _NORMALIZATION_MEAN
    ).clamp(0.0, 1.0)
    amount = _SEVERITY_VALUES[corruption][severity_index]
    if corruption == "blur":
        raw = _gaussian_blur(raw, sigma=amount, torch=torch)
    elif corruption == "noise":
        generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 + severity_index)
        noise = torch.randn(raw.shape, generator=generator, dtype=raw.dtype)
        raw = raw + noise * amount
    elif corruption == "occlusion":
        raw = raw.clone()
        side = int(amount)
        rng = np.random.default_rng(seed + 20_000)
        tops = rng.integers(0, 29 - side, size=len(raw))
        lefts = rng.integers(0, 29 - side, size=len(raw))
        for index, (top, left) in enumerate(zip(tops, lefts)):
            raw[index, :, top : top + side, left : left + side] = 0.0
    elif corruption == "contrast":
        centers = raw.mean(dim=(2, 3), keepdim=True)
        raw = centers + amount * (raw - centers)
    elif corruption == "rotation":
        radians = np.deg2rad(amount)
        cosine, sine = float(np.cos(radians)), float(np.sin(radians))
        theta = raw.new_tensor([[cosine, -sine, 0.0], [sine, cosine, 0.0]])
        theta = theta.unsqueeze(0).repeat(len(raw), 1, 1)
        grid = torch.nn.functional.affine_grid(theta, raw.shape, align_corners=False)
        raw = torch.nn.functional.grid_sample(
            raw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    normalized = (raw.clamp(0.0, 1.0) - _NORMALIZATION_MEAN) / _NORMALIZATION_STD
    return normalized.reshape(len(values), -1).cpu().numpy().astype(np.float32, copy=False)


def _gaussian_blur(images: Any, *, sigma: float, torch: Any) -> Any:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=images.dtype)
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel).reshape(1, 1, len(kernel), len(kernel))
    padded = torch.nn.functional.pad(images, (radius, radius, radius, radius), mode="reflect")
    return torch.nn.functional.conv2d(padded, kernel_2d)


def _evaluate_condition(
    model: Any,
    extractor: Any,
    values: np.ndarray,
    labels: np.ndarray,
    *,
    corruption: str,
    severity_index: int,
    severity_value: float,
    scoring_config: OverlapScoringConfig,
    batch_size: int,
    torch: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
    class_labels = np.asarray([_FASHION_CLASS_NAMES[int(label)] for label in labels])
    dataset = BenchmarkDataset.from_arrays(
        values,
        class_labels,
        modality="image",
        metadata={
            "example": "fashion_mnist_corruption_atlas",
            "split": "official_test",
            "corruption": corruption,
            "severity_index": severity_index,
            "severity_value": severity_value,
        },
        identity=DatasetIdentity.from_content(),
    )
    result = Benchmark(
        dataset,
        [extractor],
        scoring_config=scoring_config,
        embedding_config=EmbeddingConfig(batch_size=batch_size),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()
    predictions, accuracy, loss = _behavior_metrics(model, values, labels, batch_size, torch)
    confusion = _normalized_confusion(labels, predictions)
    common = {
        "corruption": corruption,
        "corruption_label": _CORRUPTION_LABELS.get(corruption, "Clean"),
        "severity_index": severity_index,
        "severity_label": _SEVERITY_LABELS[severity_index],
        "severity_value": severity_value,
        "severity_unit": _SEVERITY_UNITS.get(corruption, "none"),
        "accuracy": accuracy,
        "cross_entropy": loss,
        "n_samples": len(labels),
    }
    rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    for item in result.extractor_results:
        overlap = item.overlap
        if overlap is None:
            raise RuntimeError(f"Expected overlap diagnostics for {item.name!r}.")
        layer = str(item.embedding_metadata["output_name"])
        rows.append({**common, "layer": layer, "overlap_index": float(overlap.score)})
        for class_name, score in overlap.per_class_scores.items():
            class_rows.append(
                {
                    **common,
                    "layer": layer,
                    "class_name": str(class_name),
                    "class_overlap_index": float(score),
                }
            )
    return rows, class_rows, confusion


def _behavior_metrics(
    model: Any,
    values: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    torch: Any,
) -> Tuple[np.ndarray, float, float]:
    previous_mode = model.training
    model.eval()
    predictions: List[np.ndarray] = []
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch_x = torch.as_tensor(values[start : start + batch_size], dtype=torch.float32)
            batch_y = torch.as_tensor(labels[start : start + batch_size], dtype=torch.long)
            logits = model(batch_x)["logits"]
            total_loss += float(torch.nn.functional.cross_entropy(logits, batch_y, reduction="sum"))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    model.train(previous_mode)
    combined = np.concatenate(predictions).astype(np.int64, copy=False)
    return combined, float(np.mean(combined == labels)), total_loss / len(labels)


def _normalized_confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(_FASHION_CLASS_NAMES), len(_FASHION_CLASS_NAMES)), dtype=float)
    np.add.at(matrix, (labels, predictions), 1.0)
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def _safe_retention(value: float, baseline: float) -> float:
    return float(value / baseline) if baseline > 1e-12 else float("nan")


def _pairwise_confusion_onset(
    confusion_by_condition: Mapping[Tuple[str, int], np.ndarray],
    clean_confusion: np.ndarray,
    *,
    threshold: float,
) -> Any:
    import pandas as pd

    clean_symmetric = (clean_confusion + clean_confusion.T) / 2.0
    rows: List[Dict[str, Any]] = []
    for first in range(len(_FASHION_CLASS_NAMES)):
        for second in range(first + 1, len(_FASHION_CLASS_NAMES)):
            candidates = []
            for corruption in _CORRUPTION_ORDER:
                for severity_index in range(1, len(_SEVERITY_LABELS)):
                    matrix = confusion_by_condition[(corruption, severity_index)]
                    symmetric = (matrix + matrix.T) / 2.0
                    gain = float(symmetric[first, second] - clean_symmetric[first, second])
                    if gain >= threshold:
                        candidates.append((severity_index, -gain, corruption, gain))
                        break
            if candidates:
                severity_index, _, corruption, gain = min(candidates)
                onset_value: Optional[int] = int(severity_index)
                onset_corruption: Optional[str] = corruption
                onset_gain: Optional[float] = float(gain)
            else:
                onset_value = None
                onset_corruption = None
                onset_gain = None
            rows.append(
                {
                    "class_a": _FASHION_CLASS_NAMES[first],
                    "class_b": _FASHION_CLASS_NAMES[second],
                    "onset_severity_index": onset_value,
                    "onset_severity_label": (
                        _SEVERITY_LABELS[onset_value] if onset_value is not None else None
                    ),
                    "onset_corruption": onset_corruption,
                    "confusion_gain_at_onset": onset_gain,
                    "clean_symmetric_confusion": float(clean_symmetric[first, second]),
                }
            )
    return pd.DataFrame(rows)


def _protocol_payload(
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_status: str,
) -> Dict[str, Any]:
    return {
        "example": "fashion_mnist_corruption_atlas",
        "checkpoint": str(checkpoint_path),
        "checkpoint_status": checkpoint_status,
        "dataset": "torchvision.datasets.FashionMNIST",
        "evaluation_split": "official_test",
        "evaluation_size": args.evaluation_size,
        "seed": args.seed,
        "overlap_k": 5,
        "corruption_order": list(_CORRUPTION_ORDER),
        "severity_labels": list(_SEVERITY_LABELS),
        "severity_values": {name: list(values) for name, values in _SEVERITY_VALUES.items()},
        "severity_units": dict(_SEVERITY_UNITS),
        "confusion_threshold": args.confusion_threshold,
    }


def _practitioner_summary(summary: Any) -> str:
    severe = summary.loc[summary["severity_index"] == len(_SEVERITY_LABELS) - 1]
    retention = severe.pivot_table(
        index="corruption", columns="layer", values="oi_retention", aggfunc="last"
    )
    earliest_layer = retention.mean(axis=0).reindex(_LAYER_ORDER).idxmin()
    worst_behavior = (
        severe.loc[:, ["corruption", "accuracy"]].drop_duplicates().sort_values("accuracy").iloc[0]
    )
    return (
        "Atlas summary: the largest mean severe-shift OI loss occurred at "
        f"{_LAYER_LABELS[earliest_layer]}, while "
        f"{_CORRUPTION_LABELS[worst_behavior['corruption']]} produced the lowest "
        f"severe-shift accuracy ({float(worst_behavior['accuracy']):.3f})."
    )


def _plot_corruption_atlas(
    summary: Any,
    per_class: Any,
    pairwise_onset: Any,
    figure_dir: Path,
    *,
    confusion_threshold: float,
    plt: Any,
) -> Tuple[Path, Path]:
    _apply_plot_style(plt)
    figure = plt.figure(figsize=(18.0, 12.5))
    grid = figure.add_gridspec(
        3,
        10,
        height_ratios=(1.0, 0.9, 1.35),
        hspace=0.62,
        wspace=0.7,
        left=0.07,
        right=0.96,
        top=0.88,
        bottom=0.09,
    )
    figure.suptitle(
        "Deployment-shift atlas: what failed, where it started, and which classes collided",
        x=0.07,
        ha="left",
        fontsize=19,
        fontweight="semibold",
    )
    figure.text(
        0.07,
        0.91,
        "One trained Fashion-MNIST CNN  •  fixed official-test probe  •  "
        "OverlapIndex retention uses the clean score for each layer as 1.0",
        color="#475569",
        fontsize=10.5,
    )

    retention_axes = []
    for index, corruption in enumerate(_CORRUPTION_ORDER):
        axis = figure.add_subplot(grid[0, index * 2 : index * 2 + 2])
        retention_axes.append(axis)
        subset = summary.loc[summary["corruption"] == corruption]
        for layer in _LAYER_ORDER:
            layer_rows = subset.loc[subset["layer"] == layer].sort_values("severity_index")
            axis.plot(
                layer_rows["severity_index"],
                layer_rows["oi_retention"],
                color=_LAYER_COLORS[layer],
                marker=_LAYER_MARKERS[layer],
                markersize=4.5,
                linewidth=2.0,
                label=_LAYER_LABELS[layer],
            )
        axis.axhline(1.0, color="#94A3B8", linewidth=1.0)
        axis.axhline(0.9, color="#CBD5E1", linewidth=1.0, linestyle="--")
        axis.set_title(_CORRUPTION_LABELS[corruption])
        axis.set_xticks(range(len(_SEVERITY_LABELS)), labels=("0", "1", "2", "3", "4"))
        axis.set_xlabel("Severity")
        axis.set_ylim(0.0, max(1.08, float(subset["oi_retention"].max()) + 0.04))
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    retention_axes[0].set_ylabel("OI retention")
    handles, labels = retention_axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.96, 0.925), ncol=3)

    accuracy_axis = figure.add_subplot(grid[1, :5])
    loss_axis = figure.add_subplot(grid[1, 5:])
    behavior = summary.drop_duplicates(subset=["corruption", "severity_index"])
    clean_accuracy = float(behavior["accuracy"].iloc[0])
    clean_loss = float(behavior["cross_entropy"].iloc[0])
    for corruption in _CORRUPTION_ORDER:
        rows = behavior.loc[behavior["corruption"] == corruption].sort_values("severity_index")
        color = _CORRUPTION_COLORS[corruption]
        accuracy_axis.plot(
            rows["severity_index"],
            rows["accuracy"],
            color=color,
            marker="o",
            linewidth=2.0,
            label=_CORRUPTION_LABELS[corruption],
        )
        loss_axis.plot(
            rows["severity_index"],
            rows["cross_entropy"],
            color=color,
            marker="o",
            linewidth=2.0,
        )
    accuracy_axis.axhline(clean_accuracy, color="#94A3B8", linewidth=1.0)
    loss_axis.axhline(clean_loss, color="#94A3B8", linewidth=1.0)
    accuracy_axis.set_title("Observed behavior: accuracy")
    loss_axis.set_title("Observed behavior: cross-entropy")
    for axis, ylabel in ((accuracy_axis, "Top-1 accuracy"), (loss_axis, "Cross-entropy")):
        axis.set_xticks(range(len(_SEVERITY_LABELS)), labels=_SEVERITY_LABELS)
        axis.set_xlabel("Standardized severity tier")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    accuracy_axis.legend(frameon=False, ncol=3, fontsize=9)

    class_axis = figure.add_subplot(grid[2, :5])
    severe_classes = per_class.loc[
        (per_class["layer"] == "embedding")
        & (per_class["severity_index"] == len(_SEVERITY_LABELS) - 1)
    ]
    class_matrix = (
        severe_classes.pivot_table(
            index="class_name",
            columns="corruption",
            values="class_oi_delta",
            aggfunc="last",
        )
        .reindex(index=_FASHION_CLASS_NAMES, columns=_CORRUPTION_ORDER)
        .to_numpy(dtype=float)
    )
    absolute_limit = max(0.1, float(np.nanmax(np.abs(class_matrix))))
    class_image = class_axis.imshow(
        class_matrix,
        cmap="RdBu",
        vmin=-absolute_limit,
        vmax=absolute_limit,
        aspect="auto",
    )
    class_axis.set_title("Who loses transferable geometry?  Severe-shift embedding OI change")
    class_axis.set_xticks(
        range(len(_CORRUPTION_ORDER)),
        labels=[_CORRUPTION_LABELS[name] for name in _CORRUPTION_ORDER],
        rotation=25,
        ha="right",
    )
    class_axis.set_yticks(range(len(_FASHION_CLASS_NAMES)), labels=_FASHION_CLASS_NAMES)
    for row in range(class_matrix.shape[0]):
        for column in range(class_matrix.shape[1]):
            value = class_matrix[row, column]
            if np.isfinite(value):
                class_axis.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > absolute_limit * 0.62 else "#0F172A",
                )
    class_colorbar = figure.colorbar(class_image, ax=class_axis, fraction=0.035, pad=0.025)
    class_colorbar.set_label("Per-class OI change from clean")

    pair_axis = figure.add_subplot(grid[2, 5:])
    onset = np.full((len(_FASHION_CLASS_NAMES), len(_FASHION_CLASS_NAMES)), np.nan)
    annotations = np.full(onset.shape, "", dtype=object)
    for row in pairwise_onset.itertuples(index=False):
        first = _FASHION_CLASS_NAMES.index(row.class_a)
        second = _FASHION_CLASS_NAMES.index(row.class_b)
        if isinstance(row.onset_corruption, str) and np.isfinite(row.onset_severity_index):
            onset[second, first] = float(row.onset_severity_index)
            annotations[second, first] = (
                f"{_CORRUPTION_SHORT[row.onset_corruption]}{int(row.onset_severity_index)}"
            )
    pair_image = pair_axis.imshow(onset, cmap="YlOrRd", vmin=1.0, vmax=4.0)
    pair_axis.set_title(
        f"Who interferes first?  Earliest +{confusion_threshold * 100:.0f} pp pair confusion"
    )
    short_labels = [name.replace("T-shirt/top", "T-shirt") for name in _FASHION_CLASS_NAMES]
    pair_axis.set_xticks(
        range(len(short_labels)), labels=short_labels, rotation=45, ha="right", fontsize=8
    )
    pair_axis.set_yticks(range(len(short_labels)), labels=short_labels, fontsize=8)
    for row in range(onset.shape[0]):
        for column in range(onset.shape[1]):
            if annotations[row, column]:
                pair_axis.text(
                    column,
                    row,
                    annotations[row, column],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if onset[row, column] >= 3 else "#0F172A",
                )
    pair_colorbar = figure.colorbar(pair_image, ax=pair_axis, fraction=0.035, pad=0.025)
    pair_colorbar.set_ticks((1, 2, 3, 4), labels=("Mild", "Medium", "Strong", "Severe"))
    pair_colorbar.set_label("First severity tier")
    figure.text(
        0.55,
        0.025,
        "Pair labels identify the first triggering corruption: "
        "B blur  •  N noise  •  O occlusion  •  C contrast  •  R rotation.\n"
        "Blank pairs did not cross the threshold.",
        color="#475569",
        fontsize=9.5,
    )
    return _save_figure(figure, figure_dir, "fashion-mnist-corruption-atlas", plt)


def _apply_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#94A3B8",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(
    figure: Any,
    figure_dir: Path,
    stem: str,
    plt: Any,
) -> Tuple[Path, Path]:
    png_path = figure_dir / f"{stem}.png"
    svg_path = figure_dir / f"{stem}.svg"
    figure.savefig(png_path, dpi=180)
    figure.savefig(svg_path)
    plt.close(figure)
    return png_path, svg_path


if __name__ == "__main__":
    main()
