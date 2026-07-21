"""Generate three visual examples from one locally trained MNIST network.

The suite demonstrates mini-batch representation monitoring, embedding compression,
and hierarchical label views with real OverlapIndex evaluations on a held-out
validation split. Separatix and stability repeats are intentionally disabled so the
figures focus on the raw representation diagnostic.

Install the optional dependencies and run from the repository root:

    poetry install -E visuals
    poetry run python examples/mnist_visual_suite.py

MNIST is downloaded through torchvision on the first run and reused from the
local data directory afterward.
"""

from __future__ import annotations

import argparse
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
    EvaluationHistoryConfig,
    LabelViewConfig,
    OverlapScoringConfig,
    RepresentationMonitor,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.config import CacheConfig
from vertebrae.extractors import PrecomputedExtractor, TorchExtractor

_OUTPUT_SPECS = (
    {"name": "hidden_1", "hidden_layer": 1, "pooling": "identity"},
    {"name": "hidden_2", "hidden_layer": 2, "pooling": "identity"},
    {"name": "embedding", "hidden_layer": 3, "pooling": "identity"},
)
_LAYER_ORDER = ("hidden_1", "hidden_2", "embedding")
_LAYER_LABELS = {
    "hidden_1": "Hidden 1",
    "hidden_2": "Hidden 2",
    "embedding": "Embedding",
}
_LAYER_DIMS = {"hidden_1": 256, "hidden_2": 128, "embedding": 64}
_LAYER_COLORS = {
    "hidden_1": "#3B82F6",
    "hidden_2": "#8B5CF6",
    "embedding": "#EC4899",
}
_LAYER_MARKERS = {"hidden_1": "o", "hidden_2": "s", "embedding": "D"}
_HIERARCHY_LEVELS = ("parity", "parity + range", "digit")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--monitor-every-batches",
        type=int,
        default=2,
        help="Evaluate held-out representations after this many optimizer steps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("examples/data"),
        help="Directory used by torchvision for the MNIST download/cache.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Figure destination; defaults to VERTABRAE_EXAMPLE_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require MNIST to already exist under --data-dir.",
    )
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        from torchvision.datasets import MNIST
    except ImportError as exc:
        print(exc)
        print("Install the visual example dependencies with: poetry install -E visuals")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_x, train_y, validation_x, validation_y = _load_mnist(
        MNIST,
        data_dir=args.data_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        seed=args.seed,
        download=not args.no_download,
    )
    validation_dataset = _validation_dataset(validation_x, validation_y, args.seed)
    model = _build_model(torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    loss_fn = torch.nn.CrossEntropyLoss()
    extractor = _multi_output_extractor(model, torch, args.seed)
    scoring_config = _scoring_config(args.seed)

    monitor = RepresentationMonitor(
        validation_dataset,
        [extractor],
        history_config=EvaluationHistoryConfig(storage="memory", detail="summary"),
        scoring_config=scoring_config,
        embedding_config=EmbeddingConfig(batch_size=args.embedding_batch_size),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
    )

    global_step = 0
    initial_loss = _mean_loss(model, train_x, train_y, loss_fn, torch)
    initial_accuracy = _accuracy(model, validation_x, validation_y, torch)
    monitor.evaluate(
        snapshot_id="initialization",
        epoch=0,
        global_step=global_step,
        metadata={
            "training_loss": initial_loss,
            "validation_accuracy": initial_accuracy,
        },
    )
    initial_hierarchy_result = _hierarchy_benchmark(
        validation_dataset,
        extractor,
        scoring_config,
        batch_size=args.embedding_batch_size,
    )

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
                global_step % args.monitor_every_batches != 0
                and batch_number != total_batches
            ):
                continue
            validation_accuracy = _accuracy(model, validation_x, validation_y, torch)
            monitor.evaluate(
                epoch=epoch,
                global_step=global_step,
                metadata={
                    "batch_in_epoch": batch_number,
                    "training_loss": training_loss,
                    "validation_accuracy": validation_accuracy,
                },
            )

    output_dir = ensure_output_dir()
    figure_dir = args.figure_dir or output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    history = monitor.history.to_dataframe()
    history.to_csv(output_dir / "mnist_representation_history.csv", index=False)

    final_outputs = {
        item.name: item.embeddings for item in extractor.transform_many(validation_x)
    }
    compression_result = _compression_benchmark(
        final_outputs["embedding"],
        validation_y,
        scoring_config,
    )
    compression_result.save_json(str(output_dir / "mnist_compression_frontier.json"))
    trained_hierarchy_result = _hierarchy_benchmark(
        validation_dataset,
        extractor,
        scoring_config,
        batch_size=args.embedding_batch_size,
    )
    initial_hierarchy_result.save_json(
        str(output_dir / "mnist_hierarchy_layers_initial.json")
    )
    trained_hierarchy_result.save_json(
        str(output_dir / "mnist_hierarchy_layers_trained.json")
    )

    monitoring_paths = _plot_monitoring(history, figure_dir, plt)
    compression_paths = _plot_compression_frontier(
        compression_result,
        n_samples=len(validation_y),
        figure_dir=figure_dir,
        plt=plt,
    )
    hierarchy_paths = _plot_hierarchy_heatmap(
        initial_hierarchy_result,
        trained_hierarchy_result,
        figure_dir,
        plt,
    )

    final_accuracy = float(
        history.sort_values("global_step")["context_metadata.validation_accuracy"]
        .dropna()
        .iloc[-1]
    )
    print(f"Final MNIST validation accuracy: {final_accuracy:.3f}")
    for path in (*monitoring_paths, *compression_paths, *hierarchy_paths):
        print(f"Wrote {path}")
    print(f"Metrics written to {output_dir}")


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_options = (
        "epochs",
        "train_size",
        "validation_size",
        "train_batch_size",
        "embedding_batch_size",
        "monitor_every_batches",
    )
    for name in positive_options:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")


def _load_mnist(
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
            "Requested MNIST training and validation subsets exceed the available "
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
    train_x = _normalize_mnist(all_values[train_indices])
    validation_x = _normalize_mnist(all_values[validation_indices])
    return (
        train_x,
        all_targets[train_indices],
        validation_x,
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


def _normalize_mnist(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(len(values), -1) / 255.0
    return ((flattened - 0.1307) / 0.3081).astype(np.float32, copy=False)


def _validation_dataset(
    values: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> BenchmarkDataset:
    identity = DatasetIdentity.from_manifest(
        "torchvision-mnist-validation",
        {
            "source_split": "train",
            "role": "validation",
            "subset": "stratified",
            "sample_count": int(len(labels)),
            "random_state": int(seed + 1),
            "normalization": {"mean": 0.1307, "std": 0.3081},
        },
    )
    dataset = BenchmarkDataset.from_arrays(
        values,
        labels.astype(str),
        modality="image",
        metadata={
            "example": "mnist_visual_suite",
            "dataset_source": "torchvision.datasets.MNIST",
            "split": "fixed_held_out_validation",
        },
        identity=identity,
    )
    return dataset.with_label_hierarchy(
        _hierarchy_paths(labels),
        level_names=_HIERARCHY_LEVELS,
    )


def _hierarchy_paths(labels: Iterable[int]) -> list[Tuple[str, str, str]]:
    paths = []
    for raw_label in labels:
        label = int(raw_label)
        parity = "even" if label % 2 == 0 else "odd"
        value_range = "0-4" if label < 5 else "5-9"
        paths.append((parity, f"{parity}, {value_range}", str(label)))
    return paths


def _build_model(torch: Any) -> Any:
    class MNISTClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_1 = torch.nn.Linear(28 * 28, _LAYER_DIMS["hidden_1"])
            self.hidden_2 = torch.nn.Linear(
                _LAYER_DIMS["hidden_1"],
                _LAYER_DIMS["hidden_2"],
            )
            self.embedding = torch.nn.Linear(
                _LAYER_DIMS["hidden_2"],
                _LAYER_DIMS["embedding"],
            )
            self.classifier = torch.nn.Linear(_LAYER_DIMS["embedding"], 10)

        def forward(self, values: Any) -> Dict[str, Any]:
            hidden_1 = torch.relu(self.hidden_1(values))
            hidden_2 = torch.relu(self.hidden_2(hidden_1))
            embedding = torch.relu(self.embedding(hidden_2))
            logits = self.classifier(embedding)
            return {
                "hidden_1": hidden_1,
                "hidden_2": hidden_2,
                "embedding": embedding,
                "logits": logits,
            }

    return MNISTClassifier()


def _multi_output_extractor(model: Any, torch: Any, seed: int) -> TorchExtractor:
    def collate_fn(batch: Any) -> Any:
        return torch.as_tensor(np.asarray(batch), dtype=torch.float32)

    def output_fn(raw_output: Dict[str, Any]) -> Dict[str, Any]:
        return {name: raw_output[name] for name in _LAYER_ORDER}

    return TorchExtractor(
        name="mnist_mlp",
        model=model,
        collate_fn=collate_fn,
        output_fn=output_fn,
        outputs=_OUTPUT_SPECS,
        device="cpu",
        modality="image",
        recipe_data={
            "example": "mnist_visual_suite",
            "architecture": [784, 256, 128, 64, 10],
            "training_seed": seed,
        },
    )


def _scoring_config(seed: int) -> OverlapScoringConfig:
    return OverlapScoringConfig(
        k="auto",
        min_k=8,
        max_k=24,
        min_samples_per_cluster=8,
        kmeans_kwargs={"random_state": seed, "batch_size": 512, "n_init": 3},
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
        batch_number = start // batch_size + 1
        yield batch_number, total_batches, total_loss / seen


def _mean_loss(
    model: Any,
    values: np.ndarray,
    labels: np.ndarray,
    loss_fn: Any,
    torch: Any,
) -> float:
    previous_mode = model.training
    model.eval()
    with torch.inference_mode():
        logits = model(torch.as_tensor(values, dtype=torch.float32))["logits"]
        loss = float(loss_fn(logits, torch.as_tensor(labels, dtype=torch.long)))
    model.train(previous_mode)
    return loss


def _accuracy(model: Any, values: np.ndarray, labels: np.ndarray, torch: Any) -> float:
    previous_mode = model.training
    model.eval()
    with torch.inference_mode():
        logits = model(torch.as_tensor(values, dtype=torch.float32))["logits"]
        predictions = logits.argmax(dim=1).cpu().numpy()
    model.train(previous_mode)
    return float(np.mean(predictions == labels))


def _compression_benchmark(
    embeddings: np.ndarray,
    labels: np.ndarray,
    scoring_config: OverlapScoringConfig,
) -> Any:
    dataset = BenchmarkDataset.from_embeddings(
        embeddings,
        labels.astype(str),
        metadata={
            "example": "mnist_visual_suite_compression",
            "source_output": "embedding",
        },
        identity=DatasetIdentity.from_content(),
    )
    compression_configs = [EmbeddingCompressionConfig()]
    compression_configs.extend(
        EmbeddingCompressionConfig(
            enabled=True,
            method="pca",
            n_components=dimension,
            random_state=42,
        )
        for dimension in (2, 4, 8, 16, 32)
    )
    compression_configs.extend(
        [
            EmbeddingCompressionConfig(
                enabled=True,
                method="quantize",
                precision="float16",
            ),
            EmbeddingCompressionConfig(
                enabled=True,
                method="quantize",
                precision="int8",
            ),
        ]
    )
    return Benchmark(
        dataset,
        [PrecomputedExtractor("mnist_penultimate_embedding")],
        compression_configs=compression_configs,
        scoring_config=scoring_config,
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()


def _hierarchy_benchmark(
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
        label_view_config=LabelViewConfig(
            enabled=True,
            hierarchy_levels=_HIERARCHY_LEVELS,
            skip_invalid_levels=False,
        ),
        embedding_config=EmbeddingConfig(batch_size=batch_size),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()


def _plot_monitoring(history: Any, figure_dir: Path, plt: Any) -> Tuple[Path, Path]:
    _apply_plot_style(plt)
    successful = history.loc[history["status"] == "success"].copy()
    pivot = successful.pivot_table(
        index="global_step",
        columns="output_name",
        values="overlap_macro",
        aggfunc="last",
    ).sort_index()
    metric_rows = (
        successful.loc[
            :,
            ["epoch", "global_step", "context_metadata.validation_accuracy"],
        ]
        .drop_duplicates(subset="global_step", keep="last")
        .sort_values("global_step")
    )
    epoch_boundaries = (
        metric_rows.loc[metric_rows["epoch"] > 0]
        .groupby("epoch")["global_step"]
        .max()
        .sort_index()
        .tolist()[:-1]
    )
    final_scores = {name: float(pivot[name].iloc[-1]) for name in _LAYER_ORDER}

    figure = plt.figure(figsize=(14.5, 7.2))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 2.45),
        height_ratios=(2.5, 1.0),
        hspace=0.48,
        wspace=0.22,
    )
    architecture_axis = figure.add_subplot(grid[:, 0])
    curve_axis = figure.add_subplot(grid[0, 1])
    accuracy_axis = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(left=0.055, right=0.96, top=0.86, bottom=0.14)
    figure.suptitle(
        "Representation quality changes across mini-batch MNIST training",
        fontsize=18,
        fontweight="semibold",
        x=0.055,
        ha="left",
    )
    _draw_architecture(architecture_axis, final_scores, plt)

    for output_name in _LAYER_ORDER:
        curve_axis.plot(
            pivot.index,
            pivot[output_name],
            color=_LAYER_COLORS[output_name],
            marker=_LAYER_MARKERS[output_name],
            markersize=5,
            linewidth=2.4,
            label=_LAYER_LABELS[output_name],
        )
    curve_axis.set_title("OverlapIndex on held-out validation data", loc="left", pad=12)
    curve_axis.set_ylabel("OverlapIndex macro score")
    curve_axis.set_ylim(max(0.0, float(pivot.min().min()) - 0.08), 1.01)
    curve_axis.set_xlim(float(pivot.index.min()), float(pivot.index.max()))
    curve_axis.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    curve_axis.legend(frameon=False, loc="lower right", ncol=3)

    accuracy_axis.plot(
        metric_rows["global_step"],
        metric_rows["context_metadata.validation_accuracy"],
        color="#475569",
        marker="o",
        markersize=4,
        linewidth=2,
    )
    accuracy_axis.set_title("Validation accuracy", loc="left", pad=8)
    accuracy_axis.set_xlabel("Optimizer step (0 = initialization)")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.02)
    accuracy_axis.set_xlim(float(pivot.index.min()), float(pivot.index.max()))
    accuracy_axis.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    for boundary_index, boundary in enumerate(epoch_boundaries, start=1):
        for axis in (curve_axis, accuracy_axis):
            axis.axvline(boundary, color="#94A3B8", linestyle=":", linewidth=1.3)
        curve_axis.text(
            boundary,
            0.995,
            f" epoch {boundary_index} ",
            color="#64748B",
            fontsize=9,
            ha="right",
            va="top",
        )
    figure.text(
        0.055,
        0.055,
        "Fixed held-out MNIST validation set  •  optimizer-step checkpoints  •  "
        "Separatix and stability repeats disabled",
        color="#475569",
        fontsize=10,
    )
    return _save_figure(figure, figure_dir, "mnist-representation-monitoring", plt)


def _draw_architecture(axis: Any, final_scores: Dict[str, float], plt: Any) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.set_title("Monitored network", loc="left", pad=12)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    blocks = (
        ("input", "Input pixels", "784", None),
        ("hidden_1", "Hidden 1", "256", final_scores["hidden_1"]),
        ("hidden_2", "Hidden 2", "128", final_scores["hidden_2"]),
        ("embedding", "Embedding", "64", final_scores["embedding"]),
        ("output", "Classifier", "10", None),
    )
    y_positions = np.linspace(0.88, 0.10, len(blocks))
    for index, ((key, label, width, score), y_position) in enumerate(
        zip(blocks, y_positions)
    ):
        color = _LAYER_COLORS.get(key, "#E2E8F0")
        text_color = "white" if key in _LAYER_COLORS else "#0F172A"
        patch = FancyBboxPatch(
            (0.12, y_position - 0.055),
            0.72,
            0.11,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=color,
            edgecolor="none",
        )
        axis.add_patch(patch)
        axis.text(
            0.17,
            y_position,
            label,
            color=text_color,
            fontweight="semibold",
            va="center",
        )
        axis.text(
            0.17,
            y_position - 0.028,
            f"{width} units",
            color=text_color,
            fontsize=9,
            va="center",
        )
        if score is not None:
            axis.text(
                0.79,
                y_position,
                f"OI {score:.3f}",
                color=text_color,
                va="center",
                ha="right",
            )
        if index < len(blocks) - 1:
            next_y = y_positions[index + 1]
            axis.annotate(
                "",
                xy=(0.48, next_y + 0.065),
                xytext=(0.48, y_position - 0.065),
                arrowprops={"arrowstyle": "-|>", "color": "#94A3B8", "lw": 1.5},
            )


def _plot_compression_frontier(
    result: Any,
    *,
    n_samples: int,
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    _apply_plot_style(plt)
    points = _compression_points(result, n_samples)
    frontier_indices = _pareto_frontier_indices(
        [point["bytes_per_sample"] for point in points],
        [point["score"] for point in points],
    )
    figure, axis = plt.subplots(figsize=(11.8, 6.3))
    figure.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.18)
    figure.suptitle(
        "Compression can reduce storage without erasing class separation",
        fontsize=18,
        fontweight="semibold",
        x=0.09,
        ha="left",
    )
    styles = {
        "none": ("#0F172A", "s"),
        "pca": ("#3B82F6", "o"),
        "quantize": ("#EC4899", "D"),
    }
    for method, (color, marker) in styles.items():
        method_points = [point for point in points if point["method"] == method]
        axis.scatter(
            [point["bytes_per_sample"] for point in method_points],
            [point["score"] for point in method_points],
            color=color,
            marker=marker,
            s=72,
            label={"none": "Raw", "pca": "PCA", "quantize": "Quantized"}[method],
            zorder=3,
        )
    frontier = [points[index] for index in frontier_indices]
    axis.plot(
        [point["bytes_per_sample"] for point in frontier],
        [point["score"] for point in frontier],
        color="#0F766E",
        linewidth=2,
        linestyle="--",
        label="Pareto frontier",
        zorder=2,
    )
    annotation_offsets = {
        "PCA 16d": (-12, 12),
        "int8": (8, -19),
        "PCA 32d": (-15, 12),
        "float16": (8, -19),
    }
    for point in points:
        offset = annotation_offsets.get(point["label"], (5, 7))
        axis.annotate(
            point["label"],
            (point["bytes_per_sample"], point["score"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xscale("log", base=2)
    byte_ticks = sorted({point["bytes_per_sample"] for point in points})
    axis.set_xticks(byte_ticks, labels=[f"{value:g}" for value in byte_ticks])
    axis.set_xlabel("Encoded embedding bytes per sample (log₂ scale)")
    axis.set_ylabel("OverlapIndex macro score")
    minimum = min(point["score"] for point in points)
    axis.set_ylim(max(0.0, minimum - 0.06), 1.01)
    axis.grid(color="#CBD5E1", linewidth=0.8, alpha=0.7)
    axis.legend(frameon=False, loc="lower right", ncol=2)
    figure.text(
        0.09,
        0.06,
        "Same trained 64-dimensional MNIST embedding  •  lower storage is better  •  "
        "higher overlap is better",
        color="#475569",
        fontsize=10,
    )
    return _save_figure(figure, figure_dir, "mnist-compression-frontier", plt)


def _compression_points(result: Any, n_samples: int) -> list[Dict[str, Any]]:
    points = []
    for item in result.extractor_results:
        metadata = item.compression_metadata
        method = metadata.get("method", "none")
        precision = metadata.get("precision")
        dimension = int(metadata.get("compressed_dim", item.embedding_metadata["embedding_dim"]))
        total_bytes = metadata.get("estimated_encoded_bytes") or metadata.get("resident_bytes")
        if total_bytes is None:
            dtype = np.dtype(metadata.get("dtype", "float32"))
            total_bytes = n_samples * dimension * dtype.itemsize
        if method == "none":
            label = f"Raw {dimension}d"
        elif method == "pca":
            label = f"PCA {dimension}d"
        else:
            label = str(precision)
        points.append(
            {
                "method": method,
                "label": label,
                "score": float(item.overlap.macro_score),
                "bytes_per_sample": float(total_bytes) / float(n_samples),
            }
        )
    return sorted(points, key=lambda point: (point["bytes_per_sample"], point["score"]))


def _pareto_frontier_indices(
    bytes_per_sample: Sequence[float],
    scores: Sequence[float],
) -> list[int]:
    order = sorted(range(len(scores)), key=lambda index: (bytes_per_sample[index], -scores[index]))
    frontier = []
    best_score = float("-inf")
    for index in order:
        if scores[index] > best_score:
            frontier.append(index)
            best_score = scores[index]
    return frontier


def _plot_hierarchy_heatmap(
    initial_result: Any,
    trained_result: Any,
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    _apply_plot_style(plt)
    matrices = [
        _hierarchy_matrix(initial_result),
        _hierarchy_matrix(trained_result),
    ]
    minimum = min(float(np.nanmin(matrix)) for matrix in matrices)
    color_minimum = max(0.0, np.floor(minimum * 10.0) / 10.0)
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), sharey=True)
    figure.subplots_adjust(left=0.14, right=0.9, top=0.76, bottom=0.24, wspace=0.08)
    figure.suptitle(
        "Training reshapes separation across layer depth and label granularity",
        fontsize=18,
        fontweight="semibold",
        x=0.14,
        ha="left",
    )
    image = None
    for axis, values, title in zip(
        axes,
        matrices,
        ("Before training", "After training"),
    ):
        image = axis.imshow(
            values,
            cmap="viridis",
            vmin=color_minimum,
            vmax=1.0,
            aspect="auto",
        )
        axis.set_title(title, pad=10)
        axis.set_xticks(
            range(len(_HIERARCHY_LEVELS)),
            labels=("Parity\n2 classes", "Parity + range\n4 classes", "Digit\n10 classes"),
        )
        axis.set_yticks(
            range(len(_LAYER_ORDER)),
            labels=[_LAYER_LABELS[name] for name in _LAYER_ORDER],
        )
        axis.set_xlabel("Label hierarchy level")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                normalized = (value - color_minimum) / max(1e-12, 1.0 - color_minimum)
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="black" if normalized >= 0.58 else "white",
                    fontweight="semibold",
                )
    axes[0].set_ylabel("Network representation")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.035)
    colorbar.set_label("OverlapIndex macro score")
    figure.text(
        0.14,
        0.07,
        "Nested views: even/odd → parity-specific 0–4 or 5–9 range → exact digit",
        color="#475569",
        fontsize=10,
    )
    return _save_figure(figure, figure_dir, "mnist-hierarchy-heatmap", plt)


def _hierarchy_matrix(result: Any) -> np.ndarray:
    frame = result.to_dataframe()
    matrix = frame.pivot_table(
        index="output_name",
        columns="label_view",
        values="overlap_macro",
        aggfunc="last",
    ).reindex(index=_LAYER_ORDER, columns=_HIERARCHY_LEVELS)
    return matrix.to_numpy(dtype=float)


def _apply_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
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
