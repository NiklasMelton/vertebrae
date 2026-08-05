"""Compare control and shortcut-trained CNN representations on colored Fashion-MNIST.

Two CNNs start from identical parameters and receive the same garments, labels,
mini-batch order, optimizer, and schedule. The control receives colors balanced within
every garment class. The shortcut treatment receives a canonical class color with high
probability. Only the class-color relationship differs between their training data.

OverlapIndex is measured on a separate, exactly class-color-balanced audit probe with
two named target views: ``intended_class`` and ``nuisance_color``. Accuracy is measured
independently on correlated, balanced, reversed-color, and grayscale environments. This
separates what each representation organizes from whether color reliance is harmful.

Install and run from the repository root:

    poetry install -E visuals
    poetry run python examples/colored_fashion_mnist_shortcut.py

Fashion-MNIST is downloaded through torchvision on the first run and then reused.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from _common import ensure_output_dir

from vertebrae import (
    BenchmarkDataset,
    DatasetIdentity,
    EmbeddingConfig,
    EvaluationHistoryConfig,
    OverlapScoringConfig,
    RepresentationMonitor,
    SeparatixConfig,
    StabilityConfig,
    TargetView,
    TargetViewConfig,
)
from vertebrae.extractors import TorchExtractor

_OUTPUT_SPECS = (
    {"name": "conv_1", "hidden_layer": 1, "pooling": "adaptive_avg_2x2"},
    {"name": "conv_2", "hidden_layer": 2, "pooling": "adaptive_avg_2x2"},
    {"name": "embedding", "hidden_layer": 3, "pooling": "identity"},
)
_LAYER_ORDER = ("conv_1", "conv_2", "embedding")
_LAYER_LABELS = {
    "conv_1": "Conv block 1",
    "conv_2": "Conv block 2",
    "embedding": "Embedding",
}
_REGIMES = ("control", "shortcut")
_REGIME_COLORS = {"control": "#0F766E", "shortcut": "#E11D48"}
_REGIME_LABELS = {"control": "Independent-color control", "shortcut": "Shortcut treatment"}
_ENVIRONMENTS = ("correlated", "balanced", "reversed", "grayscale")
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
_COLOR_NAMES = (
    "red",
    "orange",
    "yellow",
    "lime",
    "green",
    "cyan",
    "blue",
    "violet",
    "pink",
    "magenta",
)
# Fixed-saturation/value hues reduce brightness as an unintended second nuisance.
_PALETTE = np.asarray(
    (
        (0.900, 0.225, 0.225),
        (0.900, 0.630, 0.225),
        (0.765, 0.900, 0.225),
        (0.360, 0.900, 0.225),
        (0.225, 0.900, 0.495),
        (0.225, 0.900, 0.900),
        (0.225, 0.495, 0.900),
        (0.360, 0.225, 0.900),
        (0.765, 0.225, 0.900),
        (0.900, 0.225, 0.630),
    ),
    dtype=np.float32,
)
_NEUTRAL_RGB = np.full(3, float(_PALETTE.mean()), dtype=np.float32)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-size", type=int, default=2_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of paired training seeds used for aggregate trajectories.",
    )
    parser.add_argument(
        "--shortcut-strength",
        type=float,
        default=0.85,
        help="Probability that a shortcut-training sample uses its canonical class color.",
    )
    parser.add_argument(
        "--color-opacity",
        type=float,
        default=0.45,
        help="Tint strength relative to a neutral grayscale rendering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("examples/data"))
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:
        print(exc)
        print("Install the visual example dependencies with: poetry install -E visuals")
        return

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    train_gray, train_y, validation_gray, validation_y = _load_fashion_mnist(
        FashionMNIST,
        data_dir=args.data_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        seed=args.seed,
        download=not args.no_download,
    )
    evaluation_values, evaluation_colors = _evaluation_environments(
        validation_gray,
        validation_y,
        shortcut_strength=args.shortcut_strength,
        opacity=args.color_opacity,
        seed=args.seed + 100,
    )
    balanced_colors = evaluation_colors["balanced"]
    if balanced_colors is None:
        raise RuntimeError("The balanced evaluation environment must have color ids.")
    audit_dataset = _monitoring_dataset(
        evaluation_values["balanced"],
        validation_y,
        balanced_colors,
        color_opacity=args.color_opacity,
        seed=args.seed,
    )

    output_dir = ensure_output_dir()
    figure_dir = args.figure_dir or output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    histories = []
    final_records = []
    for repeat_index in range(args.repeats):
        training_seed = args.seed + repeat_index * 1_000
        history, final_metrics = _run_repeat(
            training_seed=training_seed,
            repeat_index=repeat_index,
            epochs=args.epochs,
            train_gray=train_gray,
            train_y=train_y,
            validation_gray=validation_gray,
            validation_y=validation_y,
            audit_dataset=audit_dataset,
            evaluation_values=evaluation_values,
            balanced_colors=balanced_colors,
            shortcut_strength=args.shortcut_strength,
            color_opacity=args.color_opacity,
            train_batch_size=args.train_batch_size,
            embedding_batch_size=args.embedding_batch_size,
            torch=torch,
        )
        histories.append(history)
        final_records.append(
            {
                "training_seed": training_seed,
                "repeat_index": repeat_index,
                **final_metrics,
            }
        )
    import pandas as pd

    history = pd.concat(histories, ignore_index=True)
    history_path = output_dir / "colored_fashion_mnist_shortcut_history.csv"
    history.to_csv(history_path, index=False)
    final_frame = _final_metric_frame(final_records)
    metrics_path = output_dir / "colored_fashion_mnist_shortcut_final_metrics.csv"
    final_frame.to_csv(metrics_path, index=False)
    summary_path = output_dir / "colored_fashion_mnist_shortcut_final_metrics_summary.csv"
    _final_metric_summary(final_frame).to_csv(summary_path, index=False)
    figure_path = _plot_shortcut_experiment(history, final_frame, figure_dir, plt)
    effect_path = _plot_paired_effects(history, figure_dir, plt)
    exemplar_paths = _plot_environment_exemplars(
        validation_gray,
        validation_y,
        opacity=args.color_opacity,
        seed=args.seed,
        figure_dir=figure_dir,
        plt=plt,
    )
    shortcut_rows = final_frame.loc[final_frame["training_regime"] == "shortcut"]
    shortcut_summary = shortcut_rows.groupby("metric")["value"].mean()
    print(f"Mean shortcut ID accuracy: {shortcut_summary['correlated_accuracy']:.3f}")
    print(f"Mean shortcut balanced accuracy: {shortcut_summary['balanced_accuracy']:.3f}")
    print(f"Mean shortcut reversed accuracy: {shortcut_summary['reversed_accuracy']:.3f}")
    print(f"Mean shortcut color-following rate: {shortcut_summary['color_following_rate']:.3f}")
    print(f"Wrote {history_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {effect_path}")
    for path in exemplar_paths:
        print(f"Wrote {path}")


def _run_repeat(
    *,
    training_seed: int,
    repeat_index: int,
    epochs: int,
    train_gray: np.ndarray,
    train_y: np.ndarray,
    validation_gray: np.ndarray,
    validation_y: np.ndarray,
    audit_dataset: BenchmarkDataset,
    evaluation_values: Mapping[str, np.ndarray],
    balanced_colors: np.ndarray,
    shortcut_strength: float,
    color_opacity: float,
    train_batch_size: int,
    embedding_batch_size: int,
    torch: Any,
) -> Tuple[Any, Dict[str, float]]:
    train_colors = {
        "control": _assign_colors(
            train_y,
            environment="balanced",
            seed=training_seed + 10,
        ),
        "shortcut": _assign_colors(
            train_y,
            environment="correlated",
            shortcut_strength=shortcut_strength,
            seed=training_seed + 11,
        ),
    }
    training_values = {
        regime: _colorize(train_gray, colors, opacity=color_opacity)
        for regime, colors in train_colors.items()
    }
    models = dict(zip(_REGIMES, _build_paired_models(torch, seed=training_seed)))
    optimizers = {
        regime: torch.optim.Adam(model.parameters(), lr=0.001) for regime, model in models.items()
    }
    extractors = {
        regime: _multi_output_extractor(
            model,
            torch,
            training_seed,
            training_regime=regime,
        )
        for regime, model in models.items()
    }
    monitor_options = {
        "target_view_config": TargetViewConfig(
            enabled=True,
            views=("intended_class", "nuisance_color"),
        ),
        "history_config": EvaluationHistoryConfig(storage="memory", detail="summary"),
        "scoring_config": _scoring_config(training_seed),
        "embedding_config": EmbeddingConfig(batch_size=embedding_batch_size),
        "stability_config": StabilityConfig(enabled=False),
        "separatix_config": SeparatixConfig(enabled=False),
    }
    monitors = {
        regime: RepresentationMonitor(audit_dataset, [extractors[regime]], **monitor_options)
        for regime in _REGIMES
    }
    loss_fn = torch.nn.CrossEntropyLoss()
    global_step = 0
    metrics = _evaluate_monitors(
        monitors,
        epoch=0,
        global_step=global_step,
        models=models,
        evaluation_values=evaluation_values,
        labels=validation_y,
        balanced_colors=balanced_colors,
        torch=torch,
    )
    for epoch in range(1, epochs + 1):
        global_step += _train_paired_epoch(
            models,
            optimizers,
            loss_fn,
            training_values,
            train_y,
            batch_size=train_batch_size,
            seed=training_seed + epoch,
            torch=torch,
        )
        metrics = _evaluate_monitors(
            monitors,
            epoch=epoch,
            global_step=global_step,
            models=models,
            evaluation_values=evaluation_values,
            labels=validation_y,
            balanced_colors=balanced_colors,
            torch=torch,
        )
        print(
            f"Seed {training_seed} ({repeat_index + 1}) epoch {epoch:>2}/{epochs}: "
            f"control balanced={metrics['control_balanced_accuracy']:.3f}, "
            f"shortcut ID={metrics['shortcut_correlated_accuracy']:.3f}, "
            f"shortcut balanced={metrics['shortcut_balanced_accuracy']:.3f}"
        )
    metrics.update(
        _all_color_metrics(
            models,
            validation_gray,
            validation_y,
            opacity=color_opacity,
            torch=torch,
        )
    )
    history = _combined_history(monitors)
    history.insert(0, "repeat_index", repeat_index)
    history.insert(0, "training_seed", training_seed)
    return history, metrics


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in (
        "epochs",
        "repeats",
        "train_size",
        "validation_size",
        "train_batch_size",
        "embedding_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    for name in ("train_size", "validation_size"):
        if getattr(args, name) % 100:
            parser.error(
                f"--{name.replace('_', '-')} must be a multiple of 100 so every "
                "class×color cell has equal support."
            )
    if not np.isfinite(args.shortcut_strength) or not 0.0 < args.shortcut_strength <= 1.0:
        parser.error("--shortcut-strength must be finite and in (0, 1].")
    if not np.isfinite(args.color_opacity) or not 0.0 < args.color_opacity <= 1.0:
        parser.error("--color-opacity must be finite and in (0, 1].")


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
    values = np.asarray(full_dataset.data, dtype=np.float32) / 255.0
    labels = np.asarray(full_dataset.targets, dtype=np.int64)
    if train_size + validation_size > len(labels):
        raise ValueError(
            "Requested training and validation subsets exceed the Fashion-MNIST training split."
        )
    validation_indices = _stratified_indices(labels, validation_size, seed + 1)
    available = np.ones(len(labels), dtype=bool)
    available[validation_indices] = False
    remaining_indices = np.flatnonzero(available)
    train_indices = remaining_indices[
        _stratified_indices(labels[remaining_indices], train_size, seed)
    ]
    return (
        values[train_indices],
        labels[train_indices],
        values[validation_indices],
        labels[validation_indices],
    )


def _stratified_indices(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    base, remainder = divmod(size, len(classes))
    rng = np.random.default_rng(seed)
    selected = []
    for offset, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        count = base + int(offset < remainder)
        if count > len(candidates):
            raise ValueError(f"Class {label!r} does not contain {count} requested samples.")
        selected.extend(rng.choice(candidates, size=count, replace=False))
    return np.asarray(selected, dtype=np.int64)[rng.permutation(size)]


def _assign_colors(
    labels: np.ndarray,
    *,
    environment: str,
    seed: int,
    shortcut_strength: float = 0.85,
) -> np.ndarray:
    """Assign exactly balanced, correlated, or reversed colors."""
    labels = np.asarray(labels, dtype=np.int64)
    n_colors = len(_COLOR_NAMES)
    if labels.ndim != 1 or (len(labels) and (labels.min() < 0 or labels.max() >= n_colors)):
        raise ValueError(f"labels must be one-dimensional ids in [0, {n_colors}).")
    if environment == "reversed":
        return (labels + n_colors // 2) % n_colors
    if environment == "balanced":
        rng = np.random.default_rng(seed)
        colors = np.empty_like(labels)
        for class_offset, label in enumerate(np.unique(labels)):
            indices = np.flatnonzero(labels == label)
            shuffled_indices = indices[rng.permutation(len(indices))]
            cycle = np.roll(np.arange(n_colors, dtype=np.int64), class_offset)
            colors[shuffled_indices] = np.resize(cycle, len(indices))
        return colors
    if environment != "correlated":
        raise ValueError("environment must be 'balanced', 'correlated', or 'reversed'.")
    if not np.isfinite(shortcut_strength) or not 0.0 <= shortcut_strength <= 1.0:
        raise ValueError("shortcut_strength must be finite and in [0, 1].")
    rng = np.random.default_rng(seed)
    colors = labels.copy()
    alternatives = (labels + rng.integers(1, n_colors, size=len(labels))) % n_colors
    noncanonical = rng.random(len(labels)) > shortcut_strength
    colors[noncanonical] = alternatives[noncanonical]
    return colors


def _colorize(
    grayscale_images: np.ndarray,
    color_ids: np.ndarray,
    *,
    opacity: float = 0.45,
) -> np.ndarray:
    images = np.asarray(grayscale_images, dtype=np.float32)
    color_ids = np.asarray(color_ids, dtype=np.int64)
    if images.ndim != 3 or len(images) != len(color_ids):
        raise ValueError(
            "grayscale_images must be [samples, height, width] aligned with color_ids."
        )
    if len(color_ids) and (color_ids.min() < 0 or color_ids.max() >= len(_PALETTE)):
        raise ValueError("color_ids are outside the palette.")
    if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be finite and in [0, 1].")
    tints = (1.0 - opacity) * _NEUTRAL_RGB + opacity * _PALETTE[color_ids]
    return (images[:, None, :, :] * tints[:, :, None, None]).reshape(len(images), -1)


def _grayscale_rgb(grayscale_images: np.ndarray) -> np.ndarray:
    images = np.asarray(grayscale_images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("grayscale_images must have shape [samples, height, width].")
    return (images[:, None, :, :] * _NEUTRAL_RGB[None, :, None, None]).reshape(len(images), -1)


def _evaluation_environments(
    grayscale_images: np.ndarray,
    labels: np.ndarray,
    *,
    shortcut_strength: float,
    opacity: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[np.ndarray]]]:
    colors: Dict[str, Optional[np.ndarray]] = {
        "correlated": _assign_colors(
            labels,
            environment="correlated",
            shortcut_strength=shortcut_strength,
            seed=seed,
        ),
        "balanced": _assign_colors(labels, environment="balanced", seed=seed + 1),
        "reversed": _assign_colors(labels, environment="reversed", seed=seed + 2),
        "grayscale": None,
    }
    values = {
        environment: (
            _grayscale_rgb(grayscale_images)
            if color_ids is None
            else _colorize(grayscale_images, color_ids, opacity=opacity)
        )
        for environment, color_ids in colors.items()
    }
    return values, colors


def _monitoring_dataset(
    values: np.ndarray,
    class_ids: np.ndarray,
    color_ids: np.ndarray,
    *,
    color_opacity: float,
    seed: int,
) -> BenchmarkDataset:
    class_targets = np.asarray([_FASHION_CLASS_NAMES[int(label)] for label in class_ids])
    color_targets = np.asarray([_COLOR_NAMES[int(color)] for color in color_ids])
    identity = DatasetIdentity.from_manifest(
        "colored-fashion-mnist-balanced-audit-probe",
        {
            "sample_count": int(len(class_ids)),
            "color_environment": "balanced",
            "color_opacity": float(color_opacity),
            "seed": int(seed),
        },
    )
    return BenchmarkDataset.from_arrays(
        values,
        class_targets,
        modality="image",
        metadata={
            "example": "colored_fashion_mnist_shortcut",
            "color_environment": "balanced",
            "color_opacity": float(color_opacity),
        },
        identity=identity,
    ).with_target_views(
        [
            TargetView(
                name="intended_class",
                targets=class_targets,
                metadata={"semantic_role": "intended"},
            ),
            TargetView(
                name="nuisance_color",
                targets=color_targets,
                metadata={"semantic_role": "nuisance"},
            ),
        ]
    )


def _build_model(torch: Any) -> Any:
    class ColoredFashionCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_1 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )
            self.conv_2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )
            self.embedding = torch.nn.Linear(64 * 7 * 7, 128)
            self.classifier = torch.nn.Linear(128, len(_FASHION_CLASS_NAMES))

        def forward(self, values: Any) -> Dict[str, Any]:
            images = values.reshape(-1, 3, 28, 28)
            maps_1 = self.conv_1(images)
            maps_2 = self.conv_2(maps_1)
            embedding = torch.relu(self.embedding(maps_2.flatten(1)))
            return {
                "conv_1": torch.nn.functional.adaptive_avg_pool2d(maps_1, (2, 2)).flatten(1),
                "conv_2": torch.nn.functional.adaptive_avg_pool2d(maps_2, (2, 2)).flatten(1),
                "embedding": embedding,
                "logits": self.classifier(embedding),
            }

    return ColoredFashionCNN()


def _build_paired_models(torch: Any, *, seed: int) -> Tuple[Any, Any]:
    torch.manual_seed(seed)
    control = _build_model(torch)
    shortcut = _build_model(torch)
    shortcut.load_state_dict(control.state_dict())
    return control, shortcut


def _multi_output_extractor(
    model: Any,
    torch: Any,
    seed: int,
    *,
    training_regime: str,
) -> TorchExtractor:
    if training_regime not in _REGIMES:
        raise ValueError("training_regime must be 'control' or 'shortcut'.")

    def collate_fn(batch: Any) -> Any:
        return torch.as_tensor(np.asarray(batch), dtype=torch.float32)

    def output_fn(raw_output: Dict[str, Any]) -> Dict[str, Any]:
        return {name: raw_output[name] for name in _LAYER_ORDER}

    return TorchExtractor(
        name=f"colored_fashion_mnist_{training_regime}_cnn",
        model=model,
        collate_fn=collate_fn,
        output_fn=output_fn,
        outputs=_OUTPUT_SPECS,
        device="cpu",
        modality="image",
        recipe_data={
            "example": "colored_fashion_mnist_shortcut",
            "architecture": "two-block RGB CNN",
            "training_seed": seed,
            "training_regime": training_regime,
        },
    )


def _scoring_config(seed: int) -> OverlapScoringConfig:
    return OverlapScoringConfig(
        k=5,
        min_samples_per_cluster=5,
        kmeans_kwargs={"random_state": seed, "batch_size": 512, "n_init": 3},
    )


def _train_paired_epoch(
    models: Mapping[str, Any],
    optimizers: Mapping[str, Any],
    loss_fn: Any,
    training_values: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    torch: Any,
) -> int:
    for model in models.values():
        model.train()
    order = np.random.default_rng(seed).permutation(len(labels))
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch_y = torch.as_tensor(labels[indices], dtype=torch.long)
        for regime in _REGIMES:
            optimizer = optimizers[regime]
            optimizer.zero_grad()
            batch_x = torch.as_tensor(training_values[regime][indices], dtype=torch.float32)
            loss = loss_fn(models[regime](batch_x)["logits"], batch_y)
            loss.backward()
            optimizer.step()
    return int(np.ceil(len(order) / batch_size))


def _predictions(
    model: Any,
    values: np.ndarray,
    torch: Any,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    previous_mode = model.training
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            logits = model(torch.as_tensor(values[start:stop], dtype=torch.float32))["logits"]
            predictions.append(logits.argmax(1).cpu().numpy())
    model.train(previous_mode)
    return np.concatenate(predictions)


def _cell_accuracy_scores(
    predictions: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    scores = []
    for label in np.unique(labels):
        for color in np.unique(colors):
            mask = (labels == label) & (colors == color)
            if mask.any():
                scores.append(float(np.mean(predictions[mask] == labels[mask])))
    if not scores:
        raise ValueError("At least one populated class-color cell is required.")
    return np.asarray(scores, dtype=float)


def _cell_accuracy_summary(scores: np.ndarray) -> Dict[str, float]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Cell accuracies must be a non-empty finite one-dimensional array.")
    return {
        "cell_mean_accuracy": float(values.mean()),
        "cell_p10_accuracy": float(np.quantile(values, 0.10)),
        "bottom5_cell_mean_accuracy": float(np.sort(values)[: min(5, len(values))].mean()),
    }


def _checkpoint_metrics(
    models: Mapping[str, Any],
    evaluation_values: Mapping[str, np.ndarray],
    labels: np.ndarray,
    balanced_colors: np.ndarray,
    torch: Any,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for regime in _REGIMES:
        predictions = {
            environment: _predictions(models[regime], evaluation_values[environment], torch)
            for environment in _ENVIRONMENTS
        }
        for environment, values in predictions.items():
            metrics[f"{regime}_{environment}_accuracy"] = float(np.mean(values == labels))
        metrics[f"{regime}_balanced_gap"] = (
            metrics[f"{regime}_correlated_accuracy"] - metrics[f"{regime}_balanced_accuracy"]
        )
        cell_summary = _cell_accuracy_summary(
            _cell_accuracy_scores(predictions["balanced"], labels, balanced_colors)
        )
        metrics.update({f"{regime}_balanced_{name}": value for name, value in cell_summary.items()})
    return metrics


def _all_color_predictions(
    model: Any,
    grayscale_images: np.ndarray,
    *,
    opacity: float,
    torch: Any,
) -> np.ndarray:
    predictions = []
    for color in range(len(_COLOR_NAMES)):
        color_ids = np.full(len(grayscale_images), color, dtype=np.int64)
        values = _colorize(grayscale_images, color_ids, opacity=opacity)
        predictions.append(_predictions(model, values, torch))
    return np.column_stack(predictions)


def _all_color_metrics(
    models: Mapping[str, Any],
    grayscale_images: np.ndarray,
    labels: np.ndarray,
    *,
    opacity: float,
    torch: Any,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    color_grid = np.broadcast_to(
        np.arange(len(_COLOR_NAMES), dtype=np.int64),
        (len(labels), len(_COLOR_NAMES)),
    )
    label_grid = labels[:, None]
    noncanonical = color_grid != label_grid
    reversed_colors = (labels + len(_COLOR_NAMES) // 2) % len(_COLOR_NAMES)
    rows = np.arange(len(labels))
    for regime in _REGIMES:
        predictions = _all_color_predictions(
            models[regime],
            grayscale_images,
            opacity=opacity,
            torch=torch,
        )
        correct = predictions == label_grid
        noncanonical_errors = (~correct) & noncanonical
        follows_color = predictions == color_grid
        metrics[f"{regime}_all_color_accuracy"] = float(correct.mean())
        metrics[f"{regime}_all_colors_correct_rate"] = float(correct.all(axis=1).mean())
        metrics[f"{regime}_prediction_flip_rate"] = float(
            np.any(predictions != predictions[:, :1], axis=1).mean()
        )
        metrics[f"{regime}_mean_unique_predictions"] = float(
            np.mean([len(np.unique(row)) for row in predictions])
        )
        metrics[f"{regime}_color_following_rate"] = float(follows_color[noncanonical].mean())
        error_count = int(noncanonical_errors.sum())
        metrics[f"{regime}_color_following_error_share"] = (
            float((follows_color & noncanonical_errors).sum() / error_count) if error_count else 0.0
        )
        metrics[f"{regime}_reversed_color_following_rate"] = float(
            np.mean(predictions[rows, reversed_colors] == reversed_colors)
        )
        cell_scores = []
        for label in np.unique(labels):
            label_mask = labels == label
            for color in range(len(_COLOR_NAMES)):
                cell_scores.append(float(np.mean(predictions[label_mask, color] == label)))
        summary = _cell_accuracy_summary(np.asarray(cell_scores))
        metrics.update({f"{regime}_all_color_{name}": value for name, value in summary.items()})
    return metrics


def _evaluate_monitors(
    monitors: Mapping[str, RepresentationMonitor],
    *,
    epoch: int,
    global_step: int,
    models: Mapping[str, Any],
    evaluation_values: Mapping[str, np.ndarray],
    labels: np.ndarray,
    balanced_colors: np.ndarray,
    torch: Any,
) -> Dict[str, float]:
    metrics = _checkpoint_metrics(
        models,
        evaluation_values,
        labels,
        balanced_colors,
        torch,
    )
    for monitor in monitors.values():
        monitor.evaluate(
            snapshot_id=f"epoch-{epoch:03d}",
            epoch=epoch,
            global_step=global_step,
            metadata=metrics,
        )
    return metrics


def _combined_history(monitors: Mapping[str, RepresentationMonitor]) -> Any:
    import pandas as pd

    frames = []
    for regime in _REGIMES:
        frame = monitors[regime].history.to_dataframe()
        frame.insert(0, "training_regime", regime)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _metric_rows(history: Any) -> Any:
    metric_columns = [
        f"context_metadata.{regime}_{environment}_accuracy"
        for regime in _REGIMES
        for environment in _ENVIRONMENTS
    ]
    metric_columns += [f"context_metadata.{regime}_balanced_gap" for regime in _REGIMES]
    for regime in _REGIMES:
        metric_columns.extend(
            f"context_metadata.{regime}_balanced_{name}"
            for name in (
                "cell_mean_accuracy",
                "cell_p10_accuracy",
                "bottom5_cell_mean_accuracy",
            )
        )
    rows = (
        history.loc[
            history["status"] == "success",
            ["training_seed", "repeat_index", "epoch", *metric_columns],
        ]
        .drop_duplicates(["training_seed", "epoch"], keep="last")
        .sort_values(["training_seed", "epoch"])
        .reset_index(drop=True)
    )
    return rows.rename(
        columns={column: column.removeprefix("context_metadata.") for column in rows}
    )


def _final_metric_frame(final_records: Sequence[Mapping[str, Any]]) -> Any:
    import pandas as pd

    rows = []
    for record in final_records:
        for regime in _REGIMES:
            prefix = f"{regime}_"
            for name, value in record.items():
                if not name.startswith(prefix):
                    continue
                rows.append(
                    {
                        "training_seed": int(record["training_seed"]),
                        "repeat_index": int(record["repeat_index"]),
                        "training_regime": regime,
                        "metric": name.removeprefix(prefix),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _final_metric_summary(final_frame: Any) -> Any:
    import pandas as pd

    absolute = final_frame.groupby(["training_regime", "metric"], as_index=False).agg(
        mean=("value", "mean"), std=("value", "std"), count=("value", "count")
    )
    paired = final_frame.pivot_table(
        index=["training_seed", "metric"],
        columns="training_regime",
        values="value",
        aggfunc="last",
    ).reset_index()
    paired["value"] = paired["shortcut"] - paired["control"]
    paired_summary = paired.groupby("metric", as_index=False).agg(
        mean=("value", "mean"),
        std=("value", "std"),
        count=("value", "count"),
    )
    paired_summary.insert(0, "training_regime", "shortcut_minus_control")
    return pd.concat([absolute, paired_summary], ignore_index=True)


def _overlap_rows(history: Any, *, target_view: str, layer: str, regime: str) -> Any:
    return history.loc[
        (history["status"] == "success")
        & (history["target_view"] == target_view)
        & (history["output_name"] == layer)
        & (history["training_regime"] == regime)
    ].sort_values("epoch")


def _series_summary(rows: Any, value_column: str) -> Any:
    summary = rows.groupby("epoch")[value_column].agg(["mean", "std", "count"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def _plot_mean_with_spread(
    axis: Any,
    summary: Any,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    marker: Optional[str] = "o",
) -> None:
    axis.plot(
        summary["epoch"],
        summary["mean"],
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=3.5,
        linewidth=2,
        label=label,
    )
    if int(summary["count"].max()) > 1:
        lower = summary["mean"] - summary["std"]
        upper = summary["mean"] + summary["std"]
        axis.fill_between(summary["epoch"], lower, upper, color=color, alpha=0.14, linewidth=0)


def _plot_shortcut_experiment(
    history: Any,
    final_frame: Any,
    figure_dir: Path,
    plt: Any,
) -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 11.0))
    figure.suptitle(
        "A decorrelated audit reveals harmful shortcut organization",
        x=0.06,
        ha="left",
        fontsize=17,
        fontweight="semibold",
    )
    for row, (target_view, row_label) in enumerate(
        (("intended_class", "Intended garment class"), ("nuisance_color", "Nuisance color"))
    ):
        for column, layer in enumerate(_LAYER_ORDER):
            axis = axes[row, column]
            axis.set_title(_LAYER_LABELS[layer], loc="left")
            for regime in _REGIMES:
                rows = _overlap_rows(
                    history,
                    target_view=target_view,
                    layer=layer,
                    regime=regime,
                )
                _plot_mean_with_spread(
                    axis,
                    _series_summary(rows, "overlap_macro"),
                    color=_REGIME_COLORS[regime],
                    label=_REGIME_LABELS[regime],
                )
            axis.set_ylim(-0.02, 1.02)
            axis.grid(axis="y", alpha=0.3)
            if column == 0:
                axis.set_ylabel(f"{row_label}\nOI macro")
                axis.legend(frameon=False, fontsize=9)
            if row == 1:
                axis.set_xlabel("Training epoch")

    metrics = _metric_rows(history)
    accuracy_axis = axes[2, 0]
    for regime in _REGIMES:
        for environment, linestyle in (("correlated", "-"), ("balanced", "--")):
            _plot_mean_with_spread(
                accuracy_axis,
                _series_summary(metrics, f"{regime}_{environment}_accuracy"),
                color=_REGIME_COLORS[regime],
                linestyle=linestyle,
                marker=None,
                label=f"{_REGIME_LABELS[regime]} — {environment}",
            )
    accuracy_axis.set_title("Expected vs decorrelated accuracy", loc="left")
    accuracy_axis.set_xlabel("Training epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(-0.02, 1.02)
    accuracy_axis.grid(axis="y", alpha=0.3)
    accuracy_axis.legend(frameon=False, fontsize=8)

    gap_axis = axes[2, 1]
    for regime in _REGIMES:
        _plot_mean_with_spread(
            gap_axis,
            _series_summary(metrics, f"{regime}_balanced_gap"),
            color=_REGIME_COLORS[regime],
            label=_REGIME_LABELS[regime],
        )
    gap_axis.axhline(0, color="#64748B", linewidth=1)
    gap_axis.set_title("Shortcut harm: ID − balanced", loc="left")
    gap_axis.set_xlabel("Training epoch")
    gap_axis.set_ylabel("Accuracy gap")
    gap_axis.grid(axis="y", alpha=0.3)
    gap_axis.legend(frameon=False, fontsize=8)

    final_axis = axes[2, 2]
    positions = np.arange(len(_ENVIRONMENTS), dtype=float)
    width = 0.36
    for offset, regime in zip((-width / 2, width / 2), _REGIMES):
        means = []
        errors = []
        for environment in _ENVIRONMENTS:
            values = final_frame.loc[
                (final_frame["training_regime"] == regime)
                & (final_frame["metric"] == f"{environment}_accuracy"),
                "value",
            ]
            means.append(float(values.mean()))
            errors.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        final_axis.bar(
            positions + offset,
            means,
            width,
            color=_REGIME_COLORS[regime],
            label=_REGIME_LABELS[regime],
            yerr=errors,
            capsize=3,
        )
    final_axis.set_title("Final environment stress test", loc="left")
    final_axis.set_xticks(positions, ("ID", "Balanced", "Reversed", "Grayscale"), rotation=20)
    final_axis.set_ylabel("Accuracy")
    final_axis.set_ylim(0, 1)
    final_axis.grid(axis="y", alpha=0.3)
    final_axis.legend(frameon=False, fontsize=8)

    figure.text(
        0.06,
        0.94,
        "OI uses the balanced class×color probe; accuracy supplies the independent intervention.",
        color="#475569",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.0, w_pad=1.2)
    path = figure_dir / "colored_fashion_mnist_shortcut_monitoring.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _paired_overlap_effects(history: Any, *, target_view: str, layer: str) -> Any:
    rows = history.loc[
        (history["status"] == "success")
        & (history["target_view"] == target_view)
        & (history["output_name"] == layer),
        ["training_seed", "epoch", "training_regime", "overlap_macro"],
    ]
    paired = rows.pivot_table(
        index=["training_seed", "epoch"],
        columns="training_regime",
        values="overlap_macro",
        aggfunc="last",
    ).reset_index()
    paired["treatment_minus_control"] = paired["shortcut"] - paired["control"]
    return _series_summary(paired, "treatment_minus_control")


def _plot_paired_effects(history: Any, figure_dir: Path, plt: Any) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 7.5), sharex=True)
    figure.suptitle(
        "Paired representation effects isolate the color-correlation treatment",
        x=0.06,
        ha="left",
        fontsize=17,
        fontweight="semibold",
    )
    for row, (target_view, label, color) in enumerate(
        (
            ("intended_class", "Intended garment OI", "#2563EB"),
            ("nuisance_color", "Nuisance color OI", "#E11D48"),
        )
    ):
        for column, layer in enumerate(_LAYER_ORDER):
            axis = axes[row, column]
            summary = _paired_overlap_effects(
                history,
                target_view=target_view,
                layer=layer,
            )
            _plot_mean_with_spread(
                axis,
                summary,
                color=color,
                label="Shortcut − control",
            )
            axis.axhline(0, color="#64748B", linewidth=1)
            axis.set_title(_LAYER_LABELS[layer], loc="left")
            axis.grid(axis="y", alpha=0.3)
            if column == 0:
                axis.set_ylabel(f"{label}\npaired difference")
                axis.legend(frameon=False, fontsize=9)
            if row == 1:
                axis.set_xlabel("Training epoch")
    figure.text(
        0.06,
        0.925,
        "Negative intended effects indicate weaker semantic organization; "
        "positive nuisance effects indicate retained color structure.",
        color="#475569",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.0, w_pad=1.2)
    path = figure_dir / "colored_fashion_mnist_shortcut_paired_effects.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_environment_exemplars(
    grayscale_images: np.ndarray,
    labels: np.ndarray,
    *,
    opacity: float,
    seed: int,
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    selected_classes = (0, 1, 4, 5, 8, 9)
    indices = [int(rng.choice(np.flatnonzero(labels == label))) for label in selected_classes]
    figure, axes = plt.subplots(len(indices), 4, figsize=(12.0, 13.0))
    column_titles = (
        "Color removed",
        "Canonical\n(correlated)",
        "Independent\ncounterexample",
        "Reversed\nmapping",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11, fontweight="semibold", pad=10)
    for row, (index, label) in enumerate(zip(indices, selected_classes)):
        image = grayscale_images[index : index + 1]
        colors = (label, (label + 2) % len(_COLOR_NAMES), (label + 5) % len(_COLOR_NAMES))
        rendered = [
            _grayscale_rgb(image),
            *[
                _colorize(
                    image,
                    np.asarray([color], dtype=np.int64),
                    opacity=opacity,
                )
                for color in colors
            ],
        ]
        for column, values in enumerate(rendered):
            rgb = values.reshape(3, image.shape[1], image.shape[2]).transpose(1, 2, 0)
            axes[row, column].imshow(np.clip(rgb, 0.0, 1.0))
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(
            _FASHION_CLASS_NAMES[label],
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
            fontsize=9.5,
        )
        axes[row, 1].set_xlabel(_COLOR_NAMES[colors[0]], fontsize=8.5)
        axes[row, 2].set_xlabel(_COLOR_NAMES[colors[1]], fontsize=8.5)
        axes[row, 3].set_xlabel(_COLOR_NAMES[colors[2]], fontsize=8.5)
    figure.suptitle(
        "The intervention changes color while preserving garment evidence",
        x=0.08,
        ha="left",
        fontsize=17,
        fontweight="semibold",
    )
    figure.text(
        0.08,
        0.945,
        "The balanced OI audit uses one independent tint per image; "
        "behavioral tests recolor every image with all ten hues.",
        color="#475569",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.07, 0, 1, 0.92), h_pad=1.1, w_pad=0.8)
    png_path = figure_dir / "colored_fashion_mnist_shortcut_exemplars.png"
    svg_path = png_path.with_suffix(".svg")
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


if __name__ == "__main__":
    main()
