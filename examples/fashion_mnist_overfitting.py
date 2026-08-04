"""Compare clean and noisy-label CNN representations during Fashion-MNIST training.

Two compact CNNs start from identical weights and receive the same images in the same
mini-batch order. The clean-label control trains on the original targets, while the
overfitting treatment receives deterministic symmetric label corruption. Both models
are monitored on the same clean validation probe and on the same corrupted-target probe.

The paired plot shows whether representation geometry changes beyond ordinary clean
training and whether a layer begins organizing the deliberately corrupted examples by
their incorrect targets. A shaded region starts at the noisy model's minimum clean
validation loss, so overfitting is defined independently of OverlapIndex.

Install the optional dependencies and run from the repository root:

    poetry install -E visuals
    poetry run python examples/fashion_mnist_overfitting.py

Fashion-MNIST is downloaded through torchvision on the first run and reused from the
local data directory afterward.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

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
_REGIME_COLORS = {
    "clean": "#0F766E",
    "noisy": "#E11D48",
}
_REGIME_LABELS = {
    "clean": "Clean-label control",
    "noisy": "Noisy-label treatment",
}
_REGIME_MARKERS = {"clean": "o", "noisy": "D"}
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
_NORMALIZATION_MEAN = 0.2860
_NORMALIZATION_STD = 0.3530


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=1_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--label-noise", type=float, default=0.40)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--monitor-every-epochs",
        type=int,
        default=1,
        help="Evaluate both representation probes after this many completed epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("examples/data"),
        help="Directory used by torchvision for the Fashion-MNIST download/cache.",
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
        help="Require Fashion-MNIST to already exist under --data-dir.",
    )
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

    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_x, clean_train_y, validation_x, validation_y = _load_fashion_mnist(
        FashionMNIST,
        data_dir=args.data_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        seed=args.seed,
        download=not args.no_download,
    )
    observed_train_y, corrupted_mask = _corrupt_labels(
        clean_train_y,
        noise_rate=args.label_noise,
        n_classes=len(_FASHION_CLASS_NAMES),
        seed=args.seed + 101,
    )
    corruption_probe = _monitoring_dataset(
        train_x[corrupted_mask],
        observed_train_y[corrupted_mask],
        split="corrupted_target_probe",
        subset_seed=args.seed,
        label_noise_seed=args.seed + 101,
        label_noise=args.label_noise,
        corrupted_count=int(corrupted_mask.sum()),
    )
    validation_dataset = _monitoring_dataset(
        validation_x,
        validation_y,
        split="clean_held_out_validation",
        subset_seed=args.seed + 1,
        label_noise_seed=None,
        label_noise=0.0,
        corrupted_count=0,
    )

    clean_model, noisy_model = _build_paired_models(torch, seed=args.seed)
    clean_optimizer = torch.optim.Adam(clean_model.parameters(), lr=0.001, weight_decay=0.0)
    noisy_optimizer = torch.optim.Adam(noisy_model.parameters(), lr=0.001, weight_decay=0.0)
    loss_fn = torch.nn.CrossEntropyLoss()
    clean_extractor = _multi_output_extractor(
        clean_model,
        torch,
        args.seed,
        training_regime="clean",
        label_noise=0.0,
    )
    noisy_extractor = _multi_output_extractor(
        noisy_model,
        torch,
        args.seed,
        training_regime="noisy",
        label_noise=args.label_noise,
    )
    scoring_config = _scoring_config(args.seed)
    monitor_options = {
        "scoring_config": scoring_config,
        "embedding_config": EmbeddingConfig(batch_size=args.embedding_batch_size),
        "stability_config": StabilityConfig(enabled=False),
        "separatix_config": SeparatixConfig(enabled=False),
    }
    monitors = {
        ("clean", "validation"): _monitor(
            validation_dataset,
            clean_extractor,
            monitor_options,
        ),
        ("noisy", "validation"): _monitor(
            validation_dataset,
            noisy_extractor,
            monitor_options,
        ),
        ("clean", "corruption"): _monitor(
            corruption_probe,
            clean_extractor,
            monitor_options,
        ),
        ("noisy", "corruption"): _monitor(
            corruption_probe,
            noisy_extractor,
            monitor_options,
        ),
    }

    global_step = 0
    initial_metrics = _checkpoint_metrics(
        clean_model,
        noisy_model,
        train_x,
        clean_train_y,
        observed_train_y,
        validation_x,
        validation_y,
        loss_fn,
        batch_size=args.embedding_batch_size,
        torch=torch,
    )
    _evaluate_monitors(
        monitors,
        epoch=0,
        global_step=global_step,
        snapshot_id="initialization",
        metrics=initial_metrics,
    )

    for epoch in range(1, args.epochs + 1):
        global_step += _train_paired_epoch(
            clean_model,
            noisy_model,
            clean_optimizer,
            noisy_optimizer,
            loss_fn,
            train_x,
            clean_train_y,
            observed_train_y,
            batch_size=args.train_batch_size,
            seed=args.seed + epoch,
            torch=torch,
        )
        if epoch % args.monitor_every_epochs != 0 and epoch != args.epochs:
            continue
        metrics = _checkpoint_metrics(
            clean_model,
            noisy_model,
            train_x,
            clean_train_y,
            observed_train_y,
            validation_x,
            validation_y,
            loss_fn,
            batch_size=args.embedding_batch_size,
            torch=torch,
        )
        _evaluate_monitors(
            monitors,
            epoch=epoch,
            global_step=global_step,
            snapshot_id=f"epoch-{epoch:03d}",
            metrics=metrics,
        )
        print(
            f"Epoch {epoch:>3}/{args.epochs}: "
            f"clean val loss={metrics['clean_validation_loss']:.3f}, "
            f"noisy val loss={metrics['noisy_validation_loss']:.3f}, "
            f"noisy val accuracy={metrics['noisy_validation_accuracy']:.3f}"
        )

    output_dir = ensure_output_dir()
    figure_dir = args.figure_dir or output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    histories = {key: monitor.history.to_dataframe() for key, monitor in monitors.items()}
    history_path = output_dir / "fashion_mnist_paired_overfitting_history.csv"
    _combined_history(histories).to_csv(history_path, index=False)
    figure_paths = _plot_paired_overfitting(
        histories,
        noise_rate=args.label_noise,
        figure_dir=figure_dir,
        plt=plt,
    )

    metric_rows = _metric_rows(histories[("noisy", "validation")])
    noisy_best_epoch = _best_validation_epoch(
        metric_rows["epoch"],
        metric_rows["context_metadata.noisy_validation_loss"],
    )
    clean_best_epoch = _best_validation_epoch(
        metric_rows["epoch"],
        metric_rows["context_metadata.clean_validation_loss"],
    )
    print(f"Clean control's best validation loss occurred at epoch {clean_best_epoch}.")
    print(f"Noisy treatment's best validation loss occurred at epoch {noisy_best_epoch}.")
    print(f"Corrupted {int(corrupted_mask.sum())} of {len(observed_train_y)} training labels.")
    print(f"Wrote {history_path}")
    for path in figure_paths:
        print(f"Wrote {path}")


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in (
        "epochs",
        "train_size",
        "validation_size",
        "train_batch_size",
        "embedding_batch_size",
        "monitor_every_epochs",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if not np.isfinite(args.label_noise) or not 0.0 < args.label_noise < 1.0:
        parser.error("--label-noise must be finite and in (0, 1)")


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


def _corrupt_labels(
    labels: np.ndarray,
    *,
    noise_rate: float,
    n_classes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")
    if not np.isfinite(noise_rate) or not 0.0 <= noise_rate < 1.0:
        raise ValueError("noise_rate must be finite and in [0, 1).")
    if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
        raise ValueError("labels must be integer class ids in [0, n_classes).")

    rng = np.random.default_rng(seed)
    corrupted = labels.copy()
    mask = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        count = int(np.floor(noise_rate * len(class_indices) + 0.5))
        if count == 0:
            continue
        selected = rng.choice(class_indices, size=count, replace=False)
        offsets = rng.integers(1, n_classes, size=count)
        corrupted[selected] = (corrupted[selected] + offsets) % n_classes
        mask[selected] = True
    return corrupted, mask


def _normalize_fashion_mnist(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(len(values), -1) / 255.0
    return ((flattened - _NORMALIZATION_MEAN) / _NORMALIZATION_STD).astype(
        np.float32,
        copy=False,
    )


def _monitoring_dataset(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
    subset_seed: int,
    label_noise_seed: Optional[int],
    label_noise: float,
    corrupted_count: int,
) -> BenchmarkDataset:
    identity = DatasetIdentity.from_manifest(
        f"torchvision-fashion-mnist-{split}",
        {
            "source_split": "train",
            "role": split,
            "subset": "stratified",
            "sample_count": int(len(labels)),
            "subset_seed": int(subset_seed),
            "label_noise_seed": label_noise_seed,
            "label_noise": float(label_noise),
            "corrupted_count": int(corrupted_count),
            "normalization": {
                "mean": _NORMALIZATION_MEAN,
                "std": _NORMALIZATION_STD,
            },
        },
    )
    return BenchmarkDataset.from_arrays(
        values,
        np.asarray([_FASHION_CLASS_NAMES[int(label)] for label in labels]),
        modality="image",
        metadata={
            "example": "fashion_mnist_overfitting",
            "dataset_source": "torchvision.datasets.FashionMNIST",
            "split": split,
            "label_noise": float(label_noise),
            "label_noise_seed": label_noise_seed,
            "corrupted_count": int(corrupted_count),
        },
        identity=identity,
    )


def _build_model(torch: Any) -> Any:
    class FashionMNISTOverfitClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
            )
            self.conv_2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
            )
            self.embedding = torch.nn.Linear(64 * 7 * 7, 128)
            self.classifier = torch.nn.Linear(128, len(_FASHION_CLASS_NAMES))

        def forward(self, values: Any) -> Dict[str, Any]:
            images = values.reshape(-1, 1, 28, 28)
            conv_1_maps = self.conv_1(images)
            conv_2_maps = self.conv_2(conv_1_maps)
            conv_1 = torch.nn.functional.adaptive_avg_pool2d(
                conv_1_maps,
                output_size=(2, 2),
            ).flatten(start_dim=1)
            conv_2 = torch.nn.functional.adaptive_avg_pool2d(
                conv_2_maps,
                output_size=(2, 2),
            ).flatten(start_dim=1)
            embedding = torch.relu(self.embedding(conv_2_maps.flatten(start_dim=1)))
            return {
                "conv_1": conv_1,
                "conv_2": conv_2,
                "embedding": embedding,
                "logits": self.classifier(embedding),
            }

    return FashionMNISTOverfitClassifier()


def _build_paired_models(torch: Any, *, seed: int) -> Tuple[Any, Any]:
    torch.manual_seed(seed)
    clean_model = _build_model(torch)
    noisy_model = _build_model(torch)
    noisy_model.load_state_dict(clean_model.state_dict())
    return clean_model, noisy_model


def _multi_output_extractor(
    model: Any,
    torch: Any,
    seed: int,
    *,
    training_regime: str,
    label_noise: float,
) -> TorchExtractor:
    if training_regime not in _REGIME_LABELS:
        raise ValueError("training_regime must be either 'clean' or 'noisy'.")

    def collate_fn(batch: Any) -> Any:
        return torch.as_tensor(np.asarray(batch), dtype=torch.float32)

    def output_fn(raw_output: Dict[str, Any]) -> Dict[str, Any]:
        return {name: raw_output[name] for name in _LAYER_ORDER}

    return TorchExtractor(
        name=f"fashion_mnist_{training_regime}_cnn",
        model=model,
        collate_fn=collate_fn,
        output_fn=output_fn,
        outputs=_OUTPUT_SPECS,
        device="cpu",
        modality="image",
        recipe_data={
            "example": "fashion_mnist_overfitting",
            "architecture": {
                "input": [1, 28, 28],
                "conv_1": [32, 14, 14],
                "conv_2": [64, 7, 7],
                "embedding": 128,
                "classes": len(_FASHION_CLASS_NAMES),
                "dropout": 0.0,
            },
            "training_seed": seed,
            "training_regime": training_regime,
            "label_noise": float(label_noise),
        },
    )


def _scoring_config(seed: int) -> OverlapScoringConfig:
    return OverlapScoringConfig(
        k=5,
        min_samples_per_cluster=5,
        kmeans_kwargs={"random_state": seed, "batch_size": 512, "n_init": 3},
    )


def _monitor(
    dataset: BenchmarkDataset,
    extractor: TorchExtractor,
    monitor_options: Dict[str, Any],
) -> RepresentationMonitor:
    return RepresentationMonitor(
        dataset,
        [extractor],
        history_config=EvaluationHistoryConfig(storage="memory", detail="summary"),
        **monitor_options,
    )


def _train_paired_epoch(
    clean_model: Any,
    noisy_model: Any,
    clean_optimizer: Any,
    noisy_optimizer: Any,
    loss_fn: Any,
    values: np.ndarray,
    clean_labels: np.ndarray,
    observed_labels: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    torch: Any,
) -> int:
    clean_model.train()
    noisy_model.train()
    order = np.random.default_rng(seed).permutation(len(clean_labels))
    batch_count = 0
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch_x = torch.as_tensor(values[indices], dtype=torch.float32)
        for model, optimizer, labels in (
            (clean_model, clean_optimizer, clean_labels),
            (noisy_model, noisy_optimizer, observed_labels),
        ):
            batch_y = torch.as_tensor(labels[indices], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(batch_x)["logits"]
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
        batch_count += 1
    return batch_count


def _checkpoint_metrics(
    clean_model: Any,
    noisy_model: Any,
    train_x: np.ndarray,
    clean_train_y: np.ndarray,
    observed_train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    loss_fn: Any,
    *,
    batch_size: int,
    torch: Any,
) -> Dict[str, float]:
    clean_training_loss, clean_training_accuracy = _loss_accuracy(
        clean_model,
        train_x,
        clean_train_y,
        loss_fn,
        batch_size=batch_size,
        torch=torch,
    )
    clean_validation_loss, clean_validation_accuracy = _loss_accuracy(
        clean_model,
        validation_x,
        validation_y,
        loss_fn,
        batch_size=batch_size,
        torch=torch,
    )
    noisy_training_loss, noisy_training_accuracy = _loss_accuracy(
        noisy_model,
        train_x,
        observed_train_y,
        loss_fn,
        batch_size=batch_size,
        torch=torch,
    )
    _, noisy_training_accuracy_clean = _loss_accuracy(
        noisy_model,
        train_x,
        clean_train_y,
        loss_fn,
        batch_size=batch_size,
        torch=torch,
    )
    noisy_validation_loss, noisy_validation_accuracy = _loss_accuracy(
        noisy_model,
        validation_x,
        validation_y,
        loss_fn,
        batch_size=batch_size,
        torch=torch,
    )
    return {
        "clean_training_loss": clean_training_loss,
        "clean_training_accuracy": clean_training_accuracy,
        "clean_validation_loss": clean_validation_loss,
        "clean_validation_accuracy": clean_validation_accuracy,
        "noisy_training_loss_observed": noisy_training_loss,
        "noisy_training_accuracy_observed": noisy_training_accuracy,
        "noisy_training_accuracy_clean": noisy_training_accuracy_clean,
        "noisy_validation_loss": noisy_validation_loss,
        "noisy_validation_accuracy": noisy_validation_accuracy,
        "clean_accuracy_generalization_gap": (
            clean_training_accuracy - clean_validation_accuracy
        ),
        "noisy_accuracy_generalization_gap": (
            noisy_training_accuracy - noisy_validation_accuracy
        ),
    }


def _loss_accuracy(
    model: Any,
    values: np.ndarray,
    labels: np.ndarray,
    loss_fn: Any,
    *,
    batch_size: int,
    torch: Any,
) -> Tuple[float, float]:
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            batch_y = torch.as_tensor(labels[start:stop], dtype=torch.long)
            logits = model(torch.as_tensor(values[start:stop], dtype=torch.float32))["logits"]
            total_loss += float(loss_fn(logits, batch_y)) * (stop - start)
            correct += int((logits.argmax(dim=1) == batch_y).sum())
    model.train(previous_mode)
    return total_loss / len(labels), correct / len(labels)


def _evaluate_monitors(
    monitors: Dict[Tuple[str, str], RepresentationMonitor],
    *,
    epoch: int,
    global_step: int,
    snapshot_id: str,
    metrics: Dict[str, float],
) -> None:
    context = {
        "snapshot_id": snapshot_id,
        "epoch": epoch,
        "global_step": global_step,
        "metadata": metrics,
    }
    for monitor in monitors.values():
        monitor.evaluate(**context)


def _combined_history(histories: Dict[Tuple[str, str], Any]) -> Any:
    import pandas as pd

    frames = []
    for (training_regime, probe), history in histories.items():
        frame = history.copy()
        frame.insert(0, "probe", probe)
        frame.insert(0, "training_regime", training_regime)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _metric_rows(history: Any) -> Any:
    successful = history.loc[history["status"] == "success"].copy()
    columns = [
        "epoch",
        "global_step",
        "context_metadata.clean_training_loss",
        "context_metadata.clean_training_accuracy",
        "context_metadata.clean_validation_loss",
        "context_metadata.clean_validation_accuracy",
        "context_metadata.noisy_training_loss_observed",
        "context_metadata.noisy_training_accuracy_observed",
        "context_metadata.noisy_training_accuracy_clean",
        "context_metadata.noisy_validation_loss",
        "context_metadata.noisy_validation_accuracy",
    ]
    return (
        successful.loc[:, columns]
        .drop_duplicates(subset="epoch", keep="last")
        .sort_values("epoch")
        .reset_index(drop=True)
    )


def _overlap_pivot(history: Any) -> Any:
    successful = history.loc[history["status"] == "success"].copy()
    return (
        successful.pivot_table(
            index="epoch",
            columns="output_name",
            values="overlap_macro",
            aggfunc="last",
        )
        .sort_index()
        .reindex(columns=_LAYER_ORDER)
    )


def _best_validation_epoch(epochs: Iterable[int], losses: Iterable[float]) -> int:
    epoch_values = np.asarray(list(epochs))
    loss_values = np.asarray(list(losses), dtype=float)
    if epoch_values.ndim != 1 or loss_values.ndim != 1 or len(epoch_values) != len(loss_values):
        raise ValueError("epochs and losses must be one-dimensional sequences of equal length.")
    if len(epoch_values) == 0:
        raise ValueError("At least one validation loss is required.")
    if not np.all(np.isfinite(loss_values)):
        raise ValueError("Validation losses must be finite.")
    return int(epoch_values[int(np.argmin(loss_values))])


def _plot_paired_overfitting(
    histories: Dict[Tuple[str, str], Any],
    *,
    noise_rate: float,
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    _apply_plot_style(plt)
    pivots = {key: _overlap_pivot(history) for key, history in histories.items()}
    metrics = _metric_rows(histories[("noisy", "validation")])
    best_epoch = _best_validation_epoch(
        metrics["epoch"],
        metrics["context_metadata.noisy_validation_loss"],
    )
    last_epoch = int(metrics["epoch"].max())

    figure = plt.figure(figsize=(16.0, 11.0))
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.9),
        left=0.075,
        right=0.97,
        top=0.82,
        bottom=0.13,
        hspace=0.68,
        wspace=0.22,
    )
    validation_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    corruption_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]
    loss_axis = figure.add_subplot(grid[2, :2])
    accuracy_axis = figure.add_subplot(grid[2, 2])
    all_axes = (*validation_axes, *corruption_axes, loss_axis, accuracy_axis)
    figure.suptitle(
        "A clean-label control reveals where noisy-target memorization changes representations",
        fontsize=18,
        fontweight="semibold",
        x=0.075,
        ha="left",
        y=0.965,
    )
    figure.text(
        0.075,
        0.915,
        "Identical initialization, images, mini-batch order, optimizer, and schedule; "
        "only the training targets differ",
        color="#475569",
        fontsize=11,
    )
    figure.text(
        0.075,
        0.855,
        "Clean validation probe — transferable class geometry",
        color="#0F172A",
        fontsize=12,
        fontweight="semibold",
    )
    figure.text(
        0.075,
        0.585,
        "Corrupted-target probe — alignment with deliberately incorrect labels",
        color="#0F172A",
        fontsize=12,
        fontweight="semibold",
    )

    for column, output_name in enumerate(_LAYER_ORDER):
        validation_axis = validation_axes[column]
        corruption_axis = corruption_axes[column]
        validation_axis.set_title(_LAYER_LABELS[output_name], loc="left", pad=9)
        for regime in ("clean", "noisy"):
            validation_pivot = pivots[(regime, "validation")]
            corruption_pivot = pivots[(regime, "corruption")]
            plot_options = {
                "color": _REGIME_COLORS[regime],
                "marker": _REGIME_MARKERS[regime],
                "markersize": 4.5,
                "linewidth": 2.2,
                "label": _REGIME_LABELS[regime],
            }
            validation_axis.plot(
                validation_pivot.index,
                validation_pivot[output_name],
                **plot_options,
            )
            corruption_axis.plot(
                corruption_pivot.index,
                corruption_pivot[output_name],
                **plot_options,
            )
        for axis in (validation_axis, corruption_axis):
            axis.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
        if column == 0:
            validation_axis.set_ylabel("OverlapIndex macro score")
            corruption_axis.set_ylabel("OverlapIndex macro score")
            validation_axis.legend(frameon=False, loc="best")
        corruption_axis.set_xlabel("Training epoch")

    validation_limits = _overlap_limits(
        pivots[("clean", "validation")],
        pivots[("noisy", "validation")],
    )
    corruption_limits = _overlap_limits(
        pivots[("clean", "corruption")],
        pivots[("noisy", "corruption")],
    )
    for axis in validation_axes:
        axis.set_ylim(*validation_limits)
    for axis in corruption_axes:
        axis.set_ylim(*corruption_limits)

    loss_series = (
        ("context_metadata.clean_training_loss", "clean", "--", "Clean training"),
        ("context_metadata.clean_validation_loss", "clean", "-", "Clean validation"),
        (
            "context_metadata.noisy_training_loss_observed",
            "noisy",
            "--",
            "Noisy training (observed labels)",
        ),
        (
            "context_metadata.noisy_validation_loss",
            "noisy",
            "-",
            "Noisy validation (clean labels)",
        ),
    )
    for metric_column, regime, linestyle, label in loss_series:
        loss_axis.plot(
            metrics["epoch"],
            metrics[metric_column],
            color=_REGIME_COLORS[regime],
            linestyle=linestyle,
            linewidth=2.3,
            marker=_REGIME_MARKERS[regime],
            markersize=4,
            label=label,
        )
    loss_axis.set_title(
        "Cross-entropy defines the noisy treatment's overfitting region",
        loc="left",
    )
    loss_axis.set_xlabel("Training epoch (0 = initialization)")
    loss_axis.set_ylabel("Cross-entropy")
    loss_axis.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    loss_axis.legend(frameon=False, loc="best", ncol=2)

    accuracy_series = (
        ("context_metadata.clean_training_accuracy", "clean", "--", "Clean train"),
        ("context_metadata.clean_validation_accuracy", "clean", "-", "Clean validation"),
        (
            "context_metadata.noisy_training_accuracy_observed",
            "noisy",
            "--",
            "Noisy train",
        ),
        (
            "context_metadata.noisy_validation_accuracy",
            "noisy",
            "-",
            "Noisy validation",
        ),
    )
    for metric_column, regime, linestyle, label in accuracy_series:
        accuracy_axis.plot(
            metrics["epoch"],
            metrics[metric_column],
            color=_REGIME_COLORS[regime],
            linestyle=linestyle,
            linewidth=2.3,
            marker=_REGIME_MARKERS[regime],
            markersize=4,
            label=label,
        )
    accuracy_axis.set_title("Accuracy", loc="left")
    accuracy_axis.set_xlabel("Training epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.02)
    accuracy_axis.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    accuracy_axis.legend(frameon=False, loc="best", fontsize=9)

    for axis in all_axes:
        axis.set_xlim(float(metrics["epoch"].min()), float(last_epoch))
        _mark_post_best_region(axis, best_epoch=best_epoch, last_epoch=last_epoch)
    validation_axes[-1].text(
        best_epoch,
        0.98,
        f" noisy best validation loss (epoch {best_epoch}) ",
        color="#991B1B",
        fontsize=9,
        ha="right" if best_epoch >= last_epoch / 2 else "left",
        va="top",
        transform=validation_axes[-1].get_xaxis_transform(),
    )
    figure.text(
        0.075,
        0.055,
        f"{noise_rate:.0%} deterministic symmetric label corruption  •  fixed k=5  •  "
        "both models use the same clean and corrupted-target probes  •  shaded region "
        "begins at the noisy model's minimum clean validation loss",
        color="#475569",
        fontsize=10,
    )
    return _save_figure(
        figure,
        figure_dir,
        "fashion-mnist-paired-overfitting-monitoring",
        plt,
    )


def _overlap_limits(*pivots: Any) -> Tuple[float, float]:
    values = np.concatenate([pivot.to_numpy(dtype=float).ravel() for pivot in pivots])
    values = values[np.isfinite(values)]
    lower = max(0.0, float(values.min()) - 0.06)
    upper = min(1.01, float(values.max()) + 0.06)
    if upper <= lower:
        upper = min(1.01, lower + 0.1)
    return lower, upper


def _mark_post_best_region(axis: Any, *, best_epoch: int, last_epoch: int) -> None:
    axis.axvline(best_epoch, color="#B91C1C", linestyle="--", linewidth=1.4, zorder=1)
    if best_epoch < last_epoch:
        axis.axvspan(best_epoch, last_epoch, color="#FEE2E2", alpha=0.48, zorder=0)


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
