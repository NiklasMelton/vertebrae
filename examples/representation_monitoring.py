"""Monitor two representations from a small local Torch model during training.

This example is network-free and requires optional Torch support:

    poetry install -E torch

Start a new four-epoch run:

    poetry run python examples/representation_monitoring.py

Continue a caller-managed checkpoint/history pair to a larger total epoch count:

    poetry run python examples/representation_monitoring.py --resume --epochs 8
"""

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from _common import ensure_output_dir

from vertebrae import (
    BenchmarkDataset,
    ConsoleReporter,
    DatasetIdentity,
    EvaluationHistoryConfig,
    OverlapScoringConfig,
    RepresentationMonitor,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.extractors import TorchExtractor


def _make_data(seed: int, samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=-1.0, scale=0.8, size=(samples_per_class, 4))
    right = rng.normal(loc=1.0, scale=0.8, size=(samples_per_class, 4))
    features = np.vstack([left, right]).astype(np.float32)
    labels = np.concatenate(
        [
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(samples_per_class, dtype=np.int64),
        ]
    )
    order = rng.permutation(len(labels))
    return features[order], labels[order]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Total number of epochs the run should contain.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore the example training state and append to its matching JSONL history.",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        print(exc)
        print("Install optional PyTorch support with: poetry install -E torch")
        return

    torch.manual_seed(17)
    train_x, train_y = _make_data(seed=17, samples_per_class=80)
    probe_x, probe_y = _make_data(seed=29, samples_per_class=30)

    class TinyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = nn.Linear(4, 8)
            self.embedding = nn.Linear(8, 3)
            self.classifier = nn.Linear(3, 2)

        def forward(self, values):
            hidden = torch.relu(self.hidden(values))
            embedding = torch.tanh(self.embedding(hidden))
            logits = self.classifier(embedding)
            return {
                "hidden": hidden,
                "embedding": embedding,
                "logits": logits,
            }

    model = TinyClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()
    output_dir = ensure_output_dir()
    history_path = Path(output_dir) / "representation_monitoring.jsonl"
    state_path = Path(output_dir) / "representation_monitoring_state.pt"
    start_epoch = 0
    global_step = 0
    restored_state: Optional[dict[str, Any]] = None
    if args.resume:
        missing = [path for path in (history_path, state_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Resume requires both history and training state; missing: "
                + ", ".join(str(path) for path in missing)
            )
        restored = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(restored, dict):
            raise ValueError("The saved training state must contain a mapping.")
        required = {
            "model_state",
            "optimizer_state",
            "completed_epoch",
            "next_epoch",
            "global_step",
            "protocol_hash",
        }
        missing_fields = sorted(required - set(restored))
        if missing_fields:
            raise ValueError(f"The saved training state is incomplete: {missing_fields}.")
        completed_epoch = restored["completed_epoch"]
        start_epoch = restored["next_epoch"]
        global_step = restored["global_step"]
        if (
            isinstance(completed_epoch, bool)
            or not isinstance(completed_epoch, int)
            or isinstance(start_epoch, bool)
            or not isinstance(start_epoch, int)
            or isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or start_epoch != completed_epoch + 1
            or global_step < 0
        ):
            raise ValueError("The saved training coordinates are invalid.")
        model.load_state_dict(restored["model_state"])
        optimizer.load_state_dict(restored["optimizer_state"])
        restored_state = restored
    elif history_path.exists() or state_path.exists():
        raise FileExistsError(
            "The monitoring example already has history or training state. "
            "Use --resume to continue it or choose a fresh VERTABRAE_EXAMPLE_OUTPUT_DIR."
        )

    probe_dataset = BenchmarkDataset.from_arrays(
        probe_x,
        probe_y,
        modality="tabular",
        identity=DatasetIdentity.declared("monitoring-example-probe", "1"),
        metadata={"split": "fixed_probe"},
    )

    def collate_fn(batch):
        return torch.as_tensor(np.asarray(batch), dtype=torch.float32)

    def output_fn(raw_output):
        return {
            "hidden": raw_output["hidden"],
            "embedding": raw_output["embedding"],
        }

    extractor = TorchExtractor(
        name="live_classifier",
        model=model,
        collate_fn=collate_fn,
        output_fn=output_fn,
        outputs=[
            {
                "name": "hidden",
                "hidden_layer": 1,
                "pooling": "identity",
            },
            {
                "name": "embedding",
                "hidden_layer": 2,
                "pooling": "identity",
            },
        ],
        device="cpu",
        recipe_data={"example": "representation_monitoring"},
    )

    monitor = RepresentationMonitor(
        probe_dataset,
        [extractor],
        history_config=EvaluationHistoryConfig(
            storage="disk",
            path=history_path,
            detail="summary",
            resume=args.resume,
        ),
        reporters=[ConsoleReporter()],
        scoring_config=OverlapScoringConfig(k=3, min_samples_per_cluster=4),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
    )
    protocol_hash = monitor.history.monitor_metadata["protocol_hash"]
    if restored_state is not None:
        if restored_state["protocol_hash"] != protocol_hash:
            raise ValueError("The saved training state does not match the monitoring protocol.")
        latest = monitor.history.latest_dataframe()
        if latest.empty:
            raise ValueError("The resumed history contains no completed evaluation.")
        epochs = set(latest["epoch"].dropna().astype(int))
        steps = set(latest["global_step"].dropna().astype(int))
        expected_epoch = start_epoch - 1
        if epochs != {expected_epoch} or steps != {global_step}:
            raise ValueError(
                "The saved training coordinates do not match the latest history record."
            )

    train_features = torch.as_tensor(train_x, dtype=torch.float32)
    train_targets = torch.as_tensor(train_y, dtype=torch.long)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = loss_fn(outputs["logits"], train_targets)
        loss.backward()
        optimizer.step()
        global_step += 1

        monitor.evaluate(
            epoch=epoch,
            global_step=global_step,
            checkpoint=f"{state_path}#epoch={epoch}",
            metadata={"training_loss": float(loss.detach())},
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "completed_epoch": epoch,
                "next_epoch": epoch + 1,
                "global_step": global_step,
                "protocol_hash": protocol_hash,
            },
            state_path,
        )

    history = monitor.history.to_dataframe()
    pivot = history.pivot_table(
        index="epoch",
        columns="hidden_layer",
        values="overlap_score",
        aggfunc="last",
    )
    print("\nOverlap by epoch and hidden layer:")
    print(pivot)
    print(f"\nSummary history written to {history_path}")
    print(f"Training state written to {state_path}")


if __name__ == "__main__":
    main()
