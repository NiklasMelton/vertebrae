"""TorchExtractor example for a locally trained or checkpointed PyTorch model.

Requires optional dependencies:

    poetry install -E torch
"""

import numpy as np
from _common import ensure_output_dir, make_separated_blobs, print_ranking

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import TorchExtractor


def main() -> None:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        print(exc)
        print("Install optional PyTorch support with: poetry install -E torch")
        return

    class TinyTabularBackbone(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 8, embedding_dim: int = 4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.classifier = nn.Linear(embedding_dim, 3)

        def forward(self, x):
            embeddings = self.encoder(x)
            logits = self.classifier(embeddings)
            return {"embeddings": embeddings, "logits": logits}

    output_dir = ensure_output_dir()
    X, labels = make_separated_blobs(samples_per_class=30, n_features=6, random_state=23)
    dataset = BenchmarkDataset.from_arrays(
        X,
        labels,
        modality="tabular",
        metadata={"example": "torch_local_model"},
        identity=DatasetIdentity.ephemeral(),
    )

    checkpoint_path = output_dir / "torch_local_model_state_dict.pt"
    model = TinyTabularBackbone(input_dim=X.shape[1])
    torch.save(model.state_dict(), checkpoint_path)

    loaded_model = TinyTabularBackbone(input_dim=X.shape[1])
    loaded_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    loaded_model.eval()

    def collate_fn(batch):
        return torch.as_tensor(np.asarray(batch), dtype=torch.float32)

    def output_fn(raw_output):
        return raw_output["embeddings"]

    extractor = TorchExtractor(
        name="local_torch_checkpoint",
        model=loaded_model,
        collate_fn=collate_fn,
        output_fn=output_fn,
        device="cpu",
        checkpoint_paths=[str(checkpoint_path)],
        recipe_data={"checkpoint": str(checkpoint_path), "input_dim": X.shape[1]},
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=3, min_samples_per_cluster=4),
        stability_config=StabilityConfig(repeats=3, random_state=13),
        cache_config=CacheConfig(enabled=False),
    ).run()

    result.save_json(str(output_dir / "torch_local_model.json"))
    result.save_markdown(str(output_dir / "torch_local_model.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
