"""KerasExtractor example for a locally trained or checkpointed Keras model.

Requires optional dependencies:

    poetry install -E keras
    # or
    poetry install -E tensorflow
"""

import numpy as np
from _common import ensure_output_dir, make_separated_blobs, print_ranking

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import KerasExtractor


def main() -> None:
    try:
        try:
            import keras as keras_module
        except ImportError:
            from tensorflow import keras as keras_module
    except ImportError as exc:
        print(exc)
        print("Install optional Keras support with: poetry install -E keras")
        print("Or install TensorFlow-backed Keras with: poetry install -E tensorflow")
        return

    layers = keras_module.layers

    def build_model(input_dim: int, hidden_dim: int = 8, embedding_dim: int = 4):
        inputs = keras_module.Input(shape=(input_dim,))
        embeddings = layers.Dense(hidden_dim, activation="relu")(inputs)
        embeddings = layers.Dense(embedding_dim, name="embeddings")(embeddings)
        logits = layers.Dense(3, name="logits")(embeddings)
        return keras_module.Model(
            inputs=inputs,
            outputs={"embeddings": embeddings, "logits": logits},
            name="tiny_tabular_backbone",
        )

    output_dir = ensure_output_dir()
    X, labels = make_separated_blobs(samples_per_class=30, n_features=6, random_state=23)
    dataset = BenchmarkDataset.from_arrays(
        X,
        labels,
        modality="tabular",
        metadata={"example": "keras_local_model"},
    )

    checkpoint_path = output_dir / "keras_local_model.keras"
    model = build_model(input_dim=X.shape[1])
    model.save(checkpoint_path)

    loaded_model = keras_module.models.load_model(checkpoint_path)

    def collate_fn(batch):
        return np.asarray(batch, dtype=np.float32)

    def output_fn(raw_output):
        return raw_output["embeddings"]

    extractor = KerasExtractor(
        name="local_keras_checkpoint",
        model=loaded_model,
        collate_fn=collate_fn,
        output_fn=output_fn,
        call_method="call",
        recipe_data={"checkpoint": str(checkpoint_path), "input_dim": X.shape[1]},
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=3, min_samples_per_cluster=4),
        stability_config=StabilityConfig(repeats=3, random_state=13),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    result.save_json(str(output_dir / "keras_local_model.json"))
    result.save_markdown(str(output_dir / "keras_local_model.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
