"""Minimal Hugging Face video extractor example."""

import numpy as np

from vertebrae import BenchmarkDataset, CacheConfig, DatasetIdentity, Evaluator
from vertebrae.extractors import HFVideoExtractor


def main() -> None:
    rng = np.random.default_rng(11)
    clips = [
        rng.integers(0, 64, size=(8, 32, 32, 3), dtype=np.uint8),
        rng.integers(0, 64, size=(8, 32, 32, 3), dtype=np.uint8),
        rng.integers(192, 255, size=(8, 32, 32, 3), dtype=np.uint8),
        rng.integers(192, 255, size=(8, 32, 32, 3), dtype=np.uint8),
    ]
    labels = np.array(["dim", "dim", "bright", "bright"])

    dataset = BenchmarkDataset.from_video_arrays(
        frames=clips,
        labels=labels,
        frame_rate=24.0,
        metadata={"dataset": "synthetic_video"},
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFVideoExtractor(
        name="videomae_base",
        model_id="MCG-NJU/videomae-base",
        pooling="mean",
        num_frames=8,
    )

    # The remote model name is intentionally unpinned in this introductory example.
    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        cache_config=CacheConfig(enabled=False),
    ).run()
    print(result.to_dataframe())


if __name__ == "__main__":
    main()
