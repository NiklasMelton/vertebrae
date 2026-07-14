"""Minimal Hugging Face audio extractor example."""

import numpy as np

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.extractors import HFAudioExtractor


def main() -> None:
    rng = np.random.default_rng(42)
    waveforms = [
        rng.normal(loc=0.0, scale=0.05, size=16_000).astype(np.float32),
        rng.normal(loc=0.0, scale=0.05, size=16_000).astype(np.float32),
        rng.normal(loc=0.5, scale=0.05, size=16_000).astype(np.float32),
        rng.normal(loc=0.5, scale=0.05, size=16_000).astype(np.float32),
    ]
    labels = np.array(["class_a", "class_a", "class_b", "class_b"])

    dataset = BenchmarkDataset.from_audio_arrays(
        audio=waveforms,
        labels=labels,
        sampling_rate=16_000,
        metadata={"dataset": "synthetic_audio"},
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFAudioExtractor(
        name="wav2vec2_base",
        model_id="facebook/wav2vec2-base",
        pooling="mean",
        sampling_rate=16_000,
    )

    result = Evaluator(dataset=dataset, extractor=extractor).run()
    print(result.to_dataframe())


if __name__ == "__main__":
    main()
