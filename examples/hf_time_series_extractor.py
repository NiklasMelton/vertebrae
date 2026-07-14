"""Minimal Hugging Face time-series extractor example."""

import numpy as np

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.extractors import HFTimeSeriesExtractor


def main() -> None:
    rng = np.random.default_rng(7)
    series = np.stack(
        [
            rng.normal(loc=0.0, scale=0.1, size=32),
            rng.normal(loc=0.0, scale=0.1, size=32),
            rng.normal(loc=1.0, scale=0.1, size=32),
            rng.normal(loc=1.0, scale=0.1, size=32),
        ]
    ).astype(np.float32)
    labels = np.array(["low", "low", "high", "high"])

    dataset = BenchmarkDataset.from_time_series(
        series=series,
        labels=labels,
        metadata={"dataset": "synthetic_time_series"},
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFTimeSeriesExtractor(
        name="patchtst",
        model_id="some-local-or-hf-timeseries-model",
        pooling="mean",
    )

    result = Evaluator(dataset=dataset, extractor=extractor).run()
    print(result.to_dataframe())


if __name__ == "__main__":
    main()
