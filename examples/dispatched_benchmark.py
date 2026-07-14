"""Run a labeled benchmark through the artifact-backed local execution backend."""

import numpy as np
from _common import CACHE_DIR, ensure_output_dir, make_separated_blobs, print_ranking

from vertebrae import Benchmark, BenchmarkDataset, DatasetIdentity
from vertebrae.config import (
    CacheConfig,
    ExecutionConfig,
    OverlapScoringConfig,
    StabilityConfig,
)
from vertebrae.execution import LocalBackend
from vertebrae.extractors import CallableExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    values, labels = make_separated_blobs(
        samples_per_class=32,
        n_features=8,
        random_state=41,
    )
    dataset = BenchmarkDataset.from_arrays(
        values,
        labels,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
        metadata={"example": "dispatched_benchmark"},
    )
    extractor = CallableExtractor(
        "first_six_features",
        lambda batch: np.asarray(batch)[:, :6],
        modality="tabular",
        streaming_safe=True,
    )

    result = Benchmark(
        dataset,
        [extractor],
        scoring_config=OverlapScoringConfig(k=4),
        stability_config=StabilityConfig(repeats=3),
        cache_config=CacheConfig(cache_dir=str(CACHE_DIR)),
        execution=LocalBackend(n_jobs=2, joblib_backend="threading"),
        execution_config=ExecutionConfig(total_shards=2),
    ).run()
    result.save_json(str(output_dir / "dispatched_benchmark.json"))
    result.save_markdown(str(output_dir / "dispatched_benchmark.md"))
    print_ranking(result)


if __name__ == "__main__":
    main()
