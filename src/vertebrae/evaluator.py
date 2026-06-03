"""Single-extractor evaluator wrapper."""

from typing import Any

from vertebrae.benchmark import Benchmark


class Evaluator:
    """Single-extractor convenience wrapper around `Benchmark`.

    Args:
        dataset: Dataset object with inputs and labels.
        extractor: Feature extractor to evaluate.
        **kwargs: Additional keyword arguments forwarded to `Benchmark`.
    """

    def __init__(self, dataset: Any, extractor: Any, **kwargs: Any) -> None:
        self.benchmark = Benchmark(dataset=dataset, extractors=[extractor], **kwargs)

    def run(self) -> Any:
        """Run the single-extractor benchmark.

        Returns:
            A `BenchmarkResult`.
        """

        return self.benchmark.run()
