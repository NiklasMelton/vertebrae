"""Single-extractor evaluator wrapper."""

from typing import Any

from vertebrae.benchmark import Benchmark


class Evaluator:
    def __init__(self, dataset: Any, extractor: Any, **kwargs: Any) -> None:
        self.benchmark = Benchmark(dataset=dataset, extractors=[extractor], **kwargs)

    def run(self) -> Any:
        return self.benchmark.run()
