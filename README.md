# vertebrae

`vertebrae` is a practical benchmarking package for evaluating feature extractors and
transfer-learning backbones on labeled datasets. It accepts precomputed embeddings or
generates embeddings from extractors, scores them with `overlapindex`, and renders JSON
or Markdown reports for practitioner-facing diagnostics.

Version 1 uses MiniBatchKMeans-backed OverlapIndex scoring internally. Backend selection
is intentionally not part of the public API.

## Installation

```bash
pip install vertebrae
```

For local development:

```bash
poetry install --with dev
```

Optional text-model integrations are installed with:

```bash
poetry install -E hf
```

## Precomputed Embeddings

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import PrecomputedExtractor

dataset = BenchmarkDataset.from_embeddings(embeddings=Z, labels=y)

result = Evaluator(
    dataset=dataset,
    extractor=PrecomputedExtractor(name="embeddings"),
).run()

print(result.to_dataframe())
result.save_markdown("report.md")
```

## Scikit-learn Pipeline

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import SklearnExtractor

dataset = BenchmarkDataset.from_dataframe(
    df,
    input_col="text",
    label_col="label",
    modality="text",
)

extractor = SklearnExtractor(name="tfidf_svd", pipeline=my_pipeline)
result = Evaluator(dataset=dataset, extractor=extractor).run()
```

## Multi-extractor Comparison

```python
from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.extractors import SklearnExtractor

dataset = BenchmarkDataset.from_arrays(X, y, modality="tabular")

benchmark = Benchmark(dataset)
benchmark.add_extractor(SklearnExtractor("pca", pca_pipeline))
benchmark.add_extractor(SklearnExtractor("svd", svd_pipeline))

result = benchmark.run()
result.save_json("results.json")
result.save_markdown("report.md")
```

Reports include dataset summary, ranking, per-class overlap diagnostics, stability
summaries, probe results when available, warnings, recommendations, and reproducibility
metadata.
