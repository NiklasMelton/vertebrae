# vertebrae

`vertebrae` is a Python package for evaluating feature extractors and transfer-learning backbones on labeled datasets. It supports precomputed embeddings, scikit-learn pipelines, custom callable extractors, and optional Hugging Face and sentence-transformers workflows.

The package uses the `overlapindex` library as its separation metric and wraps the full evaluation flow around practical dataset handling, caching, stability analysis, probe classifiers, and report generation.

## Installation

```bash
pip install vertebrae
```

For local development:

```bash
poetry install --with dev
```

Optional Hugging Face and sentence-transformers support:

```bash
pip install "vertebrae[hf]"
```

## Quick Start

### Precomputed embeddings

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import PrecomputedExtractor

dataset = BenchmarkDataset.from_embeddings(embeddings=Z, labels=y)
extractor = PrecomputedExtractor(name="baseline_embeddings")

result = Evaluator(dataset=dataset, extractor=extractor).run()

print(result.to_dataframe())
result.save_json("result.json")
result.save_markdown("report.md")
```

Sparse matrices are supported as embedding inputs as well.

### Scikit-learn pipelines

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import SklearnExtractor

pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("svd", TruncatedSVD(n_components=128, random_state=42)),
        ("norm", Normalizer()),
    ]
)

dataset = BenchmarkDataset.from_arrays(texts, labels, modality="text")
extractor = SklearnExtractor(name="tfidf_svd", pipeline=pipeline)

result = Evaluator(dataset=dataset, extractor=extractor).run()
```

### Multi-extractor comparison

```python
from vertebrae import Benchmark

benchmark = Benchmark(dataset)
benchmark.add_extractor(tfidf_extractor)
benchmark.add_extractor(sentence_transformer_extractor)
benchmark.add_extractor(custom_extractor)

result = benchmark.run()
print(result.to_dataframe())
```

## Supported Workflows

`vertebrae` is designed for:

- precomputed dense or sparse embeddings,
- NumPy arrays and pandas DataFrames,
- scikit-learn transformers and pipelines,
- custom Python callable extractors,
- single-extractor evaluation,
- multi-extractor comparisons,
- JSON and Markdown reports,
- repeated-run stability analysis,
- lightweight probe classifier checks,
- local embedding caching and reproducible artifacts,
- artifact-backed distributed embedding and scoring through the `vertebrae` CLI.

Distributed CLI commands include `vertebrae plan`, `vertebrae embed-shard`,
`vertebrae merge-embeddings`, `vertebrae write-labels`, `vertebrae score`, and
`vertebrae slurm-array`.

## Reports and Results

Each benchmark run returns structured results that include:

- dataset summary,
- extractor summary and recipe metadata,
- overlap scores and per-class scores,
- stability summaries,
- probe results when enabled,
- warnings and recommendations,
- reproducibility metadata.

Results can be rendered directly to Markdown or JSON:

```python
result.save_json("result.json")
result.save_markdown("report.md")
```

You can also convert rankings into a DataFrame with `result.to_dataframe()`.

## Optional Extractors

Optional integrations are available through the `hf` extra:

- `SentenceTransformerExtractor`
- `HFTextExtractor`
- `HFVisionExtractor`

These workflows rely on optional dependencies and lazy imports, so the core package stays lightweight.

## Command Line Interface

`vertebrae` includes a CLI for deterministic embedding shard planning and artifact merging in local or batch-style workflows. Run `vertebrae --help` to see the available commands.

## Notes

- The package targets Python `>=3.9,<3.13`.
- `overlapindex>=0.1.1` is required.
- The public API is centered on `BenchmarkDataset`, `Evaluator`, `Benchmark`, extractor wrappers, and structured result objects.

## License

MIT
