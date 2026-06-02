# vertebrae

`vertebrae` benchmarks feature extractors and frozen transfer-learning backbones on
labeled datasets. It can score precomputed embeddings or generate embeddings from
local pipelines, callable feature functions, Hugging Face text/vision models, and
sentence-transformers models.

Scoring uses the existing `overlapindex` package. `vertebrae` intentionally exposes
only MiniBatchKMeans-backed OverlapIndex scoring internally; ART, ARTMAP, Fuzzy ART,
Hypersphere ART, `model_type`, `rho`, `r_hat`, and `match_tracking` are not public
`vertebrae` options.

## Install

```bash
pip install vertebrae
```

For local development:

```bash
poetry install --with dev
```

Optional Hugging Face and sentence-transformers support:

```bash
poetry install -E hf
```

## Precomputed embeddings

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import PrecomputedExtractor

dataset = BenchmarkDataset.from_embeddings(embeddings=Z, labels=y)

result = Evaluator(
    dataset=dataset,
    extractor=PrecomputedExtractor(name="my_embeddings"),
).run()

print(result.to_dataframe())
result.save_json("result.json")
result.save_markdown("report.md")
```

## Scikit-learn text pipeline

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import SklearnExtractor

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)),
    ("svd", TruncatedSVD(n_components=128, random_state=42)),
    ("norm", Normalizer()),
])

dataset = BenchmarkDataset.from_arrays(texts, labels, modality="text")
extractor = SklearnExtractor(name="tfidf_bigram_svd128", pipeline=pipeline)
result = Evaluator(dataset=dataset, extractor=extractor).run()
```

## Hugging Face text

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import HFTextExtractor

dataset = BenchmarkDataset.from_arrays(texts, labels, modality="text")

extractor = HFTextExtractor(
    name="distilbert_mean_pool",
    model_id="distilbert-base-uncased",
    pooling="mean",
    batch_size=16,
)

result = Evaluator(dataset=dataset, extractor=extractor).run()
```

## Sentence-transformers

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import SentenceTransformerExtractor

dataset = BenchmarkDataset.from_arrays(texts, labels, modality="text")

extractor = SentenceTransformerExtractor(
    name="minilm",
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=32,
    normalize_embeddings=True,
)

result = Evaluator(dataset=dataset, extractor=extractor).run()
```

## Multi-extractor comparison

```python
from vertebrae import Benchmark

benchmark = Benchmark(dataset)
benchmark.add_extractor(tfidf_extractor)
benchmark.add_extractor(minilm_extractor)
benchmark.add_extractor(distilbert_extractor)

result = benchmark.run()
print(result.to_dataframe())
result.save_markdown("comparison.md")
```

Reports include dataset summary, extractor recipes, ranking, per-class overlap
diagnostics, stability summaries, probe results when enabled, warnings,
recommendations, and reproducibility metadata.

Distributed execution is planned but not implemented. The current implementation keeps
embeddings as artifacts and scoring/reporting separate from live model objects so that
future distributed backends can shard embedding jobs without changing the public API.
