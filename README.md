# Vertebrae

<table>
  <tr>
    <td width="100" valign="top">
      <a href="https://github.com/NiklasMelton/vertebrae">
        <img
          src="https://github.com/NiklasMelton/vertebrae/blob/develop/img/vertebrae_logo.png?raw=true"
          alt="Vertebrae logo"
          width="140"
        />
      </a>
    </td>
    <td valign="top">

`vertebrae` is a Python package for evaluating feature extractors and 
transfer-learning backbones on labeled datasets. It supports precomputed embeddings, 
scikit-learn pipelines, custom callable extractors, ONNX models, local PyTorch 
and 
Keras modules, optional embedding compression, and optional Hugging Face and
sentence-transformers workflows.

The package uses the `overlapindex` library as its separation metric and wraps the full evaluation flow around practical dataset handling, caching, stability analysis, probe classifiers, and report generation.

</tr>
</table>



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

Optional Hugging Face audio support only:

```bash
pip install "vertebrae[audio]"
```

Optional Hugging Face time-series support only:

```bash
pip install "vertebrae[timeseries]"
```

Optional local PyTorch model support:

```bash
pip install "vertebrae[torch]"
```

Optional local Keras model support:

```bash
pip install "vertebrae[keras]"
pip install "vertebrae[tensorflow]"
```

Optional ONNX Runtime support:

```bash
pip install "vertebrae[onnx]"
```

Optional distributed execution backends:

```bash
pip install "vertebrae[ray]"
pip install "vertebrae[dask]"
pip install "vertebrae[distributed]"
```

Optional cloud artifact stores:

```bash
pip install "vertebrae[s3]"
pip install "vertebrae[gcs]"
pip install "vertebrae[cloud]"
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

### Optional embedding compression

```python
from vertebrae import BenchmarkDataset, EmbeddingCompressionConfig, Evaluator
from vertebrae.extractors import PrecomputedExtractor

dataset = BenchmarkDataset.from_embeddings(embeddings=Z, labels=y)
extractor = PrecomputedExtractor(name="baseline_embeddings")

compression = EmbeddingCompressionConfig(
    enabled=True,
    method="prefix_truncate",
    n_components=256,
    assume_matryoshka=True,
)

result = Evaluator(
    dataset=dataset,
    extractor=extractor,
    compression_config=compression,
).run()
```

Supported compression methods include `pca`, `incremental_pca`,
`truncated_svd`, random projections, `prefix_truncate`, and `quantize`.

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

### Local PyTorch models

```python
import numpy as np
import torch

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import TorchExtractor

model = torch.load("/path/to/local_model.pt", map_location="cpu")
model.eval()


def collate_fn(batch):
    return torch.as_tensor(np.asarray(batch), dtype=torch.float32)


def output_fn(raw_output):
    return raw_output if isinstance(raw_output, torch.Tensor) else raw_output["embeddings"]


dataset = BenchmarkDataset.from_arrays(features, labels, modality="tabular")
extractor = TorchExtractor(
    name="local_torch",
    model=model,
    collate_fn=collate_fn,
    output_fn=output_fn,
    device="cpu",
    recipe_data={"checkpoint": "/path/to/local_model.pt"},
)

result = Evaluator(dataset=dataset, extractor=extractor).run()
```

### Local Keras models

```python
import numpy as np

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import KerasExtractor


def collate_fn(batch):
    return np.asarray(batch, dtype=np.float32)


dataset = BenchmarkDataset.from_arrays(features, labels, modality="tabular")
extractor = KerasExtractor(
    name="local_keras",
    model=model,
    collate_fn=collate_fn,
    call_method="call",
    recipe_data={"checkpoint": "/path/to/model.keras"},
)

result = Evaluator(dataset=dataset, extractor=extractor).run()
```

### Hugging Face audio backbones

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import HFAudioExtractor

dataset = BenchmarkDataset.from_audio_arrays(
    audio=waveforms,
    labels=labels,
    sampling_rate=16_000,
)
extractor = HFAudioExtractor(
    name="wav2vec2_base",
    model_id="facebook/wav2vec2-base",
    pooling="mean",
)

result = Evaluator(dataset=dataset, extractor=extractor).run()
```

### Hugging Face time-series backbones

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import HFTimeSeriesExtractor

dataset = BenchmarkDataset.from_time_series(
    series=series,
    labels=labels,
)
extractor = HFTimeSeriesExtractor(
    name="patchtst",
    model_id="some-local-or-hf-timeseries-model",
    pooling="mean",
)

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

You can also benchmark multiple embedding outputs from the same backbone without
duplicating extractor classes. This is useful for comparing intermediate layers,
pooling strategies, or multi-head outputs from one model:

```python
from vertebrae import Benchmark
from vertebrae.extractors import HFVisionExtractor

benchmark = Benchmark(dataset)
benchmark.add_extractor(
    HFVisionExtractor(
        name="mnist_vit",
        model_id="farleyknight-org-username/vit-base-mnist",
        outputs=[
            {"name": "final_cls", "pooling": "cls"},
            {"name": "mid_cls", "pooling": "cls", "hidden_layer": 6},
        ],
        image_mode="rgb",
        batch_size=8,
    )
)

result = benchmark.run()
print(result.to_dataframe()[["name", "overlap_macro"]])
```

Each configured output is scored as its own result variant, so this run produces
rows named `mnist_vit:final_cls` and `mnist_vit:mid_cls`. See
`examples/hf_vision_mnist.py` for a fuller example that compares multi-output
Hugging Face vision embeddings alongside a classical scikit-learn image
baseline.

## Supported Workflows

`vertebrae` is designed for:

- precomputed dense or sparse embeddings,
- NumPy arrays and pandas DataFrames,
- scikit-learn transformers and pipelines,
- custom Python callable extractors,
- Hugging Face audio backbones through `HFAudioExtractor`,
- Hugging Face time-series backbones through `HFTimeSeriesExtractor`,
- local PyTorch modules through `TorchExtractor`,
- local Keras modules through `KerasExtractor`,
- single-extractor evaluation,
- multi-extractor comparisons,
- JSON and Markdown reports,
- repeated-run stability analysis,
- lightweight probe classifier checks,
- optional embedding compression and compressed-variant comparisons,
- local embedding caching and reproducible artifacts,
- artifact-backed distributed embedding and scoring through the `vertebrae` CLI.
- optional Ray and Dask backends for distributed execution,
- local paths, `s3://...`, and `gs://...` artifact stores.

Distributed CLI commands include `vertebrae plan`, `vertebrae embed-shard`,
`vertebrae merge-embeddings`, `vertebrae write-labels`, `vertebrae compress`, `vertebrae score`, and
`vertebrae score-repeats`, `vertebrae collect-scores`, `vertebrae benchmark-from-artifacts`,
`vertebrae slurm-array`, `vertebrae slurm-score-array`, and `vertebrae run-embedding-shards`.

For Ray or Dask cluster runs, the configured `cache_dir` can be either a shared local
path or a cloud object-store URI such as `s3://team-bucket/vertebrae/run-001` or
`gs://team-bucket/vertebrae/run-001`. Workers need credentials for the selected store.

## Reports and Results

Each benchmark run returns structured results that include:

- dataset summary,
- extractor summary and recipe metadata,
- overlap scores and per-class scores,
- compression metadata and compressed dimensions,
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

Compression-aware results include the compression method and compressed
dimension, and quantized runs preserve calibration metadata in the structured
result payload.

The key component of the report is the performance and comparison table. An example 
markdown table as generated by `examples\sklearn_wine_pipeline.py` is shown below.

| rank | extractor | extractor_type | overlap_macro | stability_interval | weakest_class | probe_accuracy | embedding_dim | compression | compressed_dim | recommendation |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| 1 | wine_minmax_pca_2 | unsupervised_fitted | 0.9373 | 0.9366-0.9455 | class_1 | 1.0000 | 2 | none | 2 | strong_candidate |
| 2 | wine_standard_scaler_all_features | unsupervised_fitted | 0.9248 | 0.9058-0.9279 | class_1 | 0.9722 | 13 | none | 13 | strong_candidate |
| 3 | wine_standard_scaler_pca_6 | unsupervised_fitted | 0.9128 | 0.9051-0.9368 | class_1 | 1.0000 | 6 | none | 6 | strong_candidate |
| 4 | wine_quantile_pca_1 | unsupervised_fitted | 0.4554 | 0.3455-0.4554 | class_0 | 0.8056 | 1 | none | 1 | poor_frozen_representation_weak_class_attention |

Extractors are ranked according to their `overlap_macro` performance.
## Optional Extractors

Optional integrations are available through the `torch`, `keras`, `tensorflow`, `onnx`, and `hf` extras:

- `TorchExtractor`
- `KerasExtractor`
- `ONNXExtractor`
- `SentenceTransformerExtractor`
- `HFTextExtractor`
- `HFVisionExtractor`

These workflows rely on optional dependencies and lazy imports, so the core package stays lightweight.
See `examples/onnx_extractor.py` for a local ONNX export workflow.
See `examples/hf_vision_mnist.py` for a laptop-friendly comparison that runs
MNIST handwritten digit data through final and mid-layer Hugging Face vision
embeddings, and
`examples/sklearn_wine_pipeline.py` for a network-free real-data scikit-learn
pipeline comparison.
See the compression guide in `docs/compression.md` for compression options and guidance.

## Command Line Interface

`vertebrae` includes a CLI for deterministic embedding shard planning and artifact
merging in local or batch-style workflows. Distributed orchestration commands accept
`--backend local|ray|dask`, with `--ray-address` and `--dask-address` available for
cluster connections. Cloud artifact stores use the same `--cache-dir` flag, plus
provider options such as `--s3-endpoint-url`, `--s3-profile`, `--s3-region`, and
`--gcs-project`. The CLI can also derive compressed embedding artifacts with
`vertebrae compress`. Run `vertebrae --help` to see the available commands.

## Notes

- The package targets Python `>=3.9,<3.15`.
- The public API is centered on `BenchmarkDataset`, `Evaluator`, `Benchmark`, extractor wrappers, and structured result objects.

## License

MIT
