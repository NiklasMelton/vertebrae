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
transfer-learning backbones on labeled datasets. It supports dense and sparse
precomputed embeddings, scikit-learn pipelines, custom callable extractors, ONNX
models, local PyTorch and Keras modules, segmentation token workflows, optional
embedding compression, and optional model families spanning Hugging Face,
sentence-transformers, timm, torchvision, OpenCLIP/SigLIP, TensorFlow Hub,
JAX/Flax, tree ensembles, graph models, and hosted embedding APIs. It can also
evaluate labeled embedding units such as document regions, tokens, frames,
keypoints, depth cells, and latent slots emitted directly by a model.

The package uses the `overlapindex` library as its primary separation metric and
adds a `separatix` complexity diagnostic to reports when an evaluated embedding
clears a configurable overlap-quality threshold. The full evaluation flow wraps
those diagnostics with practical dataset handling, named target and hierarchy
views, caching, memory-aware subsampling, stability analysis, artifact-backed
execution, custom embedding metrics, and report generation.

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

Optional Hugging Face video support only:

```bash
pip install "vertebrae[video]"
```

Optional local PyTorch model support:

```bash
pip install "vertebrae[torch]"
pip install "vertebrae[timm]"
pip install "vertebrae[torchvision]"
pip install "vertebrae[openclip]"
```

Optional local Keras model support:

```bash
pip install "vertebrae[keras]"
pip install "vertebrae[tensorflow]"
pip install "vertebrae[tensorflow-hub]"
```

Optional ONNX Runtime support:

```bash
pip install "vertebrae[onnx]"
```

Optional JAX/Flax, tree ensemble, and graph model support:

```bash
pip install "vertebrae[jax]"
pip install "vertebrae[trees]"
pip install "vertebrae[graph]"
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

By default, vertebrae also runs a Separatix complexity diagnostic when the
evaluated embedding reaches `overlap_macro >= 0.80`. That extra diagnostic does
not affect ranking. It adds report guidance about what kind of downstream
classifier complexity the labeled geometry appears to need.

Multi-label classification datasets are supported through the same constructors.
Use per-sample label sequences or a binary indicator matrix with `label_names`:

```python
dataset = BenchmarkDataset.from_embeddings(
    embeddings=Z,
    labels=[
        ("outdoor", "vehicle"),
        ("outdoor", "vehicle"),
        ("indoor",),
        ("indoor",),
        ("outdoor", "animal"),
        ("animal",),
    ],
)

result = Evaluator(dataset=dataset, extractor=PrecomputedExtractor()).run()
```

OverlapIndex receives a dense multi-label indicator target internally, and
Separatix runs with `target_mode="multilabel"`.

Regression targets are supported when explicitly requested so numeric class
identifiers are not accidentally interpreted as continuous targets:

```python
dataset = BenchmarkDataset.from_arrays(
    X=samples,
    y=targets,
    modality="tabular",
    target_type="regression",
    target_names=["quality_score"],
)

result = Evaluator(dataset=dataset, extractor=extractor).run()
```

Regression scoring uses `ContinuousOverlapIndex` through vertebrae's internal
scoring adapter and appears in reports as continuous overlap diagnostics.

### Multiple target views

When one embedding should be compared against several aligned targets, register
named target views on the dataset and enable them in `Benchmark`. Views can be
classification, multi-label, or regression targets; each is reported as a
separate result variant.

```python
from vertebrae import Benchmark, BenchmarkDataset, TargetView, TargetViewConfig

dataset = BenchmarkDataset.from_embeddings(embeddings=Z, labels=leaf_labels)
dataset = dataset.with_target_views(
    [
        TargetView(name="coarse", targets=coarse_labels),
        TargetView(name="quality", targets=quality_scores, target_type="regression"),
    ]
)

result = Benchmark(
    dataset,
    [extractor],
    target_view_config=TargetViewConfig(enabled=True, views=("coarse", "quality")),
).run()
```

For taxonomies represented as label paths, use `with_label_hierarchy(...)` and
`LabelViewConfig` instead. `output_views` and `output_levels` can route named
extractor outputs to the target or hierarchy view that they should evaluate.

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

### Hugging Face multi-modal models

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import HFMultimodalExtractor

dataset = BenchmarkDataset.from_multimodal(
    inputs={"image": images, "caption": captions},
    labels=labels,
    modalities={"image": "image", "caption": "text"},
)

extractor = HFMultimodalExtractor(
    name="clip_like",
    model_id="openai/clip-vit-base-patch32",
    input_modalities={"image": "image", "caption": "text"},
    outputs=[
        {"name": "image_branch", "source": "image", "model_output": "image_embeds"},
        {"name": "text_branch", "source": "text", "model_output": "text_embeds"},
        {"name": "fused", "source": "fused", "model_output": "pooler_output"},
    ],
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

### Hugging Face video backbones

```python
from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.extractors import HFVideoExtractor

dataset = BenchmarkDataset.from_video_arrays(
    frames=clips,
    labels=labels,
    frame_rate=24.0,
)
extractor = HFVideoExtractor(
    name="videomae_base",
    model_id="MCG-NJU/videomae-base",
    pooling="mean",
    num_frames=16,
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
pooling strategies, or multi-head outputs from one model. When the dataset has
hierarchy metadata from `with_label_hierarchy(...)`, outputs can be routed to
different hierarchy levels:

```python
from vertebrae import Benchmark, LabelViewConfig
from vertebrae.extractors import HFVisionExtractor

benchmark = Benchmark(
    dataset,
    label_view_config=LabelViewConfig(
        output_levels={
            "mid_cls": "family",
            "final_cls": "leaf",
        },
    ),
)
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
print(result.to_dataframe()[["extractor", "label_view", "overlap_macro"]])
```

Each configured output is scored as its own result variant, so this run produces
rows such as `mnist_vit:mid_cls[level=family]` and
`mnist_vit:final_cls[level=leaf]`. See `examples/hf_vision_mnist.py` for a fuller
example that compares multi-output Hugging Face vision embeddings alongside a
classical scikit-learn image baseline.
For a more realistic image workflow, `examples/caltech101_vision_foundation_models.py`
downloads a laptop-sized Caltech-101 subset with a few related category pairs,
compares DINOv2 with a tiny supervised ViT baseline, and can include gated DINOv3
embeddings when `VERTABRAE_INCLUDE_DINOV3=1` is set.

### Custom embedding metrics

Every benchmark always records the built-in overlap metric. You can score the
same full embedding batch with additional metrics and choose one as the ranking
criterion. A custom metric returns a finite aggregate `score` and may include
JSON-safe diagnostics, warnings, and metadata.

```python
from vertebrae import Benchmark, CallableMetric

def domain_margin(embeddings, labels, *, groups=None, seed=None):
    return {"score": 0.87, "diagnostics": {"rule": "domain margin"}}

result = Benchmark(
    dataset,
    [extractor],
    metrics=[CallableMetric("domain_margin", domain_margin)],
    primary_metric="domain_margin",
).run()
```

The overlap result remains available in every `ExtractorResult` and continues
to drive stability and Separatix. For artifact or CLI workflows, use an
importable callable path such as `my_project.metrics:domain_margin`; see
[`docs/scoring.md`](docs/scoring.md) and
[`docs/distributed_readiness.md`](docs/distributed_readiness.md).

### Dense segmentation tokens

Dense segmentation evaluation scores spatial feature cells after they are aligned
to semantic mask labels. It measures representation organization for retained
tokens; it is not an IoU, mask-accuracy, or boundary-quality metric.

```python
from vertebrae import (
    Benchmark,
    CallableSpatialExtractor,
    SegmentationConfig,
    SegmentationDataset,
    SpatialLayout,
    SpatialOutputSpec,
)

dataset = SegmentationDataset.from_arrays(
    images=images,
    semantic_masks=masks,
)

extractor = CallableSpatialExtractor(
    "encoder",
    transform_fn=extract_spatial_features,
    output_specs=[
        SpatialOutputSpec(
            name="stage_4",
            layout=SpatialLayout(grid_height=14, grid_width=14),
        )
    ],
)

result = Benchmark(
    dataset=dataset,
    extractors=[extractor],
    segmentation_config=SegmentationConfig(max_tokens_per_class=10_000),
).run()
```

See `docs/segmentation.md` for background handling, ambiguity filtering,
instance caps, grouped diagnostics, and precomputed segmentation embeddings.

### Structured units from native model outputs

Structured extractors flatten one declared per-parent unit matrix into a grouped
embedding dataset, preserving unit provenance and parent groups. This supports
representation diagnostics for regions, tokens, frames, keypoints, depth cells,
and latent slots. It does not substitute task-native metrics such as mAP, IoU,
WER/CER, OKS, depth error, or reconstruction quality.

```python
from vertebrae import Benchmark, BenchmarkDataset, UnitAnnotation
from vertebrae.extractors import CallableStructuredExtractor, StructuredOutputSpec

dataset = BenchmarkDataset.from_arrays(X=pages, y=document_labels, modality="image")
dataset = dataset.with_unit_annotations(
    [
        UnitAnnotation(labels=["heading", "body"]),
        UnitAnnotation(labels=["heading", "body"]),
    ],
    unit_type="document_region",
)

extractor = CallableStructuredExtractor(
    name="layout_encoder",
    transform_fn=extract_region_embeddings,  # one 2D matrix per page
    output_specs=[StructuredOutputSpec(name="regions", unit_type="document_region")],
)
result = Benchmark(dataset, [extractor]).run()
```

When the model emits unmatched rows (for example special tokens or sampled
frames), supply an explicit alignment rule such as
`drop_special_rows(leading=1)` or `select_frame_rows(...)`. Typed adapters are
also available for detection/layout, sequence labeling, keypoints, depth, and
latent slots. See [`docs/datasets.md`](docs/datasets.md),
[`docs/feature_extractors.md`](docs/feature_extractors.md), and the runnable
`examples/structured_*.py` workflows.

## Supported Workflows

`vertebrae` is designed for:

- precomputed dense or sparse embeddings,
- NumPy arrays and pandas DataFrames,
- single-label classification, multi-label classification, and explicit regression targets,
- hierarchy label views and named target views for scoring the same embeddings against
  different targets,
- graph-node, graph-edge, entity, pair, triplet-derived, and generic labeled-unit
  embedding diagnostics,
- scikit-learn transformers and pipelines,
- custom Python callable extractors,
- dense segmentation token materialization from spatial feature maps,
- structured unit materialization from native token, frame, region, keypoint, depth,
  or latent-slot outputs, with explicit alignment when rows do not already match,
- Hugging Face text backbones through `HFTextExtractor`,
- Hugging Face vision backbones through `HFVisionExtractor`,
- Hugging Face audio backbones through `HFAudioExtractor`,
- Hugging Face image-text and other structured multi-modal backbones through
  `HFMultimodalExtractor`,
- Hugging Face video backbones through `HFVideoExtractor`,
- Hugging Face time-series backbones through `HFTimeSeriesExtractor`,
- sentence-transformers through `SentenceTransformerExtractor`,
- timm, torchvision, OpenCLIP, SigLIP, TensorFlow Hub, JAX/Flax, tree-leaf,
  graph, and hosted embedding API extractors,
- local PyTorch modules through `TorchExtractor`,
- local Keras modules through `KerasExtractor`,
- local ONNX Runtime sessions through `ONNXExtractor`,
- single-output and multi-output extractor evaluation,
- single-extractor evaluation,
- multi-extractor comparisons,
- JSON and Markdown reports,
- repeated-run stability analysis,
- optional Separatix complexity diagnostics in local and artifact-backed reports,
- custom full-batch embedding metrics with a selectable primary ranking metric,
- optional embedding compression and compressed-variant comparisons,
- local embedding caching and reproducible artifacts,
- artifact-backed distributed embedding and scoring through the `vertebrae` CLI,
- optional Ray and Dask backends for distributed execution,
- local paths, `s3://...`, and `gs://...` artifact stores.

Distributed CLI commands include `vertebrae plan`, `vertebrae embed-shard`,
`vertebrae merge-embeddings`, `vertebrae write-labels`, `vertebrae write-groups`,
`vertebrae materialize-segmentation`, `vertebrae materialize-structured`,
`vertebrae compress`, `vertebrae score`,
`vertebrae diagnose-complexity`, `vertebrae score-repeats`,
`vertebrae collect-scores`, `vertebrae benchmark-from-artifacts`,
`vertebrae slurm-array`, `vertebrae slurm-score-array`, and
`vertebrae run-embedding-shards`.

For Ray or Dask cluster runs, the configured `cache_dir` can be either a shared local
path or a cloud object-store URI such as `s3://team-bucket/vertebrae/run-001` or
`gs://team-bucket/vertebrae/run-001`. Workers need credentials for the selected store.

## Reports and Results

Each benchmark run returns structured results that include:

- dataset summary,
- extractor summary and recipe metadata,
- overlap scores plus per-class, per-label, or per-target diagnostics,
- Separatix recommendation, confidence, and report details when the overlap gate passes,
- label-view, target-view, segmentation, structured-unit, grouping, and target-type
  metadata when present,
- every configured metric result and the selected primary ranking metric,
- compression metadata and compressed dimensions,
- stability summaries,
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

Separatix is stored as the default classifier-complexity report field. JSON
output preserves the full Separatix report. Markdown and DataFrame views
surface the main recommendation, confidence, and compact explanation fields
that are usually the most actionable.

Separatix is also the source of probe-style summary fields, so vertebrae does not
fit a second probe system alongside the complexity diagnostic.

The key component of the report is the performance and comparison table. An example 
markdown table as generated by `examples\sklearn_wine_pipeline.py` is shown below.

| rank | extractor | extractor_type | overlap_macro | stability_interval | weakest_class | probe_accuracy | embedding_dim | compression | compressed_dim | recommendation | separatix_recommendation | separatix_confidence |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 1 | wine_standard_scaler_pca_6 | unsupervised_fitted | 0.9511 | 0.9051-0.9368 | class_1 | 0.9775 | 6 | none | 6 | strong_candidate | linear_likely_sufficient | high |
| 2 | wine_minmax_pca_2 | unsupervised_fitted | 0.9484 | 0.9366-0.9455 | class_1 | 0.9719 | 2 | none | 2 | strong_candidate | inconclusive | high |
| 3 | wine_standard_scaler_all_features | unsupervised_fitted | 0.9235 | 0.9058-0.9279 | class_1 | 0.9775 | 13 | none | 13 | strong_candidate | linear_likely_sufficient | high |
| 4 | wine_quantile_pca_1 | unsupervised_fitted | 0.4554 | 0.3455-0.4554 | class_2 |  | 1 | none | 1 | poor_frozen_representation_weak_class_attention |  |  |

By default, extractors are ranked by overlap. When a custom `primary_metric` is
configured, they are ranked by that metric instead; the overlap columns remain
available for representation diagnostics and Separatix gating.

The easiest way to interpret the report is:

- Start with `primary_metric` and `primary_score`. By default these are overlap;
  with a custom metric they identify the configured ranking signal.
- Inspect `overlap_macro` and per-class overlap scores as the standard vertebrae
  representation diagnostic, even when another metric ranks the candidates.
- Use the vertebrae `recommendation` field as a quick summary of representation quality under the benchmark protocol.
- Use `separatix_recommendation` and `separatix_confidence` to understand what kind of downstream classifier complexity the labeled embedding seems to imply once the representation is already reasonably separated.
- When Separatix columns are blank, the embedding did not clear the overlap gate, so vertebrae skipped the extra diagnostic rather than over-interpreting a weak representation.
- Treat `probe_accuracy` as a Separatix-derived quick-check column. It is blank when Separatix is disabled, skipped, or does not report a baseline probe score.

In the per-extractor Markdown section, Separatix also adds:

- a plain-language recommendation text,
- a decision path showing the main diagnostic branches,
- normalized summary scores such as signal, overlap, linearity, nonlinearity, and reliability,
- warnings and skipped diagnostics when part of the complexity audit did not run.

As a rule of thumb, a strong overlap score plus `linear_likely_sufficient`
usually points to an embedding that should work well with simple downstream
classifiers, while a strong overlap score plus
`smooth_nonlinear_recommended` or `kernel_or_local_recommended` suggests the
embedding is promising but may benefit from a more flexible decision boundary.

## Optional Extractors

Optional integrations are available through extras such as `torch`, `keras`,
`tensorflow`, `onnx`, `hf`, `timm`, `torchvision`, `openclip`,
`tensorflow-hub`, `jax`, `trees`, and `graph`:

- `TorchExtractor`
- `KerasExtractor`
- `ONNXExtractor`
- `SentenceTransformerExtractor`
- `HFTextExtractor`
- `HFVisionExtractor`
- `HFAudioExtractor`
- `HFTimeSeriesExtractor`
- `HFVideoExtractor`
- `HFMultimodalExtractor`
- `TimmVisionExtractor`
- `TorchvisionVisionExtractor`
- `OpenCLIPExtractor`
- `SigLIPExtractor`
- `TFHubExtractor`
- `JAXFlaxExtractor`
- `TreeLeafEmbeddingExtractor`
- `GraphModelExtractor`
- `HostedEmbeddingExtractor`
- `CallableSpatialExtractor`
- `PrecomputedSpatialExtractor`
- `CallableStructuredExtractor`
- `PrecomputedStructuredExtractor`

These workflows rely on optional dependencies and lazy imports, so the core package stays lightweight.
See `examples/onnx_extractor.py` for a local ONNX export workflow.
See `examples/hf_vision_mnist.py` for a laptop-friendly comparison that runs
MNIST handwritten digit data through final and mid-layer Hugging Face vision
embeddings, and
`examples/caltech101_vision_foundation_models.py` for a single-label Caltech-101
workflow with automatic local data reuse/downloads and a less trivial default
class slice. See
`examples/sklearn_wine_pipeline.py` for a network-free real-data scikit-learn
pipeline comparison.
See `docs/feature_extractors.md` for the full extractor matrix and install
mapping, `docs/segmentation.md` for dense spatial workflows, and
`docs/compression.md` for compression options and guidance.

## Command Line Interface

`vertebrae` includes a CLI for deterministic embedding shard planning and artifact
merging in local or batch-style workflows. Distributed orchestration commands accept
`--backend local|ray|dask`, with `--ray-address` and `--dask-address` available for
cluster connections. Cloud artifact stores use the same `--cache-dir` flag, plus
provider options such as `--s3-endpoint-url`, `--s3-profile`, `--s3-region`, and
`--gcs-project`. The CLI can also derive compressed embedding artifacts with
`vertebrae compress`, materialize structured unit artifacts with
`vertebrae materialize-structured`, and evaluate importable custom metrics through
repeatable `vertebrae score --metric module:callable` options. Run
`vertebrae --help` to see the available commands.

## Notes

- The package targets Python `>=3.9,<3.15`.
- The public API is centered on `BenchmarkDataset`, `SegmentationDataset`,
  `Evaluator`, `Benchmark`, extractor wrappers, metric adapters, config
  dataclasses, and structured result objects.

## License

MIT
