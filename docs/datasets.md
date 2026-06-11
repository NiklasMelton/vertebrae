# datasets

`vertebrae` centers the benchmark workflow on a single labeled dataset abstraction:
`BenchmarkDataset`. The dataset object keeps raw inputs, labels, modality, and
construction metadata together so extraction, caching, scoring, and reporting can
share one validated input contract.

## Supported constructors

Use the constructor that matches the form of your source data:

- `BenchmarkDataset.from_arrays(...)` for array-like samples and labels.
- `BenchmarkDataset.from_dataframe(...)` for pandas DataFrames with explicit input
  and label columns.
- `BenchmarkDataset.from_embeddings(...)` for dense or sparse precomputed
  embeddings.
- `BenchmarkDataset.from_image_paths(...)` for image classification datasets stored
  as filesystem paths.
- `BenchmarkDataset.from_audio_paths(...)` for audio classification datasets stored
  as filesystem paths.
- `BenchmarkDataset.from_audio_arrays(...)` for waveform arrays with a shared
  sampling rate.
- `BenchmarkDataset.from_time_series(...)` for aligned time-series arrays with
  optional masks and time features.

Examples:

```python
from vertebrae import BenchmarkDataset

text_dataset = BenchmarkDataset.from_arrays(
    X=texts,
    y=labels,
    modality="text",
    metadata={"source": "support_tickets"},
)
```

```python
from vertebrae import BenchmarkDataset

tabular_dataset = BenchmarkDataset.from_dataframe(
    df=dataframe,
    input_col=["age", "income", "region"],
    label_col="segment",
    modality="tabular",
)
```

```python
from vertebrae import BenchmarkDataset

embedding_dataset = BenchmarkDataset.from_embeddings(
    embeddings=Z,
    labels=y,
    metadata={"backbone": "resnet50"},
)
```

## Validation rules

Every constructor validates the dataset immediately. Current validation checks
include:

- `X` and `y` must have the same number of samples.
- labels must be one-dimensional and non-missing.
- the dataset must contain at least one sample.
- the dataset must contain at least two classes.
- each class must contain at least two samples.

These checks keep downstream scoring failures readable and early. If class counts
are too small for the selected protocol, fix the dataset or rebalance it before
starting a benchmark run.

## Modalities and preserved metadata

The `modality` field is a lightweight routing hint for extractors and reports.
Common values are:

- `"text"`
- `"tabular"`
- `"image"`
- `"audio"`
- `"time_series"`
- `"embeddings"`

`metadata` is preserved through benchmarking so reports can retain source context
such as dataset name, split, backbone provenance, or collection notes.

When using `from_dataframe(...)`, the dataset also records the original column names
and chosen input columns. When using `from_embeddings(...)`, metadata is tagged with
`precomputed_embeddings=True`. Audio-array datasets preserve their shared
`sampling_rate` in metadata. Time-series datasets preserve structured fields such as
`observed_mask`, `time_features`, and `timestamps` when provided.

## Hierarchical label views

When a dataset has a category hierarchy, keep your primary labels in `y` and attach
the hierarchy separately:

```python
dataset = BenchmarkDataset.from_arrays(
    X=samples,
    y=leaf_labels,
    modality="text",
).with_label_hierarchy(
    label_paths=[
        ("support", "billing", "refund"),
        ("support", "billing", "invoice"),
        ("support", "technical", "latency"),
    ],
    level_names=("domain", "group", "leaf"),
)
```

You can then project the dataset to a single hierarchy level with
`BenchmarkDataset.label_view(...)`:

```python
group_dataset = dataset.label_view("group")
```

Derived label views behave like ordinary benchmark datasets. They preserve the same
inputs, carry label-view metadata into reports and artifacts, and still use the
standard dataset validation rules for class counts and minimum samples per class.

For multi-output extractors, `LabelViewConfig.output_levels` can route different
embedding outputs to different hierarchy levels during benchmarking:

```python
from vertebrae import Benchmark, LabelViewConfig

result = Benchmark(
    dataset=dataset,
    extractors=[extractor],
    label_view_config=LabelViewConfig(
        output_levels={"layer_6": "group", "final": "leaf"},
    ),
).run()
```

In this mode, embeddings are materialized from the base dataset and each mapped
output is scored against its configured label view.

## Batching and sharding

`BenchmarkDataset.iter_batches(...)` yields deterministic sample batches. This is
used by streaming-safe extractors and by the artifact-backed distributed embedding
flow described in the distributed-readiness guide.

```python
for batch in dataset.iter_batches(batch_size=128):
    print(batch.indices, batch.X)
```

The dataset object also exposes `stratified_subsample_indices(...)` for
class-preserving sampling without replacement. That method is used by memory-aware
and stability-related workflows when the full dataset should not be embedded or
scored in one pass.

## Working with precomputed embeddings

Precomputed embeddings are the simplest path and the best starting point for new
benchmarks. `BenchmarkDataset.from_embeddings(...)` accepts:

- dense NumPy-like matrices, or
- scipy sparse matrices.

Sparse embeddings are preserved as sparse artifacts until the scoring boundary.
Because the current metric backend is MiniBatchKMeans-backed OverlapIndex, sparse
inputs are densified only inside the internal scorer, with a configurable memory
guard.

## Practical guidance

- Start with `from_embeddings(...)` when you already trust your feature pipeline.
- Use `from_dataframe(...)` when you want reports to preserve column provenance.
- Keep labels in their original semantic form when possible; `vertebrae` preserves
  them in class counts, per-class scores, and weakest-class summaries.
- Add meaningful `metadata` up front so JSON and Markdown reports stay useful later.
