# datasets

`vertebrae` centers the benchmark workflow on a single labeled dataset abstraction:
`BenchmarkDataset`. The dataset object keeps raw inputs, labels, modality, and
construction metadata together so extraction, caching, scoring, and reporting can
share one validated input contract.

## Supported constructors

Use the constructor that matches the form of your source data:

- `BenchmarkDataset.from_arrays(...)` for array-like samples and labels.
- `BenchmarkDataset.from_dataframe(...)` for pandas DataFrames with explicit input
  and label columns, including multi-label indicator columns or explicit
  regression target columns.
- `BenchmarkDataset.from_embeddings(...)` for dense or sparse precomputed
  embeddings.
- `BenchmarkDataset.from_image_paths(...)` for image classification datasets stored
  as filesystem paths.
- `BenchmarkDataset.from_audio_paths(...)` for audio classification datasets stored
  as filesystem paths.
- `BenchmarkDataset.from_audio_arrays(...)` for waveform arrays with a shared
  sampling rate.
- `BenchmarkDataset.from_video_paths(...)` for video classification datasets stored
  as filesystem paths.
- `BenchmarkDataset.from_video_arrays(...)` for predecoded frame arrays with an
  optional shared frame rate.
- `BenchmarkDataset.from_time_series(...)` for aligned time-series arrays with
  optional masks and time features.
- `BenchmarkDataset.from_multimodal(...)` for aligned multi-modal sample fields
  such as image-text or audio-text classification datasets.
- `BenchmarkDataset.from_graphs(...)` for aligned graph objects used by
  graph-level PyG or DGL extractors.
- `BenchmarkDataset.from_node_embeddings(...)` for labeled graph-node embeddings.
- `BenchmarkDataset.from_edge_embeddings(...)` for labeled graph-edge embeddings
  or edge rows composed from node embeddings.
- `BenchmarkDataset.from_entity_embeddings(...)` for labeled user, item, query,
  document, or other entity embeddings.
- `BenchmarkDataset.from_pair_embeddings(...)` for labeled pair embeddings or pair
  rows composed from entity embeddings.
- `BenchmarkDataset.from_triplet_embeddings(...)` for supervised triplet-derived
  embedding rows.
- `BenchmarkDataset.from_embedding_units(...)` for generic labeled unit embeddings
  such as detection boxes, document regions, tokens, keypoints, depth cells, or
  latent slots.

Regression datasets are opt-in. Pass `target_type="regression"` so numeric class
identifiers are not reinterpreted as continuous targets by accident:

```python
dataset = BenchmarkDataset.from_arrays(
    X=samples,
    y=targets,
    modality="tabular",
    target_type="regression",
    target_names=["score"],
)
```

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

Relational embedding constructors are still ordinary supervised embedding
datasets. They are intended for transfer-learning diagnostics where each node,
edge, entity, pair, or triplet-derived row has an aligned label or regression
target. They do not run retrieval, recommender, ranking, mAP, NDCG, MRR, or
candidate-set evaluation.

Generic unit embeddings follow the same philosophy: each row is a labeled unit
embedding, not a task-specific metric evaluation. This supports structured
representation diagnostics for boxes, OCR/document layout regions, ASR or NLP
tokens, pose/keypoints, dense depth cells, and generative latents as long as the
workflow can materialize one embedding row per supervised unit.

```python
from vertebrae import BenchmarkDataset, TargetView

dataset = BenchmarkDataset.from_embedding_units(
    embeddings=unit_embeddings,
    labels=primary_labels,
    unit_ids=unit_ids,
    parent_ids=page_ids,
    unit_type="document_region",
    target_views=[
        TargetView(name="role", targets=region_roles),
        TargetView(
            name="quality",
            targets=region_scores,
            target_type="regression",
            target_names=["quality"],
        ),
    ],
)
```

```python
from vertebrae import BenchmarkDataset

node_dataset = BenchmarkDataset.from_node_embeddings(
    embeddings=node_z,
    labels=node_labels,
    node_ids=node_ids,
    edge_index=edge_index,
)
```

For edge or pair labels, you can either pass precomputed row embeddings or compose
them from endpoint/entity embeddings:

```python
edge_dataset = BenchmarkDataset.from_edge_embeddings(
    labels=edge_labels,
    edge_index=edge_index,
    node_embeddings=node_z,
    node_ids=node_ids,
    composition="hadamard",
)
```

Supported composition methods are `concat`, `hadamard`, `abs_diff`, and
`average`. Triplet construction composes anchor-positive and anchor-negative pairs
with the selected method, then concatenates those two pair representations into one
supervised row.

Multi-label datasets can use per-sample label sequences:

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
```

They can also use a binary indicator matrix. Pass `label_names` when you want
report fields and `k` mappings to use semantic names:

```python
dataset = BenchmarkDataset.from_arrays(
    X=samples,
    y=indicator_matrix,
    modality="image",
    label_names=["animal", "vehicle", "outdoor"],
)
```

For DataFrames, pass multiple label columns to `label_col`; those columns are
treated as 0/1 indicator columns and their names become `label_names`.

## Named target views

Datasets can also carry multiple aligned target projections through
`with_target_views(...)`. Each `TargetView` is validated like a normal dataset
target and can be single-label, multi-label, or explicit regression.

```python
from vertebrae import BenchmarkDataset, TargetView

dataset = BenchmarkDataset.from_embeddings(embeddings, labels).with_target_views(
    [
        TargetView(name="coarse", targets=coarse_labels),
        TargetView(
            name="score",
            targets=scores,
            target_type="regression",
            target_names=["score"],
        ),
    ]
)

score_dataset = dataset.target_view("score")
```

This keeps one shared embedding sample axis while exposing multiple transfer
targets for the same representation. Named target views still describe embedding
efficacy only; they do not add mAP, IoU, WER, CER, OKS, depth-error, or other
task-native metrics.

## Raw structured unit annotations

When you want `vertebrae` to materialize raw token, frame, region, keypoint, or
other structured outputs from an extractor directly, attach per-parent unit
annotations to the source dataset with `with_unit_annotations(...)`. Each
annotation describes the aligned supervised unit labels plus optional unit ids,
positions, spans, coordinates, and provenance.

```python
from vertebrae import BenchmarkDataset, UnitAnnotation

dataset = BenchmarkDataset.from_arrays(
    X=texts,
    y=document_labels,
    modality="text",
).with_unit_annotations(
    [
        UnitAnnotation(labels=["title", "body"], spans=[[0, 5], [6, 42]]),
        UnitAnnotation(labels=["body", "footer"], spans=[[0, 30], [31, 40]]),
    ],
    unit_type="token",
)
```

This keeps the original parent-sample dataset intact until a structured
extractor materializes one embedding row per retained unit. The same pattern
works across several structured domains:

- OCR/document layout regions: use `coordinates`, stable `unit_ids`, and page or
  document provenance per region.
- ASR or NLP tokens: use `positions`, `spans`, and utterance or sequence
  provenance per token.
- Pose or keypoint workflows: use `coordinates` plus frame/person provenance per
  keypoint.

```python
layout_dataset = BenchmarkDataset.from_arrays(
    X=page_images,
    y=document_labels,
    modality="image",
).with_unit_annotations(
    [
        UnitAnnotation(
            labels=["header", "table", "footer"],
            unit_ids=["page-0:h", "page-0:t", "page-0:f"],
            coordinates=[
                [0.05, 0.05, 0.95, 0.18],
                [0.08, 0.22, 0.92, 0.76],
                [0.10, 0.82, 0.85, 0.92],
            ],
            provenance=[{"page": 0}] * 3,
        ),
    ],
    unit_type="document_region",
)
```

```python
asr_dataset = BenchmarkDataset.from_arrays(
    X=waveforms,
    y=utterance_labels,
    modality="audio",
).with_unit_annotations(
    [
        UnitAnnotation(
            labels=["greeting", "intent", "entity"],
            positions=[0, 1, 2],
            spans=[[0.0, 0.2], [0.2, 0.5], [0.5, 0.9]],
            provenance=[{"utterance_id": "utt-0"}] * 3,
        ),
    ],
    unit_type="token",
)
```

```python
pose_dataset = BenchmarkDataset.from_arrays(
    X=frames,
    y=activity_labels,
    modality="image",
).with_unit_annotations(
    [
        UnitAnnotation(
            labels=["shoulder", "elbow", "wrist"],
            coordinates=[[0.30, 0.20], [0.42, 0.34], [0.55, 0.48]],
            provenance=[{"frame": 0, "person": "p0"}] * 3,
        ),
    ],
    unit_type="keypoint",
)
```

These materialized unit datasets still evaluate embedding efficacy only. They do
not add native task metrics such as mAP, IoU, WER/CER, OKS, or depth error. The
downstream workflow is still an embedding-efficacy diagnostic over the flattened
units, not a task-native OCR, ASR, detection, pose, or depth benchmark.

For the most common structured task families, `vertebrae` also exposes typed
adapter helpers that normalize domain annotations into the same
`with_unit_annotations(...)` foundation:

- `DetectionLayoutAdapter` + `RegionAnnotation`
- `SequenceLabelingAdapter` + `SequenceAnnotation`
- `KeypointAdapter` + `KeypointAnnotation`
- `DepthAdapter` + `DepthAnnotation`
- `LatentSlotAdapter` + `LatentSlotAnnotation`

```python
from vertebrae import (
    BenchmarkDataset,
    DetectionLayoutAdapter,
    RegionAnnotation,
)

dataset = DetectionLayoutAdapter(unit_type="document_region").attach(
    BenchmarkDataset.from_arrays(
        X=page_images,
        y=document_labels,
        modality="image",
    ),
    [
        RegionAnnotation(
            labels=["header", "table", "footer"],
            unit_ids=["page-0:h", "page-0:t", "page-0:f"],
            boxes=[
                [0.05, 0.05, 0.95, 0.18],
                [0.08, 0.22, 0.92, 0.76],
                [0.10, 0.82, 0.85, 0.92],
            ],
            page_id="page-0",
            document_id="doc-0",
        ),
    ] * len(page_images),
)
```

The adapters preserve domain conventions in metadata and per-unit provenance,
but they still normalize into the same unit-annotation schema so the standard
structured benchmark, artifact, report, and scoring paths stay unchanged.

## Explicit structured alignment

Some models emit raw structured outputs with extra rows or a different unit
order, such as a leading special token, unmatched latent slots, or filtered
frame outputs. In those cases, pass an explicit `StructuredUnitAligner` to
`materialize_structured_outputs(...)`, `materialize_structured_artifacts(...)`,
or `Benchmark(..., structured_aligners=...)`. Most deterministic selection
patterns can now use the built-in helper factories directly.

```python
from vertebrae import drop_special_rows, select_frame_rows

token_aligner = drop_special_rows(leading=1, trailing=1)
frame_aligner = select_frame_rows(indices_metadata_key="sampled_frames")
```

Aligners must return explicit one-to-one annotation and embedding row matches.
`vertebrae` validates bounds and uniqueness and records the alignment recipe in
materialized metadata and provenance. It does not perform automatic IoU-based,
temporal, Hungarian, or learned matching on the user's behalf.

## Validation rules

Every constructor validates the dataset immediately. Current validation checks
include:

- `X` and `y` must have the same number of samples.
- labels must be non-missing.
- the dataset must contain at least one sample.
- the dataset must contain at least two classes or labels.
- each class or label must contain at least two samples.

Multi-label targets additionally require every sample to have at least one label,
no duplicate labels within a sample, and binary values for indicator matrices.
For multi-label summaries, `class_counts` means per-label occurrence counts.

Explicit regression targets must be finite numeric 1D or 2D arrays, must contain
at least three samples, and must include at least one non-constant target column.
Regression summaries include `n_targets`, `target_names`, target means, target
variances, and any constant targets detected during validation.

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
- `"video"`
- `"time_series"`
- `"embeddings"`
- `"multimodal"`
- `"graph"`

Relational embedding constructors use modality `"embeddings"` and preserve a
`relational_unit` metadata field such as `"node"`, `"edge"`, `"entity"`,
`"pair"`, or `"triplet"`.

`metadata` is preserved through benchmarking so reports can retain source context
such as dataset name, split, backbone provenance, or collection notes.

Multi-modal datasets also preserve:

- `input_fields`: the declared aligned fields, such as `["image", "caption"]`,
- `modalities`: the per-field modality mapping, such as
  `{"image": "image", "caption": "text"}`.

Example:

```python
from vertebrae import BenchmarkDataset

dataset = BenchmarkDataset.from_multimodal(
    inputs={
        "image": image_paths,
        "caption": captions,
    },
    labels=labels,
    modalities={
        "image": "image",
        "caption": "text",
    },
)
```

V1 requires every declared modality to be present for every sample. Filter or
impute missing modalities before constructing the dataset.

Dataset summaries include `target_type`. For multi-label targets they also include
`label_names`, `labelset_counts`, mean label cardinality, and label density. For
regression targets they include `target_names`, `n_targets`, target statistics,
and constant-target diagnostics.

When using `from_dataframe(...)`, the dataset also records the original column names
and chosen input columns. When using `from_embeddings(...)`, metadata is tagged with
`precomputed_embeddings=True`. Audio-array datasets preserve their shared
`sampling_rate` in metadata. Video-array datasets preserve an optional shared
`frame_rate` in metadata. Time-series datasets preserve structured fields such as
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

Hierarchy-derived label views are classification-only. They are not available for
explicit regression datasets.

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

For dense semantic, instance, or panoptic evaluation, use `SegmentationDataset`
and the spatial extractor contracts documented in
[segmentation.md](segmentation.md).

Ordinary benchmark datasets can declare independence units with
`dataset.with_groups(groups, name="image_id")`. Groups remain aligned through
subsetting and are forwarded to grouped diagnostics without being serialized as
raw IDs in result reports.
