# Segmentation datasets

Segmentation evaluation treats each retained spatial feature cell as an embedding
sample and its aligned semantic class as the scoring label. The score describes
semantic organization in dense representation space; it is not IoU, mask
accuracy, boundary accuracy, or a segmentation quality metric.

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
    instance_masks=instance_masks,
    class_metadata={
        0: {"background": True, "is_thing": False},
        1: {"is_thing": True},
        2: {"is_thing": False},
    },
    identity=DatasetIdentity.declared("example-dataset", "1"),
)

extractor = CallableSpatialExtractor(
    "backbone",
    transform_fn=extract_spatial_features,
    output_specs=[
        SpatialOutputSpec(
            name="encoder_8",
            layout=SpatialLayout(grid_height=14, grid_width=14, special_tokens=1),
        )
    ],
)

result = Benchmark(
    dataset=dataset,
    extractors=[extractor],
    segmentation_config=SegmentationConfig(
        background_mode="include_excluded",
        max_tokens_per_class=10_000,
        max_tokens_per_instance=128,
    ),
).run()
```

The default background mode is `ignore`. `include` scores background as an
ordinary class. `include_excluded` includes background samples in prototype and
pairwise computation while excluding background from global aggregation.

Dataset construction validates raster contracts before extraction. Each semantic
mask is a nonempty 2D raster whose cells are non-missing, hashable scalar labels;
every optional instance raster has the identical shape and the same scalar-value
requirements. `class_metadata` must be a mapping keyed only by semantic labels that
actually occur, image and annotation counts must align, and `ignore_labels` is
materialized once so generator inputs behave deterministically. Spatial grid heights
and widths must be positive integers.

Coverage is computed over each declared uniform spatial cell. Cells are retained
when the leading class reaches `coverage_threshold` and exceeds the second class
by `ambiguity_margin`. Sampling is deterministic and can be capped per instance,
class, and background.

Instance sampling applies only to non-background classes declared or inferred as
things. Stuff and background tokens always have `instance_id=None`, even when an
instance raster uses a numeric fill value for those pixels. By default,
`ignore_instance_ids=(0,)` treats zero as a sentinel for thing regions as well;
set `ignore_instance_ids=()` when zero is a legitimate instance ID. Instance
counts and per-instance caps use the full `(image_id, semantic_label, instance_id)`
identity, so the same raster ID in two semantic classes represents two distinct
instances.

Each retained token carries its source image as an independence group. Separatix
therefore uses image-disjoint evaluation. If grouped Separatix support is
insufficient, vertebrae records a structured skipped diagnostic and never retries
row-wise.

Segmentation materialization stages retained token embeddings and candidate
metadata incrementally. With `MemoryConfig(allow_disk_spill=True)`, records and
the deterministic priority/cap selection state use indexed local temporary
storage, so sampling does not require an in-memory candidate sort. An over-budget
dense result is exposed as a disk-backed array, and sparse rows remain sparse
throughout staging and final assembly. Returned labels, image groups, token metadata,
and provenance remain resident; their estimated final footprint is admitted before
allocation even when spill is enabled. With disk spill disabled, staged and final
embeddings and metadata share one peak admission budget and materialization fails
before retaining or allocating more than that budget.

Subsetting a `SegmentationDataset` records the selected original image indices in
`metadata["sample_indices"]`. Nested subsets compose those indices, and token
groups and provenance continue to report the original dataset positions rather
than subset-local positions. Empty subsets and misaligned source-index metadata
fail during dataset construction.

Torch and Keras wrappers accept `spatial_output_fn` plus explicit
`spatial_output_specs`. `HFVisionExtractor` accepts `spatial_outputs` with
`grid_shape`, `special_tokens`, and optional hidden-layer selection. Supply an
`annotation_transform` whenever preprocessing crops or otherwise changes image
geometry; vertebrae does not guess arbitrary processor transforms.

If spatial embeddings are already flattened, use
`BenchmarkDataset.from_segmentation_embeddings(embeddings, labels, image_ids,
identity=...)`.
This bypasses alignment while retaining image-disjoint diagnostic groups.
