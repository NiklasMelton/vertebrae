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

Coverage is computed over each declared uniform spatial cell. Cells are retained
when the leading class reaches `coverage_threshold` and exceeds the second class
by `ambiguity_margin`. Sampling is deterministic and can be capped per instance,
class, and background.

Each retained token carries its source image as an independence group. Separatix
therefore uses image-disjoint evaluation. If grouped Separatix support is
insufficient, vertebrae records a structured skipped diagnostic and never retries
row-wise.

Torch and Keras wrappers accept `spatial_output_fn` plus explicit
`spatial_output_specs`. `HFVisionExtractor` accepts `spatial_outputs` with
`grid_shape`, `special_tokens`, and optional hidden-layer selection. Supply an
`annotation_transform` whenever preprocessing crops or otherwise changes image
geometry; vertebrae does not guess arbitrary processor transforms.

If spatial embeddings are already flattened, use
`BenchmarkDataset.from_segmentation_embeddings(embeddings, labels, image_ids,
identity=...)`.
This bypasses alignment while retaining image-disjoint diagnostic groups.
