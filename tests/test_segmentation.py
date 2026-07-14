import numpy as np

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CacheConfig,
    CallableSpatialExtractor,
    DatasetIdentity,
    ResourceProfilingConfig,
    SegmentationConfig,
    SegmentationDataset,
    SeparatixConfig,
    SpatialLayout,
    SpatialOutputSpec,
    StabilityConfig,
)
from vertebrae.cache import LocalArtifactStore
from vertebrae.execution import materialize_segmentation_artifacts
from vertebrae.segmentation import materialize_segmentation_outputs


def _dataset():
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    masks = np.array(
        [
            [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1]],
        ]
    )
    return SegmentationDataset.from_arrays(
        images,
        masks,
        class_metadata={
            0: {"background": True, "is_thing": False},
            1: {"is_thing": True},
            2: {"is_thing": False},
        },
        identity=DatasetIdentity.ephemeral(),
    )


def _extractor():
    values = np.arange(2 * 2 * 2 * 3, dtype=float).reshape(2, 2, 2, 3)
    return CallableSpatialExtractor(
        "spatial",
        transform_fn=lambda batch: values[: len(batch)],
        output_specs=[
            SpatialOutputSpec(
                name="layer",
                layout=SpatialLayout(grid_height=2, grid_width=2),
            )
        ],
    )


def test_segmentation_alignment_materializes_grouped_tokens():
    materialized = materialize_segmentation_outputs(
        _dataset(),
        _extractor(),
        SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include_excluded",
        ),
    )[0]

    assert materialized.dataset.X.shape == (8, 3)
    assert materialized.dataset.groups().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert materialized.metadata["background_tokens"] == 2
    assert materialized.metadata["retained_tokens"] == 8
    assert "groups" not in materialized.dataset.summary()["metadata"]
    assert materialized.dataset.summary()["grouping"]["n_groups"] == 2


def test_precomputed_segmentation_embeddings_use_image_groups():
    dataset = SegmentationDataset.from_arrays(
        np.zeros((2, 2, 2, 3)),
        np.array([[[1, 1], [2, 2]], [[1, 1], [2, 2]]]),
        identity=DatasetIdentity.ephemeral(),
    )
    grouped = dataset  # keep the constructor smoke test close to segmentation fixtures
    token_dataset = BenchmarkDataset.from_segmentation_embeddings(
        np.eye(4),
        [1, 1, 2, 2],
        ["image-a", "image-a", "image-b", "image-b"],
        identity=DatasetIdentity.ephemeral(),
    )

    assert grouped.summary()["n_images"] == 2
    assert token_dataset.groups().tolist() == [
        "image-a",
        "image-a",
        "image-b",
        "image-b",
    ]


def test_segmentation_benchmark_reuses_standard_scoring_pipeline(
    tmp_path,
    fake_overlapindex,
):
    result = Benchmark(
        dataset=_dataset(),
        extractors=[_extractor()],
        segmentation_config=SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include_excluded",
        ),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        resource_profiling_config=ResourceProfilingConfig(enabled=True),
    ).run()

    item = result.extractor_results[0]
    assert item.name == "spatial:layer"
    assert item.embedding_metadata["segmentation"]["retained_tokens"] == 8
    assert item.overlap.metadata["exclude_classes"] == [0]
    assert fake_overlapindex.calls[-1]["exclude_classes"] == [0]
    assert result.dataset_summary["n_images"] == 2
    assert item.resource_profile.inference.status == "measured"
    assert item.resource_profile.context["call_types"] == ["transform_spatial"]
    assert item.resource_profile.embedding.evaluated_bytes == 8 * 3 * 8


def test_segmentation_artifacts_have_independent_output_boundaries(tmp_path):
    store = LocalArtifactStore(tmp_path)

    bundle = materialize_segmentation_artifacts(
        _dataset(),
        _extractor(),
        store,
        segmentation_config=SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include_excluded",
        ),
    )

    output = bundle["outputs"][0]
    assert output["artifact_type"] == "segmentation_embedding"
    assert store.get_array(output["output_key"]).shape == (8, 3)
    assert store.get_labels(output["labels_key"]).shape == (8,)
    assert store.get_labels(output["groups_key"]).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert len(store.get_json(output["provenance_key"])["rows"]) == 8
