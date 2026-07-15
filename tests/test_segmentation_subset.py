import numpy as np
import pytest

from vertebrae import (
    CallableSpatialExtractor,
    DatasetIdentity,
    SegmentationAnnotation,
    SegmentationConfig,
    SegmentationDataset,
    SpatialLayout,
    SpatialOutputSpec,
)
from vertebrae.segmentation import materialize_segmentation_outputs


def test_segmentation_dataset_validates_direct_construction_and_sample_indices():
    identity = DatasetIdentity.declared("segmentation", "1")
    annotation = SegmentationAnnotation(semantic=[[1, 1], [2, 2]])

    dataset = SegmentationDataset(
        X=[np.zeros((2, 2, 3))],
        annotations=[annotation],
        identity=identity,
        metadata={"sample_indices": [7]},
    )

    assert isinstance(dataset.annotations[0].semantic, np.ndarray)
    assert dataset.metadata["sample_indices"] == [7]
    with pytest.raises(ValueError, match="at least one image"):
        SegmentationDataset(X=[], annotations=[], identity=identity)
    with pytest.raises(ValueError, match="one entry per image"):
        SegmentationDataset(
            X=[np.zeros((2, 2, 3))],
            annotations=[annotation],
            identity=identity,
            metadata={"sample_indices": [1, 2]},
        )


def test_segmentation_nested_subsets_preserve_original_image_indices_and_identity():
    images = np.zeros((4, 2, 2, 3))
    masks = np.array([[[1, 2], [1, 2]]] * 4)
    dataset = SegmentationDataset.from_arrays(
        images,
        masks,
        identity=DatasetIdentity.declared("segmentation-subsets", "1"),
    )

    first = dataset.subset([3, 1, 2])
    nested = first.subset([0, 2])
    same = first.subset([0, 2])
    different = first.subset([1, 2])

    assert first.metadata["sample_indices"] == [3, 1, 2]
    assert nested.metadata["sample_indices"] == [3, 2]
    assert nested.metadata["subset"] is True
    assert nested.metadata["parent_n_images"] == 3
    assert nested.identity_key() == same.identity_key()
    assert nested.identity_key() != different.identity_key()
    with pytest.raises(ValueError, match="at least one image"):
        dataset.subset([])


def test_segmentation_materialization_uses_original_image_indices_after_subset():
    images = np.zeros((4, 2, 2, 3))
    masks = np.array([[[1, 2], [1, 2]]] * 4)
    dataset = SegmentationDataset.from_arrays(
        images,
        masks,
        identity=DatasetIdentity.declared("segmentation-provenance", "1"),
    ).subset([3, 1])
    values = np.arange(2 * 1 * 2 * 3, dtype=float).reshape(2, 1, 2, 3)
    extractor = CallableSpatialExtractor(
        "source-indices",
        transform_fn=lambda batch: values[: len(batch)],
        output_specs=[
            SpatialOutputSpec(
                name="layer",
                layout=SpatialLayout(grid_height=1, grid_width=2),
            )
        ],
    )

    materialized = materialize_segmentation_outputs(
        dataset,
        extractor,
        SegmentationConfig(coverage_threshold=1.0, ambiguity_margin=0.0),
    )[0]

    assert materialized.dataset.groups().tolist() == [1, 1, 3, 3]
    assert [row["image_id"] for row in materialized.provenance] == [1, 1, 3, 3]
