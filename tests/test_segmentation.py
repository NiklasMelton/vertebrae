from dataclasses import asdict

import numpy as np
import pytest

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CacheConfig,
    CallableSpatialExtractor,
    DatasetIdentity,
    ExecutionConfig,
    LocalBackend,
    ResourceProfilingConfig,
    SegmentationConfig,
    SegmentationDataset,
    SeparatixConfig,
    SpatialLayout,
    SpatialOutputSpec,
    StabilityConfig,
)
from vertebrae.cache import LocalArtifactStore
from vertebrae.cache.fingerprint import hash_json_exact
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


def test_segmentation_config_normalizes_serializable_ignored_instance_ids():
    config = SegmentationConfig(ignore_instance_ids=[0, 255])

    assert config.ignore_instance_ids == (0, 255)
    with pytest.raises(TypeError, match="iterable"):
        SegmentationConfig(ignore_instance_ids=0)
    with pytest.raises(TypeError, match="unsupported object"):
        SegmentationConfig(ignore_instance_ids=(object(),))


def test_segmentation_stuff_and_background_never_receive_instance_ids():
    dataset = SegmentationDataset.from_arrays(
        np.zeros((2, 2, 2, 3)),
        np.array([[[0, 2], [0, 2]], [[0, 2], [0, 2]]]),
        instance_masks=np.zeros((2, 2, 2), dtype=int),
        class_metadata={
            0: {"background": True, "is_thing": False},
            2: {"is_thing": False},
        },
        identity=DatasetIdentity.declared("stuff-sentinel", "1"),
    )
    values = np.arange(2 * 1 * 2 * 3, dtype=float).reshape(2, 1, 2, 3)
    extractor = CallableSpatialExtractor(
        "stuff-sentinel",
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
        SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include",
            max_instances_per_class=1,
            max_tokens_per_instance=1,
        ),
    )[0]

    assert materialized.metadata["retained_tokens"] == 4
    assert materialized.metadata["n_instances"] == 0
    assert all(row["instance_id"] is None for row in materialized.provenance)


def test_segmentation_ignored_thing_instance_zero_can_be_enabled_explicitly():
    dataset = SegmentationDataset.from_arrays(
        np.zeros((2, 1, 2, 3)),
        np.array([[[1, 2]], [[1, 2]]]),
        instance_masks=np.array([[[0, 5]], [[0, 5]]]),
        class_metadata={1: {"is_thing": True}, 2: {"is_thing": True}},
        identity=DatasetIdentity.declared("thing-zero", "1"),
    )
    values = np.arange(2 * 1 * 2 * 3, dtype=float).reshape(2, 1, 2, 3)

    def materialize(config):
        extractor = CallableSpatialExtractor(
            "thing-zero",
            transform_fn=lambda batch: values[: len(batch)],
            output_specs=[
                SpatialOutputSpec(
                    name="layer",
                    layout=SpatialLayout(grid_height=1, grid_width=2),
                )
            ],
        )
        return materialize_segmentation_outputs(dataset, extractor, config)[0]

    default = materialize(SegmentationConfig(coverage_threshold=1.0, ambiguity_margin=0.0))
    enabled = materialize(
        SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            ignore_instance_ids=(),
        )
    )

    assert default.metadata["n_instances"] == 2
    assert enabled.metadata["n_instances"] == 4
    assert [row["instance_id"] for row in default.provenance if row["label"] == 1] == [
        None,
        None,
    ]
    assert [row["instance_id"] for row in enabled.provenance if row["label"] == 1] == [
        0,
        0,
    ]


def test_segmentation_equal_instance_ids_in_different_classes_are_capped_independently():
    dataset = SegmentationDataset.from_arrays(
        np.zeros((2, 2, 2, 3)),
        np.array([[[1, 1], [2, 2]], [[1, 1], [2, 2]]]),
        instance_masks=np.full((2, 2, 2), 7),
        class_metadata={1: {"is_thing": True}, 2: {"is_thing": True}},
        identity=DatasetIdentity.declared("cross-class-instances", "1"),
    )
    values = np.arange(2 * 2 * 2 * 3, dtype=float).reshape(2, 2, 2, 3)
    extractor = CallableSpatialExtractor(
        "cross-class-instances",
        transform_fn=lambda batch: values[: len(batch)],
        output_specs=[
            SpatialOutputSpec(
                name="layer",
                layout=SpatialLayout(grid_height=2, grid_width=2),
            )
        ],
    )

    materialized = materialize_segmentation_outputs(
        dataset,
        extractor,
        SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            max_tokens_per_instance=1,
        ),
    )[0]

    assert materialized.metadata["retained_tokens"] == 4
    assert materialized.metadata["n_instances"] == 4
    assert materialized.dataset.y.tolist().count(1) == 2
    assert materialized.dataset.y.tolist().count(2) == 2


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


def test_segmentation_benchmark_dispatches_as_an_extractor_job(tmp_path, fake_overlapindex):
    result = Benchmark(
        dataset=_dataset(),
        extractors=[_extractor()],
        execution=LocalBackend(),
        execution_config=ExecutionConfig(total_shards=4),
        segmentation_config=SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include_excluded",
        ),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
    ).run()

    assert result.extractor_results[0].name == "spatial:layer"
    assert result.dataset_summary["segmentation_outputs"][0]["retained_tokens"] == 8
    assert result.metadata["execution"]["artifact_backed"] is True
    assert result.metadata["execution"]["effective_total_shards"] == [1]


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


def test_segmentation_artifact_keys_and_manifest_include_exact_resolved_config(tmp_path):
    store = LocalArtifactStore(tmp_path)
    common = {
        "coverage_threshold": 1.0,
        "ambiguity_margin": 0.0,
        "background_mode": "include_excluded",
    }
    default_sentinel = SegmentationConfig(**common)
    zero_is_valid = SegmentationConfig(**common, ignore_instance_ids=())
    dataset = _dataset()
    extractor = _extractor()

    first = materialize_segmentation_artifacts(
        dataset,
        extractor,
        store,
        segmentation_config=default_sentinel,
    )
    second = materialize_segmentation_artifacts(
        dataset,
        extractor,
        store,
        segmentation_config=zero_is_valid,
    )

    assert first["output_key"] != second["output_key"]
    assert first["segmentation_config_hash"] == hash_json_exact(asdict(default_sentinel))
    assert second["segmentation_config_hash"] == hash_json_exact(asdict(zero_is_valid))
    assert first["segmentation_config"]["ignore_instance_ids"] == [0]
    assert second["segmentation_config"]["ignore_instance_ids"] == []
    assert store.get_json(first["output_key"])["output_key"] == first["output_key"]
    assert store.get_json(second["output_key"])["output_key"] == second["output_key"]


def test_segmentation_artifacts_keep_formerly_colliding_names_independent(tmp_path):
    values = np.arange(2 * 2 * 2 * 3, dtype=float).reshape(2, 2, 2, 3)
    layout = SpatialLayout(grid_height=2, grid_width=2)
    extractor = CallableSpatialExtractor(
        "spatial-collisions",
        transform_fn=lambda batch: {
            "a/b": values[: len(batch)],
            "a_b": values[: len(batch)] + 100,
        },
        output_specs=[
            SpatialOutputSpec(name="a/b", layout=layout),
            SpatialOutputSpec(name="a_b", layout=layout),
        ],
    )
    store = LocalArtifactStore(tmp_path)

    bundle = materialize_segmentation_artifacts(
        _dataset(),
        extractor,
        store,
        segmentation_config=SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include_excluded",
        ),
    )

    assert [output["output_name"] for output in bundle["outputs"]] == ["a/b", "a_b"]
    keys = [output["output_key"] for output in bundle["outputs"]]
    assert len(set(keys)) == 2
    assert np.array_equal(store.get_array(keys[1]), store.get_array(keys[0]) + 100)
