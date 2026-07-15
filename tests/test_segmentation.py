from dataclasses import asdict

import numpy as np
import pytest

import vertebrae.utils.memory as memory_module
from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CacheConfig,
    CallableSpatialExtractor,
    DatasetIdentity,
    ExecutionConfig,
    LocalBackend,
    MemoryConfig,
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
from vertebrae.utils.labels import semantic_label_key


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


def _large_segmentation_case(size=24):
    row, column = np.indices((size, size))
    semantic = np.where((row + column) % 2 == 0, 1, 2)
    instances = row // 4 + 1
    features = np.stack(
        [row.astype(float), column.astype(float), semantic.astype(float)],
        axis=-1,
    )[None, ...]
    dataset = SegmentationDataset.from_arrays(
        np.zeros((1, size, size, 3), dtype=np.uint8),
        semantic[None, ...],
        instance_masks=instances[None, ...],
        class_metadata={
            1: {"is_thing": True},
            2: {"is_thing": True},
        },
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableSpatialExtractor(
        "large-spatial",
        transform_fn=lambda batch: features[: len(batch)],
        output_specs=[
            SpatialOutputSpec(
                name="layer",
                layout=SpatialLayout(grid_height=size, grid_width=size),
            )
        ],
        streaming_safe=True,
    )
    config = SegmentationConfig(
        coverage_threshold=1.0,
        ambiguity_margin=0.0,
        max_tokens_per_class=17,
        max_instances_per_class=3,
        max_tokens_per_instance=4,
        ignore_instance_ids=(),
        random_state=73,
    )
    return dataset, extractor, config


def test_segmentation_materialization_honors_non_streaming_extractors():
    calls = []
    values = np.arange(2 * 2 * 2 * 3, dtype=float).reshape(2, 2, 2, 3)

    def transform(batch):
        calls.append(len(batch))
        if len(batch) != 2:
            raise AssertionError("full image context is required")
        return values

    extractor = CallableSpatialExtractor(
        "contextual",
        transform_fn=transform,
        output_specs=[SpatialOutputSpec("layer", SpatialLayout(grid_height=2, grid_width=2))],
        streaming_safe=False,
    )

    materialized = materialize_segmentation_outputs(
        _dataset(),
        extractor,
        SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include_excluded",
        ),
        batch_size=1,
    )

    assert len(materialized) == 1
    assert calls == [2]


def test_segmentation_materialization_identity_includes_source_extractor_recipe():
    values = np.arange(2 * 2 * 2 * 3, dtype=float).reshape(2, 2, 2, 3)

    def make_extractor(revision):
        return CallableSpatialExtractor(
            "same-name",
            transform_fn=lambda batch: values[: len(batch)],
            output_specs=[SpatialOutputSpec("layer", SpatialLayout(grid_height=2, grid_width=2))],
            recipe_data={"revision": revision},
        )

    config = SegmentationConfig(
        coverage_threshold=1.0,
        ambiguity_margin=0.0,
        background_mode="include_excluded",
    )
    first = materialize_segmentation_outputs(_dataset(), make_extractor("first"), config)[0]
    second = materialize_segmentation_outputs(_dataset(), make_extractor("second"), config)[0]

    assert first.dataset.identity_key() != second.dataset.identity_key()


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


def test_segmentation_materialization_spills_dense_final_assembly_under_tiny_budget():
    materialized = materialize_segmentation_outputs(
        _dataset(),
        _extractor(),
        SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include_excluded",
        ),
        memory_config=MemoryConfig(max_memory_bytes=64_000, allow_disk_spill=True),
    )[0]

    assert materialized.dataset.X.shape == (8, 3)
    assert materialized.metadata["memory"]["strategy"] == "disk_spill"
    assert materialized.metadata["memory"]["required_bytes"] > 1
    assert materialized.metadata["memory"]["final_metadata_required_bytes"] <= 64_000


def test_segmentation_final_metadata_is_admitted_even_when_spill_is_enabled():
    with pytest.raises(ValueError, match="Segmentation final row metadata.*remain resident"):
        materialize_segmentation_outputs(
            _dataset(),
            _extractor(),
            SegmentationConfig(
                coverage_threshold=1.0,
                ambiguity_margin=0.0,
                background_mode="include_excluded",
            ),
            memory_config=MemoryConfig(max_memory_bytes=1, allow_disk_spill=True),
        )

    with pytest.raises(ValueError, match="fixed model/raw-batch memory"):
        materialize_segmentation_outputs(
            _dataset(),
            _extractor(),
            SegmentationConfig(
                coverage_threshold=1.0,
                ambiguity_margin=0.0,
                background_mode="include_excluded",
            ),
            memory_config=MemoryConfig(
                max_memory_bytes=64_000,
                raw_batch_memory_bytes=63_000,
                allow_disk_spill=True,
            ),
        )


def test_segmentation_final_metadata_peak_is_admitted_without_spill():
    with pytest.raises(ValueError, match="Segmentation final row metadata.*memory budget"):
        materialize_segmentation_outputs(
            _dataset(),
            _extractor(),
            SegmentationConfig(
                coverage_threshold=1.0,
                ambiguity_margin=0.0,
                background_mode="include_excluded",
            ),
            memory_config=MemoryConfig(
                max_memory_bytes=60_000,
                allow_disk_spill=False,
            ),
        )


def test_segmentation_streams_multiple_outputs_to_disk_without_dense_row_accumulation(
    monkeypatch,
):
    calls = []
    values = np.arange(2 * 2 * 2 * 3, dtype=float).reshape(2, 2, 2, 3)

    def transform(batch):
        calls.append(len(batch))
        return {
            "early": values[: len(batch)],
            "late": values[: len(batch)] + 100,
        }

    layout = SpatialLayout(grid_height=2, grid_width=2)
    extractor = CallableSpatialExtractor(
        "streaming-multi-output",
        transform_fn=transform,
        output_specs=[
            SpatialOutputSpec(name="early", layout=layout),
            SpatialOutputSpec(name="late", layout=layout),
        ],
        streaming_safe=True,
    )
    original_append = memory_module.IncrementalMatrixStager.append

    def append_and_assert_staged(self, output_name, row):
        reference = original_append(self, output_name, row)
        assert not self._entries
        assert not self._tokens_by_output
        return reference

    def reject_vstack(*_args, **_kwargs):
        raise AssertionError("disk-staged dense rows must not be accumulated for vstack")

    monkeypatch.setattr(memory_module.IncrementalMatrixStager, "append", append_and_assert_staged)
    monkeypatch.setattr(memory_module.np, "vstack", reject_vstack)

    materialized = materialize_segmentation_outputs(
        _dataset(),
        extractor,
        SegmentationConfig(
            coverage_threshold=1.0,
            ambiguity_margin=0.0,
            background_mode="include_excluded",
        ),
        batch_size=1,
        memory_config=MemoryConfig(max_memory_bytes=100_000, allow_disk_spill=True),
    )

    assert calls == [1, 1]
    assert [output.name for output in materialized] == ["early", "late"]
    assert all(isinstance(output.dataset.X.base, np.memmap) for output in materialized)
    assert all(output.metadata["memory"]["strategy"] == "disk_spill" for output in materialized)
    assert all(output.metadata["memory"]["staging_strategy"] == "disk" for output in materialized)
    assert np.array_equal(materialized[1].dataset.X, materialized[0].dataset.X + 100)

    with pytest.raises(
        ValueError,
        match="Segmentation final row metadata.*estimated cumulative.*memory budget",
    ):
        materialize_segmentation_outputs(
            _dataset(),
            extractor,
            SegmentationConfig(
                coverage_threshold=1.0,
                ambiguity_margin=0.0,
                background_mode="include_excluded",
            ),
            batch_size=1,
            memory_config=MemoryConfig(
                max_memory_bytes=64_000,
                allow_disk_spill=True,
            ),
        )


def test_segmentation_materialization_rejects_over_budget_assembly_without_spill():
    with pytest.raises(ValueError, match="allow_disk_spill"):
        materialize_segmentation_outputs(
            _dataset(),
            _extractor(),
            SegmentationConfig(
                coverage_threshold=1.0,
                ambiguity_margin=0.0,
                background_mode="include_excluded",
            ),
            memory_config=MemoryConfig(max_memory_bytes=1, allow_disk_spill=False),
        )


def test_segmentation_metadata_spill_preserves_sampling_and_provenance(monkeypatch):
    dataset, extractor, config = _large_segmentation_case()
    baseline = materialize_segmentation_outputs(
        dataset,
        extractor,
        config,
        batch_size=1,
        memory_config=MemoryConfig(
            max_memory_bytes=50_000_000,
            allow_disk_spill=False,
        ),
    )[0]
    original_append = memory_module.IncrementalMetadataStager.append
    original_dense_assembly = memory_module._assemble_staged_dense_entries
    observed_disk_appends = 0
    observed_streaming_assemblies = 0

    def append_and_assert_bounded(self, *args, **kwargs):
        nonlocal observed_disk_appends
        reference = original_append(self, *args, **kwargs)
        if self.strategy == "disk":
            observed_disk_appends += 1
            assert self.resident_bytes == 0
            assert not self._entries
            assert not self.matrix_stager._entries
            assert not self.matrix_stager._tokens_by_output
        return reference

    def assemble_from_stream(stage, entries, **kwargs):
        nonlocal observed_streaming_assemblies
        observed_streaming_assemblies += 1
        assert not isinstance(entries, (list, tuple))
        return original_dense_assembly(stage, entries, **kwargs)

    monkeypatch.setattr(
        memory_module.IncrementalMetadataStager,
        "append",
        append_and_assert_bounded,
    )
    monkeypatch.setattr(
        memory_module,
        "_assemble_staged_dense_entries",
        assemble_from_stream,
    )
    spilled = materialize_segmentation_outputs(
        dataset,
        extractor,
        config,
        batch_size=1,
        memory_config=MemoryConfig(max_memory_bytes=200_000, allow_disk_spill=True),
    )[0]

    assert observed_disk_appends == 24 * 24
    assert observed_streaming_assemblies == 1
    assert np.array_equal(spilled.dataset.X, baseline.dataset.X)
    assert np.array_equal(spilled.dataset.y, baseline.dataset.y)
    assert spilled.dataset.groups().tolist() == baseline.dataset.groups().tolist()
    assert spilled.provenance == baseline.provenance
    assert spilled.dataset.identity_key() == baseline.dataset.identity_key()
    assert {key: value for key, value in spilled.metadata.items() if key != "memory"} == {
        key: value for key, value in baseline.metadata.items() if key != "memory"
    }
    assert spilled.metadata["memory"]["metadata_staging_strategy"] == "disk"
    assert baseline.metadata["memory"]["metadata_staging_strategy"] == "memory"


def test_segmentation_metadata_fails_before_unbounded_retention_without_spill():
    dataset, extractor, config = _large_segmentation_case()

    with pytest.raises(ValueError, match="Segmentation candidate metadata.*allow_disk_spill"):
        materialize_segmentation_outputs(
            dataset,
            extractor,
            config,
            batch_size=1,
            memory_config=MemoryConfig(
                max_memory_bytes=40_000,
                allow_disk_spill=False,
            ),
        )


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
    assert item.overlap.metadata["exclude_classes"] == [semantic_label_key(0)]
    assert fake_overlapindex.calls[-1]["exclude_classes"] == [semantic_label_key(0)]
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
    assert store.get_labels(output["groups_key"]).tolist() == [
        semantic_label_key(value) for value in [0, 0, 0, 0, 1, 1, 1, 1]
    ]
    assert len(store.get_json(output["provenance_key"])["rows"]) == 8


def test_segmentation_artifacts_pass_memory_config_to_spill_assembly(tmp_path):
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
        memory_config=MemoryConfig(max_memory_bytes=64_000, allow_disk_spill=True),
    )

    output = bundle["outputs"][0]
    assert output["segmentation"]["memory"]["strategy"] == "disk_spill"
    assert store.get_array(output["output_key"]).shape == (8, 3)


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
