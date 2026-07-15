from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import vertebrae.structured as structured_module
import vertebrae.utils.memory as memory_module
from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CallableStructuredExtractor,
    DatasetIdentity,
    ExecutionConfig,
    LocalBackend,
    TargetView,
    UnitAnnotation,
    drop_special_rows,
    select_frame_rows,
)
from vertebrae.cache import LocalArtifactStore
from vertebrae.config import (
    CacheConfig,
    MemoryConfig,
    ResourceProfilingConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.execution import materialize_structured_artifacts
from vertebrae.extractors import StructuredOutputSpec
from vertebrae.structured import (
    iter_materialize_structured_outputs,
    materialize_structured_outputs,
)
from vertebrae.utils.semantic_labels import semantic_label_key


def _annotations():
    return [
        UnitAnnotation(
            labels=["x", "y"],
            unit_ids=["a:0", "a:1"],
            spans=[[0, 1], [1, 2]],
        ),
        UnitAnnotation(
            labels=["x", "y"],
            unit_ids=["b:0", "b:1"],
            spans=[[0, 1], [1, 2]],
        ),
        UnitAnnotation(
            labels=["y", "x"],
            unit_ids=["c:0", "c:1"],
            spans=[[0, 1], [1, 2]],
        ),
        UnitAnnotation(
            labels=["y", "x"],
            unit_ids=["d:0", "d:1"],
            spans=[[0, 1], [1, 2]],
        ),
    ]


def _dataset():
    dataset = BenchmarkDataset.from_arrays(
        np.array(["a", "b", "c", "d"], dtype=object),
        ["doc_a", "doc_a", "doc_b", "doc_b"],
        modality="text",
        identity=DatasetIdentity.ephemeral(),
    ).with_target_views([TargetView(name="coarse", targets=["left", "left", "right", "right"])])
    return dataset.with_unit_annotations(_annotations(), unit_type="token", task_family="sequence")


def _extractor():
    values = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.5, 0.5], [1.0, 1.0]]),
        np.array([[0.5, 0.5], [1.0, 1.0]]),
    ]
    return CallableStructuredExtractor(
        "structured",
        transform_fn=lambda batch: values[: len(batch)],
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
    )


def _dataset_with_annotations(annotations):
    return BenchmarkDataset.from_arrays(
        np.array(["a", "b", "c", "d"], dtype=object),
        ["doc_a", "doc_a", "doc_b", "doc_b"],
        modality="text",
        identity=DatasetIdentity.ephemeral(),
    ).with_unit_annotations(annotations, unit_type="token")


def _two_row_extractor(transform_fn):
    return CallableStructuredExtractor(
        "structured",
        transform_fn=transform_fn,
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
    )


def _large_structured_case(n_units=96):
    parent_values = np.array(["large-a", "large-b", "large-c", "large-d"], dtype=object)
    annotations = []
    matrices = {}
    for parent_index, parent_value in enumerate(parent_values.tolist()):
        labels = ["x" if index % 2 == 0 else "y" for index in range(n_units)]
        annotations.append(
            UnitAnnotation(
                labels=labels,
                unit_ids=[f"{parent_value}:{index}" for index in range(n_units)],
                positions=list(range(n_units)),
                spans=[[index, index + 1] for index in range(n_units)],
                provenance=[
                    {"parent": parent_value, "source_unit": index} for index in range(n_units)
                ],
            )
        )
        matrices[parent_value] = (
            np.arange(n_units * 3, dtype=float).reshape(n_units, 3) + parent_index * 10_000
        )
    dataset = BenchmarkDataset.from_arrays(
        parent_values,
        ["left", "left", "right", "right"],
        modality="text",
        identity=DatasetIdentity.ephemeral(),
    ).with_unit_annotations(annotations, unit_type="token")
    extractor = CallableStructuredExtractor(
        "large-structured",
        transform_fn=lambda batch: [matrices[str(value)] for value in batch],
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
        streaming_safe=True,
    )
    return dataset, extractor


def test_structured_materialization_honors_non_streaming_extractors():
    calls = []
    values = [np.eye(2, dtype=float) for _ in range(4)]

    def transform(batch):
        calls.append(len(batch))
        if len(batch) != 4:
            raise AssertionError("full parent context is required")
        return values

    extractor = CallableStructuredExtractor(
        "contextual",
        transform_fn=transform,
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
        streaming_safe=False,
    )

    materialized = materialize_structured_outputs(_dataset(), extractor, batch_size=1)

    assert len(materialized) == 1
    assert calls == [4]


def test_structured_materialization_identity_includes_source_extractor_recipe():
    first = CallableStructuredExtractor(
        "same-name",
        transform_fn=lambda batch: [np.eye(2, dtype=float) for _ in batch],
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
        recipe_data={"revision": "first"},
    )
    second = CallableStructuredExtractor(
        "same-name",
        transform_fn=lambda batch: [np.eye(2, dtype=float) for _ in batch],
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
        recipe_data={"revision": "second"},
    )

    first_dataset = materialize_structured_outputs(_dataset(), first)[0].dataset
    second_dataset = materialize_structured_outputs(_dataset(), second)[0].dataset

    assert first_dataset.identity_key() != second_dataset.identity_key()


def test_structured_materialization_preserves_source_parent_ids_after_reordering():
    reordered = _dataset().subset([3, 1, 2, 0])

    materialized = materialize_structured_outputs(reordered, _extractor())[0]

    assert materialized.dataset.groups().tolist() == [3, 3, 1, 1, 2, 2, 0, 0]
    assert [row["parent_position"] for row in materialized.provenance] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_dataset_with_unit_annotations_survives_subset_and_summary():
    dataset = BenchmarkDataset.from_arrays(
        np.array(["a", "b", "c", "d"], dtype=object),
        [0.0, 0.1, 0.2, 0.3],
        modality="text",
        target_type="regression",
        target_names=["score"],
        identity=DatasetIdentity.ephemeral(),
    ).with_unit_annotations(_annotations(), unit_type="token")
    subset = dataset.subset([0, 1, 2])

    assert dataset.summary()["structured_units"]["n_units"] == 8
    assert subset.unit_annotations()[0]["unit_ids"] == ["a:0", "a:1"]
    assert subset.summary()["structured_units"]["n_parents"] == 3


def test_structured_materialization_flattens_units_and_target_views():
    materialized = materialize_structured_outputs(_dataset(), _extractor())[0]

    assert materialized.dataset.X.shape == (8, 2)
    assert materialized.dataset.groups().tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert materialized.dataset.metadata["unit_type"] == "token"
    assert materialized.dataset.target_view_names() == ["coarse"]
    assert materialized.dataset.target_view("coarse").y.tolist() == [
        "left",
        "left",
        "left",
        "left",
        "right",
        "right",
        "right",
        "right",
    ]


def test_structured_materialization_spills_dense_final_assembly_under_tiny_budget():
    materialized = materialize_structured_outputs(
        _dataset(),
        _extractor(),
        memory_config=MemoryConfig(max_memory_bytes=100_000, allow_disk_spill=True),
    )[0]

    assert materialized.dataset.X.shape == (8, 2)
    assert materialized.metadata["memory"]["strategy"] == "disk_spill"
    assert materialized.metadata["memory"]["required_bytes"] > 1
    assert materialized.metadata["memory"]["final_metadata_required_bytes"] <= 100_000
    assert materialized.metadata["memory"]["target_view_metadata_required_bytes"] > 0
    assert np.all(np.isfinite(materialized.dataset.X))


def test_structured_final_metadata_is_admitted_even_when_spill_is_enabled():
    with pytest.raises(ValueError, match="Structured final row metadata.*remain resident"):
        materialize_structured_outputs(
            _dataset(),
            _extractor(),
            memory_config=MemoryConfig(max_memory_bytes=1, allow_disk_spill=True),
        )

    with pytest.raises(ValueError, match="fixed model/raw-batch memory"):
        materialize_structured_outputs(
            _dataset(),
            _extractor(),
            memory_config=MemoryConfig(
                max_memory_bytes=100_000,
                model_memory_bytes=99_000,
                allow_disk_spill=True,
            ),
        )


def test_structured_target_views_are_admitted_before_materialization():
    dataset = _dataset().with_target_views(
        [
            TargetView(
                name=f"view-{index}",
                targets=["left", "left", "right", "right"],
            )
            for index in range(200)
        ]
    )

    with pytest.raises(
        ValueError,
        match="Structured final row metadata.*estimated cumulative.*memory budget",
    ):
        materialize_structured_outputs(
            dataset,
            _extractor(),
            memory_config=MemoryConfig(
                max_memory_bytes=100_000,
                allow_disk_spill=True,
            ),
        )


def test_structured_multi_output_accumulates_only_retained_target_view_bytes():
    dataset = _dataset().with_target_views(
        [
            TargetView(
                name=f"view-{index}",
                targets=["left", "left", "right", "right"],
            )
            for index in range(1000)
        ]
    )
    values = [np.eye(2, dtype=float) for _ in range(4)]
    extractor = CallableStructuredExtractor(
        "target-view-multi-output",
        transform_fn=lambda batch: {
            "early": values[: len(batch)],
            "late": [value * 2 for value in values[: len(batch)]],
        },
        output_specs=[
            StructuredOutputSpec(name="early", unit_type="token"),
            StructuredOutputSpec(name="late", unit_type="token"),
        ],
    )

    materialized = materialize_structured_outputs(
        dataset,
        extractor,
        memory_config=MemoryConfig(
            max_memory_bytes=9_000_000,
            allow_disk_spill=True,
        ),
    )

    assert [item.name for item in materialized] == ["early", "late"]
    assert (
        materialized[1].metadata["memory"]["cumulative_final_metadata_required_bytes"] < 9_000_000
    )


def test_structured_final_metadata_peak_is_admitted_without_spill():
    with pytest.raises(ValueError, match="Structured final row metadata.*memory budget"):
        materialize_structured_outputs(
            _dataset(),
            _extractor(),
            memory_config=MemoryConfig(
                max_memory_bytes=100_000,
                allow_disk_spill=False,
            ),
        )


def test_structured_materialization_rejects_over_budget_assembly_without_spill():
    with pytest.raises(ValueError, match="allow_disk_spill"):
        materialize_structured_outputs(
            _dataset(),
            _extractor(),
            memory_config=MemoryConfig(max_memory_bytes=1, allow_disk_spill=False),
        )


def test_structured_metadata_spill_preserves_many_rows_and_provenance(
    monkeypatch,
):
    dataset, extractor = _large_structured_case()
    baseline = materialize_structured_outputs(
        dataset,
        extractor,
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
    spilled = materialize_structured_outputs(
        dataset,
        extractor,
        batch_size=1,
        memory_config=MemoryConfig(max_memory_bytes=4_000_000, allow_disk_spill=True),
    )[0]

    assert observed_disk_appends == 4 * 96
    assert observed_streaming_assemblies == 1
    assert np.array_equal(spilled.dataset.X, baseline.dataset.X)
    assert np.array_equal(spilled.dataset.y, baseline.dataset.y)
    assert spilled.provenance == baseline.provenance
    assert spilled.dataset.metadata["unit_ids"] == baseline.dataset.metadata["unit_ids"]
    assert spilled.dataset.metadata["unit_spans"] == baseline.dataset.metadata["unit_spans"]
    assert (
        spilled.dataset.metadata["unit_provenance"] == baseline.dataset.metadata["unit_provenance"]
    )
    assert spilled.dataset.identity_key() == baseline.dataset.identity_key()
    assert spilled.metadata["memory"]["metadata_staging_strategy"] == "disk"
    assert baseline.metadata["memory"]["metadata_staging_strategy"] == "memory"


def test_structured_candidates_fail_before_unbounded_retention_without_spill():
    dataset, extractor = _large_structured_case()

    with pytest.raises(
        ValueError,
        match=r"Structured candidate (?:metadata|embeddings).*allow_disk_spill",
    ):
        materialize_structured_outputs(
            dataset,
            extractor,
            batch_size=1,
            memory_config=MemoryConfig(
                max_memory_bytes=40_000,
                allow_disk_spill=False,
            ),
        )


def test_sparse_structured_materialization_spills_without_densifying(monkeypatch):
    calls = []

    def transform(batch):
        calls.append(len(batch))
        return {
            "early": [sparse.coo_matrix(np.eye(2)) for _ in range(len(batch))],
            "late": [sparse.csc_matrix(np.eye(2) * 2) for _ in range(len(batch))],
        }

    extractor = CallableStructuredExtractor(
        "sparse-multi-output",
        transform_fn=transform,
        output_specs=[
            StructuredOutputSpec(name="early", unit_type="token"),
            StructuredOutputSpec(name="late", unit_type="token"),
        ],
        streaming_safe=True,
    )
    original_append = memory_module.IncrementalMatrixStager.append

    def append_and_assert_staged(self, output_name, row):
        reference = original_append(self, output_name, row)
        assert not self._entries
        assert not self._tokens_by_output
        return reference

    def reject_densification(*_args, **_kwargs):
        raise AssertionError("sparse materialization must not densify")

    monkeypatch.setattr(memory_module.IncrementalMatrixStager, "append", append_and_assert_staged)
    for sparse_type in (sparse.coo_matrix, sparse.csc_matrix, sparse.csr_matrix):
        monkeypatch.setattr(sparse_type, "toarray", reject_densification)

    materialized = materialize_structured_outputs(
        _dataset(),
        extractor,
        batch_size=1,
        memory_config=MemoryConfig(max_memory_bytes=160_000, allow_disk_spill=True),
    )

    assert calls == [1, 1, 1, 1]
    assert [output.name for output in materialized] == ["early", "late"]
    assert all(sparse.issparse(output.dataset.X) for output in materialized)
    assert all(isinstance(output.dataset.X.data, np.memmap) for output in materialized)
    assert all(output.metadata["memory"]["strategy"] == "disk_spill" for output in materialized)
    assert all(output.metadata["memory"]["staging_strategy"] == "disk" for output in materialized)
    assert (materialized[1].dataset.X != materialized[0].dataset.X * 2).nnz == 0

    with pytest.raises(
        ValueError,
        match="Structured final row metadata.*estimated cumulative.*memory budget",
    ):
        materialize_structured_outputs(
            _dataset(),
            extractor,
            batch_size=1,
            memory_config=MemoryConfig(
                max_memory_bytes=100_000,
                allow_disk_spill=True,
            ),
        )


def test_structured_staging_cleanup_runs_when_output_iteration_closes_early(
    tmp_path,
    monkeypatch,
):
    values = [np.eye(2, dtype=float) for _ in range(4)]
    extractor = CallableStructuredExtractor(
        "cleanup-multi-output",
        transform_fn=lambda batch: {
            "early": values[: len(batch)],
            "late": [value + 1 for value in values[: len(batch)]],
        },
        output_specs=[
            StructuredOutputSpec(name="early", unit_type="token"),
            StructuredOutputSpec(name="late", unit_type="token"),
        ],
    )
    original_temporary_directory = memory_module.tempfile.TemporaryDirectory
    created = []

    def tracked_temporary_directory(*args, **kwargs):
        kwargs["dir"] = tmp_path
        directory = original_temporary_directory(*args, **kwargs)
        created.append(Path(directory.name))
        return directory

    monkeypatch.setattr(
        memory_module.tempfile,
        "TemporaryDirectory",
        tracked_temporary_directory,
    )
    materializations = iter_materialize_structured_outputs(
        _dataset(),
        extractor,
        batch_size=1,
        memory_config=MemoryConfig(max_memory_bytes=100_000, allow_disk_spill=True),
    )

    first = next(materializations)
    assert first.name == "early"
    assert created and created[0].exists()

    materializations.close()

    assert created
    assert all(not path.exists() for path in created)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: drop_special_rows(leading=True),
        lambda: drop_special_rows(trailing=1.5),
        lambda: select_frame_rows(every_n=True),
        lambda: select_frame_rows(every_n=2, start=0.5),
    ],
)
def test_structured_alignment_helpers_require_exact_integer_options(factory):
    with pytest.raises(TypeError, match="non-boolean integer"):
        factory()


def test_structured_benchmark_reuses_standard_scoring_pipeline(tmp_path, fake_overlapindex):
    result = Benchmark(
        dataset=_dataset(),
        extractors=[_extractor()],
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        structured_aligners={"tokens": drop_special_rows()},
        resource_profiling_config=ResourceProfilingConfig(enabled=True),
    ).run()

    item = result.extractor_results[0]
    assert item.name == "structured:tokens"
    assert item.embedding_metadata["structured"]["n_units"] == 8
    assert result.dataset_summary["structured_outputs"][0]["unit_type"] == "token"
    assert result.dataset_summary["structured_outputs"][0]["task_family"] == "sequence"
    assert result.dataset_summary["structured_outputs"][0]["alignment_mode"] == "explicit"
    assert item.embedding_metadata["structured"]["alignment_recipe"]["name"] == "drop_special_rows"
    frame = result.to_dataframe()
    assert frame.loc[0, "task_family"] == "sequence"
    assert frame.loc[0, "alignment_mode"] == "explicit"
    assert frame.loc[0, "alignment_recipe"]["name"] == "drop_special_rows"
    assert item.resource_profile.inference.status == "measured"
    assert item.resource_profile.context["call_types"] == ["transform_structured"]
    assert item.resource_profile.embedding.evaluated_bytes == 8 * 2 * 8

    markdown_path = tmp_path / "structured_report.md"
    result.save_markdown(str(markdown_path))
    report = markdown_path.read_text(encoding="utf-8")
    assert "## Structured outputs" in report
    assert "Structured task family: sequence" in report
    assert "Alignment mode: explicit" in report
    assert "drop_special_rows (drop_special_rows)" in report


def test_structured_benchmark_slices_provenance_with_memory_subsampling(
    tmp_path,
    fake_overlapindex,
):
    result = Benchmark(
        dataset=_dataset(),
        extractors=[_extractor()],
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        memory_config=MemoryConfig(subsample_rate=0.5),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
    ).run()

    metadata = result.extractor_results[0].embedding_metadata
    assert metadata["n_samples"] == 4
    assert metadata["provenance_rows"] == 4
    assert metadata["cumulative_subsample_rate"] == pytest.approx(0.5)


def test_structured_benchmark_dispatches_as_an_extractor_job(tmp_path, fake_overlapindex):
    result = Benchmark(
        dataset=_dataset(),
        extractors=[_extractor()],
        execution=LocalBackend(),
        execution_config=ExecutionConfig(total_shards=3),
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        structured_aligners={"tokens": drop_special_rows()},
    ).run()

    assert result.extractor_results[0].name == "structured:tokens"
    assert result.dataset_summary["structured_outputs"][0]["n_units"] == 8
    assert result.metadata["execution"]["artifact_backed"] is True
    assert result.metadata["execution"]["effective_total_shards"] == [1]


def test_structured_artifacts_have_independent_output_boundaries(tmp_path):
    store = LocalArtifactStore(tmp_path)

    bundle = materialize_structured_artifacts(_dataset(), _extractor(), store)

    output = bundle["outputs"][0]
    assert output["artifact_type"] == "structured_embedding"
    assert store.get_array(output["output_key"]).shape == (8, 2)
    assert store.get_labels(output["labels_key"]).shape == (8,)
    assert store.get_labels(output["groups_key"]).tolist() == [
        semantic_label_key(value) for value in [0, 0, 1, 1, 2, 2, 3, 3]
    ]
    assert len(store.get_json(output["provenance_key"])["rows"]) == 8
    assert output["task_family"] == "sequence"
    assert output["alignment_mode"] == "strict"
    assert output["alignment_recipe"] is None


def test_structured_artifact_identity_includes_aligner_recipe_and_safety(tmp_path):
    store = LocalArtifactStore(tmp_path)

    strict = materialize_structured_artifacts(_dataset(), _extractor(), store)
    aligned = materialize_structured_artifacts(
        _dataset(),
        _extractor(),
        store,
        aligners={"tokens": drop_special_rows()},
    )
    assert strict["output_key"] != aligned["output_key"]
    assert aligned["aligner_recipes"]["tokens"]["cache_safe"] is True


def test_structured_artifacts_pass_memory_config_to_spill_assembly(tmp_path):
    store = LocalArtifactStore(tmp_path)

    bundle = materialize_structured_artifacts(
        _dataset(),
        _extractor(),
        store,
        memory_config=MemoryConfig(max_memory_bytes=100_000, allow_disk_spill=True),
    )

    output = bundle["outputs"][0]
    assert output["structured"]["memory"]["strategy"] == "disk_spill"
    assert store.get_array(output["output_key"]).shape == (8, 2)


def test_structured_artifacts_persist_sparse_spilled_outputs_without_densifying(
    tmp_path,
    monkeypatch,
):
    extractor = CallableStructuredExtractor(
        "sparse-artifacts",
        transform_fn=lambda batch: {
            "early": [sparse.csr_matrix(np.eye(2)) for _ in range(len(batch))],
            "late": [sparse.csr_matrix(np.eye(2) * 3) for _ in range(len(batch))],
        },
        output_specs=[
            StructuredOutputSpec(name="early", unit_type="token"),
            StructuredOutputSpec(name="late", unit_type="token"),
        ],
        streaming_safe=True,
    )

    def reject_densification(*_args, **_kwargs):
        raise AssertionError("sparse artifact persistence must not densify")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_densification)
    store = LocalArtifactStore(tmp_path)

    bundle = materialize_structured_artifacts(
        _dataset(),
        extractor,
        store,
        batch_size=1,
        memory_config=MemoryConfig(max_memory_bytes=160_000, allow_disk_spill=True),
    )

    assert [output["output_name"] for output in bundle["outputs"]] == ["early", "late"]
    stored = [store.get_array(output["output_key"]) for output in bundle["outputs"]]
    assert all(sparse.issparse(matrix) for matrix in stored)
    assert all(output["sparse"] is True for output in bundle["outputs"])
    assert all(
        output["structured"]["memory"]["strategy"] == "disk_spill" for output in bundle["outputs"]
    )
    assert (stored[1] != stored[0] * 3).nnz == 0


def test_structured_artifacts_validate_row_alignment_before_writes(tmp_path, monkeypatch):
    materialization = materialize_structured_outputs(_dataset(), _extractor())[0]
    materialization.provenance = materialization.provenance[:-1]
    monkeypatch.setattr(
        structured_module,
        "iter_materialize_structured_outputs",
        lambda *args, **kwargs: iter([materialization]),
    )
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="provenance rows"):
        materialize_structured_artifacts(_dataset(), _extractor(), store)

    assert not list(tmp_path.rglob("array-manifest.json"))


def test_structured_artifacts_keep_formerly_colliding_names_independent(tmp_path):
    values = _extractor().transform_fn(_dataset().X)
    extractor = CallableStructuredExtractor(
        "structured-collisions",
        transform_fn=lambda batch: {
            "a/b": values[: len(batch)],
            "a_b": [value + 10 for value in values[: len(batch)]],
        },
        output_specs=[
            StructuredOutputSpec(name="a/b", unit_type="token"),
            StructuredOutputSpec(name="a_b", unit_type="token"),
        ],
    )
    store = LocalArtifactStore(tmp_path)

    bundle = materialize_structured_artifacts(_dataset(), extractor, store)

    assert [output["output_name"] for output in bundle["outputs"]] == ["a/b", "a_b"]
    keys = [output["output_key"] for output in bundle["outputs"]]
    assert len(set(keys)) == 2
    assert np.array_equal(store.get_array(keys[1]), store.get_array(keys[0]) + 10)


def test_unit_annotations_support_indicator_and_labelset_multilabel_targets():
    indicator_annotations = [
        UnitAnnotation(
            labels=np.array([[1, 0], [0, 1]], dtype=int),
            label_names=["red", "blue"],
            target_type="multi_label",
        )
        for _ in range(4)
    ]
    indicator = materialize_structured_outputs(
        _dataset_with_annotations(indicator_annotations),
        _two_row_extractor(lambda batch: [np.eye(2, dtype=float) for _ in range(len(batch))]),
    )[0]

    assert indicator.dataset.metadata["target_type"] == "multi_label"
    assert indicator.dataset.metadata["label_names"] == ["red", "blue"]
    assert indicator.dataset.y.tolist() == [("red",), ("blue",)] * 4

    labelset_annotations = [
        UnitAnnotation(
            labels=[("red", "blue"), ("blue",)],
            label_names=["red", "blue"],
            target_type="multi_label",
        )
        for _ in range(4)
    ]
    attached = _dataset_with_annotations(labelset_annotations)
    assert attached.unit_annotations()[0]["labels"] == [["red", "blue"], ["blue"]]
    assert attached.unit_annotations()[0]["target_type"] == "multi_label"


def test_unit_annotations_support_multi_target_and_canonical_single_target_regression():
    multi_target_annotations = [
        UnitAnnotation(
            labels=np.array([[index, index + 1], [index + 2, index + 3]], dtype=float),
            target_type="regression",
            target_names=["depth", "uncertainty"],
        )
        for index in range(4)
    ]
    multi_target = materialize_structured_outputs(
        _dataset_with_annotations(multi_target_annotations),
        _two_row_extractor(lambda batch: [np.eye(2, dtype=float) for _ in range(len(batch))]),
    )[0]

    assert multi_target.dataset.y.shape == (8, 2)
    assert multi_target.dataset.metadata["target_type"] == "regression"
    assert multi_target.dataset.metadata["target_names"] == ["depth", "uncertainty"]

    single_target_annotations = [
        UnitAnnotation(
            labels=([index, index + 1] if index % 2 == 0 else [[index], [index + 1]]),
            target_type="regression",
            target_names=["depth"],
        )
        for index in range(4)
    ]
    attached = _dataset_with_annotations(single_target_annotations)
    assert attached.unit_annotations()[0]["labels"] == [0.0, 1.0]
    assert attached.unit_annotations()[1]["labels"] == [1.0, 2.0]


@pytest.mark.parametrize(
    ("first", "second", "error"),
    [
        (
            UnitAnnotation(labels=["a", "b"]),
            UnitAnnotation(labels=[1.0, 2.0], target_type="regression", target_names=["score"]),
            "target_type for sample 1",
        ),
        (
            UnitAnnotation(
                labels=[[1, 0], [0, 1]],
                label_names=["a", "b"],
                target_type="multi_label",
            ),
            UnitAnnotation(
                labels=[[1, 0], [0, 1]],
                label_names=["b", "a"],
                target_type="multi_label",
            ),
            "label_names for sample 1",
        ),
        (
            UnitAnnotation(labels=[1.0, 2.0], target_type="regression", target_names=["score"]),
            UnitAnnotation(
                labels=[[1.0, 2.0], [3.0, 4.0]],
                target_type="regression",
                target_names=["score", "weight"],
            ),
            "target_names for sample 1",
        ),
    ],
)
def test_unit_annotation_schemas_must_match_across_parents(first, second, error):
    with pytest.raises(ValueError, match=error):
        _dataset_with_annotations([first, second, first, first])


def test_materialized_unit_ids_are_parent_aware_and_preserve_local_ids():
    annotations = [
        UnitAnnotation(labels=["x", "y"], unit_ids=[1, "1"]),
        UnitAnnotation(labels=["x", "y"], unit_ids=[1, "1"]),
        UnitAnnotation(labels=["y", "x"], unit_ids=[1, "1"]),
        UnitAnnotation(labels=["y", "x"], unit_ids=[1, "1"]),
    ]
    dataset = _dataset_with_annotations(annotations)
    extractor = _two_row_extractor(
        lambda batch: [np.eye(2, dtype=float) for _ in range(len(batch))]
    )

    first = materialize_structured_outputs(dataset, extractor)[0]
    second = materialize_structured_outputs(dataset, extractor)[0]
    unit_ids = first.dataset.metadata["unit_ids"]

    assert len(set(unit_ids)) == 8
    assert all(unit_id.startswith("structured-unit-v1-") for unit_id in unit_ids)
    assert unit_ids == second.dataset.metadata["unit_ids"]
    assert first.provenance[0]["local_unit_id"] == 1
    assert first.provenance[1]["local_unit_id"] == "1"
    assert first.provenance[0]["unit_id"] != first.provenance[1]["unit_id"]


def test_duplicate_local_unit_ids_are_rejected_within_parent():
    duplicate = UnitAnnotation(labels=["x", "y"], unit_ids=["same", "same"])
    with pytest.raises(ValueError, match="unique within the parent"):
        _dataset_with_annotations([duplicate] * 4)


@pytest.mark.parametrize("aligner", [None, drop_special_rows(leading=1)])
def test_sparse_structured_materialization_preserves_sparse_matrices(aligner):
    dataset = _dataset_with_annotations(_annotations())

    def transform(_batch):
        rows = 3 if aligner is not None else 2
        return sparse.csr_matrix(np.arange(rows * 2, dtype=float).reshape(rows, 2))

    materialized = materialize_structured_outputs(
        dataset,
        _two_row_extractor(transform),
        batch_size=1,
        aligners={"tokens": aligner} if aligner is not None else None,
    )[0]

    assert sparse.isspmatrix_csr(materialized.dataset.X)
    assert materialized.dataset.X.shape == (8, 2)


@pytest.mark.parametrize("change", ["dimension", "dtype", "sparse"])
def test_structured_outputs_reject_embedding_contract_changes(change):
    dataset = _dataset_with_annotations(_annotations())

    def transform(batch):
        first = batch[0] == "a"
        if change == "dimension":
            return np.ones((2, 2 if first else 3), dtype=float)
        if change == "dtype":
            return np.ones((2, 2), dtype=np.float64 if first else np.float32)
        matrix = np.ones((2, 2), dtype=float)
        return sparse.csr_matrix(matrix) if first else matrix

    with pytest.raises(ValueError, match="changed embedding format"):
        materialize_structured_outputs(
            dataset,
            _two_row_extractor(transform),
            batch_size=1,
        )


def test_sparse_structured_artifacts_and_binary_regression_labels_round_trip(tmp_path):
    annotations = [
        UnitAnnotation(
            labels=np.array([[0, 1], [1, 0]], dtype=float),
            target_type="regression",
            target_names=["left", "right"],
            unit_ids=["cell-0", "cell-1"],
        )
        for _ in range(4)
    ]
    dataset = _dataset_with_annotations(annotations)
    extractor = _two_row_extractor(lambda _batch: sparse.csr_matrix(np.eye(2, dtype=float)))
    store = LocalArtifactStore(tmp_path)

    bundle = materialize_structured_artifacts(
        dataset,
        extractor,
        store,
        batch_size=1,
    )
    output = bundle["outputs"][0]
    loaded_embeddings = store.get_array(output["output_key"])
    loaded_labels = store.get_labels(output["labels_key"])
    label_manifest = store.get_json(output["labels_key"])

    assert sparse.isspmatrix_csr(loaded_embeddings)
    assert output["sparse"] is True
    assert output["nnz"] == 8
    assert output["storage_format"] == "csr"
    assert loaded_labels.shape == (8, 2)
    assert np.array_equal(loaded_labels, np.tile([[0.0, 1.0], [1.0, 0.0]], (4, 1)))
    assert label_manifest["target_type"] == "regression"
    assert label_manifest["target_names"] == ["left", "right"]


def test_multilabel_structured_artifacts_preserve_ordered_label_names(tmp_path):
    annotations = [
        UnitAnnotation(
            labels=[[1, 0], [0, 1]],
            target_type="multi_label",
            label_names=["primary", "secondary"],
        )
        for _ in range(4)
    ]
    store = LocalArtifactStore(tmp_path)
    bundle = materialize_structured_artifacts(
        _dataset_with_annotations(annotations),
        _two_row_extractor(lambda batch: [np.eye(2, dtype=float) for _ in range(len(batch))]),
        store,
    )
    output = bundle["outputs"][0]
    labels = store.get_labels(output["labels_key"])
    label_manifest = store.get_json(output["labels_key"])

    assert labels.tolist() == [("primary",), ("secondary",)] * 4
    assert label_manifest["target_type"] == "multi_label"
    assert label_manifest["label_names"] == ["primary", "secondary"]
