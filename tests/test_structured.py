import numpy as np

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CallableStructuredExtractor,
    TargetView,
    UnitAnnotation,
    drop_special_rows,
)
from vertebrae.cache import LocalArtifactStore
from vertebrae.config import CacheConfig, SeparatixConfig, StabilityConfig
from vertebrae.execution import materialize_structured_artifacts
from vertebrae.extractors import StructuredOutputSpec
from vertebrae.structured import materialize_structured_outputs


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


def test_dataset_with_unit_annotations_survives_subset_and_summary():
    dataset = BenchmarkDataset.from_arrays(
        np.array(["a", "b", "c", "d"], dtype=object),
        [0.0, 0.1, 0.2, 0.3],
        modality="text",
        target_type="regression",
        target_names=["score"],
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


def test_structured_benchmark_reuses_standard_scoring_pipeline(tmp_path, fake_overlapindex):
    result = Benchmark(
        dataset=_dataset(),
        extractors=[_extractor()],
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        structured_aligners={"tokens": drop_special_rows()},
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

    markdown_path = tmp_path / "structured_report.md"
    result.save_markdown(str(markdown_path))
    report = markdown_path.read_text(encoding="utf-8")
    assert "## Structured outputs" in report
    assert "Structured task family: sequence" in report
    assert "Alignment mode: explicit" in report
    assert "drop_special_rows (drop_special_rows)" in report


def test_structured_artifacts_have_independent_output_boundaries(tmp_path):
    store = LocalArtifactStore(tmp_path)

    bundle = materialize_structured_artifacts(_dataset(), _extractor(), store)

    output = bundle["outputs"][0]
    assert output["artifact_type"] == "structured_embedding"
    assert store.get_array(output["output_key"]).shape == (8, 2)
    assert store.get_labels(output["labels_key"]).shape == (8,)
    assert store.get_labels(output["groups_key"]).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(store.get_json(output["provenance_key"])["rows"]) == 8
    assert output["task_family"] == "sequence"
    assert output["alignment_mode"] == "strict"
    assert output["alignment_recipe"] is None
