import pickle

import numpy as np

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CallableStructuredExtractor,
    DepthAdapter,
    DepthAnnotation,
    DetectionLayoutAdapter,
    KeypointAdapter,
    KeypointAnnotation,
    LatentSlotAdapter,
    LatentSlotAnnotation,
    RegionAnnotation,
    SequenceAnnotation,
    SequenceLabelingAdapter,
    StructuredOutputSpec,
    StructuredUnitAligner,
    drop_special_rows,
    keep_row_indices,
    select_frame_rows,
)
from vertebrae.cache import LocalArtifactStore
from vertebrae.config import CacheConfig, SeparatixConfig, StabilityConfig
from vertebrae.execution import materialize_structured_artifacts
from vertebrae.structured import materialize_structured_outputs


def test_detection_layout_adapter_attaches_region_annotations():
    dataset = BenchmarkDataset.from_arrays(
        np.array(["page-0", "page-1", "page-2", "page-3"], dtype=object),
        ["invoice", "invoice", "resume", "resume"],
        modality="image",
    )
    adapted = DetectionLayoutAdapter(unit_type="document_region").attach(
        dataset,
        [
            RegionAnnotation(
                labels=["header", "table"],
                unit_ids=["p0:h", "p0:t"],
                boxes=[[0.0, 0.0, 1.0, 0.2], [0.1, 0.2, 0.9, 0.7]],
                page_id="page-0",
                document_id="doc-0",
            ),
            RegionAnnotation(
                labels=["header", "footer"],
                unit_ids=["p1:h", "p1:f"],
                boxes=[[0.0, 0.0, 1.0, 0.2], [0.1, 0.8, 0.9, 0.95]],
                page_id="page-1",
                document_id="doc-1",
            ),
            RegionAnnotation(
                labels=["title", "body"],
                unit_ids=["p2:t", "p2:b"],
                boxes=[[0.0, 0.0, 1.0, 0.2], [0.1, 0.2, 0.9, 0.8]],
                page_id="page-2",
                document_id="doc-2",
            ),
            RegionAnnotation(
                labels=["title", "body"],
                unit_ids=["p3:t", "p3:b"],
                boxes=[[0.0, 0.0, 1.0, 0.2], [0.1, 0.2, 0.9, 0.8]],
                page_id="page-3",
                document_id="doc-3",
            ),
        ],
    )

    assert adapted.summary()["structured_units"]["task_family"] == "detection_layout"
    assert adapted.metadata["unit_annotation_unit_type"] == "document_region"
    assert adapted.unit_annotations()[0]["provenance"][0]["page_id"] == "page-0"
    assert adapted.unit_annotations()[0]["provenance"][1]["box"] == [0.1, 0.2, 0.9, 0.7]


def test_sequence_and_keypoint_adapters_preserve_typed_metadata():
    text_dataset = BenchmarkDataset.from_arrays(
        np.array(["a", "b", "c", "d"], dtype=object),
        ["intent", "intent", "command", "command"],
        modality="text",
    )
    sequence_dataset = SequenceLabelingAdapter().attach(
        text_dataset,
        [
            SequenceAnnotation(labels=["greeting", "entity"], tokens=["hi", "sam"]),
            SequenceAnnotation(labels=["greeting", "entity"], tokens=["hey", "alex"]),
            SequenceAnnotation(labels=["verb", "object"], tokens=["open", "calendar"]),
            SequenceAnnotation(labels=["verb", "object"], tokens=["show", "weather"]),
        ],
    )

    assert sequence_dataset.unit_annotations()[0]["positions"] == [0, 1]
    assert sequence_dataset.unit_annotations()[0]["provenance"][1]["token_text"] == "sam"

    image_dataset = BenchmarkDataset.from_arrays(
        np.array(["f0", "f1", "f2", "f3"], dtype=object),
        ["walk", "walk", "run", "run"],
        modality="image",
    )
    keypoint_dataset = KeypointAdapter().attach(
        image_dataset,
        [
            KeypointAnnotation(
                labels=["shoulder", "wrist"],
                coordinates=[[0.2, 0.3], [0.5, 0.6]],
                visibility=[1, 0],
                frame_id="f0",
            ),
            KeypointAnnotation(
                labels=["shoulder", "wrist"],
                coordinates=[[0.2, 0.3], [0.5, 0.6]],
                visibility=[1, 1],
                frame_id="f1",
            ),
            KeypointAnnotation(
                labels=["shoulder", "wrist"],
                coordinates=[[0.25, 0.35], [0.55, 0.65]],
                visibility=[1, 1],
                frame_id="f2",
            ),
            KeypointAnnotation(
                labels=["shoulder", "wrist"],
                coordinates=[[0.25, 0.35], [0.55, 0.65]],
                visibility=[1, 1],
                frame_id="f3",
            ),
        ],
    )

    assert keypoint_dataset.summary()["structured_units"]["task_family"] == "keypoint"
    assert keypoint_dataset.unit_annotations()[0]["provenance"][1]["visibility"] == 0


def test_depth_and_latent_slot_adapters_build_structured_unit_datasets():
    image_dataset = BenchmarkDataset.from_arrays(
        np.array(["im0", "im1", "im2", "im3"], dtype=object),
        [0.1, 0.2, 0.8, 0.9],
        modality="image",
        target_type="regression",
        target_names=["scene_depth"],
    )
    depth_dataset = DepthAdapter().attach(
        image_dataset,
        [
            DepthAnnotation(
                labels=[1.0, 2.0, 9.0],
                coordinates=[[0, 0], [1, 1], [2, 2]],
                valid=[1, 1, 0],
            ),
            DepthAnnotation(
                labels=[1.2, 2.2, 9.2],
                coordinates=[[0, 0], [1, 1], [2, 2]],
                valid=[1, 1, 0],
            ),
            DepthAnnotation(
                labels=[3.0, 4.0, 9.3],
                coordinates=[[0, 0], [1, 1], [2, 2]],
                valid=[1, 1, 0],
            ),
            DepthAnnotation(
                labels=[3.2, 4.2, 9.4],
                coordinates=[[0, 0], [1, 1], [2, 2]],
                valid=[1, 1, 0],
            ),
        ],
    )

    assert depth_dataset.summary()["structured_units"]["task_family"] == "depth"
    assert len(depth_dataset.unit_annotations()[0]["labels"]) == 2
    assert depth_dataset.unit_annotations()[0]["target_type"] == "regression"

    latent_dataset = BenchmarkDataset.from_arrays(
        np.array(["z0", "z1", "z2", "z3"], dtype=object),
        ["scene", "scene", "object", "object"],
        modality="embeddings",
    )
    adapted_latents = LatentSlotAdapter().attach(
        latent_dataset,
        [
            LatentSlotAnnotation(
                labels=["bg", "fg"],
                slot_ids=["s0", "s1"],
                source_component_ids=["decoder", "decoder"],
            ),
            LatentSlotAnnotation(
                labels=["bg", "fg"],
                slot_ids=["s0", "s1"],
                source_component_ids=["decoder", "decoder"],
            ),
            LatentSlotAnnotation(
                labels=["bg", "fg"],
                slot_ids=["s0", "s1"],
                source_component_ids=["decoder", "decoder"],
            ),
            LatentSlotAnnotation(
                labels=["bg", "fg"],
                slot_ids=["s0", "s1"],
                source_component_ids=["decoder", "decoder"],
            ),
        ],
    )

    assert adapted_latents.summary()["structured_units"]["task_family"] == "latent_slot"
    assert (
        adapted_latents.unit_annotations()[0]["provenance"][0]["source_component_id"]
        == "decoder"
    )


def test_structured_aligner_materializes_explicit_subset_and_records_recipe(
    tmp_path,
    fake_overlapindex,
):
    dataset = SequenceLabelingAdapter().attach(
        BenchmarkDataset.from_arrays(
            np.array(["utt-0", "utt-1", "utt-2", "utt-3"], dtype=object),
            ["speech", "speech", "music", "music"],
            modality="audio",
        ),
        [
            SequenceAnnotation(labels=["a", "b"], tokens=["hello", "world"]),
            SequenceAnnotation(labels=["a", "b"], tokens=["hello", "world"]),
            SequenceAnnotation(labels=["c", "d"], tokens=["alpha", "beta"]),
            SequenceAnnotation(labels=["c", "d"], tokens=["alpha", "beta"]),
        ],
    )
    extractor = CallableStructuredExtractor(
        "frames",
        transform_fn=lambda batch: [
            np.array([[100.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            for _ in range(len(batch))
        ],
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
    )
    aligner = StructuredUnitAligner(
        "drop_special",
        align_fn=lambda embeddings, annotation: [(0, 1), (1, 2)],
        recipe_data={"policy": "drop_leading_special"},
    )

    materialized = materialize_structured_outputs(
        dataset,
        extractor,
        aligners={"tokens": aligner},
    )[0]
    assert materialized.dataset.X.shape == (8, 2)
    assert materialized.metadata["alignment_mode"] == "explicit"
    assert (
        materialized.metadata["alignment_recipe"]["recipe_data"]["policy"]
        == "drop_leading_special"
    )
    assert materialized.provenance[0]["annotation_index"] == 0
    assert materialized.provenance[0]["embedding_index"] == 1

    bundle = materialize_structured_artifacts(
        dataset,
        extractor,
        LocalArtifactStore(tmp_path),
        aligners={"tokens": aligner},
    )
    assert bundle["outputs"][0]["structured"]["alignment_mode"] == "explicit"

    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        structured_aligners={"tokens": aligner},
    ).run()
    assert (
        result.extractor_results[0].embedding_metadata["structured"]["alignment_mode"]
        == "explicit"
    )


def test_drop_special_rows_helper_materializes_and_is_pickle_safe():
    dataset = SequenceLabelingAdapter().attach(
        BenchmarkDataset.from_arrays(
            np.array(["utt-0", "utt-1", "utt-2", "utt-3"], dtype=object),
            ["speech", "speech", "music", "music"],
            modality="audio",
        ),
        [
            SequenceAnnotation(labels=["a", "b"], tokens=["hello", "world"]),
            SequenceAnnotation(labels=["a", "b"], tokens=["hello", "world"]),
            SequenceAnnotation(labels=["c", "d"], tokens=["alpha", "beta"]),
            SequenceAnnotation(labels=["c", "d"], tokens=["alpha", "beta"]),
        ],
    )
    extractor = CallableStructuredExtractor(
        "frames",
        transform_fn=lambda batch: [
            np.array([[100.0, 0.0], [1.0, 0.0], [0.0, 1.0], [200.0, 0.0]])
            for _ in range(len(batch))
        ],
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
    )
    aligner = drop_special_rows(leading=1, trailing=1)
    restored = pickle.loads(pickle.dumps(aligner))

    materialized = materialize_structured_outputs(
        dataset,
        extractor,
        aligners={"tokens": restored},
    )[0]

    assert materialized.dataset.X.shape == (8, 2)
    assert materialized.metadata["alignment_recipe"]["name"] == "drop_special_rows"
    assert materialized.metadata["alignment_recipe"]["recipe_data"] == {
        "policy": "drop_special_rows",
        "leading": 1,
        "trailing": 1,
    }
    assert materialized.provenance[0]["embedding_index"] == 1
    assert materialized.provenance[0]["alignment_metadata"]["selected_embedding_indices"] == [
        1,
        2,
    ]


def test_keep_row_indices_helper_supports_reordered_annotation_mapping():
    dataset = SequenceLabelingAdapter().attach(
        BenchmarkDataset.from_arrays(
            np.array(["utt-0", "utt-1", "utt-2", "utt-3"], dtype=object),
            ["speech", "speech", "music", "music"],
            modality="audio",
        ),
        [
            SequenceAnnotation(labels=["a", "b"], tokens=["hello", "world"]),
            SequenceAnnotation(labels=["a", "b"], tokens=["hello", "world"]),
            SequenceAnnotation(labels=["c", "d"], tokens=["alpha", "beta"]),
            SequenceAnnotation(labels=["c", "d"], tokens=["alpha", "beta"]),
        ],
    )
    extractor = CallableStructuredExtractor(
        "frames",
        transform_fn=lambda batch: [
            np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
            for _ in range(len(batch))
        ],
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
    )

    materialized = materialize_structured_outputs(
        dataset,
        extractor,
        aligners={
            "tokens": keep_row_indices(
                embedding_indices=[2, 1],
                annotation_indices=[0, 1],
            )
        },
    )[0]

    assert materialized.dataset.X[0].tolist() == [30.0, 0.0]
    assert materialized.dataset.y.tolist()[:2] == ["a", "b"]
    assert materialized.provenance[0]["embedding_index"] == 2
    assert materialized.provenance[1]["embedding_index"] == 1


def test_select_frame_rows_helper_supports_stride_and_metadata_modes():
    dataset = SequenceLabelingAdapter().attach(
        BenchmarkDataset.from_arrays(
            np.array(["utt-0", "utt-1", "utt-2", "utt-3"], dtype=object),
            ["speech", "speech", "music", "music"],
            modality="audio",
        ),
        [
            SequenceAnnotation(
                labels=["a", "b"],
                tokens=["hello", "world"],
                metadata={"sampled_frames": [1, 3]},
            ),
            SequenceAnnotation(
                labels=["a", "b"],
                tokens=["hello", "world"],
                metadata={"sampled_frames": [1, 3]},
            ),
            SequenceAnnotation(
                labels=["c", "d"],
                tokens=["alpha", "beta"],
                metadata={"sampled_frames": [0, 2]},
            ),
            SequenceAnnotation(
                labels=["c", "d"],
                tokens=["alpha", "beta"],
                metadata={"sampled_frames": [0, 2]},
            ),
        ],
    )
    extractor = CallableStructuredExtractor(
        "frames",
        transform_fn=lambda batch: [
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
            for _ in range(len(batch))
        ],
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
    )

    stride_materialized = materialize_structured_outputs(
        dataset,
        extractor,
        aligners={"tokens": select_frame_rows(every_n=2, start=0)},
    )[0]
    metadata_materialized = materialize_structured_outputs(
        dataset,
        extractor,
        aligners={"tokens": select_frame_rows(indices_metadata_key="sampled_frames")},
    )[0]

    assert stride_materialized.provenance[0]["alignment_metadata"][
        "selected_embedding_indices"
    ] == [0, 2]
    assert stride_materialized.dataset.X[1].tolist() == [2.0, 0.0]
    assert metadata_materialized.provenance[0]["alignment_metadata"][
        "selected_embedding_indices"
    ] == [1, 3]
    assert metadata_materialized.provenance[4]["alignment_metadata"][
        "selected_embedding_indices"
    ] == [0, 2]
    assert metadata_materialized.dataset.X[0].tolist() == [1.0, 0.0]


def test_standard_aligner_helpers_validate_invalid_inputs():
    with np.testing.assert_raises_regex(ValueError, "leading >= 0"):
        drop_special_rows(leading=-1)
    with np.testing.assert_raises_regex(ValueError, "same length"):
        keep_row_indices(embedding_indices=[0, 1], annotation_indices=[0])
    with np.testing.assert_raises_regex(ValueError, "exactly one"):
        select_frame_rows(every_n=2, indices=[0, 2])
    with np.testing.assert_raises_regex(ValueError, "every_n > 0"):
        select_frame_rows(every_n=0)


def test_standard_aligner_helpers_raise_contextual_alignment_errors():
    dataset = SequenceLabelingAdapter().attach(
        BenchmarkDataset.from_arrays(
            np.array(["utt-0", "utt-1", "utt-2", "utt-3"], dtype=object),
            ["speech", "speech", "music", "music"],
            modality="audio",
        ),
        [
            SequenceAnnotation(labels=["a", "b"], tokens=["hello", "world"]),
            SequenceAnnotation(labels=["a", "b"], tokens=["hello", "world"]),
            SequenceAnnotation(labels=["c", "d"], tokens=["alpha", "beta"]),
            SequenceAnnotation(labels=["c", "d"], tokens=["alpha", "beta"]),
        ],
    )
    extractor = CallableStructuredExtractor(
        "frames",
        transform_fn=lambda batch: [
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
            for _ in range(len(batch))
        ],
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
    )

    with np.testing.assert_raises_regex(ValueError, "retained 3 embedding rows"):
        materialize_structured_outputs(
            dataset,
            extractor,
            aligners={"tokens": drop_special_rows(leading=1)},
        )
    with np.testing.assert_raises_regex(ValueError, "could not find sampled frame indices"):
        materialize_structured_outputs(
            dataset,
            extractor,
            aligners={"tokens": select_frame_rows(indices_metadata_key="sampled_frames")},
        )
