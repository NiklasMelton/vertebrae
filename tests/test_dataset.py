import numpy as np
import pandas as pd
import pytest

from vertebrae import BenchmarkDataset, DatasetIdentity, EmbeddingUnitDataset, TargetView

COARSE_TARGETS = [
    "mammal",
    "mammal",
    "mammal",
    "mammal",
    "avian",
    "avian",
    "mammal",
    "mammal",
]


def test_from_arrays_validates_lengths():
    with pytest.raises(ValueError, match="same length"):
        BenchmarkDataset.from_arrays(
            np.zeros((3, 2)),
            np.array(["a", "b"]),
            modality="tabular",
            identity=DatasetIdentity.ephemeral(),
        )


def test_from_arrays_rejects_single_class():
    with pytest.raises(ValueError, match="at least two classes"):
        BenchmarkDataset.from_arrays(
            np.zeros((3, 2)),
            np.array(["a", "a", "a"]),
            modality="tabular",
            identity=DatasetIdentity.ephemeral(),
        )


def test_from_arrays_rejects_missing_labels():
    with pytest.raises(ValueError, match="non-missing"):
        BenchmarkDataset.from_arrays(
            np.zeros((4, 2)),
            np.array(["a", "a", None, "b"], dtype=object),
            modality="tabular",
            identity=DatasetIdentity.ephemeral(),
        )


def test_from_graphs_preserves_modality_and_metadata():
    graphs = [{"nodes": 3}, {"nodes": 4}, {"nodes": 5}, {"nodes": 6}]

    dataset = BenchmarkDataset.from_graphs(
        graphs,
        ["a", "a", "b", "b"],
        metadata={"split": "train"},
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.modality == "graph"
    assert dataset.metadata["source"] == "graphs"
    assert dataset.metadata["split"] == "train"
    assert dataset.X.dtype == object


def test_from_node_embeddings_preserves_node_provenance_and_subsets():
    embeddings = np.arange(24, dtype=float).reshape(6, 4)
    labels = ["a", "a", "a", "b", "b", "b"]

    dataset = BenchmarkDataset.from_node_embeddings(
        embeddings,
        labels,
        node_ids=["n0", "n1", "n2", "n3", "n4", "n5"],
        edge_index=[["n0", "n1"], ["n2", "n3"]],
        identity=DatasetIdentity.ephemeral(),
    )
    subset = dataset.subset([1, 2, 4, 5])

    assert dataset.modality == "embeddings"
    assert dataset.metadata["relational_unit"] == "node"
    assert dataset.metadata["node_ids"] == ["n0", "n1", "n2", "n3", "n4", "n5"]
    assert dataset.metadata["edge_index"] == [["n0", "n1"], ["n2", "n3"]]
    assert subset.metadata["node_ids"] == ["n1", "n2", "n4", "n5"]
    assert subset.summary()["metadata"]["modality_detail"] == "graph_node_embeddings"


def test_from_entity_embeddings_supports_regression_targets():
    dataset = BenchmarkDataset.from_entity_embeddings(
        np.arange(18, dtype=float).reshape(6, 3),
        np.linspace(0.0, 1.0, 6),
        entity_ids=["u0", "u1", "u2", "u3", "u4", "u5"],
        entity_type="user",
        target_type="regression",
        target_names=["retention"],
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.metadata["relational_unit"] == "entity"
    assert dataset.metadata["entity_type"] == "user"
    assert dataset.metadata["entity_ids"] == ["u0", "u1", "u2", "u3", "u4", "u5"]
    assert dataset.summary()["target_type"] == "regression"


def test_from_edge_embeddings_composes_node_pairs():
    node_embeddings = np.array(
        [
            [1.0, 2.0],
            [3.0, 5.0],
            [2.0, 7.0],
            [11.0, 13.0],
        ]
    )

    dataset = BenchmarkDataset.from_edge_embeddings(
        labels=["same", "same", "diff", "diff"],
        edge_index=[["a", "b"], ["b", "c"], ["a", "d"], ["c", "d"]],
        node_embeddings=node_embeddings,
        node_ids=["a", "b", "c", "d"],
        composition="hadamard",
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.metadata["relational_unit"] == "edge"
    assert dataset.metadata["embedding_source"] == "composed_node_embeddings"
    assert dataset.X.tolist()[0] == [3.0, 10.0]
    assert dataset.X.shape == (4, 2)


def test_from_pair_embeddings_composes_entity_pairs_with_abs_diff():
    entity_embeddings = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 7.0],
            [4.0, 4.0, 4.0],
            [9.0, 1.0, 2.0],
        ]
    )

    dataset = BenchmarkDataset.from_pair_embeddings(
        labels=["near", "near", "far", "far"],
        pairs=[["q0", "d0"], ["q0", "d1"], ["q1", "d0"], ["d1", "d0"]],
        entity_embeddings=entity_embeddings,
        entity_ids=["q0", "d0", "q1", "d1"],
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.metadata["relational_unit"] == "pair"
    assert dataset.metadata["pair_ids"][0] == ["q0", "d0"]
    assert dataset.X.tolist()[0] == [2.0, 1.0, 4.0]


def test_from_triplet_embeddings_composes_positive_and_negative_pairs():
    entity_embeddings = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [5.0, 8.0],
            [3.0, 1.0],
        ]
    )

    dataset = BenchmarkDataset.from_triplet_embeddings(
        labels=["ok", "ok", "bad", "bad"],
        triplets=[[0, 1, 2], [0, 3, 2], [2, 1, 0], [2, 3, 0]],
        entity_embeddings=entity_embeddings,
        composition="abs_diff",
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.metadata["relational_unit"] == "triplet"
    assert dataset.X.shape == (4, 4)
    assert dataset.X.tolist()[0] == [1.0, 0.0, 4.0, 7.0]


def test_relational_constructors_validate_alignment():
    with pytest.raises(ValueError, match="node_ids must have length"):
        BenchmarkDataset.from_node_embeddings(
            np.zeros((4, 2)),
            ["a", "a", "b", "b"],
            node_ids=["n0"],
            identity=DatasetIdentity.ephemeral(),
        )
    with pytest.raises(ValueError, match="unknown ids"):
        BenchmarkDataset.from_pair_embeddings(
            labels=["a", "a", "b", "b"],
            pairs=[
                ["known", "missing"],
                ["known", "known"],
                ["known", "known"],
                ["known", "known"],
            ],
            entity_embeddings=np.zeros((2, 2)),
            entity_ids=["known", "other"],
            identity=DatasetIdentity.ephemeral(),
        )


def test_from_dataframe_preserves_columns_and_counts():
    df = pd.DataFrame({"text": ["one", "two", "three", "four"], "label": ["a", "a", "b", "b"]})

    dataset = BenchmarkDataset.from_dataframe(
        df,
        input_col="text",
        label_col="label",
        modality="text",
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.input_col == "text"
    assert dataset.label_col == "label"
    assert dataset.class_counts() == {"a": 2, "b": 2}


def test_from_dataframe_accepts_tabular_column_list():
    df = pd.DataFrame(
        {
            "age": [20, 25, 35, 42],
            "income": [40_000, 52_000, 83_000, 91_000],
            "state": ["CA", "CA", "NY", "NY"],
            "target": ["low", "low", "high", "high"],
        }
    )

    dataset = BenchmarkDataset.from_dataframe(
        df,
        input_col=["age", "income", "state"],
        label_col="target",
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )

    assert list(dataset.X.columns) == ["age", "income", "state"]
    assert dataset.metadata["input_columns"] == ["age", "income", "state"]


def test_from_arrays_accepts_multilabel_sequences_and_summarizes_targets():
    labels = [
        ("red", "round"),
        ("red",),
        ("round",),
        ("red", "sweet"),
        ("round", "sweet"),
        ("sweet",),
    ]

    dataset = BenchmarkDataset.from_arrays(
        np.arange(12).reshape(6, 2),
        labels,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    summary = dataset.summary()

    assert dataset.metadata["target_type"] == "multi_label"
    assert dataset.metadata["label_names"] == ["red", "round", "sweet"]
    assert dataset.y.tolist() == labels
    assert dataset.class_counts() == {"red": 3, "round": 3, "sweet": 3}
    assert summary["target_type"] == "multi_label"
    assert summary["labelset_counts"]["red + round"] == 1
    assert summary["mean_label_cardinality"] == 1.5
    assert summary["label_density"] == 0.5


def test_from_arrays_accepts_indicator_multilabel_targets_with_names():
    indicator = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
        ]
    )

    dataset = BenchmarkDataset.from_arrays(
        np.arange(12).reshape(6, 2),
        indicator,
        modality="tabular",
        label_names=["red", "round", "sweet"],
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.y.tolist() == [
        ("red", "round"),
        ("red",),
        ("round",),
        ("red", "sweet"),
        ("round", "sweet"),
        ("sweet",),
    ]
    assert dataset.class_counts() == {"red": 3, "round": 3, "sweet": 3}


def test_from_dataframe_accepts_multilabel_indicator_columns():
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "red": [1, 1, 0, 1, 0, 0],
            "round": [1, 0, 1, 0, 1, 0],
            "sweet": [0, 0, 0, 1, 1, 1],
        }
    )

    dataset = BenchmarkDataset.from_dataframe(
        df,
        input_col="x",
        label_col=["red", "round", "sweet"],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.label_col == ["red", "round", "sweet"]
    assert dataset.metadata["label_names"] == ["red", "round", "sweet"]
    assert dataset.summary()["target_type"] == "multi_label"


def test_multilabel_validation_rejects_invalid_targets():
    X = np.arange(12).reshape(6, 2)

    with pytest.raises(ValueError, match="at least one label"):
        BenchmarkDataset.from_arrays(
            X,
            [("a",), ("a",), ("b",), ("b",), ("a", "b"), ()],
            modality="tabular",
            identity=DatasetIdentity.ephemeral(),
        )
    with pytest.raises(ValueError, match="missing label"):
        BenchmarkDataset.from_arrays(
            X,
            [("a",), ("a",), ("b",), ("b",), ("a", "b"), (None,)],
            modality="tabular",
            identity=DatasetIdentity.ephemeral(),
        )
    with pytest.raises(ValueError, match="duplicate labels"):
        BenchmarkDataset.from_arrays(
            X,
            [("a",), ("a",), ("b",), ("b",), ("a", "b"), ("a", "a")],
            modality="tabular",
            identity=DatasetIdentity.ephemeral(),
        )
    with pytest.raises(ValueError, match="0/1"):
        BenchmarkDataset.from_arrays(
            X,
            np.array([[1, 0], [1, 0], [0, 1], [0, 1], [2, 0], [0, 1]]),
            modality="tabular",
            label_names=["a", "b"],
            identity=DatasetIdentity.ephemeral(),
        )
    with pytest.raises(ValueError, match="label_names length"):
        BenchmarkDataset.from_arrays(
            X,
            np.ones((6, 3), dtype=int),
            modality="tabular",
            label_names=["a", "b"],
            identity=DatasetIdentity.ephemeral(),
        )


def test_from_arrays_supports_explicit_regression_targets():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(12, dtype=float).reshape(6, 2),
        np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        modality="tabular",
        target_type="regression",
        target_names=["score"],
        identity=DatasetIdentity.ephemeral(),
    )

    summary = dataset.summary()
    assert dataset.metadata["target_type"] == "regression"
    assert dataset.metadata["target_names"] == ["score"]
    assert summary["target_type"] == "regression"
    assert summary["n_targets"] == 1
    assert summary["target_names"] == ["score"]
    assert summary["constant_targets"] == []


def test_from_dataframe_supports_multitarget_regression_columns():
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "y1": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "y2": [1.0, 1.1, 1.2, 1.8, 1.9, 2.0],
        }
    )

    dataset = BenchmarkDataset.from_dataframe(
        df,
        input_col="x",
        label_col=["y1", "y2"],
        modality="tabular",
        target_type="regression",
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.metadata["target_type"] == "regression"
    assert dataset.metadata["target_names"] == ["y1", "y2"]
    assert dataset.y.shape == (6, 2)


def test_regression_validation_rejects_constant_and_too_small_targets():
    X = np.arange(6, dtype=float).reshape(3, 2)

    with pytest.raises(ValueError, match="at least 3 samples"):
        BenchmarkDataset.from_arrays(
            np.arange(4, dtype=float).reshape(2, 2),
            np.array([0.0, 1.0]),
            modality="tabular",
            target_type="regression",
            identity=DatasetIdentity.ephemeral(),
        )
    with pytest.raises(ValueError, match="non-constant target"):
        BenchmarkDataset.from_arrays(
            X,
            np.array([1.0, 1.0, 1.0]),
            modality="tabular",
            target_type="regression",
            identity=DatasetIdentity.ephemeral(),
        )


def test_regression_subsetting_and_roundtrip_summary_preserve_target_metadata():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(18, dtype=float).reshape(6, 3),
        np.column_stack(
            [
                np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]),
                np.array([1.0, 1.2, 1.1, 2.0, 2.2, 2.1]),
            ]
        ),
        modality="embeddings",
        target_type="regression",
        target_names=["a", "b"],
        identity=DatasetIdentity.ephemeral(),
    )

    subset = dataset.subset([0, 2, 4])
    assert subset.metadata["target_type"] == "regression"
    assert subset.metadata["target_names"] == ["a", "b"]
    assert subset.summary()["n_targets"] == 2


def test_from_image_paths_sets_image_modality():
    dataset = BenchmarkDataset.from_image_paths(
        ["a.png", "b.png", "c.png", "d.png"],
        ["cat", "cat", "dog", "dog"],
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.modality == "image"
    assert dataset.metadata["source"] == "image_paths"


def test_from_audio_arrays_preserves_sampling_rate():
    dataset = BenchmarkDataset.from_audio_arrays(
        [
            np.array([0.0, 0.1, 0.2], dtype=np.float32),
            np.array([0.2, 0.1, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, -1.0], dtype=np.float32),
            np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        ],
        ["speech", "speech", "music", "music"],
        sampling_rate=16_000,
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.modality == "audio"
    assert dataset.metadata["sampling_rate"] == 16_000
    assert dataset.X["sampling_rate"].tolist() == [16_000] * 4


def test_from_audio_paths_sets_audio_modality():
    dataset = BenchmarkDataset.from_audio_paths(
        ["a.wav", "b.wav", "c.wav", "d.wav"],
        ["speech", "speech", "music", "music"],
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.modality == "audio"
    assert dataset.metadata["source"] == "audio_paths"
    assert dataset.X["path"].tolist() == ["a.wav", "b.wav", "c.wav", "d.wav"]


def test_from_video_paths_sets_video_modality():
    dataset = BenchmarkDataset.from_video_paths(
        ["a.mp4", "b.mp4", "c.mp4", "d.mp4"],
        ["cat", "cat", "dog", "dog"],
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.modality == "video"
    assert dataset.metadata["source"] == "video_paths"
    assert dataset.X["path"].tolist() == ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]


def test_from_video_arrays_preserves_frame_rate():
    clips = [
        np.zeros((3, 2, 2, 3), dtype=np.uint8),
        np.ones((4, 2, 2, 3), dtype=np.uint8),
        np.full((5, 2, 2, 3), 2, dtype=np.uint8),
        np.full((6, 2, 2, 3), 3, dtype=np.uint8),
    ]

    dataset = BenchmarkDataset.from_video_arrays(
        clips,
        ["left", "left", "right", "right"],
        frame_rate=24.0,
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.modality == "video"
    assert dataset.metadata["source"] == "video_arrays"
    assert dataset.metadata["frame_rate"] == 24.0
    assert dataset.X["frame_rate"].tolist() == [24.0] * 4


def test_from_time_series_preserves_structured_inputs():
    series = np.arange(24, dtype=float).reshape(4, 3, 2)
    observed_mask = np.ones((4, 3, 2), dtype=float)
    time_features = np.arange(24, dtype=float).reshape(4, 3, 2)

    dataset = BenchmarkDataset.from_time_series(
        series=series,
        labels=["a", "a", "b", "b"],
        observed_mask=observed_mask,
        time_features=time_features,
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.modality == "time_series"
    assert dataset.metadata["source"] == "time_series"
    assert dataset.X["series"].shape == (4, 3, 2)
    assert dataset.X["observed_mask"].shape == (4, 3, 2)
    assert dataset.X["time_features"].shape == (4, 3, 2)


def test_from_multimodal_preserves_fields_and_modalities():
    dataset = BenchmarkDataset.from_multimodal(
        inputs={
            "image": ["a.png", "b.png", "c.png", "d.png"],
            "caption": ["sun", "moon", "cat", "dog"],
        },
        labels=["left", "left", "right", "right"],
        modalities={"image": "image", "caption": "text"},
        identity=DatasetIdentity.ephemeral(),
    )

    assert dataset.modality == "multimodal"
    assert dataset.metadata["source"] == "multimodal"
    assert dataset.metadata["input_fields"] == ["image", "caption"]
    assert dataset.metadata["modalities"] == {"image": "image", "caption": "text"}
    assert dataset.summary()["modality"] == "multimodal"


def test_from_multimodal_rejects_invalid_fields():
    with pytest.raises(ValueError, match="must not be empty"):
        BenchmarkDataset.from_multimodal(
            inputs={},
            labels=["a", "a", "b", "b"],
            modalities={},
            identity=DatasetIdentity.ephemeral(),
        )

    with pytest.raises(ValueError, match="same field names"):
        BenchmarkDataset.from_multimodal(
            inputs={"image": ["a.png", "b.png", "c.png", "d.png"]},
            labels=["a", "a", "b", "b"],
            modalities={"caption": "text"},
            identity=DatasetIdentity.ephemeral(),
        )

    with pytest.raises(ValueError, match="must have 4 samples"):
        BenchmarkDataset.from_multimodal(
            inputs={
                "image": ["a.png", "b.png", "c.png"],
                "caption": ["one", "two", "three", "four"],
            },
            labels=["a", "a", "b", "b"],
            modalities={"image": "image", "caption": "text"},
            identity=DatasetIdentity.ephemeral(),
        )

    with pytest.raises(ValueError, match="contains missing values"):
        BenchmarkDataset.from_multimodal(
            inputs={
                "image": ["a.png", None, "c.png", "d.png"],
                "caption": ["one", "two", "three", "four"],
            },
            labels=["a", "a", "b", "b"],
            modalities={"image": "image", "caption": "text"},
            identity=DatasetIdentity.ephemeral(),
        )


def test_multimodal_subset_batches_and_content_identity_are_stable():
    dataset = BenchmarkDataset.from_multimodal(
        inputs={
            "image": ["a.png", "b.png", "c.png", "d.png", "e.png", "f.png"],
            "caption": ["one", "two", "three", "four", "five", "six"],
        },
        labels=["a", "a", "a", "b", "b", "b"],
        modalities={"image": "image", "caption": "text"},
        identity=DatasetIdentity.from_content(),
    )

    subset = dataset.subset([1, 2, 4, 5])
    batches = list(subset.iter_batches(batch_size=2))
    changed = BenchmarkDataset.from_multimodal(
        inputs={
            "image": ["a.png", "b.png", "c.png", "d.png", "e.png", "f.png"],
            "caption": ["one", "two", "THREE", "four", "five", "six"],
        },
        labels=["a", "a", "a", "b", "b", "b"],
        modalities={"image": "image", "caption": "text"},
        identity=DatasetIdentity.from_content(),
    )

    assert subset.metadata["sample_indices"] == [1, 2, 4, 5]
    assert batches[0].X["image"].tolist() == ["b.png", "c.png"]
    assert batches[1].X["caption"].tolist() == ["five", "six"]
    assert dataset.identity_key() != changed.identity_key()


def test_stratified_subsample_indices_preserve_classes():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(12, 2),
        ["a"] * 6 + ["b"] * 4 + ["c"] * 2,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )

    indices = dataset.stratified_subsample_indices(rate=0.5, random_state=1)
    subset = dataset.subset(indices)

    assert len(indices) == 7
    assert subset.class_counts() == {"a": 3, "b": 2, "c": 2}
    assert subset.metadata["sample_indices"] == indices.tolist()


def test_multilabel_stratified_subsample_indices_preserve_label_counts():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(12, 2),
        [
            ("a", "b"),
            ("a",),
            ("a", "c"),
            ("a",),
            ("b",),
            ("b", "c"),
            ("b",),
            ("c",),
            ("c",),
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        ],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )

    indices = dataset.stratified_subsample_indices(rate=0.4, random_state=1)
    subset = dataset.subset(indices)

    assert all(count >= 2 for count in subset.class_counts().values())
    assert subset.summary()["target_type"] == "multi_label"
    assert subset.metadata["sample_indices"] == indices.tolist()


def test_regression_subsample_indices_expand_tiny_rate_and_preserve_variation():
    targets = np.column_stack(
        [
            np.ones(10),
            np.asarray([0.0] * 9 + [1.0]),
            np.ones(10) * 4.0,
        ]
    )
    dataset = BenchmarkDataset.from_arrays(
        np.arange(20).reshape(10, 2),
        targets,
        modality="tabular",
        target_type="regression",
        target_names=["constant_a", "signal", "constant_b"],
        identity=DatasetIdentity.ephemeral(),
    )

    first = dataset.stratified_subsample_indices(rate=0.01, random_state=7)
    second = dataset.stratified_subsample_indices(rate=0.01, random_state=7)
    subset = dataset.subset(first)

    assert len(first) == 3
    assert np.array_equal(first, second)
    assert np.any(np.var(subset.y, axis=0) > 0.0)
    assert subset.summary()["nonconstant_targets"] == ["signal"]


def test_nested_subset_preserves_original_sample_indices():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(12, 2),
        ["a"] * 6 + ["b"] * 6,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )

    first = dataset.subset([1, 2, 3, 7, 8, 9])
    second = first.subset([0, 2, 3, 5])

    assert second.metadata["sample_indices"] == [1, 3, 7, 9]


def test_structured_dataset_subset_and_batches_align_fields():
    dataset = BenchmarkDataset.from_time_series(
        series=np.arange(48, dtype=float).reshape(6, 4, 2),
        labels=["a", "a", "a", "b", "b", "b"],
        observed_mask=np.ones((6, 4, 2), dtype=float),
        time_features=np.arange(48, dtype=float).reshape(6, 4, 2),
        identity=DatasetIdentity.ephemeral(),
    )

    subset = dataset.subset([1, 2, 4, 5])
    batches = list(subset.iter_batches(batch_size=1))

    assert subset.X["series"].shape == (4, 4, 2)
    assert subset.metadata["sample_indices"] == [1, 2, 4, 5]
    assert batches[0].indices.tolist() == [0]
    assert batches[0].X["series"].shape == (1, 4, 2)


def test_video_dataset_subset_and_batches_align_fields():
    dataset = BenchmarkDataset.from_video_arrays(
        [np.full((3, 2, 2, 3), fill_value=index, dtype=np.uint8) for index in range(6)],
        labels=["a", "a", "a", "b", "b", "b"],
        frame_rate=12.0,
        identity=DatasetIdentity.ephemeral(),
    )

    subset = dataset.subset([1, 2, 4, 5])
    batches = list(subset.iter_batches(batch_size=2))

    assert subset.X["frames"].shape == (4,)
    assert subset.X["frame_rate"].tolist() == [12.0] * 4
    assert subset.metadata["sample_indices"] == [1, 2, 4, 5]
    assert batches[0].indices.tolist() == [0, 1]
    assert len(batches[0].X["frames"]) == 2


def test_with_label_hierarchy_preserves_primary_labels_and_summary():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["husky", "husky", "pug", "pug", "sedan", "sedan", "suv", "suv"],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    ).with_label_hierarchy(
        [
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
        ],
        level_names=("domain", "family", "leaf"),
    )

    assert dataset.class_counts() == {"husky": 2, "pug": 2, "sedan": 2, "suv": 2}
    assert dataset.summary()["label_view"]["name"] == "primary"
    assert dataset.metadata["label_hierarchy"]["level_names"] == ["domain", "family", "leaf"]


def test_label_view_projects_named_hierarchy_level():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["husky", "husky", "pug", "pug", "sedan", "sedan", "suv", "suv"],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    ).with_label_hierarchy(
        [
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
        ],
        level_names=("domain", "family", "leaf"),
    )

    family_view = dataset.label_view("family")

    assert family_view.active_label_view()["name"] == "family"
    assert family_view.class_counts() == {"animal > dog": 4, "vehicle > car": 4}
    assert family_view.summary()["label_view"]["level"] == 1


def test_label_view_subset_preserves_hierarchy_alignment():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(8, 3),
        ["husky", "husky", "pug", "pug", "sedan", "sedan", "suv", "suv"],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    ).with_label_hierarchy(
        [
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
        ],
        level_names=("domain", "family", "leaf"),
    )

    subset = dataset.subset([0, 1, 4, 5])
    family_view = subset.label_view("family")

    assert family_view.class_counts() == {"animal > dog": 2, "vehicle > car": 2}
    assert subset.metadata["label_hierarchy"]["paths"] == [
        ["animal", "dog", "husky"],
        ["animal", "dog", "husky"],
        ["vehicle", "car", "sedan"],
        ["vehicle", "car", "sedan"],
    ]


def test_target_views_materialize_and_subset_preserve_alignment():
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24, dtype=float).reshape(8, 3),
        ["cat", "cat", "dog", "dog", "bird", "bird", "fox", "fox"],
        identity=DatasetIdentity.ephemeral(),
    ).with_target_views(
        [
            TargetView(
                name="coarse",
                targets=COARSE_TARGETS,
            ),
            TargetView(
                name="score",
                targets=np.linspace(0.0, 1.0, 8),
                target_type="regression",
                target_names=["quality"],
            ),
        ]
    )

    coarse = dataset.target_view("coarse")
    subset = dataset.subset([0, 1, 4, 5])
    subset_score = subset.target_view("score")

    assert dataset.target_view_names() == ["coarse", "score"]
    assert coarse.active_target_view()["name"] == "coarse"
    assert coarse.class_counts() == {"avian": 2, "mammal": 6}
    assert coarse.summary()["target_view"]["name"] == "coarse"
    assert subset.target_view_names() == ["coarse", "score"]
    assert subset_score.metadata["target_type"] == "regression"
    assert subset_score.metadata["target_names"] == ["quality"]
    assert subset_score.y.tolist() == [
        0.0,
        0.14285714285714285,
        0.5714285714285714,
        0.7142857142857142,
    ]


def test_from_embedding_units_preserves_unit_metadata_groups_and_target_views():
    dataset = BenchmarkDataset.from_embedding_units(
        embeddings=np.arange(24, dtype=float).reshape(8, 3),
        labels=["header", "header", "body", "body", "footer", "footer", "caption", "caption"],
        unit_ids=[f"u{i}" for i in range(8)],
        parent_ids=["page0", "page0", "page1", "page1", "page2", "page2", "page3", "page3"],
        unit_type="document_region",
        positions=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
        target_views=[
            TargetView(
                name="score",
                targets=np.linspace(0.0, 1.0, 8),
                target_type="regression",
                target_names=["quality"],
            )
        ],
        identity=DatasetIdentity.ephemeral(),
    )

    assert isinstance(dataset, EmbeddingUnitDataset)
    assert dataset.metadata["unit_embeddings"] is True
    assert dataset.metadata["unit_type"] == "document_region"
    assert dataset.groups().tolist() == [
        "page0",
        "page0",
        "page1",
        "page1",
        "page2",
        "page2",
        "page3",
        "page3",
    ]
    assert dataset.summary()["units"]["unit_type"] == "document_region"
    assert dataset.summary()["grouping"]["name"] == "parent_id"
    assert dataset.target_view("score").metadata["target_type"] == "regression"


def test_with_label_hierarchy_validates_alignment():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(12).reshape(4, 3),
        ["a", "a", "b", "b"],
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )

    with pytest.raises(ValueError, match="same length as the dataset"):
        dataset.with_label_hierarchy([("root", "a"), ("root", "a")])
