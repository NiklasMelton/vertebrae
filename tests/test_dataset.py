import numpy as np
import pandas as pd
import pytest

from vertebrae import BenchmarkDataset


def test_from_arrays_validates_lengths():
    with pytest.raises(ValueError, match="same length"):
        BenchmarkDataset.from_arrays(np.zeros((3, 2)), np.array(["a", "b"]), modality="tabular")


def test_from_arrays_rejects_single_class():
    with pytest.raises(ValueError, match="at least two classes"):
        BenchmarkDataset.from_arrays(
            np.zeros((3, 2)),
            np.array(["a", "a", "a"]),
            modality="tabular",
        )


def test_from_arrays_rejects_missing_labels():
    with pytest.raises(ValueError, match="non-missing"):
        BenchmarkDataset.from_arrays(
            np.zeros((4, 2)),
            np.array(["a", "a", None, "b"], dtype=object),
            modality="tabular",
        )


def test_from_dataframe_preserves_columns_and_counts():
    df = pd.DataFrame({"text": ["one", "two", "three", "four"], "label": ["a", "a", "b", "b"]})

    dataset = BenchmarkDataset.from_dataframe(
        df,
        input_col="text",
        label_col="label",
        modality="text",
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
    )

    assert list(dataset.X.columns) == ["age", "income", "state"]
    assert dataset.metadata["input_columns"] == ["age", "income", "state"]


def test_from_image_paths_sets_image_modality():
    dataset = BenchmarkDataset.from_image_paths(
        ["a.png", "b.png", "c.png", "d.png"],
        ["cat", "cat", "dog", "dog"],
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
    )

    assert dataset.modality == "audio"
    assert dataset.metadata["sampling_rate"] == 16_000
    assert dataset.X["sampling_rate"].tolist() == [16_000] * 4


def test_from_audio_paths_sets_audio_modality():
    dataset = BenchmarkDataset.from_audio_paths(
        ["a.wav", "b.wav", "c.wav", "d.wav"],
        ["speech", "speech", "music", "music"],
    )

    assert dataset.modality == "audio"
    assert dataset.metadata["source"] == "audio_paths"
    assert dataset.X["path"].tolist() == ["a.wav", "b.wav", "c.wav", "d.wav"]


def test_from_video_paths_sets_video_modality():
    dataset = BenchmarkDataset.from_video_paths(
        ["a.mp4", "b.mp4", "c.mp4", "d.mp4"],
        ["cat", "cat", "dog", "dog"],
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
        )

    with pytest.raises(ValueError, match="same field names"):
        BenchmarkDataset.from_multimodal(
            inputs={"image": ["a.png", "b.png", "c.png", "d.png"]},
            labels=["a", "a", "b", "b"],
            modalities={"caption": "text"},
        )

    with pytest.raises(ValueError, match="must have 4 samples"):
        BenchmarkDataset.from_multimodal(
            inputs={
                "image": ["a.png", "b.png", "c.png"],
                "caption": ["one", "two", "three", "four"],
            },
            labels=["a", "a", "b", "b"],
            modalities={"image": "image", "caption": "text"},
        )

    with pytest.raises(ValueError, match="contains missing values"):
        BenchmarkDataset.from_multimodal(
            inputs={
                "image": ["a.png", None, "c.png", "d.png"],
                "caption": ["one", "two", "three", "four"],
            },
            labels=["a", "a", "b", "b"],
            modalities={"image": "image", "caption": "text"},
        )


def test_multimodal_subset_batches_and_fingerprint_are_stable():
    dataset = BenchmarkDataset.from_multimodal(
        inputs={
            "image": ["a.png", "b.png", "c.png", "d.png", "e.png", "f.png"],
            "caption": ["one", "two", "three", "four", "five", "six"],
        },
        labels=["a", "a", "a", "b", "b", "b"],
        modalities={"image": "image", "caption": "text"},
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
    )

    assert subset.metadata["sample_indices"] == [1, 2, 4, 5]
    assert batches[0].X["image"].tolist() == ["b.png", "c.png"]
    assert batches[1].X["caption"].tolist() == ["five", "six"]
    assert dataset.fingerprint() != changed.fingerprint()


def test_stratified_subsample_indices_preserve_classes():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(12, 2),
        ["a"] * 6 + ["b"] * 4 + ["c"] * 2,
        modality="tabular",
    )

    indices = dataset.stratified_subsample_indices(rate=0.5, random_state=1)
    subset = dataset.subset(indices)

    assert len(indices) == 7
    assert subset.class_counts() == {"a": 3, "b": 2, "c": 2}
    assert subset.metadata["sample_indices"] == indices.tolist()


def test_nested_subset_preserves_original_sample_indices():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24).reshape(12, 2),
        ["a"] * 6 + ["b"] * 6,
        modality="tabular",
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


def test_with_label_hierarchy_validates_alignment():
    dataset = BenchmarkDataset.from_arrays(
        np.arange(12).reshape(4, 3),
        ["a", "a", "b", "b"],
        modality="tabular",
    )

    with pytest.raises(ValueError, match="same length as the dataset"):
        dataset.with_label_hierarchy([("root", "a"), ("root", "a")])
