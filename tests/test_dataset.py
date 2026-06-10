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
