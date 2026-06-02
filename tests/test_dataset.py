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
