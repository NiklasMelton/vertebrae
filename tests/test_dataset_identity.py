import pickle

import numpy as np
import pytest
from scipy import sparse

from vertebrae import (
    BenchmarkDataset,
    DatasetIdentity,
    EmbeddingUnitDataset,
    RetrievalDataset,
    SegmentationDataset,
    ZeroShotDataset,
)


def test_root_dataset_factories_require_explicit_identity():
    with pytest.raises(TypeError, match="identity"):
        BenchmarkDataset.from_arrays(np.eye(4), ["a", "a", "b", "b"], modality="tabular")
    with pytest.raises(TypeError, match="identity"):
        EmbeddingUnitDataset.from_units(np.eye(4), ["a", "a", "b", "b"], unit_ids=range(4))
    with pytest.raises(TypeError, match="identity"):
        RetrievalDataset.from_embeddings(np.eye(2), np.eye(2), [(0, 0, 1), (1, 1, 1)])
    with pytest.raises(TypeError, match="identity"):
        SegmentationDataset.from_arrays(
            np.zeros((2, 2, 2, 1)),
            [np.zeros((2, 2)), np.ones((2, 2))],
        )


def test_declared_identity_requires_id_and_revision_and_is_data_independent():
    with pytest.raises(ValueError, match="dataset_id"):
        DatasetIdentity.declared("", "1")
    with pytest.raises(ValueError, match="revision"):
        DatasetIdentity.declared("dataset", "")

    first = _dataset(np.arange(8).reshape(4, 2), DatasetIdentity.declared("demo", "1"))
    reconstructed = _dataset(np.full((4, 2), 99), DatasetIdentity.declared("demo", "1"))
    revised = _dataset(np.arange(8).reshape(4, 2), DatasetIdentity.declared("demo", "2"))

    assert first.identity_key() == reconstructed.identity_key()
    assert first.identity_key() != revised.identity_key()


def test_manifest_identity_is_canonical_and_summary_omits_manifest_content():
    left = DatasetIdentity.from_manifest(
        "images", {"objects": [{"key": "a", "etag": "v1"}], "labels_revision": 3}
    )
    reordered = DatasetIdentity.from_manifest(
        "images", {"labels_revision": 3, "objects": [{"etag": "v1", "key": "a"}]}
    )
    changed = DatasetIdentity.from_manifest(
        "images", {"objects": [{"key": "a", "etag": "v2"}], "labels_revision": 3}
    )
    with pytest.raises(ValueError, match="must not be empty"):
        DatasetIdentity.from_manifest("images", {})

    first = _dataset(np.eye(4), left)
    second = _dataset(np.eye(4), reordered)
    third = _dataset(np.eye(4), changed)

    assert first.identity_key() == second.identity_key()
    assert first.identity_key() != third.identity_key()
    descriptor = first.summary()["identity"]
    assert descriptor["mode"] == "manifest"
    assert descriptor["dataset_id"] == "images"
    assert "manifest_sha256" in descriptor
    assert "objects" not in descriptor


def test_ephemeral_identity_is_unique_and_survives_pickle_round_trip():
    identity = DatasetIdentity.ephemeral()
    assert identity.resolve() != DatasetIdentity.ephemeral().resolve()
    assert pickle.loads(pickle.dumps(identity)).resolve() == identity.resolve()


def test_content_identity_is_lazy_memoized_and_rejects_unsupported_values():
    class Unsupported:
        pass

    unsupported = BenchmarkDataset.from_arrays(
        np.asarray([Unsupported(), Unsupported(), Unsupported(), Unsupported()], dtype=object),
        ["a", "a", "b", "b"],
        modality="custom",
        identity=DatasetIdentity.from_content(),
    )
    with pytest.raises(ValueError, match="DatasetIdentity.declared"):
        unsupported.identity_key()

    values = np.arange(8).reshape(4, 2)
    dataset = _dataset(values, DatasetIdentity.from_content())
    original = dataset.identity_key()
    values[0, 0] = 999
    assert dataset.identity_key() == original


def test_content_identity_hashes_complete_lists_and_arrays():
    array = np.arange(800).reshape(200, 4)
    changed_array = array.copy()
    sampled = set(np.linspace(0, array.size - 1, num=50, dtype=int).tolist())
    changed_position = next(index for index in range(array.size) if index not in sampled)
    changed_array.reshape(-1)[changed_position] = -1

    first = _dataset(array, DatasetIdentity.from_content())
    changed = _dataset(changed_array, DatasetIdentity.from_content())
    same = _dataset(array.copy(), DatasetIdentity.from_content())

    assert first.identity_key() != changed.identity_key()
    assert first.identity_key() == same.identity_key()

    items = [f"sample-{index}" for index in range(120)]
    changed_items = list(items)
    changed_items[110] = "changed-after-old-list-boundary"
    labels = ["a"] * 60 + ["b"] * 60
    list_first = BenchmarkDataset.from_arrays(
        items,
        labels,
        modality="text",
        identity=DatasetIdentity.from_content(),
    )
    list_changed = BenchmarkDataset.from_arrays(
        changed_items,
        labels,
        modality="text",
        identity=DatasetIdentity.from_content(),
    )
    assert list_first.identity_key() != list_changed.identity_key()


def test_content_identity_is_stable_for_noncontiguous_and_sparse_values():
    contiguous = np.arange(24).reshape(4, 6)
    carrier = np.empty((4, 12), dtype=contiguous.dtype)
    carrier[:, ::2] = contiguous
    noncontiguous = carrier[:, ::2]
    dense_left = _dataset(contiguous, DatasetIdentity.from_content())
    dense_right = _dataset(noncontiguous, DatasetIdentity.from_content())
    sparse_left = _dataset(sparse.csr_matrix(contiguous), DatasetIdentity.from_content())
    sparse_right = _dataset(sparse.csr_matrix(contiguous.copy()), DatasetIdentity.from_content())

    assert contiguous.flags.c_contiguous
    assert not noncontiguous.flags.c_contiguous
    assert np.array_equal(contiguous, noncontiguous)
    assert dense_left.identity_key() == dense_right.identity_key()
    assert sparse_left.identity_key() == sparse_right.identity_key()


def test_content_identity_hashes_object_arrays_exactly():
    values = np.asarray(["one", "two", "three", "four"], dtype=object)
    same = values.copy()
    changed = values.copy()
    changed[-1] = "changed"

    first = _dataset(values, DatasetIdentity.from_content())
    reconstructed = _dataset(same, DatasetIdentity.from_content())
    different = _dataset(changed, DatasetIdentity.from_content())

    assert first.identity_key() == reconstructed.identity_key()
    assert first.identity_key() != different.identity_key()


def test_derived_dataset_identities_are_stable_and_distinct():
    dataset = _dataset(np.arange(12).reshape(6, 2), DatasetIdentity.declared("demo", "1"))

    first_subset = dataset.subset([0, 1, 3, 4])
    same_subset = dataset.subset([0, 1, 3, 4])
    other_subset = dataset.subset([1, 2, 4, 5])
    grouped = dataset.with_groups([0, 0, 1, 1, 2, 2], name="image")

    assert first_subset.identity_key() == same_subset.identity_key()
    assert first_subset.identity_key() != other_subset.identity_key()
    assert grouped.identity_key() != dataset.identity_key()


def test_zero_shot_protocol_uses_source_identity_and_complete_prompt_content():
    dataset = _dataset(np.arange(8).reshape(4, 2), DatasetIdentity.declared("demo", "1"))
    first = ZeroShotDataset.from_dataset(
        dataset,
        {"a": [f"a-{index}" for index in range(101)], "b": ["b"]},
    )
    prompts = [f"a-{index}" for index in range(101)]
    prompts[-1] = "changed"
    second = ZeroShotDataset.from_dataset(dataset, {"a": prompts, "b": ["b"]})

    assert first.protocol_recipe()["source_dataset_identity_key"] == dataset.identity_key()
    assert first.protocol_fingerprint() != second.protocol_fingerprint()


def _dataset(values, identity):
    n_samples = int(values.shape[0]) if sparse.issparse(values) else len(values)
    return BenchmarkDataset.from_arrays(
        values,
        ["a"] * (n_samples // 2) + ["b"] * (n_samples - (n_samples // 2)),
        modality="tabular",
        identity=identity,
    )
