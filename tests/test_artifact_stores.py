import importlib.util
import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from scipy import sparse

from vertebrae.cache import (
    GCSArtifactStore,
    LocalArtifactStore,
    S3ArtifactStore,
    create_artifact_store,
    create_artifact_store_from_config,
)
from vertebrae.utils.semantic_labels import (
    LABEL_ENCODING,
    SemanticLabelKey,
    semantic_label_key,
)


class _StringLabel(str, Enum):
    RED = "red"


def test_create_artifact_store_returns_local_store_for_plain_paths(tmp_path):
    store = create_artifact_store(str(tmp_path / "cache"))

    assert isinstance(store, LocalArtifactStore)
    assert store.config().uri == str(tmp_path / "cache")


def test_s3_artifact_store_roundtrip_with_fake_boto3(monkeypatch):
    objects = {}
    monkeypatch.setitem(sys.modules, "boto3", _fake_boto3_module(objects))

    store = create_artifact_store(
        "s3://test-bucket/cache-prefix",
        endpoint_url="http://minio:9000",
        profile_name="dev",
        region_name="us-east-1",
    )

    assert isinstance(store, S3ArtifactStore)
    store.put_json(
        "runs/demo",
        {"ok": True, "path": Path("models/demo"), "values": np.asarray([1, 2]), "tags": {"b", "a"}},
    )
    store.put_json("runs/a..b", {"identity": "dots"})
    store.put_json("runs/a__b", {"identity": "underscores"})
    store.put_labels("labels/demo", np.array(["a", "b"]))
    store.put_array("arrays/dense", np.arange(6).reshape(2, 3))
    store.put_array("arrays/sparse", sparse.csr_matrix(np.eye(3)))
    store.put_array("arrays/rewrite", sparse.csr_matrix(np.eye(3)))
    store.put_array("arrays/rewrite", np.full((3, 3), 7))

    recreated = create_artifact_store_from_config(store.config())
    assert recreated.get_json("runs/demo")["ok"] is True
    assert recreated.get_json("runs/demo")["tags"] == ["a", "b"]
    assert recreated.get_json("runs/a..b") == {"identity": "dots"}
    assert recreated.get_json("runs/a__b") == {"identity": "underscores"}
    assert np.array_equal(recreated.get_labels("labels/demo"), np.array(["a", "b"]))
    assert np.array_equal(recreated.get_array("arrays/dense"), np.arange(6).reshape(2, 3))
    assert np.array_equal(recreated.get_array("arrays/sparse").toarray(), np.eye(3))
    assert np.array_equal(recreated.get_array("arrays/rewrite"), np.full((3, 3), 7))
    dense_stat = recreated.stat_array("arrays/dense")
    sparse_stat = recreated.stat_array("arrays/sparse")
    assert dense_stat.size_bytes > 0
    assert dense_stat.storage_format == "npy"
    assert sparse_stat.storage_format == "npz"
    before = dict(objects)
    with pytest.raises(TypeError, match=r"unsupported object at \$\.metadata\.model"):
        store.put_json("runs/invalid", {"metadata": {"model": object()}})
    assert objects == before
    recreated.delete_prefix("runs")
    assert not any("/runs/" in key for _, key in objects)
    assert recreated.exists("arrays/dense")


def test_gcs_artifact_store_roundtrip_with_fake_client(monkeypatch):
    objects = {}
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    storage_module = _fake_gcs_storage_module(objects)
    cloud_module.storage = storage_module
    google_module.cloud = cloud_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)

    store = create_artifact_store("gs://test-bucket/cache-prefix", project="demo-project")

    assert isinstance(store, GCSArtifactStore)
    store.put_json(
        "runs/demo",
        {"ok": True, "path": Path("models/demo"), "values": np.asarray([1, 2]), "tags": {"b", "a"}},
    )
    store.put_json("runs/a..b", {"identity": "dots"})
    store.put_json("runs/a__b", {"identity": "underscores"})
    store.put_array("arrays/dense", np.arange(6).reshape(2, 3))
    store.put_array("arrays/rewrite", np.eye(3))
    store.put_array("arrays/rewrite", sparse.csr_matrix(np.full((3, 3), 2)))

    recreated = create_artifact_store_from_config(store.config())
    assert recreated.get_json("runs/demo")["ok"] is True
    assert recreated.get_json("runs/demo")["tags"] == ["a", "b"]
    assert recreated.get_json("runs/a..b") == {"identity": "dots"}
    assert recreated.get_json("runs/a__b") == {"identity": "underscores"}
    assert np.array_equal(recreated.get_array("arrays/dense"), np.arange(6).reshape(2, 3))
    assert np.array_equal(recreated.get_array("arrays/rewrite").toarray(), np.full((3, 3), 2))
    stat = recreated.stat_array("arrays/dense")
    assert stat.size_bytes > 0
    assert stat.storage_format == "npy"
    before = dict(objects)
    with pytest.raises(TypeError, match=r"unsupported object at \$\.metadata\.model"):
        store.put_json("runs/invalid", {"metadata": {"model": object()}})
    assert objects == before
    recreated.delete_prefix("runs")
    assert not any("/runs/" in key for _, key in objects)
    assert recreated.exists("arrays/dense")


def test_local_json_rejects_invalid_metadata_without_overwriting_previous_value(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_json("runs/demo", {"ok": True})

    with pytest.raises(TypeError, match=r"unsupported object at \$\.metadata\.model"):
        store.put_json("runs/demo", {"metadata": {"model": object()}})

    assert store.get_json("runs/demo") == {"ok": True}


def test_local_store_preserves_benign_double_dot_identity(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_json("runs/a..b", {"identity": "dots"})
    store.put_json("runs/a__b", {"identity": "underscores"})

    assert store.get_json("runs/a..b") == {"identity": "dots"}
    assert store.get_json("runs/a__b") == {"identity": "underscores"}


@pytest.mark.parametrize(
    "invalid_key",
    ["", "/absolute", "trailing/", "double//part", "a/../b", "a/./b", "a\\b", "a\x00b"],
)
def test_artifact_stores_reject_invalid_keys_before_io(tmp_path, invalid_key):
    stores = [
        LocalArtifactStore(str(tmp_path / "local")),
        S3ArtifactStore("bucket"),
        GCSArtifactStore("bucket"),
    ]

    for store in stores:
        with pytest.raises(ValueError):
            store.put_json(invalid_key, {"ok": True})
        with pytest.raises(ValueError):
            store.put_array(invalid_key, np.eye(2))
        with pytest.raises(ValueError):
            store.exists(invalid_key)
        with pytest.raises(ValueError):
            store.delete_prefix(invalid_key)


def test_local_store_rejects_symlink_escape(tmp_path):
    root = tmp_path / "cache"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this platform.")

    store = LocalArtifactStore(str(root))
    with pytest.raises(ValueError, match="outside"):
        store.put_json("escape/result", {"ok": True})

    assert not (outside / "result").exists()


def test_local_delete_prefix_removes_only_selected_artifacts(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_json("runs/one/result", {"ok": True})
    store.put_json("runs/two/result", {"ok": True})

    store.delete_prefix("runs/one")

    assert not (tmp_path / "runs/one").exists()
    assert store.get_json("runs/two/result") == {"ok": True}
    with pytest.raises(ValueError, match="root"):
        store.delete_prefix("")


def test_local_array_stat_uses_actual_file_size(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    path = store.put_array("arrays/dense", np.arange(6).reshape(2, 3))

    stat = store.stat_array("arrays/dense")

    assert stat.size_bytes == Path(path).stat().st_size
    assert stat.uri == path
    assert stat.storage_format == "npy"
    with pytest.raises(FileNotFoundError):
        store.stat_array("missing")


def test_local_array_manifest_controls_rewrites_and_invalidates_legacy_arrays(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    key = "arrays/rewrite"

    store.put_array(key, sparse.csr_matrix(np.eye(3)))
    store.put_array(key, np.full((3, 2), 4, dtype=np.float32))
    manifest_path = tmp_path / key / "array-manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == 2
    assert manifest["filename"] == f"embeddings-v2-{manifest['sha256']}.npy"
    assert manifest["storage_format"] == "npy"
    assert manifest["shape"] == [3, 2]
    assert manifest["dtype"] == "float32"
    assert manifest["nnz"] is None
    assert (tmp_path / key / manifest["filename"]).exists()
    assert {path.name for path in (tmp_path / key).glob("embeddings-v2-*")} == {
        manifest["filename"]
    }
    assert np.array_equal(store.get_array(key), np.full((3, 2), 4, dtype=np.float32))

    store.put_array(key, sparse.csr_matrix(np.full((3, 2), 9)))
    sparse_manifest = json.loads(manifest_path.read_text())
    assert sparse_manifest["filename"] == f"embeddings-v2-{sparse_manifest['sha256']}.npz"
    assert sparse_manifest["sparse_format"] == "csr"
    assert sparse_manifest["nnz"] == 6
    assert {path.name for path in (tmp_path / key).glob("embeddings-v2-*")} == {
        sparse_manifest["filename"]
    }
    assert sparse.issparse(store.get_array(key))
    assert np.array_equal(store.get_array(key).toarray(), np.full((3, 2), 9))

    legacy = tmp_path / "arrays/legacy"
    legacy.mkdir(parents=True)
    np.save(legacy / "embeddings.npy", np.eye(2))
    assert store.exists("arrays/legacy") is False
    with pytest.raises(FileNotFoundError, match="manifest"):
        store.get_array("arrays/legacy")


def test_local_array_manifest_detects_corruption_and_missing_targets(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    key = "arrays/corrupt"
    store.put_array(key, np.eye(2))
    manifest_path = tmp_path / key / "array-manifest.json"
    manifest_path.write_text('{"filename": "../../escape.npy"}')

    with pytest.raises(ValueError, match="manifest fields"):
        store.get_array(key)

    store.put_array(key, np.eye(2))
    manifest = json.loads(manifest_path.read_text())
    (tmp_path / key / manifest["filename"]).unlink()
    assert store.exists(key) is False
    with pytest.raises(FileNotFoundError, match="missing file"):
        store.get_array(key)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2.0, "schema_version must be an integer"),
        ("schema_version", True, "schema_version must be an integer"),
        ("shape", [2, 1.5], "shape must be an array of integers"),
        ("shape", [2, True], "shape must be an array of integers"),
        ("size_bytes", 12.5, "size_bytes must be an integer"),
        ("size_bytes", False, "size_bytes must be an integer"),
        ("dtype", 7, "dtype must be a string"),
        ("sha256", 7, "sha256 must be a string"),
    ],
)
def test_local_array_manifest_rejects_wrong_json_field_types(tmp_path, field, value, message):
    store = LocalArtifactStore(str(tmp_path))
    key = "arrays/strict-manifest"
    store.put_array(key, np.eye(2))
    manifest_path = tmp_path / key / "array-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        store.get_array(key)


def test_local_array_manifest_detects_checksum_and_sparse_nnz_corruption(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    dense_key = "arrays/dense-corrupt"
    dense_path = Path(store.put_array(dense_key, np.eye(3)))
    dense_path.write_bytes(dense_path.read_bytes()[:-1] + b"x")

    with pytest.raises(ValueError, match="SHA-256"):
        store.get_array(dense_key)

    sparse_key = "arrays/sparse-corrupt"
    store.put_array(sparse_key, sparse.csr_matrix(np.eye(3)))
    manifest_path = tmp_path / sparse_key / "array-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["nnz"] = 2
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="sparse nnz"):
        store.get_array(sparse_key)


def test_failed_batch_or_manifest_commit_preserves_previous_committed_array(tmp_path, monkeypatch):
    store = LocalArtifactStore(str(tmp_path))
    key = "arrays/committed"
    original = np.arange(6).reshape(2, 3)
    store.put_array(key, original)

    with pytest.raises(ValueError, match="did not cover all samples"):
        store.put_array_batches(
            key,
            [(np.array([0]), np.ones((1, 3)))],
            n_samples=2,
        )
    assert np.array_equal(store.get_array(key), original)

    def fail_manifest(*_args, **_kwargs):
        raise OSError("simulated manifest publication failure")

    monkeypatch.setattr(store, "_write_array_manifest", fail_manifest)
    with pytest.raises(OSError, match="publication failure"):
        store.put_array(key, np.full((2, 3), 99))
    assert np.array_equal(store.get_array(key), original)


def test_concurrent_local_array_publish_preserves_one_committed_digest(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    key = "arrays/concurrent"
    values = [np.full((4, 3), value, dtype=np.int64) for value in range(8)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda value: store.put_array(key, value), values))

    committed = store.get_array(key)
    manifest = json.loads((tmp_path / key / "array-manifest.json").read_text())
    digest_files = list((tmp_path / key).glob("embeddings-v2-*"))

    assert any(np.array_equal(committed, value) for value in values)
    assert [path.name for path in digest_files] == [manifest["filename"]]


@pytest.mark.parametrize("sparse_value", [False, True])
def test_local_composite_array_roundtrip_uses_persisted_contract(tmp_path, sparse_value):
    store = LocalArtifactStore(str(tmp_path))
    key = "composite/array"
    dense = np.arange(12, dtype=np.float32).reshape(4, 3)
    value = sparse.csr_matrix(dense) if sparse_value else dense
    caller_metadata = {
        "generation": 7,
        "artifact_path": "caller-controlled",
        "shape": [999, 999],
        "n_samples": 999,
        "embedding_dim": 999,
        "dtype": "incorrect",
        "nnz": -1,
    }

    path = store.put_artifact(key, value, caller_metadata)
    loaded, metadata = store.get_artifact(key)
    manifest = json.loads((tmp_path / key / "artifact-manifest.json").read_text())

    assert caller_metadata["artifact_path"] == "caller-controlled"
    assert metadata["artifact_path"] == path
    assert metadata["shape"] == [4, 3]
    assert metadata["n_samples"] == 4
    assert metadata["embedding_dim"] == 3
    assert metadata["dtype"] == "float32"
    assert metadata["sparse"] is sparse_value
    assert metadata["nnz"] == (11 if sparse_value else None)
    assert manifest["kind"] == "array+metadata"
    assert manifest["array"]["filename"] == Path(path).name
    if sparse_value:
        assert sparse.issparse(loaded)
        assert np.array_equal(loaded.toarray(), dense)
    else:
        assert np.array_equal(loaded, dense)
    assert store.exists(key)
    with pytest.raises(ValueError, match="committed composite"):
        store.put_json(key, {"generation": 8})


def test_local_composite_labels_roundtrip_uses_actual_label_contract(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    key = "composite/labels"
    labels = np.asarray(["cat", "dog", "cat"], dtype=object)
    caller_metadata = {
        "generation": 3,
        "artifact_path": "caller-controlled",
        "n_samples": 999,
        "shape": [999],
        "dtype": "incorrect",
        "target_type": "auto",
    }

    path = store.put_labels_artifact(key, labels, caller_metadata)
    loaded, metadata = store.get_labels_artifact(key)
    manifest = json.loads((tmp_path / key / "artifact-manifest.json").read_text())

    assert caller_metadata["n_samples"] == 999
    assert metadata["artifact_path"] == path
    assert metadata["n_samples"] == 3
    assert metadata["shape"] == [3]
    assert metadata["dtype"] == "object"
    assert metadata["target_type"] == "single_label"
    assert metadata["label_names"] is None
    assert metadata["target_names"] is None
    assert manifest["kind"] == "labels+metadata"
    assert np.array_equal(loaded, labels)
    assert np.array_equal(store.get_labels(key), labels)
    assert store.get_json(key) == metadata
    assert store.exists(key)


@pytest.mark.parametrize("provider", ["local", "s3", "gcs"])
def test_composite_label_artifacts_preserve_typed_semantics(provider, tmp_path):
    store = (
        LocalArtifactStore(str(tmp_path))
        if provider == "local"
        else _fake_remote_store_pair(provider)[0]
    )
    distinct = [
        Decimal("1.25"),
        date(2026, 7, 15),
        UUID("12345678-1234-5678-1234-567812345678"),
        1,
        True,
        "1",
        "red",
        _StringLabel.RED,
    ]
    labels = np.empty(len(distinct) * 2, dtype=object)
    labels[:] = distinct + distinct

    store.put_labels_artifact(
        "composite/typed-labels",
        labels,
        {
            "target_type": "single_label",
            # The authoritative contract replaces caller-provided structural fields
            # before strict JSON serialization.
            "label_catalog": [{"value": Decimal("999")}],
        },
    )
    loaded, metadata = store.get_labels_artifact("composite/typed-labels")
    loaded_via_convenience_api = store.get_labels("composite/typed-labels")

    expected = [semantic_label_key(value) for value in distinct + distinct]
    assert [str(value) for value in loaded] == expected
    assert all(isinstance(value, SemanticLabelKey) for value in loaded)
    assert [str(value) for value in loaded_via_convenience_api] == expected
    assert all(isinstance(value, SemanticLabelKey) for value in loaded_via_convenience_api)
    assert len(set(expected[: len(distinct)])) == len(distinct)
    assert metadata["label_encoding"] == LABEL_ENCODING
    assert {item["key"] for item in metadata["label_catalog"]} == set(expected)
    assert {item["display"] for item in metadata["label_catalog"]} == {
        str(value) for value in distinct
    }

    store.put_labels_artifact(
        "composite/typed-labels-copy",
        loaded,
        metadata,
        target_type=metadata["target_type"],
    )
    copied, copied_metadata = store.get_labels_artifact("composite/typed-labels-copy")
    assert [str(value) for value in copied] == expected
    assert copied_metadata["label_catalog"] == metadata["label_catalog"]


@pytest.mark.parametrize("provider", ["local", "s3", "gcs"])
def test_explicit_label_contract_overrides_conflicting_caller_metadata(provider, tmp_path):
    store = (
        LocalArtifactStore(str(tmp_path))
        if provider == "local"
        else _fake_remote_store_pair(provider)[0]
    )

    store.put_labels_artifact(
        "composite/stale-classification",
        ["a", "a", "b", "b"],
        {"target_type": "single_label"},
    )
    _, stale_metadata = store.get_labels_artifact("composite/stale-classification")
    stale_metadata["label_names"] = ["wrong"]
    stale_metadata["target_names"] = ["wrong"]

    store.put_labels_artifact(
        "composite/explicit-regression",
        np.asarray([0.0, 0.5, 1.0]),
        stale_metadata,
        target_type="regression",
        target_names=["score"],
    )

    loaded, metadata = store.get_labels_artifact("composite/explicit-regression")
    assert np.array_equal(loaded, np.asarray([0.0, 0.5, 1.0]))
    assert metadata["target_type"] == "regression"
    assert metadata["label_names"] is None
    assert metadata["target_names"] == ["score"]
    assert metadata["label_encoding"] is None


def test_local_composite_failed_commit_preserves_previous_generations(tmp_path, monkeypatch):
    store = LocalArtifactStore(str(tmp_path))
    array_key = "composite/failure-array"
    labels_key = "composite/failure-labels"
    old_array = np.full((3, 2), 1)
    old_labels = np.asarray([1, 2, 1], dtype=object)
    store.put_artifact(array_key, old_array, {"generation": 1})
    store.put_labels_artifact(labels_key, old_labels, {"generation": 1})

    def fail_manifest(*_args, **_kwargs):
        raise OSError("simulated composite manifest failure")

    monkeypatch.setattr(store, "_write_artifact_manifest", fail_manifest)
    with pytest.raises(OSError, match="manifest failure"):
        store.put_artifact(array_key, np.full((3, 2), 2), {"generation": 2})
    with pytest.raises(OSError, match="manifest failure"):
        store.put_labels_artifact(
            labels_key,
            np.asarray([2, 1, 2], dtype=object),
            {"generation": 2},
        )

    loaded_array, array_metadata = store.get_artifact(array_key)
    loaded_labels, labels_metadata = store.get_labels_artifact(labels_key)
    assert np.array_equal(loaded_array, old_array)
    assert array_metadata["generation"] == 1
    assert loaded_labels.tolist() == [
        SemanticLabelKey(semantic_label_key(value)) for value in old_labels
    ]
    assert labels_metadata["generation"] == 1


def test_local_composite_concurrent_readers_observe_coherent_generations(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    array_key = "composite/concurrent-array"
    labels_key = "composite/concurrent-labels"
    store.put_artifact(array_key, np.zeros((4, 2), dtype=np.int64), {"generation": 0})
    store.put_labels_artifact(
        labels_key,
        np.zeros(4, dtype=np.int64),
        {"generation": 0, "target_type": "single_label"},
    )
    barrier = threading.Barrier(5)
    stop = threading.Event()

    def writer():
        barrier.wait()
        try:
            for generation in range(1, 30):
                store.put_artifact(
                    array_key,
                    np.full((4, 2), generation, dtype=np.int64),
                    {"generation": generation},
                )
                store.put_labels_artifact(
                    labels_key,
                    np.full(4, generation, dtype=np.int64),
                    {"generation": generation, "target_type": "single_label"},
                )
        finally:
            stop.set()

    def reader():
        barrier.wait()
        observations = 0
        while not stop.is_set() or observations == 0:
            array, array_metadata = store.get_artifact(array_key)
            labels, labels_metadata = store.get_labels_artifact(labels_key)
            assert np.all(array == array_metadata["generation"])
            expected_label = semantic_label_key(labels_metadata["generation"])
            assert np.all(labels == expected_label)
            observations += 1
        return observations

    with ThreadPoolExecutor(max_workers=5) as executor:
        writer_future = executor.submit(writer)
        reader_futures = [executor.submit(reader) for _ in range(4)]
        writer_future.result()
        observations = [future.result() for future in reader_futures]

    assert all(count > 0 for count in observations)


def test_local_missing_reads_do_not_create_artifact_directories(tmp_path):
    store = LocalArtifactStore(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        store.get_artifact("missing/composite")
    with pytest.raises(FileNotFoundError):
        store.get_labels_artifact("missing/labels")
    with pytest.raises(FileNotFoundError):
        store.get_array("missing/array")

    assert not (tmp_path / "missing").exists()


def test_local_composite_manifest_corruption_and_missing_components_are_detected(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    key = "composite/corrupt"
    store.put_artifact(key, np.eye(3), {"generation": 1})
    manifest_path = tmp_path / key / "artifact-manifest.json"
    original_manifest = manifest_path.read_text()
    manifest_path.write_text('{"kind": "array+metadata"}')

    with pytest.raises(ValueError, match="manifest fields"):
        store.get_artifact(key)
    with pytest.raises(ValueError, match="manifest fields"):
        store.exists(key)

    manifest_path.write_text(original_manifest)
    manifest = json.loads(original_manifest)
    (tmp_path / key / manifest["metadata"]["filename"]).unlink()
    assert store.exists(key) is False
    with pytest.raises(FileNotFoundError, match="missing metadata"):
        store.get_artifact(key)


@pytest.mark.parametrize("provider", ["s3", "gcs"])
def test_remote_composite_roundtrip_failure_and_exists_without_array_download(
    provider, monkeypatch
):
    store, _, _ = _fake_remote_store_pair(provider)
    array_key = "composite/array"
    labels_key = "composite/labels"
    old_array = np.arange(8, dtype=np.float32).reshape(4, 2)
    old_labels = np.asarray(["a", "b", "a", "b"], dtype=object)
    store.put_artifact(array_key, old_array, {"generation": 1, "n_samples": 999})
    store.put_labels_artifact(
        labels_key,
        old_labels,
        {"generation": 1, "n_samples": 999, "target_type": "auto"},
    )

    loaded_array, array_metadata = store.get_artifact(array_key)
    loaded_labels, labels_metadata = store.get_labels_artifact(labels_key)
    assert np.array_equal(loaded_array, old_array)
    assert array_metadata["n_samples"] == 4
    assert np.array_equal(loaded_labels, old_labels)
    assert labels_metadata["n_samples"] == 4
    assert labels_metadata["target_type"] == "single_label"

    original_download_file = store._download_file
    monkeypatch.setattr(
        store,
        "_download_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exists() must not download array bodies")
        ),
    )
    assert store.exists(array_key)
    assert store.exists(labels_key)
    monkeypatch.setattr(store, "_download_file", original_download_file)

    original_put_bytes = store._put_bytes

    def fail_manifest(name, payload):
        if name.endswith("/artifact-manifest.json"):
            raise OSError("simulated remote manifest failure")
        original_put_bytes(name, payload)

    monkeypatch.setattr(store, "_put_bytes", fail_manifest)
    with pytest.raises(OSError, match="manifest failure"):
        store.put_artifact(array_key, np.full((4, 2), 9), {"generation": 2})
    with pytest.raises(OSError, match="manifest failure"):
        store.put_labels_artifact(
            labels_key,
            np.asarray(["b", "a", "b", "a"], dtype=object),
            {"generation": 2},
        )
    monkeypatch.setattr(store, "_put_bytes", original_put_bytes)

    loaded_array, array_metadata = store.get_artifact(array_key)
    loaded_labels, labels_metadata = store.get_labels_artifact(labels_key)
    assert np.array_equal(loaded_array, old_array)
    assert array_metadata["generation"] == 1
    assert np.array_equal(loaded_labels, old_labels)
    assert labels_metadata["generation"] == 1


@pytest.mark.parametrize("provider", ["s3", "gcs"])
@pytest.mark.parametrize("artifact_kind", ["array", "labels"])
def test_remote_composite_interleaved_writer_cleanup_preserves_staged_generation(
    provider, artifact_kind, monkeypatch
):
    first, second, _ = _fake_remote_store_pair(provider)
    key = f"composite/interleaved-{artifact_kind}"
    if artifact_kind == "array":
        first.put_artifact(key, np.full((3, 2), 0), {"generation": 0})
    else:
        first.put_labels_artifact(
            key,
            np.full(3, 0),
            {"generation": 0, "target_type": "single_label"},
        )

    staged = threading.Event()
    release = threading.Event()
    original_put_bytes = second._put_bytes

    def pause_before_commit(name, payload):
        if name.endswith(f"/{key}/artifact-manifest.json"):
            staged.set()
            if not release.wait(timeout=10):
                raise TimeoutError("timed out waiting to publish staged manifest")
        original_put_bytes(name, payload)

    monkeypatch.setattr(second, "_put_bytes", pause_before_commit)

    def write_second():
        if artifact_kind == "array":
            second.put_artifact(key, np.full((3, 2), 2), {"generation": 2})
        else:
            second.put_labels_artifact(
                key,
                np.full(3, 2),
                {"generation": 2, "target_type": "single_label"},
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(write_second)
        assert staged.wait(timeout=10)
        if artifact_kind == "array":
            first.put_artifact(key, np.full((3, 2), 1), {"generation": 1})
        else:
            first.put_labels_artifact(
                key,
                np.full(3, 1),
                {"generation": 1, "target_type": "single_label"},
            )
        release.set()
        future.result()

    if artifact_kind == "array":
        value, metadata = first.get_artifact(key)
        assert np.all(value == 2)
    else:
        value, metadata = first.get_labels_artifact(key)
        assert np.all(value == semantic_label_key(2))
    assert metadata["generation"] == 2


@pytest.mark.parametrize("provider", ["s3", "gcs"])
def test_remote_readers_retry_generation_switches_and_legacy_transitions(provider, monkeypatch):
    writer, reader, _ = _fake_remote_store_pair(provider)

    array_key = "composite/read-switch-array"
    writer.put_artifact(array_key, np.zeros((3, 2)), {"generation": 0})
    original_load_array = reader._load_remote_array
    array_switched = False

    def load_array_then_switch(key, manifest):
        nonlocal array_switched
        value = original_load_array(key, manifest)
        if not array_switched:
            array_switched = True
            writer.put_artifact(array_key, np.full((3, 2), 1), {"generation": 1})
        return value

    monkeypatch.setattr(reader, "_load_remote_array", load_array_then_switch)
    array, array_metadata = reader.get_artifact(array_key)
    assert np.all(array == 1)
    assert array_metadata["generation"] == 1
    monkeypatch.setattr(reader, "_load_remote_array", original_load_array)

    labels_key = "composite/read-switch-labels"
    writer.put_labels_artifact(
        labels_key,
        np.zeros(3, dtype=int),
        {"generation": 0, "target_type": "single_label"},
    )
    original_load_json = reader._load_remote_json
    labels_switched = False

    def load_labels_then_switch(key, manifest, *, require_object):
        nonlocal labels_switched
        value = original_load_json(key, manifest, require_object=require_object)
        if manifest.role == "labels" and not labels_switched:
            labels_switched = True
            writer.put_labels_artifact(
                labels_key,
                np.ones(3, dtype=int),
                {"generation": 1, "target_type": "single_label"},
            )
        return value

    monkeypatch.setattr(reader, "_load_remote_json", load_labels_then_switch)
    labels, labels_metadata = reader.get_labels_artifact(labels_key)
    assert np.all(labels == semantic_label_key(1))
    assert labels_metadata["generation"] == 1
    monkeypatch.setattr(reader, "_load_remote_json", original_load_json)

    json_key = "composite/legacy-json-switch"
    writer.put_json(json_key, {"generation": 0})
    original_get_bytes = reader._get_bytes
    json_switched = False

    def get_legacy_json_then_switch(name):
        nonlocal json_switched
        payload = original_get_bytes(name)
        if name.endswith(f"/{json_key}/metadata.json") and not json_switched:
            json_switched = True
            writer.put_artifact(json_key, np.eye(2), {"generation": 1})
        return payload

    monkeypatch.setattr(reader, "_get_bytes", get_legacy_json_then_switch)
    assert reader.get_json(json_key)["generation"] == 1
    monkeypatch.setattr(reader, "_get_bytes", original_get_bytes)

    legacy_labels_key = "composite/legacy-labels-switch"
    writer.put_labels(legacy_labels_key, np.zeros(3, dtype=int))
    labels_transitioned = False

    def get_legacy_labels_then_switch(name):
        nonlocal labels_transitioned
        payload = original_get_bytes(name)
        if name.endswith(f"/{legacy_labels_key}/labels.json") and not labels_transitioned:
            labels_transitioned = True
            writer.put_labels_artifact(
                legacy_labels_key,
                np.ones(3, dtype=int),
                {"generation": 1, "target_type": "single_label"},
            )
        return payload

    monkeypatch.setattr(reader, "_get_bytes", get_legacy_labels_then_switch)
    assert np.all(reader.get_labels(legacy_labels_key) == semantic_label_key(1))


@pytest.mark.parametrize(
    "indices,match",
    [
        (np.asarray([-1, 0]), "between 0 and n_samples"),
        (np.asarray([0, 2]), "between 0 and n_samples"),
        (np.asarray([0.0, 1.0]), "must contain integers"),
        (np.asarray([True, False]), "must contain integers"),
        (np.asarray([[0, 1]]), "must be 1D"),
        (np.asarray([0, 0]), "within one batch"),
    ],
)
@pytest.mark.parametrize("sparse_batches", [False, True])
def test_local_batch_writes_reject_invalid_indices(
    tmp_path,
    indices,
    match,
    sparse_batches,
):
    store = LocalArtifactStore(str(tmp_path))
    key = "arrays/validated"
    original = np.arange(6).reshape(2, 3)
    store.put_array(key, original)
    batch = np.ones((2, 3), dtype=int)
    if sparse_batches:
        batch = sparse.csr_matrix(batch)

    with pytest.raises(ValueError, match=match):
        store.put_array_batches(key, [(indices, batch)], n_samples=2)

    assert np.array_equal(store.get_array(key), original)


@pytest.mark.parametrize("sparse_batches", [False, True])
def test_local_batch_writes_reject_duplicates_across_batches(tmp_path, sparse_batches):
    store = LocalArtifactStore(str(tmp_path))
    first = np.ones((1, 2), dtype=np.float32)
    second = np.ones((1, 2), dtype=np.float32)
    if sparse_batches:
        first = sparse.csr_matrix(first)
        second = sparse.csr_matrix(second)

    with pytest.raises(ValueError, match="across batches"):
        store.put_array_batches(
            "arrays/duplicate",
            [(np.asarray([0]), first), (np.asarray([0]), second)],
            n_samples=2,
            require_complete=False,
        )


@pytest.mark.parametrize("sparse_batches", [False, True])
@pytest.mark.parametrize("mismatch", ["width", "dtype"])
def test_local_batch_writes_require_consistent_shapes_and_dtypes(
    tmp_path,
    sparse_batches,
    mismatch,
):
    store = LocalArtifactStore(str(tmp_path))
    first = np.ones((1, 2), dtype=np.float32)
    second = np.ones(
        (1, 3) if mismatch == "width" else (1, 2),
        dtype=np.float32 if mismatch == "width" else np.float64,
    )
    if sparse_batches:
        first = sparse.csr_matrix(first)
        second = sparse.csr_matrix(second)

    with pytest.raises(ValueError, match="column count|consistent dtype"):
        store.put_array_batches(
            "arrays/inconsistent",
            [(np.asarray([0]), first), (np.asarray([1]), second)],
            n_samples=2,
        )


def test_local_batch_writes_allow_intentional_gaps_but_validate_sample_count(tmp_path):
    store = LocalArtifactStore(str(tmp_path))

    path = store.put_array_batches(
        "arrays/gaps",
        [(np.asarray([1]), np.asarray([[3.0, 4.0]]))],
        n_samples=3,
        require_complete=False,
    )

    assert Path(path).exists()
    assert store.get_array("arrays/gaps")[1].tolist() == [3.0, 4.0]
    with pytest.raises(ValueError, match="n_samples"):
        store.put_array_batches(
            "arrays/invalid-count",
            [(np.asarray([0]), np.asarray([[1.0]]))],
            n_samples=0,
        )


def test_local_sparse_batch_writer_streams_one_pass_generator_in_global_row_order(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    batches = iter(
        [
            (
                np.asarray([3, 0]),
                sparse.csr_matrix(np.asarray([[3.0, 0.0, 4.0], [1.0, 0.0, 0.0]])),
            ),
            (
                np.asarray([2, 1]),
                sparse.csr_matrix(np.asarray([[0.0, 2.0, 0.0], [5.0, 6.0, 0.0]])),
            ),
        ]
    )

    class NoLengthHint:
        def __iter__(self):
            return self

        def __next__(self):
            return next(batches)

        def __length_hint__(self):
            raise AssertionError("sparse batches must not be eagerly converted to a list")

    store.put_array_batches("arrays/streamed-sparse", NoLengthHint(), n_samples=4)
    result = store.get_array("arrays/streamed-sparse")

    assert result.getformat() == "csr"
    assert result.nnz == 6
    assert np.array_equal(
        result.toarray(),
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [5.0, 6.0, 0.0],
                [0.0, 2.0, 0.0],
                [3.0, 0.0, 4.0],
            ]
        ),
    )


def test_s3_artifact_store_missing_dependency_raises_clear_error():
    if importlib.util.find_spec("boto3") is not None:
        pytest.skip("boto3 is installed in this environment.")

    store = create_artifact_store("s3://bucket/cache")
    with pytest.raises(ImportError, match="optional 's3' extra"):
        store.put_json("runs/demo", {"ok": True})


def test_gcs_artifact_store_missing_dependency_raises_clear_error():
    try:
        spec = importlib.util.find_spec("google.cloud.storage")
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        pytest.skip("google-cloud-storage is installed in this environment.")

    store = create_artifact_store("gs://bucket/cache")
    with pytest.raises(ImportError, match="optional 'gcs' extra"):
        store.put_json("runs/demo", {"ok": True})


def _fake_boto3_module(objects):
    class FakeClientError(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code}}

    class FakeBody:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return self._payload

    class FakeClient:
        def put_object(self, Bucket, Key, Body):
            objects[(Bucket, Key)] = Body

        def get_object(self, Bucket, Key):
            return {"Body": FakeBody(objects[(Bucket, Key)])}

        def upload_file(self, Filename, Bucket, Key):
            with open(Filename, "rb") as f:
                objects[(Bucket, Key)] = f.read()

        def download_file(self, Bucket, Key, Filename):
            with open(Filename, "wb") as f:
                f.write(objects[(Bucket, Key)])

        def head_object(self, Bucket, Key):
            if (Bucket, Key) not in objects:
                raise FakeClientError("404")
            return {
                "ResponseMetadata": {"HTTPStatusCode": 200},
                "ContentLength": len(objects[(Bucket, Key)]),
            }

        def delete_object(self, Bucket, Key):
            objects.pop((Bucket, Key), None)

        def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
            del ContinuationToken
            return {
                "Contents": [
                    {"Key": key}
                    for bucket, key in objects
                    if bucket == Bucket and key.startswith(Prefix)
                ],
                "IsTruncated": False,
            }

        def delete_objects(self, Bucket, Delete):
            for item in Delete["Objects"]:
                objects.pop((Bucket, item["Key"]), None)

    class FakeSession:
        def __init__(self, profile_name=None, region_name=None):
            self.profile_name = profile_name
            self.region_name = region_name

        def client(self, name, endpoint_url=None):
            assert name == "s3"
            return FakeClient()

    module = types.ModuleType("boto3")
    module.Session = FakeSession
    return module


def _fake_gcs_storage_module(objects):
    class FakeBlob:
        def __init__(self, bucket_name, blob_name):
            self.bucket_name = bucket_name
            self.blob_name = blob_name

        def upload_from_string(self, payload):
            objects[(self.bucket_name, self.blob_name)] = payload

        def download_as_bytes(self):
            return objects[(self.bucket_name, self.blob_name)]

        def upload_from_filename(self, filename):
            with open(filename, "rb") as f:
                objects[(self.bucket_name, self.blob_name)] = f.read()

        def download_to_filename(self, filename):
            with open(filename, "wb") as f:
                f.write(objects[(self.bucket_name, self.blob_name)])

        def exists(self):
            return (self.bucket_name, self.blob_name) in objects

        def reload(self):
            return None

        def delete(self):
            objects.pop((self.bucket_name, self.blob_name), None)

        @property
        def size(self):
            payload = objects.get((self.bucket_name, self.blob_name))
            return len(payload) if payload is not None else None

    class FakeBucket:
        def __init__(self, name):
            self.name = name

        def blob(self, blob_name):
            return FakeBlob(self.name, blob_name)

        def list_blobs(self, prefix=""):
            return [
                FakeBlob(self.name, blob_name)
                for bucket_name, blob_name in objects
                if bucket_name == self.name and blob_name.startswith(prefix)
            ]

    class FakeClient:
        def __init__(self, project=None):
            self.project = project

        def bucket(self, name):
            return FakeBucket(name)

    module = types.ModuleType("google.cloud.storage")
    module.Client = FakeClient
    return module


def _fake_remote_store_pair(provider):
    objects = {}
    if provider == "s3":
        module = _fake_boto3_module(objects)
        first = S3ArtifactStore("bucket", prefix="prefix")
        second = S3ArtifactStore("bucket", prefix="prefix")
        first._client = module.Session().client("s3")
        second._client = module.Session().client("s3")
        return first, second, objects
    if provider == "gcs":
        module = _fake_gcs_storage_module(objects)
        first = GCSArtifactStore("bucket", prefix="prefix")
        second = GCSArtifactStore("bucket", prefix="prefix")
        first._bucket = module.Client().bucket("bucket")
        second._bucket = module.Client().bucket("bucket")
        return first, second, objects
    raise AssertionError(f"Unsupported fake provider {provider!r}.")
