import importlib.util
import json
import sys
import types
from pathlib import Path

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
    store.put_labels("labels/demo", np.array(["a", "b"]))
    store.put_array("arrays/dense", np.arange(6).reshape(2, 3))
    store.put_array("arrays/sparse", sparse.csr_matrix(np.eye(3)))
    store.put_array("arrays/rewrite", sparse.csr_matrix(np.eye(3)))
    store.put_array("arrays/rewrite", np.full((3, 3), 7))

    recreated = create_artifact_store_from_config(store.config())
    assert recreated.get_json("runs/demo")["ok"] is True
    assert recreated.get_json("runs/demo")["tags"] == ["a", "b"]
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
    store.put_array("arrays/dense", np.arange(6).reshape(2, 3))
    store.put_array("arrays/rewrite", np.eye(3))
    store.put_array("arrays/rewrite", sparse.csr_matrix(np.full((3, 3), 2)))

    recreated = create_artifact_store_from_config(store.config())
    assert recreated.get_json("runs/demo")["ok"] is True
    assert recreated.get_json("runs/demo")["tags"] == ["a", "b"]
    assert np.array_equal(recreated.get_array("arrays/dense"), np.arange(6).reshape(2, 3))
    assert np.array_equal(recreated.get_array("arrays/rewrite").toarray(), np.full((3, 3), 2))
    stat = recreated.stat_array("arrays/dense")
    assert stat.size_bytes > 0
    assert stat.storage_format == "npy"
    before = dict(objects)
    with pytest.raises(TypeError, match=r"unsupported object at \$\.metadata\.model"):
        store.put_json("runs/invalid", {"metadata": {"model": object()}})
    assert objects == before


def test_local_json_rejects_invalid_metadata_without_overwriting_previous_value(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    store.put_json("runs/demo", {"ok": True})

    with pytest.raises(TypeError, match=r"unsupported object at \$\.metadata\.model"):
        store.put_json("runs/demo", {"metadata": {"model": object()}})

    assert store.get_json("runs/demo") == {"ok": True}


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

    assert manifest["filename"] == "embeddings.npy"
    assert manifest["storage_format"] == "npy"
    assert manifest["shape"] == [3, 2]
    assert manifest["dtype"] == "float32"
    assert not (tmp_path / key / "embeddings.npz").exists()
    assert np.array_equal(store.get_array(key), np.full((3, 2), 4, dtype=np.float32))

    store.put_array(key, sparse.csr_matrix(np.full((3, 2), 9)))
    assert sparse.issparse(store.get_array(key))
    assert np.array_equal(store.get_array(key).toarray(), np.full((3, 2), 9))
    assert not (tmp_path / key / "embeddings.npy").exists()

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
    (tmp_path / key / "embeddings.npy").unlink()
    assert store.exists(key) is False
    with pytest.raises(FileNotFoundError, match="missing file"):
        store.get_array(key)


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
        store.put_array(key, sparse.csr_matrix(np.eye(2)))
    assert np.array_equal(store.get_array(key), original)


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

    class FakeClient:
        def __init__(self, project=None):
            self.project = project

        def bucket(self, name):
            return FakeBucket(name)

    module = types.ModuleType("google.cloud.storage")
    module.Client = FakeClient
    return module
