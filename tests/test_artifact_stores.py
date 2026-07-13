import importlib.util
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
    store.put_json("runs/demo", {"ok": True})
    store.put_labels("labels/demo", np.array(["a", "b"]))
    store.put_array("arrays/dense", np.arange(6).reshape(2, 3))
    store.put_array("arrays/sparse", sparse.csr_matrix(np.eye(3)))

    recreated = create_artifact_store_from_config(store.config())
    assert recreated.get_json("runs/demo")["ok"] is True
    assert np.array_equal(recreated.get_labels("labels/demo"), np.array(["a", "b"]))
    assert np.array_equal(recreated.get_array("arrays/dense"), np.arange(6).reshape(2, 3))
    assert np.array_equal(recreated.get_array("arrays/sparse").toarray(), np.eye(3))
    dense_stat = recreated.stat_array("arrays/dense")
    sparse_stat = recreated.stat_array("arrays/sparse")
    assert dense_stat.size_bytes > 0
    assert dense_stat.storage_format == "npy"
    assert sparse_stat.storage_format == "npz"


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
    store.put_json("runs/demo", {"ok": True})
    store.put_array("arrays/dense", np.arange(6).reshape(2, 3))

    recreated = create_artifact_store_from_config(store.config())
    assert recreated.get_json("runs/demo")["ok"] is True
    assert np.array_equal(recreated.get_array("arrays/dense"), np.arange(6).reshape(2, 3))
    stat = recreated.stat_array("arrays/dense")
    assert stat.size_bytes > 0
    assert stat.storage_format == "npy"


def test_local_array_stat_uses_actual_file_size(tmp_path):
    store = LocalArtifactStore(str(tmp_path))
    path = store.put_array("arrays/dense", np.arange(6).reshape(2, 3))

    stat = store.stat_array("arrays/dense")

    assert stat.size_bytes == Path(path).stat().st_size
    assert stat.uri == path
    assert stat.storage_format == "npy"
    with pytest.raises(FileNotFoundError):
        store.stat_array("missing")


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
