import importlib
import os

import numpy as np
import pytest
from scipy import sparse

from vertebrae.cache import create_artifact_store, create_artifact_store_from_config

pytestmark = [
    pytest.mark.cloudemulators,
    pytest.mark.skipif(
        os.environ.get("VERTABRAE_RUN_CLOUD_EMULATORS") != "1",
        reason="set VERTABRAE_RUN_CLOUD_EMULATORS=1 to run cloud emulator tests",
    ),
]


def test_s3_artifact_store_roundtrip_with_moto_emulator():
    boto3 = _require_module("boto3")
    moto = _require_module("moto")
    mock_aws = getattr(moto, "mock_aws", None)
    if mock_aws is None:
        raise AssertionError("Required dependency 'moto' does not expose mock_aws.")

    with mock_aws():
        bucket = "vertebrae-emulator"
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
        store = create_artifact_store(
            f"s3://{bucket}/integration-prefix",
            region_name="us-east-1",
        )

        _assert_artifact_store_roundtrip(store)
        recreated = create_artifact_store_from_config(store.config())
        assert np.array_equal(recreated.get_array("arrays/dense"), np.arange(12).reshape(3, 4))


def test_gcs_artifact_store_roundtrip_with_fake_gcs_server():
    _require_module("google.cloud.storage")
    emulator_host = os.environ.get("STORAGE_EMULATOR_HOST")
    if not emulator_host:
        raise AssertionError("STORAGE_EMULATOR_HOST must point at fake-gcs-server.")

    from google.auth.credentials import AnonymousCredentials
    from google.cloud import storage

    bucket_name = "vertebrae-emulator"
    client = storage.Client(
        project="vertebrae-test",
        credentials=AnonymousCredentials(),
        client_options={"api_endpoint": emulator_host},
    )
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        client.create_bucket(bucket)

    store = create_artifact_store(
        f"gs://{bucket_name}/integration-prefix",
        project="vertebrae-test",
        emulator_host=emulator_host,
    )

    _assert_artifact_store_roundtrip(store)
    recreated = create_artifact_store_from_config(store.config())
    assert recreated.get_json("runs/demo")["provider"] == "emulator"


def _assert_artifact_store_roundtrip(store):
    dense = np.arange(12).reshape(3, 4)
    sparse_matrix = sparse.csr_matrix(np.eye(4))
    labels = np.array(["left", "left", "right", "right"])
    batches = [
        (np.ones((2, 3), dtype=np.float32), np.array([0, 1])),
        (np.zeros((2, 3), dtype=np.float32), np.array([2, 3])),
    ]

    store.put_json("runs/demo", {"provider": "emulator", "ok": True})
    store.put_labels("labels/demo", labels)
    store.put_array("arrays/dense", dense)
    store.put_array("arrays/sparse", sparse_matrix)
    store.put_array_batches("arrays/batched", batches, n_samples=4)

    assert store.get_json("runs/demo")["ok"] is True
    assert np.array_equal(store.get_labels("labels/demo"), labels)
    assert np.array_equal(store.get_array("arrays/dense"), dense)
    assert np.array_equal(store.get_array("arrays/sparse").toarray(), sparse_matrix.toarray())
    assert np.array_equal(
        store.get_array("arrays/batched"),
        np.vstack([np.ones((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]),
    )
    assert store.exists("arrays/dense")


def _require_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AssertionError(
            f"Required dependency {module_name!r} is not installed for enabled "
            "cloud emulator tests."
        ) from exc
