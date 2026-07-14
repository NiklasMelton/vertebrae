"""GCS-backed artifact store."""

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from vertebrae.cache.artifact_store import (
    ARRAY_MANIFEST_FILENAME,
    ArrayArtifactManifest,
    ArtifactStat,
    ArtifactStoreConfig,
)
from vertebrae.cache.keys import validate_artifact_key
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.utils.labels import labels_from_jsonable, labels_to_jsonable
from vertebrae.utils.serialization import json_dumps_strict


class GCSArtifactStore:
    """Store artifacts in Google Cloud Storage.

    Args:
        bucket: GCS bucket name.
        prefix: Optional object key prefix.
        project: Optional GCP project override.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        project: Optional[str] = None,
        emulator_host: Optional[str] = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.project = project
        self.emulator_host = emulator_host
        self._bucket = None

    @classmethod
    def from_uri(cls, uri: str, **options: Any) -> "GCSArtifactStore":
        """Build a GCS store from a `gs://bucket/prefix` URI."""

        parsed = urlparse(uri)
        if parsed.scheme != "gs" or not parsed.netloc:
            raise ValueError(f"Invalid GCS artifact store URI: {uri}.")
        return cls(
            bucket=parsed.netloc,
            prefix=parsed.path.lstrip("/"),
            project=options.get("project"),
            emulator_host=options.get("emulator_host"),
        )

    def config(self) -> ArtifactStoreConfig:
        """Return a serializable config for reconstructing this store."""

        options = {
            key: value
            for key, value in {
                "project": self.project,
                "emulator_host": self.emulator_host,
            }.items()
            if value is not None
        }
        return ArtifactStoreConfig(
            uri=f"gs://{self.bucket}/{self.prefix}" if self.prefix else f"gs://{self.bucket}",
            options=options,
        )

    def exists(self, key: str) -> bool:
        """Return whether an embedding artifact exists for `key`."""

        manifest_name = self._artifact_blob_name(key, ARRAY_MANIFEST_FILENAME)
        if not self._blob_exists(manifest_name):
            return False
        manifest = self._read_array_manifest(key)
        return self._blob_exists(self._artifact_blob_name(key, manifest.filename))

    def put_array(self, key: str, arr: Any) -> str:
        """Store a dense or sparse embedding matrix."""

        validate_artifact_key(key)
        with tempfile.TemporaryDirectory() as tmpdir:
            local = LocalArtifactStore(tmpdir)
            local_path = Path(local.put_array(key, arr))
            manifest = local._read_array_manifest(local._path(key))
            blob_name = self._publish_local_array(key, local_path, manifest)
        return self._uri_for(blob_name)

    def put_array_batches(
        self,
        key: str,
        batches: Iterable[Tuple[np.ndarray, Any]],
        n_samples: int,
        require_complete: bool = True,
    ) -> str:
        """Store embeddings from deterministic batches."""

        validate_artifact_key(key)
        with tempfile.TemporaryDirectory() as tmpdir:
            local = LocalArtifactStore(tmpdir)
            local_path = Path(
                local.put_array_batches(
                    key,
                    batches,
                    n_samples=n_samples,
                    require_complete=require_complete,
                )
            )
            manifest = local._read_array_manifest(local._path(key))
            blob_name = self._publish_local_array(key, local_path, manifest)
        return self._uri_for(blob_name)

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse embedding matrix."""

        manifest = self._read_array_manifest(key)
        blob_name = self._artifact_blob_name(key, manifest.filename)
        if not self._blob_exists(blob_name):
            raise FileNotFoundError(
                f"Array manifest for key {key} references missing file {manifest.filename}."
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / manifest.filename
            self._download_file(blob_name, local_path)
            if manifest.storage_format == "npz":
                from scipy import sparse

                return sparse.load_npz(local_path)
            return np.load(local_path, allow_pickle=False)

    def stat_array(self, key: str) -> ArtifactStat:
        """Return blob size using GCS metadata without downloading it."""

        manifest = self._read_array_manifest(key)
        blob_name = self._artifact_blob_name(key, manifest.filename)
        blob = self._bucket_or_raise().blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(
                f"Array manifest for key {key} references missing file {manifest.filename}."
            )
        blob.reload()
        return ArtifactStat(
            uri=self._uri_for(blob_name),
            size_bytes=int(blob.size or 0),
            storage_format=manifest.storage_format,
        )

    def put_labels(self, key: str, labels: Any) -> str:
        """Store labels as JSON."""

        payload = json_dumps_strict(labels_to_jsonable(labels), indent=2, sort_keys=True).encode(
            "utf-8"
        )
        blob_name = self._artifact_blob_name(key, "labels.json")
        self._put_bytes(blob_name, payload)
        return self._uri_for(blob_name)

    def get_labels(self, key: str) -> np.ndarray:
        """Load labels from JSON."""

        payload = self._get_bytes(self._artifact_blob_name(key, "labels.json"))
        label_names = None
        target_type = "auto"
        target_names = None
        metadata_name = self._artifact_blob_name(key, "metadata.json")
        if self._blob_exists(metadata_name):
            metadata_payload = self._get_bytes(metadata_name)
            metadata = json.loads(metadata_payload.decode("utf-8"))
            label_names = metadata.get("label_names")
            target_type = metadata.get("target_type", "auto")
            target_names = metadata.get("target_names")
        return labels_from_jsonable(
            json.loads(payload.decode("utf-8")),
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    def put_json(self, key: str, obj: dict) -> str:
        """Store JSON metadata for an artifact key."""

        payload = json_dumps_strict(obj, indent=2, sort_keys=True).encode("utf-8")
        blob_name = self._artifact_blob_name(key, "metadata.json")
        self._put_bytes(blob_name, payload)
        return self._uri_for(blob_name)

    def get_json(self, key: str) -> dict:
        """Load JSON metadata for an artifact key."""

        payload = self._get_bytes(self._artifact_blob_name(key, "metadata.json"))
        return json.loads(payload.decode("utf-8"))

    def delete_prefix(self, prefix: str) -> None:
        """Delete every GCS blob beneath an artifact key prefix."""

        if prefix == "":
            raise ValueError("Refusing to delete the artifact-store root.")
        clean = validate_artifact_key(prefix)
        blob_prefix = "/".join(part for part in (self.prefix, clean) if part) + "/"
        bucket = self._bucket_or_raise()
        for blob in list(bucket.list_blobs(prefix=blob_prefix)):
            blob.delete()

    def _read_array_manifest(self, key: str) -> ArrayArtifactManifest:
        manifest_name = self._artifact_blob_name(key, ARRAY_MANIFEST_FILENAME)
        if not self._blob_exists(manifest_name):
            raise FileNotFoundError(f"No committed array manifest found for key {key}.")
        payload = self._get_bytes(manifest_name)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Array manifest for key {key} is not valid JSON.") from exc
        return ArrayArtifactManifest.from_dict(value)

    def _publish_local_array(
        self,
        key: str,
        local_path: Path,
        manifest: ArrayArtifactManifest,
    ) -> str:
        blob_name = self._artifact_blob_name(key, manifest.filename)
        self._upload_file(local_path, blob_name)
        manifest_name = self._artifact_blob_name(key, ARRAY_MANIFEST_FILENAME)
        self._put_bytes(
            manifest_name,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        stale_name = "embeddings.npy" if manifest.storage_format == "npz" else "embeddings.npz"
        stale_blob_name = self._artifact_blob_name(key, stale_name)
        stale_blob = self._bucket_or_raise().blob(stale_blob_name)
        if stale_blob.exists():
            try:
                stale_blob.delete()
            except Exception as exc:
                warnings.warn(
                    f"Committed {self._uri_for(blob_name)} but could not remove stale "
                    f"{self._uri_for(stale_blob_name)}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return blob_name

    def _artifact_blob_name(self, key: str, filename: str) -> str:
        validated_key = validate_artifact_key(key)
        return "/".join(part for part in (self.prefix, validated_key, filename) if part)

    def _uri_for(self, blob_name: str) -> str:
        return f"gs://{self.bucket}/{blob_name}"

    def _bucket_or_raise(self) -> Any:
        if self._bucket is not None:
            return self._bucket
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError(
                "GCS artifact storage requires the optional 'gcs' extra. Install with "
                "`poetry install --extras gcs`."
            ) from exc
        client_kwargs: dict[str, Any] = {"project": self.project}
        emulator_host = self.emulator_host or os.environ.get("STORAGE_EMULATOR_HOST")
        if emulator_host:
            from google.auth.credentials import AnonymousCredentials

            client_kwargs["credentials"] = AnonymousCredentials()
            client_kwargs["client_options"] = {"api_endpoint": emulator_host}
        client = storage.Client(**client_kwargs)
        self._bucket = client.bucket(self.bucket)
        return self._bucket

    def _put_bytes(self, blob_name: str, payload: bytes) -> None:
        self._bucket_or_raise().blob(blob_name).upload_from_string(payload)

    def _get_bytes(self, blob_name: str) -> bytes:
        return self._bucket_or_raise().blob(blob_name).download_as_bytes()

    def _upload_file(self, local_path: Path, blob_name: str) -> None:
        self._bucket_or_raise().blob(blob_name).upload_from_filename(str(local_path))

    def _download_file(self, blob_name: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._bucket_or_raise().blob(blob_name).download_to_filename(str(local_path))

    def _blob_exists(self, blob_name: str) -> bool:
        return bool(self._bucket_or_raise().blob(blob_name).exists())
