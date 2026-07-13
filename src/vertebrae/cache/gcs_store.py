"""GCS-backed artifact store."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from vertebrae.cache.artifact_store import ArtifactStat, ArtifactStoreConfig
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.utils.labels import labels_from_jsonable, labels_to_jsonable
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import is_sparse_matrix


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

        return self._blob_exists(self._artifact_blob_name(key, "embeddings.npy")) or (
            self._blob_exists(self._artifact_blob_name(key, "embeddings.npz"))
        )

    def put_array(self, key: str, arr: Any) -> str:
        """Store a dense or sparse embedding matrix."""

        filename = "embeddings.npz" if is_sparse_matrix(arr) else "embeddings.npy"
        with tempfile.TemporaryDirectory() as tmpdir:
            local = LocalArtifactStore(tmpdir)
            local_path = Path(local.put_array(key, arr))
            blob_name = self._artifact_blob_name(key, filename)
            self._upload_file(local_path, blob_name)
        return self._uri_for(blob_name)

    def put_array_batches(
        self,
        key: str,
        batches: Iterable[Tuple[np.ndarray, Any]],
        n_samples: int,
        require_complete: bool = True,
    ) -> str:
        """Store embeddings from deterministic batches."""

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
            self._upload_file(local_path, self._artifact_blob_name(key, local_path.name))
        return self._uri_for(self._artifact_blob_name(key, local_path.name))

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse embedding matrix."""

        candidates = ("embeddings.npz", "embeddings.npy")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = None
            for filename in candidates:
                blob_name = self._artifact_blob_name(key, filename)
                if self._blob_exists(blob_name):
                    local_path = Path(tmpdir) / filename
                    self._download_file(blob_name, local_path)
                    break
            if local_path is None:
                raise FileNotFoundError(f"No array artifact found for key {key}.")
            if local_path.suffix == ".npz":
                from scipy import sparse

                return sparse.load_npz(local_path)
            return np.load(local_path, allow_pickle=False)

    def stat_array(self, key: str) -> ArtifactStat:
        """Return blob size using GCS metadata without downloading it."""

        for filename, storage_format in (("embeddings.npz", "npz"), ("embeddings.npy", "npy")):
            blob_name = self._artifact_blob_name(key, filename)
            blob = self._bucket_or_raise().blob(blob_name)
            if not blob.exists():
                continue
            blob.reload()
            return ArtifactStat(
                uri=self._uri_for(blob_name),
                size_bytes=int(blob.size or 0),
                storage_format=storage_format,
            )
        raise FileNotFoundError(f"No array artifact found for key {key}.")

    def put_labels(self, key: str, labels: Any) -> str:
        """Store labels as JSON."""

        payload = json.dumps(
            make_json_safe(labels_to_jsonable(labels)),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
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

        payload = json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
        blob_name = self._artifact_blob_name(key, "metadata.json")
        self._put_bytes(blob_name, payload)
        return self._uri_for(blob_name)

    def get_json(self, key: str) -> dict:
        """Load JSON metadata for an artifact key."""

        payload = self._get_bytes(self._artifact_blob_name(key, "metadata.json"))
        return json.loads(payload.decode("utf-8"))

    def _artifact_blob_name(self, key: str, filename: str) -> str:
        clean_key = key.strip("/").replace("..", "__")
        return "/".join(part for part in (self.prefix, clean_key, filename) if part)

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
