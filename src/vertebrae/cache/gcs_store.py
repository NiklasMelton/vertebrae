"""GCS-backed artifact store."""

import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from vertebrae.cache.artifact_store import ArtifactStoreConfig
from vertebrae.cache.local_store import LocalArtifactStore
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
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.project = project
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
        )

    def config(self) -> ArtifactStoreConfig:
        """Return a serializable config for reconstructing this store."""

        options = {"project": self.project} if self.project is not None else {}
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

    def put_labels(self, key: str, labels: Any) -> str:
        """Store labels as JSON."""

        payload = json.dumps(
            make_json_safe(list(np.asarray(labels))),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        blob_name = self._artifact_blob_name(key, "labels.json")
        self._put_bytes(blob_name, payload)
        return self._uri_for(blob_name)

    def get_labels(self, key: str) -> np.ndarray:
        """Load labels from JSON."""

        payload = self._get_bytes(self._artifact_blob_name(key, "labels.json"))
        return np.asarray(json.loads(payload.decode("utf-8")))

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
        client = storage.Client(project=self.project)
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
