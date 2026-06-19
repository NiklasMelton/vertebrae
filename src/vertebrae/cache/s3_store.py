"""S3-backed artifact store."""

import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from vertebrae.cache.artifact_store import ArtifactStoreConfig
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.utils.labels import labels_from_jsonable, labels_to_jsonable
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import is_sparse_matrix


class S3ArtifactStore:
    """Store artifacts in S3-compatible object storage.

    Args:
        bucket: S3 bucket name.
        prefix: Optional object key prefix.
        endpoint_url: Optional S3-compatible endpoint URL.
        profile_name: Optional boto3 profile name.
        region_name: Optional AWS region name.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: Optional[str] = None,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url
        self.profile_name = profile_name
        self.region_name = region_name
        self._client = None

    @classmethod
    def from_uri(cls, uri: str, **options: Any) -> "S3ArtifactStore":
        """Build an S3 store from a `s3://bucket/prefix` URI."""

        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid S3 artifact store URI: {uri}.")
        return cls(
            bucket=parsed.netloc,
            prefix=parsed.path.lstrip("/"),
            endpoint_url=options.get("endpoint_url"),
            profile_name=options.get("profile_name"),
            region_name=options.get("region_name"),
        )

    def config(self) -> ArtifactStoreConfig:
        """Return a serializable config for reconstructing this store."""

        options = {
            key: value
            for key, value in {
                "endpoint_url": self.endpoint_url,
                "profile_name": self.profile_name,
                "region_name": self.region_name,
            }.items()
            if value is not None
        }
        return ArtifactStoreConfig(
            uri=f"s3://{self.bucket}/{self.prefix}" if self.prefix else f"s3://{self.bucket}",
            options=options,
        )

    def exists(self, key: str) -> bool:
        """Return whether an embedding artifact exists for `key`."""

        return self._object_exists(self._artifact_object_key(key, "embeddings.npy")) or (
            self._object_exists(self._artifact_object_key(key, "embeddings.npz"))
        )

    def put_array(self, key: str, arr: Any) -> str:
        """Store a dense or sparse embedding matrix."""

        filename = "embeddings.npz" if is_sparse_matrix(arr) else "embeddings.npy"
        with tempfile.TemporaryDirectory() as tmpdir:
            local = LocalArtifactStore(tmpdir)
            local_path = Path(local.put_array(key, arr))
            object_key = self._artifact_object_key(key, filename)
            self._upload_file(local_path, object_key)
        return self._uri_for(object_key)

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
            self._upload_file(local_path, self._artifact_object_key(key, local_path.name))
        return self._uri_for(self._artifact_object_key(key, local_path.name))

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse embedding matrix."""

        candidates = ("embeddings.npz", "embeddings.npy")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = None
            for filename in candidates:
                object_key = self._artifact_object_key(key, filename)
                if self._object_exists(object_key):
                    local_path = Path(tmpdir) / filename
                    self._download_file(object_key, local_path)
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
            make_json_safe(labels_to_jsonable(labels)),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        object_key = self._artifact_object_key(key, "labels.json")
        self._put_bytes(object_key, payload)
        return self._uri_for(object_key)

    def get_labels(self, key: str) -> np.ndarray:
        """Load labels from JSON."""

        payload = self._get_bytes(self._artifact_object_key(key, "labels.json"))
        label_names = None
        metadata_key = self._artifact_object_key(key, "metadata.json")
        if self._object_exists(metadata_key):
            metadata_payload = self._get_bytes(metadata_key)
            label_names = json.loads(metadata_payload.decode("utf-8")).get("label_names")
        return labels_from_jsonable(json.loads(payload.decode("utf-8")), label_names=label_names)

    def put_json(self, key: str, obj: dict) -> str:
        """Store JSON metadata for an artifact key."""

        payload = json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
        object_key = self._artifact_object_key(key, "metadata.json")
        self._put_bytes(object_key, payload)
        return self._uri_for(object_key)

    def get_json(self, key: str) -> dict:
        """Load JSON metadata for an artifact key."""

        payload = self._get_bytes(self._artifact_object_key(key, "metadata.json"))
        return json.loads(payload.decode("utf-8"))

    def _artifact_object_key(self, key: str, filename: str) -> str:
        clean_key = key.strip("/").replace("..", "__")
        return "/".join(part for part in (self.prefix, clean_key, filename) if part)

    def _uri_for(self, object_key: str) -> str:
        return f"s3://{self.bucket}/{object_key}"

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "S3 artifact storage requires the optional 's3' extra. Install with "
                "`poetry install --extras s3`."
            ) from exc
        session = boto3.Session(
            profile_name=self.profile_name,
            region_name=self.region_name,
        )
        self._client = session.client("s3", endpoint_url=self.endpoint_url)
        return self._client

    def _put_bytes(self, object_key: str, payload: bytes) -> None:
        self._client_or_raise().put_object(Bucket=self.bucket, Key=object_key, Body=payload)

    def _get_bytes(self, object_key: str) -> bytes:
        body = self._client_or_raise().get_object(Bucket=self.bucket, Key=object_key)["Body"]
        return body.read()

    def _upload_file(self, local_path: Path, object_key: str) -> None:
        self._client_or_raise().upload_file(str(local_path), self.bucket, object_key)

    def _download_file(self, object_key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client_or_raise().download_file(self.bucket, object_key, str(local_path))

    def _object_exists(self, object_key: str) -> bool:
        try:
            self._client_or_raise().head_object(Bucket=self.bucket, Key=object_key)
            return True
        except Exception as exc:
            response = getattr(exc, "response", {})
            code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
