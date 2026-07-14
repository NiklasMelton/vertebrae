"""S3-backed artifact store."""

import json
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
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.utils.labels import labels_from_jsonable, labels_to_jsonable
from vertebrae.utils.serialization import json_dumps_strict


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

        manifest_key = self._artifact_object_key(key, ARRAY_MANIFEST_FILENAME)
        if not self._object_exists(manifest_key):
            return False
        manifest = self._read_array_manifest(key)
        return self._object_exists(self._artifact_object_key(key, manifest.filename))

    def put_array(self, key: str, arr: Any) -> str:
        """Store a dense or sparse embedding matrix."""

        with tempfile.TemporaryDirectory() as tmpdir:
            local = LocalArtifactStore(tmpdir)
            local_path = Path(local.put_array(key, arr))
            manifest = local._read_array_manifest(local._path(key))
            object_key = self._publish_local_array(key, local_path, manifest)
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
            manifest = local._read_array_manifest(local._path(key))
            object_key = self._publish_local_array(key, local_path, manifest)
        return self._uri_for(object_key)

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse embedding matrix."""

        manifest = self._read_array_manifest(key)
        object_key = self._artifact_object_key(key, manifest.filename)
        if not self._object_exists(object_key):
            raise FileNotFoundError(
                f"Array manifest for key {key} references missing file {manifest.filename}."
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / manifest.filename
            self._download_file(object_key, local_path)
            if manifest.storage_format == "npz":
                from scipy import sparse

                return sparse.load_npz(local_path)
            return np.load(local_path, allow_pickle=False)

    def stat_array(self, key: str) -> ArtifactStat:
        """Return object size using S3 metadata without downloading it."""

        manifest = self._read_array_manifest(key)
        object_key = self._artifact_object_key(key, manifest.filename)
        try:
            response = self._client_or_raise().head_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:
            error = getattr(exc, "response", {})
            code = error.get("Error", {}).get("Code") if isinstance(error, dict) else None
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(
                    f"Array manifest for key {key} references missing file {manifest.filename}."
                ) from exc
            raise
        return ArtifactStat(
            uri=self._uri_for(object_key),
            size_bytes=int(response["ContentLength"]),
            storage_format=manifest.storage_format,
        )

    def put_labels(self, key: str, labels: Any) -> str:
        """Store labels as JSON."""

        payload = json_dumps_strict(labels_to_jsonable(labels), indent=2, sort_keys=True).encode(
            "utf-8"
        )
        object_key = self._artifact_object_key(key, "labels.json")
        self._put_bytes(object_key, payload)
        return self._uri_for(object_key)

    def get_labels(self, key: str) -> np.ndarray:
        """Load labels from JSON."""

        payload = self._get_bytes(self._artifact_object_key(key, "labels.json"))
        label_names = None
        target_type = "auto"
        target_names = None
        metadata_key = self._artifact_object_key(key, "metadata.json")
        if self._object_exists(metadata_key):
            metadata_payload = self._get_bytes(metadata_key)
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
        object_key = self._artifact_object_key(key, "metadata.json")
        self._put_bytes(object_key, payload)
        return self._uri_for(object_key)

    def get_json(self, key: str) -> dict:
        """Load JSON metadata for an artifact key."""

        payload = self._get_bytes(self._artifact_object_key(key, "metadata.json"))
        return json.loads(payload.decode("utf-8"))

    def delete_prefix(self, prefix: str) -> None:
        """Delete every S3 object beneath an artifact key prefix."""

        clean = prefix.strip("/").replace("..", "__")
        if not clean:
            raise ValueError("Refusing to delete the artifact-store root.")
        object_prefix = "/".join(part for part in (self.prefix, clean) if part) + "/"
        client = self._client_or_raise()
        continuation = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": object_prefix}
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = client.list_objects_v2(**kwargs)
            objects = [{"Key": item["Key"]} for item in response.get("Contents", [])]
            if objects:
                client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")

    def _read_array_manifest(self, key: str) -> ArrayArtifactManifest:
        manifest_key = self._artifact_object_key(key, ARRAY_MANIFEST_FILENAME)
        if not self._object_exists(manifest_key):
            raise FileNotFoundError(f"No committed array manifest found for key {key}.")
        payload = self._get_bytes(manifest_key)
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
        object_key = self._artifact_object_key(key, manifest.filename)
        self._upload_file(local_path, object_key)
        manifest_key = self._artifact_object_key(key, ARRAY_MANIFEST_FILENAME)
        self._put_bytes(
            manifest_key,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        stale_name = "embeddings.npy" if manifest.storage_format == "npz" else "embeddings.npz"
        stale_key = self._artifact_object_key(key, stale_name)
        if self._object_exists(stale_key):
            try:
                self._client_or_raise().delete_object(Bucket=self.bucket, Key=stale_key)
            except Exception as exc:
                warnings.warn(
                    f"Committed {self._uri_for(object_key)} but could not remove stale "
                    f"{self._uri_for(stale_key)}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return object_key

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
