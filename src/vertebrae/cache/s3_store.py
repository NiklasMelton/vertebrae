"""S3-backed artifact store."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np

from vertebrae.cache.artifact_store import (
    ARRAY_MANIFEST_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
    ArrayArtifactManifest,
    ArtifactManifest,
    ArtifactStat,
    ArtifactStoreConfig,
    JSONArtifactManifest,
    LabelsArtifactManifest,
)
from vertebrae.cache.keys import validate_artifact_key
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.utils.labels import (
    decode_label_artifact_metadata,
    labels_from_jsonable,
    labels_to_jsonable,
)
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

        artifact_manifest_key = self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)
        if self._object_exists(artifact_manifest_key):
            artifact_manifest = self._read_artifact_manifest(key)
            primary = (
                artifact_manifest.array
                if isinstance(artifact_manifest, ArtifactManifest)
                else artifact_manifest.labels
            )
            primary_size = self._object_size(self._artifact_object_key(key, primary.filename))
            metadata_size = self._object_size(
                self._artifact_object_key(key, artifact_manifest.metadata.filename)
            )
            return primary_size == primary.size_bytes and (
                metadata_size == artifact_manifest.metadata.size_bytes
            )
        array_manifest_key = self._artifact_object_key(key, ARRAY_MANIFEST_FILENAME)
        if not self._object_exists(array_manifest_key):
            return False
        array_manifest = self._read_array_manifest(key)
        return (
            self._object_size(self._artifact_object_key(key, array_manifest.filename))
            == array_manifest.size_bytes
        )

    def put_array(self, key: str, arr: Any) -> str:
        """Store a dense or sparse embedding matrix."""

        validate_artifact_key(key)
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
            object_key = self._publish_local_array(key, local_path, manifest)
        return self._uri_for(object_key)

    def put_artifact(
        self,
        key: str,
        arr: Any,
        metadata: dict,
        *,
        metadata_finalizer: Optional[
            Callable[[dict, ArrayArtifactManifest, ArtifactStat], dict]
        ] = None,
    ) -> str:
        """Commit an array and immutable metadata with a last-written manifest."""

        if not isinstance(metadata, dict):
            raise TypeError("Artifact metadata must be a dictionary.")
        validate_artifact_key(key)
        with tempfile.TemporaryDirectory() as tmpdir:
            local = LocalArtifactStore(tmpdir)
            local_path = Path(local.put_array("staged", arr))
            array_manifest = local._read_array_manifest(local._path("staged"))
            array_key = self._publish_local_artifact(
                key,
                local_path,
                array_manifest=array_manifest,
                metadata=metadata,
                metadata_finalizer=metadata_finalizer,
            )
        return self._uri_for(array_key)

    def put_artifact_batches(
        self,
        key: str,
        batches: Iterable[Tuple[np.ndarray, Any]],
        n_samples: int,
        metadata: dict,
        require_complete: bool = True,
        *,
        metadata_finalizer: Optional[
            Callable[[dict, ArrayArtifactManifest, ArtifactStat], dict]
        ] = None,
    ) -> str:
        """Commit batched arrays and immutable metadata as one generation."""

        LocalArtifactStore._validate_metadata(metadata)
        validate_artifact_key(key)
        with tempfile.TemporaryDirectory() as tmpdir:
            local = LocalArtifactStore(tmpdir)
            local_path = Path(
                local.put_array_batches(
                    "staged",
                    batches,
                    n_samples=n_samples,
                    require_complete=require_complete,
                )
            )
            array_manifest = local._read_array_manifest(local._path("staged"))
            array_key = self._publish_local_artifact(
                key,
                local_path,
                array_manifest=array_manifest,
                metadata=metadata,
                metadata_finalizer=metadata_finalizer,
            )
        return self._uri_for(array_key)

    def get_artifact(self, key: str) -> tuple[Any, dict]:
        """Load a validated array/metadata pair, retrying one manifest switch."""

        last_error: Optional[BaseException] = None
        for _ in range(2):
            manifest = self._read_artifact_manifest(key)
            if not isinstance(manifest, ArtifactManifest):
                raise ValueError(f"Committed artifact for key {key} does not contain an array.")
            try:
                value = self._load_remote_array(key, manifest.array)
                metadata = self._load_remote_json(key, manifest.metadata, require_object=True)
                if self._read_artifact_manifest(key) != manifest:
                    continue
                return value, metadata
            except FileNotFoundError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Artifact manifest for key {key} changed repeatedly while reading.")

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse embedding matrix."""

        if self._object_exists(self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)):
            return self.get_artifact(key)[0]
        last_error: Optional[BaseException] = None
        for _ in range(2):
            try:
                manifest = self._read_array_manifest(key)
                value = self._load_remote_array(key, manifest)
                if self._object_exists(self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)):
                    return self.get_artifact(key)[0]
                if self._read_array_manifest(key) != manifest:
                    continue
                return value
            except FileNotFoundError as exc:
                last_error = exc
                if self._object_exists(self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)):
                    return self.get_artifact(key)[0]
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Array manifest for key {key} changed repeatedly while reading.")

    def stat_array(self, key: str) -> ArtifactStat:
        """Return object size using S3 metadata without downloading it."""

        if self._object_exists(self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)):
            artifact_manifest = self._read_artifact_manifest(key)
            if not isinstance(artifact_manifest, ArtifactManifest):
                raise ValueError(f"Committed artifact for key {key} does not contain an array.")
            manifest = artifact_manifest.array
        else:
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
        size_bytes = int(response["ContentLength"])
        if size_bytes != manifest.size_bytes:
            raise ValueError(
                f"Array artifact {self._uri_for(object_key)} has size {size_bytes}, expected "
                f"{manifest.size_bytes} from its manifest."
            )
        return ArtifactStat(
            uri=self._uri_for(object_key),
            size_bytes=size_bytes,
            storage_format=manifest.storage_format,
        )

    def put_labels(
        self,
        key: str,
        labels: Any,
        *,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> str:
        """Store labels as JSON."""

        payload = json_dumps_strict(
            labels_to_jsonable(
                labels,
                label_names=label_names,
                target_type=target_type,
                target_names=target_names,
            ),
            indent=2,
            sort_keys=True,
        ).encode(
            "utf-8",
        )
        object_key = self._artifact_object_key(key, "labels.json")
        self._put_bytes(object_key, payload)
        self._delete_object(self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME))
        return self._uri_for(object_key)

    def put_labels_artifact(
        self,
        key: str,
        labels: Any,
        metadata: dict,
        *,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> str:
        """Commit labels and decoding metadata under one manifest."""

        if not isinstance(metadata, dict):
            raise TypeError("Artifact metadata must be a dictionary.")
        validate_artifact_key(key)
        normalized_label_names = None if label_names is None else tuple(label_names)
        normalized_target_names = None if target_names is None else tuple(target_names)
        use_metadata_contract = target_type == "auto"
        effective_target_type = (
            str(metadata.get("target_type", "auto")) if use_metadata_contract else target_type
        )
        effective_label_names = (
            normalized_label_names
            if normalized_label_names is not None
            else (
                metadata.get("label_names")
                if use_metadata_contract and effective_target_type != "regression"
                else None
            )
        )
        effective_target_names = (
            normalized_target_names
            if normalized_target_names is not None
            else (
                metadata.get("target_names")
                if use_metadata_contract and effective_target_type == "regression"
                else None
            )
        )
        labels_payload, labels_contract = LocalArtifactStore._prepare_labels_artifact(
            labels,
            label_names=effective_label_names,
            target_type=effective_target_type,
            target_names=effective_target_names,
        )
        labels_contract = LocalArtifactStore._preserve_v2_label_catalog(
            labels_contract,
            metadata,
        )
        labels_sha256 = hashlib.sha256(labels_payload).hexdigest()
        labels_manifest = JSONArtifactManifest(
            filename=f"labels-v2-{labels_sha256}.json",
            size_bytes=len(labels_payload),
            sha256=labels_sha256,
            role="labels",
        )
        labels_key = self._artifact_object_key(key, labels_manifest.filename)
        committed_metadata = dict(metadata)
        committed_metadata["artifact_path"] = self._uri_for(labels_key)
        committed_metadata.update(labels_contract)
        metadata_payload = LocalArtifactStore._serialize_metadata(committed_metadata)
        metadata_sha256 = hashlib.sha256(metadata_payload).hexdigest()
        metadata_manifest = JSONArtifactManifest(
            filename=f"metadata-v2-{metadata_sha256}.json",
            size_bytes=len(metadata_payload),
            sha256=metadata_sha256,
        )
        manifest = LabelsArtifactManifest(
            labels=labels_manifest,
            metadata=metadata_manifest,
        )
        self._put_bytes(labels_key, labels_payload)
        self._put_bytes(
            self._artifact_object_key(key, metadata_manifest.filename), metadata_payload
        )
        manifest_key = self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)
        self._put_bytes(
            manifest_key,
            json_dumps_strict(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        return self._uri_for(labels_key)

    def get_labels_artifact(self, key: str) -> tuple[np.ndarray, dict]:
        """Load one validated labels/metadata generation with switch retries."""

        last_error: Optional[BaseException] = None
        for _ in range(2):
            manifest = self._read_artifact_manifest(key)
            if not isinstance(manifest, LabelsArtifactManifest):
                raise ValueError(f"Committed artifact for key {key} does not contain labels.")
            try:
                labels_value = self._load_remote_json(key, manifest.labels, require_object=False)
                metadata = decode_label_artifact_metadata(
                    self._load_remote_json(key, manifest.metadata, require_object=True)
                )
                if self._read_artifact_manifest(key) != manifest:
                    continue
                labels = labels_from_jsonable(
                    labels_value,
                    label_names=metadata.get("label_names"),
                    target_type=metadata.get("target_type", "auto"),
                    target_names=metadata.get("target_names"),
                    label_encoding=metadata.get("label_encoding"),
                )
                return np.asarray(labels), metadata
            except FileNotFoundError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Artifact manifest for key {key} changed repeatedly while reading.")

    def get_labels(self, key: str) -> np.ndarray:
        """Load labels from JSON."""

        manifest_key = self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)
        if self._object_exists(manifest_key):
            return self.get_labels_artifact(key)[0]
        try:
            payload = self._get_bytes(self._artifact_object_key(key, "labels.json"))
        except FileNotFoundError:
            if self._object_exists(manifest_key):
                return self.get_labels_artifact(key)[0]
            raise
        label_names = None
        target_type = "auto"
        target_names = None
        metadata_key = self._artifact_object_key(key, "metadata.json")
        if self._object_exists(metadata_key):
            try:
                metadata_payload = self._get_bytes(metadata_key)
            except FileNotFoundError:
                if self._object_exists(manifest_key):
                    return self.get_labels_artifact(key)[0]
                raise
            metadata = json.loads(metadata_payload.decode("utf-8"))
            label_names = metadata.get("label_names")
            target_type = metadata.get("target_type", "auto")
            target_names = metadata.get("target_names")
        if self._object_exists(manifest_key):
            return self.get_labels_artifact(key)[0]
        return labels_from_jsonable(
            json.loads(payload.decode("utf-8")),
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    def put_json(self, key: str, obj: dict) -> str:
        """Store JSON metadata for an artifact key."""

        payload = json_dumps_strict(obj, indent=2, sort_keys=True).encode("utf-8")
        if self._object_exists(self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)):
            raise ValueError(
                "Cannot update low-level JSON metadata for a committed composite artifact; "
                "use put_artifact() or put_labels_artifact() to replace the full generation."
            )
        object_key = self._artifact_object_key(key, "metadata.json")
        self._put_bytes(object_key, payload)
        return self._uri_for(object_key)

    def get_json(self, key: str) -> dict:
        """Load JSON metadata for an artifact key."""

        manifest_key = self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)
        if self._object_exists(manifest_key):
            return self._get_composite_metadata(key)
        try:
            payload = self._get_bytes(self._artifact_object_key(key, "metadata.json"))
        except FileNotFoundError:
            if self._object_exists(manifest_key):
                return self._get_composite_metadata(key)
            raise
        if self._object_exists(manifest_key):
            return self._get_composite_metadata(key)
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"JSON artifact for key {key} must contain a JSON object.")
        return value

    def _get_composite_metadata(self, key: str) -> dict:
        last_error: Optional[BaseException] = None
        for _ in range(2):
            manifest = self._read_artifact_manifest(key)
            try:
                metadata = self._load_remote_json(key, manifest.metadata, require_object=True)
                if self._read_artifact_manifest(key) == manifest:
                    return metadata
            except FileNotFoundError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Artifact manifest for key {key} changed repeatedly while reading.")

    def delete_prefix(self, prefix: str) -> None:
        """Delete every S3 object beneath an artifact key prefix."""

        if prefix == "":
            raise ValueError("Refusing to delete the artifact-store root.")
        clean = validate_artifact_key(prefix)
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

    def _read_artifact_manifest(self, key: str) -> Union[ArtifactManifest, LabelsArtifactManifest]:
        manifest_key = self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)
        if not self._object_exists(manifest_key):
            raise FileNotFoundError(f"No committed artifact manifest found for key {key}.")
        payload = self._get_bytes(manifest_key)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Artifact manifest for key {key} is not valid JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError("Artifact manifest must be a JSON object.")
        kind = value.get("kind")
        if kind == "array+metadata":
            return ArtifactManifest.from_dict(value)
        if kind == "labels+metadata":
            return LabelsArtifactManifest.from_dict(value)
        raise ValueError(f"Artifact manifest for key {key} has unsupported kind {kind!r}.")

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
            json_dumps_strict(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        self._delete_object(self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME))
        return object_key

    def _publish_local_artifact(
        self,
        key: str,
        local_path: Path,
        *,
        array_manifest: ArrayArtifactManifest,
        metadata: dict,
        metadata_finalizer: Optional[Callable[[dict, ArrayArtifactManifest, ArtifactStat], dict]],
    ) -> str:
        array_key = self._artifact_object_key(key, array_manifest.filename)
        committed_metadata = LocalArtifactStore._finalize_array_metadata(
            metadata,
            manifest=array_manifest,
            stat=ArtifactStat(
                uri=self._uri_for(array_key),
                size_bytes=array_manifest.size_bytes,
                storage_format=array_manifest.storage_format,
            ),
            metadata_finalizer=metadata_finalizer,
        )
        metadata_payload = LocalArtifactStore._serialize_metadata(committed_metadata)
        metadata_sha256 = hashlib.sha256(metadata_payload).hexdigest()
        metadata_manifest = JSONArtifactManifest(
            filename=f"metadata-v2-{metadata_sha256}.json",
            size_bytes=len(metadata_payload),
            sha256=metadata_sha256,
        )
        manifest = ArtifactManifest(array=array_manifest, metadata=metadata_manifest)
        self._upload_file(local_path, array_key)
        self._put_bytes(
            self._artifact_object_key(key, metadata_manifest.filename), metadata_payload
        )
        manifest_key = self._artifact_object_key(key, ARTIFACT_MANIFEST_FILENAME)
        self._put_bytes(
            manifest_key,
            json_dumps_strict(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        # Superseded immutable digests are retained: safe reclamation requires
        # provider generation/version preconditions, not a racy check-then-delete.
        return array_key

    def _load_remote_array(self, key: str, manifest: ArrayArtifactManifest) -> Any:
        object_key = self._artifact_object_key(key, manifest.filename)
        if self._object_size(object_key) is None:
            raise FileNotFoundError(
                f"Array manifest for key {key} references missing file {manifest.filename}."
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / manifest.filename
            try:
                self._download_file(object_key, local_path)
            except (FileNotFoundError, KeyError) as exc:
                raise FileNotFoundError(
                    f"Array manifest for key {key} references missing file " f"{manifest.filename}."
                ) from exc
            local = LocalArtifactStore(tmpdir)
            return local._load_array(local_path, manifest)

    def _load_remote_json(
        self,
        key: str,
        manifest: JSONArtifactManifest,
        *,
        require_object: bool,
    ) -> Any:
        object_key = self._artifact_object_key(key, manifest.filename)
        try:
            payload = self._get_bytes(object_key)
        except (FileNotFoundError, KeyError) as exc:
            raise FileNotFoundError(
                f"Artifact manifest for key {key} references missing file {manifest.filename}."
            ) from exc
        if len(payload) != manifest.size_bytes:
            raise ValueError(
                f"JSON artifact {self._uri_for(object_key)} has size {len(payload)}, expected "
                f"{manifest.size_bytes} from its manifest."
            )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest.sha256:
            raise ValueError(
                f"JSON artifact {self._uri_for(object_key)} failed SHA-256 verification; "
                f"expected {manifest.sha256}, got {digest}."
            )
        try:
            value = json.loads(payload.decode(manifest.encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"JSON artifact {self._uri_for(object_key)} is not valid UTF-8 JSON."
            ) from exc
        if require_object and not isinstance(value, dict):
            raise ValueError(
                f"JSON artifact {self._uri_for(object_key)} must contain a JSON object."
            )
        return value

    def _artifact_object_key(self, key: str, filename: str) -> str:
        validated_key = validate_artifact_key(key)
        return "/".join(part for part in (self.prefix, validated_key, filename) if part)

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
        try:
            body = self._client_or_raise().get_object(Bucket=self.bucket, Key=object_key)["Body"]
            return body.read()
        except Exception as exc:
            response = getattr(exc, "response", {})
            code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
            if code in {"404", "NoSuchKey", "NotFound"} or isinstance(exc, KeyError):
                raise FileNotFoundError(
                    f"S3 artifact object {self._uri_for(object_key)} does not exist."
                ) from exc
            raise

    def _upload_file(self, local_path: Path, object_key: str) -> None:
        self._client_or_raise().upload_file(str(local_path), self.bucket, object_key)

    def _download_file(self, object_key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client_or_raise().download_file(self.bucket, object_key, str(local_path))
        except Exception as exc:
            response = getattr(exc, "response", {})
            code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
            if code in {"404", "NoSuchKey", "NotFound"} or isinstance(exc, KeyError):
                raise FileNotFoundError(
                    f"S3 artifact object {self._uri_for(object_key)} does not exist."
                ) from exc
            raise

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

    def _object_size(self, object_key: str) -> Optional[int]:
        try:
            response = self._client_or_raise().head_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:
            response = getattr(exc, "response", {})
            code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return int(response["ContentLength"])

    def _delete_object(self, object_key: str) -> None:
        self._client_or_raise().delete_object(Bucket=self.bucket, Key=object_key)
