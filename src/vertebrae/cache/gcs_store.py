"""GCS-backed artifact store."""

import hashlib
import json
import os
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

        artifact_manifest_name = self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)
        if self._blob_exists(artifact_manifest_name):
            artifact_manifest = self._read_artifact_manifest(key)
            primary = (
                artifact_manifest.array
                if isinstance(artifact_manifest, ArtifactManifest)
                else artifact_manifest.labels
            )
            primary_size = self._blob_size(self._artifact_blob_name(key, primary.filename))
            metadata_size = self._blob_size(
                self._artifact_blob_name(key, artifact_manifest.metadata.filename)
            )
            return primary_size == primary.size_bytes and (
                metadata_size == artifact_manifest.metadata.size_bytes
            )
        array_manifest_name = self._artifact_blob_name(key, ARRAY_MANIFEST_FILENAME)
        if not self._blob_exists(array_manifest_name):
            return False
        array_manifest = self._read_array_manifest(key)
        return (
            self._blob_size(self._artifact_blob_name(key, array_manifest.filename))
            == array_manifest.size_bytes
        )

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
            array_name = self._publish_local_artifact(
                key,
                local_path,
                array_manifest=array_manifest,
                metadata=metadata,
                metadata_finalizer=metadata_finalizer,
            )
        return self._uri_for(array_name)

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
            array_name = self._publish_local_artifact(
                key,
                local_path,
                array_manifest=array_manifest,
                metadata=metadata,
                metadata_finalizer=metadata_finalizer,
            )
        return self._uri_for(array_name)

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

        if self._blob_exists(self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)):
            return self.get_artifact(key)[0]
        last_error: Optional[BaseException] = None
        for _ in range(2):
            try:
                manifest = self._read_array_manifest(key)
                value = self._load_remote_array(key, manifest)
                if self._blob_exists(self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)):
                    return self.get_artifact(key)[0]
                if self._read_array_manifest(key) != manifest:
                    continue
                return value
            except FileNotFoundError as exc:
                last_error = exc
                if self._blob_exists(self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)):
                    return self.get_artifact(key)[0]
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Array manifest for key {key} changed repeatedly while reading.")

    def stat_array(self, key: str) -> ArtifactStat:
        """Return blob size using GCS metadata without downloading it."""

        if self._blob_exists(self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)):
            artifact_manifest = self._read_artifact_manifest(key)
            if not isinstance(artifact_manifest, ArtifactManifest):
                raise ValueError(f"Committed artifact for key {key} does not contain an array.")
            manifest = artifact_manifest.array
        else:
            manifest = self._read_array_manifest(key)
        blob_name = self._artifact_blob_name(key, manifest.filename)
        blob = self._bucket_or_raise().blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(
                f"Array manifest for key {key} references missing file {manifest.filename}."
            )
        blob.reload()
        size_bytes = int(blob.size or 0)
        if size_bytes != manifest.size_bytes:
            raise ValueError(
                f"Array artifact {self._uri_for(blob_name)} has size {size_bytes}, expected "
                f"{manifest.size_bytes} from its manifest."
            )
        return ArtifactStat(
            uri=self._uri_for(blob_name),
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
        blob_name = self._artifact_blob_name(key, "labels.json")
        self._put_bytes(blob_name, payload)
        self._delete_blob(self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME))
        return self._uri_for(blob_name)

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
        labels_name = self._artifact_blob_name(key, labels_manifest.filename)
        committed_metadata = dict(metadata)
        committed_metadata["artifact_path"] = self._uri_for(labels_name)
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
        self._put_bytes(labels_name, labels_payload)
        self._put_bytes(self._artifact_blob_name(key, metadata_manifest.filename), metadata_payload)
        manifest_name = self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)
        self._put_bytes(
            manifest_name,
            json_dumps_strict(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        return self._uri_for(labels_name)

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

        manifest_name = self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)
        if self._blob_exists(manifest_name):
            return self.get_labels_artifact(key)[0]
        try:
            payload = self._get_bytes(self._artifact_blob_name(key, "labels.json"))
        except FileNotFoundError:
            if self._blob_exists(manifest_name):
                return self.get_labels_artifact(key)[0]
            raise
        label_names = None
        target_type = "auto"
        target_names = None
        metadata_name = self._artifact_blob_name(key, "metadata.json")
        if self._blob_exists(metadata_name):
            try:
                metadata_payload = self._get_bytes(metadata_name)
            except FileNotFoundError:
                if self._blob_exists(manifest_name):
                    return self.get_labels_artifact(key)[0]
                raise
            metadata = json.loads(metadata_payload.decode("utf-8"))
            label_names = metadata.get("label_names")
            target_type = metadata.get("target_type", "auto")
            target_names = metadata.get("target_names")
        if self._blob_exists(manifest_name):
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
        if self._blob_exists(self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)):
            raise ValueError(
                "Cannot update low-level JSON metadata for a committed composite artifact; "
                "use put_artifact() or put_labels_artifact() to replace the full generation."
            )
        blob_name = self._artifact_blob_name(key, "metadata.json")
        self._put_bytes(blob_name, payload)
        return self._uri_for(blob_name)

    def get_json(self, key: str) -> dict:
        """Load JSON metadata for an artifact key."""

        manifest_name = self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)
        if self._blob_exists(manifest_name):
            return self._get_composite_metadata(key)
        try:
            payload = self._get_bytes(self._artifact_blob_name(key, "metadata.json"))
        except FileNotFoundError:
            if self._blob_exists(manifest_name):
                return self._get_composite_metadata(key)
            raise
        if self._blob_exists(manifest_name):
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

    def _read_artifact_manifest(self, key: str) -> Union[ArtifactManifest, LabelsArtifactManifest]:
        manifest_name = self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)
        if not self._blob_exists(manifest_name):
            raise FileNotFoundError(f"No committed artifact manifest found for key {key}.")
        payload = self._get_bytes(manifest_name)
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
        blob_name = self._artifact_blob_name(key, manifest.filename)
        self._upload_file(local_path, blob_name)
        manifest_name = self._artifact_blob_name(key, ARRAY_MANIFEST_FILENAME)
        self._put_bytes(
            manifest_name,
            json_dumps_strict(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        self._delete_blob(self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME))
        return blob_name

    def _publish_local_artifact(
        self,
        key: str,
        local_path: Path,
        *,
        array_manifest: ArrayArtifactManifest,
        metadata: dict,
        metadata_finalizer: Optional[Callable[[dict, ArrayArtifactManifest, ArtifactStat], dict]],
    ) -> str:
        array_name = self._artifact_blob_name(key, array_manifest.filename)
        committed_metadata = LocalArtifactStore._finalize_array_metadata(
            metadata,
            manifest=array_manifest,
            stat=ArtifactStat(
                uri=self._uri_for(array_name),
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
        self._upload_file(local_path, array_name)
        self._put_bytes(self._artifact_blob_name(key, metadata_manifest.filename), metadata_payload)
        manifest_name = self._artifact_blob_name(key, ARTIFACT_MANIFEST_FILENAME)
        self._put_bytes(
            manifest_name,
            json_dumps_strict(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        # Superseded immutable digests are retained: safe reclamation requires
        # provider generation/version preconditions, not a racy check-then-delete.
        return array_name

    def _load_remote_array(self, key: str, manifest: ArrayArtifactManifest) -> Any:
        blob_name = self._artifact_blob_name(key, manifest.filename)
        if self._blob_size(blob_name) is None:
            raise FileNotFoundError(
                f"Array manifest for key {key} references missing file {manifest.filename}."
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / manifest.filename
            try:
                self._download_file(blob_name, local_path)
            except (FileNotFoundError, KeyError) as exc:
                raise FileNotFoundError(
                    f"Array manifest for key {key} references missing file " f"{manifest.filename}."
                ) from exc
            return LocalArtifactStore(tmpdir)._load_array(local_path, manifest)

    def _load_remote_json(
        self,
        key: str,
        manifest: JSONArtifactManifest,
        *,
        require_object: bool,
    ) -> Any:
        blob_name = self._artifact_blob_name(key, manifest.filename)
        try:
            payload = self._get_bytes(blob_name)
        except (FileNotFoundError, KeyError) as exc:
            raise FileNotFoundError(
                f"Artifact manifest for key {key} references missing file {manifest.filename}."
            ) from exc
        if len(payload) != manifest.size_bytes:
            raise ValueError(
                f"JSON artifact {self._uri_for(blob_name)} has size {len(payload)}, expected "
                f"{manifest.size_bytes} from its manifest."
            )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest.sha256:
            raise ValueError(
                f"JSON artifact {self._uri_for(blob_name)} failed SHA-256 verification; "
                f"expected {manifest.sha256}, got {digest}."
            )
        try:
            value = json.loads(payload.decode(manifest.encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"JSON artifact {self._uri_for(blob_name)} is not valid UTF-8 JSON."
            ) from exc
        if require_object and not isinstance(value, dict):
            raise ValueError(
                f"JSON artifact {self._uri_for(blob_name)} must contain a JSON object."
            )
        return value

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
        blob = self._bucket_or_raise().blob(blob_name)
        try:
            return blob.download_as_bytes()
        except (FileNotFoundError, KeyError) as exc:
            raise FileNotFoundError(
                f"GCS artifact blob {self._uri_for(blob_name)} does not exist."
            ) from exc

    def _upload_file(self, local_path: Path, blob_name: str) -> None:
        self._bucket_or_raise().blob(blob_name).upload_from_filename(str(local_path))

    def _download_file(self, blob_name: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._bucket_or_raise().blob(blob_name).download_to_filename(str(local_path))
        except Exception as exc:
            if isinstance(exc, (FileNotFoundError, KeyError)) or type(exc).__name__ in {
                "NotFound",
            }:
                raise FileNotFoundError(
                    f"GCS artifact blob {self._uri_for(blob_name)} does not exist."
                ) from exc
            raise

    def _blob_exists(self, blob_name: str) -> bool:
        return bool(self._bucket_or_raise().blob(blob_name).exists())

    def _blob_size(self, blob_name: str) -> Optional[int]:
        blob = self._bucket_or_raise().blob(blob_name)
        if not blob.exists():
            return None
        blob.reload()
        return int(blob.size or 0)

    def _delete_blob(self, blob_name: str) -> None:
        blob = self._bucket_or_raise().blob(blob_name)
        if blob.exists():
            blob.delete()
