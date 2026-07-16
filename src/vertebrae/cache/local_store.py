"""Local filesystem artifact store."""

import hashlib
import itertools
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union

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
from vertebrae.utils.labels import (
    decode_label_artifact_metadata,
    labels_from_jsonable,
    labels_to_artifact_jsonable,
    labels_to_jsonable,
    normalize_targets,
)
from vertebrae.utils.semantic_labels import (
    LABEL_ENCODING,
    semantic_label_catalog,
    semantic_label_keys,
    validate_label_catalog,
)
from vertebrae.utils.serialization import json_dumps_strict
from vertebrae.utils.validation import is_sparse_matrix


class LocalArtifactStore:
    """Store dense/sparse arrays and JSON metadata under a local directory.

    Args:
        root: Root cache directory.
    """

    def __init__(self, root: str = ".vertebrae_cache") -> None:
        self.root = Path(root)

    def config(self) -> ArtifactStoreConfig:
        """Return a serializable config for reconstructing this store."""

        return ArtifactStoreConfig(uri=str(self.root))

    def exists(self, key: str) -> bool:
        """Return whether an embedding artifact exists for `key`.

        Args:
            key: Artifact key.

        Returns:
            Whether a dense `.npy` or sparse `.npz` embedding file exists.
        """

        path = self._path(key)
        if not path.exists():
            return False
        try:
            with self._artifact_lock(path, shared=True):
                artifact_manifest_path = path / ARTIFACT_MANIFEST_FILENAME
                if artifact_manifest_path.exists():
                    artifact = self._read_artifact_manifest(path)
                    primary = (
                        artifact.array
                        if isinstance(artifact, ArtifactManifest)
                        else artifact.labels
                    )
                    primary_target = path / primary.filename
                    metadata_target = path / artifact.metadata.filename
                    if not primary_target.exists() or not metadata_target.exists():
                        return False
                    return (
                        int(primary_target.stat().st_size) == primary.size_bytes
                        and int(metadata_target.stat().st_size) == artifact.metadata.size_bytes
                    )
                manifest_path = path / ARRAY_MANIFEST_FILENAME
                if not manifest_path.exists():
                    return False
                legacy_manifest, target = self._resolve_manifest_target(path)
                return target is not None and (
                    int(target.stat().st_size) == legacy_manifest.size_bytes
                )
        except FileNotFoundError:
            return False

    def put_array(self, key: str, arr: Any) -> str:
        """Store a dense or sparse embedding matrix.

        Args:
            key: Artifact key.
            arr: Dense array-like object or scipy sparse matrix.

        Returns:
            Filesystem path to the saved artifact.
        """

        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        sparse_value = is_sparse_matrix(arr)
        filename = "embeddings.npz" if sparse_value else "embeddings.npy"
        with tempfile.TemporaryDirectory(dir=path) as tmpdir:
            prepared = Path(tmpdir) / filename
            if sparse_value:
                from scipy import sparse

                sparse.save_npz(prepared, arr)
                shape = tuple(int(size) for size in arr.shape)
                dtype = str(arr.dtype)
                sparse_format = str(arr.getformat())
                nnz = int(arr.nnz)
            else:
                array = np.asarray(arr)
                with prepared.open("wb") as file:
                    np.save(file, array)
                    file.flush()
                    os.fsync(file.fileno())
                shape = tuple(int(size) for size in array.shape)
                dtype = str(array.dtype)
                sparse_format = None
                nnz = None
            return self._publish_prepared_array(
                path,
                prepared,
                shape=shape,
                dtype=dtype,
                sparse_format=sparse_format,
                nnz=nnz,
            )

    def put_array_batches(
        self,
        key: str,
        batches: Iterable[Tuple[np.ndarray, Any]],
        n_samples: int,
        require_complete: bool = True,
    ) -> str:
        """Store embeddings from deterministic batches.

        Args:
            key: Artifact key.
            batches: Iterable of `(indices, embeddings)` batch pairs.
            n_samples: Total number of rows in the full embedding artifact.
            require_complete: Whether every row must be written exactly once.

        Returns:
            Filesystem path to the saved artifact.

        Raises:
            ValueError: If batches contain duplicate indices, invalid shapes, or
                incomplete coverage when `require_complete` is true.
        """

        if (
            isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (int, np.integer))
            or int(n_samples) < 1
        ):
            raise ValueError("n_samples must be an integer >= 1.")
        n_samples = int(n_samples)
        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        iterator = iter(batches)
        try:
            first_indices, first_batch = next(iterator)
        except StopIteration as exc:
            raise ValueError("At least one embedding batch is required.") from exc

        with tempfile.TemporaryDirectory(dir=path) as tmpdir:
            staging = Path(tmpdir)
            if is_sparse_matrix(first_batch):
                prepared_path, nnz = self._put_sparse_batches(
                    staging,
                    itertools.chain([(first_indices, first_batch)], iterator),
                    n_samples=n_samples,
                    require_complete=require_complete,
                )
                prepared = Path(prepared_path)
                shape = (int(n_samples), int(first_batch.shape[1]))
                dtype = str(first_batch.dtype)
                sparse_format = "csr"
            else:
                prepared = Path(
                    self._put_dense_batches(
                        staging,
                        first_indices=first_indices,
                        first_batch=first_batch,
                        remaining=iterator,
                        n_samples=n_samples,
                        require_complete=require_complete,
                    )
                )
                first = np.asarray(first_batch)
                shape = (int(n_samples), int(first.shape[1]))
                dtype = str(first.dtype)
                sparse_format = None
                nnz = None
            return self._publish_prepared_array(
                path,
                prepared,
                shape=shape,
                dtype=dtype,
                sparse_format=sparse_format,
                nnz=nnz,
            )

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse embedding matrix.

        Args:
            key: Artifact key.

        Returns:
            Dense NumPy array or scipy sparse matrix.
        """

        path = self._path(key)
        with self._artifact_lock(path, shared=True):
            if (path / ARTIFACT_MANIFEST_FILENAME).exists():
                artifact_manifest = self._read_artifact_manifest(path)
                if not isinstance(artifact_manifest, ArtifactManifest):
                    raise ValueError(f"Committed artifact for key {key} does not contain an array.")
                manifest = artifact_manifest.array
                target = path / manifest.filename
            else:
                legacy_manifest, legacy_target = self._resolve_manifest_target(path)
                if legacy_target is None:
                    raise FileNotFoundError(
                        f"Array manifest for key {key} references missing file "
                        f"{legacy_manifest.filename}."
                    )
                manifest = legacy_manifest
                target = legacy_target
            if not target.exists():
                raise FileNotFoundError(
                    f"Array manifest for key {key} references missing file {manifest.filename}."
                )
            return self._load_array(target, manifest)

    def stat_array(self, key: str) -> ArtifactStat:
        """Return the persisted array file size without loading it."""

        path = self._path(key)
        with self._artifact_lock(path, shared=True):
            if (path / ARTIFACT_MANIFEST_FILENAME).exists():
                artifact_manifest = self._read_artifact_manifest(path)
                if not isinstance(artifact_manifest, ArtifactManifest):
                    raise ValueError(f"Committed artifact for key {key} does not contain an array.")
                manifest = artifact_manifest.array
                target = path / manifest.filename
            else:
                legacy_manifest, legacy_target = self._resolve_manifest_target(path)
                if legacy_target is None:
                    raise FileNotFoundError(
                        f"Array manifest for key {key} references missing file "
                        f"{legacy_manifest.filename}."
                    )
                manifest = legacy_manifest
                target = legacy_target
            if not target.exists():
                raise FileNotFoundError(
                    f"Array manifest for key {key} references missing file {manifest.filename}."
                )
            actual_size = int(target.stat().st_size)
            if actual_size != manifest.size_bytes:
                raise ValueError(
                    f"Array artifact {target} has size {actual_size}, expected "
                    f"{manifest.size_bytes} from its manifest."
                )
            return ArtifactStat(
                uri=str(target),
                size_bytes=actual_size,
                storage_format=manifest.storage_format,
            )

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
        """Commit an array and JSON metadata with one last-written manifest."""

        if not isinstance(metadata, dict):
            raise TypeError("Artifact metadata must be a dictionary.")
        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path) as tmpdir:
            staged = LocalArtifactStore(tmpdir)
            prepared = Path(staged.put_array("staged", arr))
            array_manifest = staged._read_array_manifest(staged._path("staged"))
            return self._publish_prepared_artifact(
                path,
                prepared,
                array_manifest=array_manifest,
                metadata=metadata,
                metadata_finalizer=metadata_finalizer,
            )

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
        """Commit batched arrays and JSON metadata with one final manifest switch."""

        self._validate_metadata(metadata)
        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path) as tmpdir:
            staged = LocalArtifactStore(tmpdir)
            prepared = Path(
                staged.put_array_batches(
                    "staged",
                    batches,
                    n_samples=n_samples,
                    require_complete=require_complete,
                )
            )
            array_manifest = staged._read_array_manifest(staged._path("staged"))
            return self._publish_prepared_artifact(
                path,
                prepared,
                array_manifest=array_manifest,
                metadata=metadata,
                metadata_finalizer=metadata_finalizer,
            )

    def get_artifact(self, key: str) -> tuple[Any, dict]:
        """Load an array and metadata guarded by the same shared read lock."""

        path = self._path(key)
        with self._artifact_lock(path, shared=True):
            manifest = self._read_artifact_manifest(path)
            if not isinstance(manifest, ArtifactManifest):
                raise ValueError(f"Committed artifact for key {key} does not contain an array.")
            array_target = path / manifest.array.filename
            metadata_target = path / manifest.metadata.filename
            if not array_target.exists():
                raise FileNotFoundError(
                    f"Artifact manifest for key {key} references missing array file "
                    f"{manifest.array.filename}."
                )
            if not metadata_target.exists():
                raise FileNotFoundError(
                    f"Artifact manifest for key {key} references missing metadata file "
                    f"{manifest.metadata.filename}."
                )
            value = self._load_array(array_target, manifest.array)
            metadata = self._load_json(metadata_target, manifest.metadata)
            return value, metadata

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
        """Commit labels and their decoding metadata as one generation."""

        if not isinstance(metadata, dict):
            raise TypeError("Artifact metadata must be a dictionary.")
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
        labels_payload, labels_contract = self._prepare_labels_artifact(
            labels,
            label_names=effective_label_names,
            target_type=effective_target_type,
            target_names=effective_target_names,
        )
        labels_contract = self._preserve_v2_label_catalog(
            labels_contract,
            metadata,
        )
        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        return self._publish_labels_artifact(
            path,
            labels_payload=labels_payload,
            metadata=metadata,
            labels_contract=labels_contract,
        )

    def get_labels_artifact(self, key: str) -> tuple[np.ndarray, dict]:
        """Load labels and decoding metadata under one shared generation lock."""

        path = self._path(key)
        with self._artifact_lock(path, shared=True):
            manifest = self._read_artifact_manifest(path)
            if not isinstance(manifest, LabelsArtifactManifest):
                raise ValueError(f"Committed artifact for key {key} does not contain labels.")
            labels_target = path / manifest.labels.filename
            metadata_target = path / manifest.metadata.filename
            if not labels_target.exists() or not metadata_target.exists():
                missing = (
                    manifest.labels.filename
                    if not labels_target.exists()
                    else manifest.metadata.filename
                )
                raise FileNotFoundError(
                    f"Artifact manifest for key {key} references missing file {missing}."
                )
            labels_value = self._load_json_value(labels_target, manifest.labels)
            metadata = decode_label_artifact_metadata(
                self._load_json(metadata_target, manifest.metadata)
            )
            labels = labels_from_jsonable(
                labels_value,
                label_names=metadata.get("label_names"),
                target_type=metadata.get("target_type", "auto"),
                target_names=metadata.get("target_names"),
                label_encoding=metadata.get("label_encoding"),
            )
            return np.asarray(labels), metadata

    def put_labels(
        self,
        key: str,
        labels: Any,
        *,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> str:
        """Store labels as a JSON artifact.

        Args:
            key: Artifact key.
            labels: One-dimensional labels.

        Returns:
            Filesystem path to the saved labels file.
        """

        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "labels.json"
        payload = json_dumps_strict(
            labels_to_jsonable(
                labels,
                label_names=label_names,
                target_type=target_type,
                target_names=target_names,
            ),
            indent=2,
            sort_keys=True,
        )
        with self._artifact_lock(path, shared=False):
            self._write_text_atomic(target, payload)
            (path / ARTIFACT_MANIFEST_FILENAME).unlink(missing_ok=True)
            self._remove_unreferenced_content_files(path)
        return str(target)

    def get_labels(self, key: str) -> np.ndarray:
        """Load labels from a JSON artifact.

        Args:
            key: Artifact key.

        Returns:
            One-dimensional label array.
        """

        path = self._path(key)
        with self._artifact_lock(path, shared=True):
            if (path / ARTIFACT_MANIFEST_FILENAME).exists():
                manifest = self._read_artifact_manifest(path)
                if not isinstance(manifest, LabelsArtifactManifest):
                    raise ValueError(f"Committed artifact for key {key} does not contain labels.")
                labels_target = path / manifest.labels.filename
                metadata_target = path / manifest.metadata.filename
                if not labels_target.exists() or not metadata_target.exists():
                    missing = (
                        manifest.labels.filename
                        if not labels_target.exists()
                        else manifest.metadata.filename
                    )
                    raise FileNotFoundError(
                        f"Artifact manifest for key {key} references missing file {missing}."
                    )
                labels_value = self._load_json_value(labels_target, manifest.labels)
                metadata = decode_label_artifact_metadata(
                    self._load_json(metadata_target, manifest.metadata)
                )
                return labels_from_jsonable(
                    labels_value,
                    label_names=metadata.get("label_names"),
                    target_type=metadata.get("target_type", "auto"),
                    target_names=metadata.get("target_names"),
                    label_encoding=metadata.get("label_encoding"),
                )
            label_names = None
            target_type = "auto"
            target_names = None
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    label_names = metadata.get("label_names")
                    target_type = metadata.get("target_type", "auto")
                    target_names = metadata.get("target_names")
            with (path / "labels.json").open("r", encoding="utf-8") as f:
                return labels_from_jsonable(
                    json.load(f),
                    label_names=label_names,
                    target_type=target_type,
                    target_names=target_names,
                )

    def put_json(self, key: str, obj: dict) -> str:
        """Store JSON metadata for an artifact key.

        Args:
            key: Artifact key.
            obj: JSON-serializable metadata.

        Returns:
            Filesystem path to the saved metadata file.
        """

        payload = json_dumps_strict(obj, indent=2, sort_keys=True)
        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "metadata.json"
        with self._artifact_lock(path, shared=False):
            if (path / ARTIFACT_MANIFEST_FILENAME).exists():
                raise ValueError(
                    "Cannot update low-level JSON metadata for a committed composite artifact; "
                    "use put_artifact() or put_labels_artifact() to replace the full generation."
                )
            self._write_text_atomic(target, payload)
        return str(target)

    def get_json(self, key: str) -> dict:
        """Load JSON metadata for an artifact key.

        Args:
            key: Artifact key.

        Returns:
            Metadata dictionary.
        """

        path = self._path(key)
        with self._artifact_lock(path, shared=True):
            if (path / ARTIFACT_MANIFEST_FILENAME).exists():
                manifest = self._read_artifact_manifest(path).metadata
                target = path / manifest.filename
                if not target.exists():
                    raise FileNotFoundError(
                        f"Artifact manifest for key {key} references missing metadata file "
                        f"{manifest.filename}."
                    )
                return self._load_json(target, manifest)
            with (path / "metadata.json").open("r", encoding="utf-8") as f:
                return json.load(f)

    def delete_prefix(self, prefix: str) -> None:
        """Delete every local artifact beneath a key prefix."""

        if prefix == "":
            raise ValueError("Refusing to delete the artifact-store root.")
        path = self._path(prefix)
        if path.exists():
            shutil.rmtree(path)

    def _path(self, key: str) -> Path:
        validated_key = validate_artifact_key(key)
        path = self.root.joinpath(*validated_key.split("/"))
        resolved_root = self.root.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        try:
            relative = resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("Artifact key resolves outside the artifact-store root.") from exc
        if not relative.parts:
            raise ValueError("Artifact key resolves to the artifact-store root.")
        return path

    def _read_array_manifest(self, path: Path) -> ArrayArtifactManifest:
        manifest_path = path / ARRAY_MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No committed array manifest found under {path}.")
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Array manifest under {path} is not valid JSON.") from exc
        return ArrayArtifactManifest.from_dict(payload)

    def _read_artifact_manifest(
        self, path: Path
    ) -> Union[ArtifactManifest, LabelsArtifactManifest]:
        manifest_path = path / ARTIFACT_MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No committed artifact manifest found under {path}.")
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Artifact manifest under {path} is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Artifact manifest must be a JSON object.")
        kind = payload.get("kind")
        if kind == "array+metadata":
            return ArtifactManifest.from_dict(payload)
        if kind == "labels+metadata":
            return LabelsArtifactManifest.from_dict(payload)
        raise ValueError(f"Artifact manifest under {path} has unsupported kind {kind!r}.")

    def _publish_prepared_array(
        self,
        path: Path,
        prepared: Path,
        *,
        shape: tuple[int, ...],
        dtype: str,
        sparse_format: Any,
        nnz: Optional[int],
    ) -> str:
        path.mkdir(parents=True, exist_ok=True)
        storage_format = "npz" if prepared.suffix == ".npz" else "npy"
        with prepared.open("rb") as file:
            os.fsync(file.fileno())
        checksum = self._sha256_file(prepared)
        target = path / f"embeddings-v2-{checksum}.{storage_format}"
        with self._artifact_lock(path, shared=False):
            os.replace(prepared, target)
            self._fsync_directory(path)
            manifest = ArrayArtifactManifest(
                filename=target.name,
                storage_format=storage_format,
                shape=shape,
                dtype=dtype,
                size_bytes=int(target.stat().st_size),
                sha256=checksum,
                sparse_format=sparse_format,
                nnz=nnz,
            )
            self._write_array_manifest(path, manifest)
            (path / ARTIFACT_MANIFEST_FILENAME).unlink(missing_ok=True)
            self._fsync_directory(path)
            self._remove_unreferenced_content_files(path)
        return str(target)

    def _publish_prepared_artifact(
        self,
        path: Path,
        prepared: Path,
        *,
        array_manifest: ArrayArtifactManifest,
        metadata: dict,
        metadata_finalizer: Optional[Callable[[dict, ArrayArtifactManifest, ArtifactStat], dict]],
    ) -> str:
        """Publish immutable components before switching the composite manifest."""

        array_target = path / array_manifest.filename
        metadata_payload = self._serialize_metadata(
            self._finalize_array_metadata(
                metadata,
                manifest=array_manifest,
                stat=ArtifactStat(
                    uri=str(array_target),
                    size_bytes=array_manifest.size_bytes,
                    storage_format=array_manifest.storage_format,
                ),
                metadata_finalizer=metadata_finalizer,
            )
        )
        metadata_sha256 = hashlib.sha256(metadata_payload).hexdigest()
        metadata_manifest = JSONArtifactManifest(
            filename=f"metadata-v2-{metadata_sha256}.json",
            size_bytes=len(metadata_payload),
            sha256=metadata_sha256,
        )
        manifest = ArtifactManifest(array=array_manifest, metadata=metadata_manifest)
        metadata_target = path / metadata_manifest.filename
        with self._artifact_lock(path, shared=False):
            os.replace(prepared, array_target)
            self._fsync_file(array_target)
            self._fsync_directory(path)
            self._write_bytes_atomic(metadata_target, metadata_payload)
            self._write_artifact_manifest(path, manifest)
            (path / ARRAY_MANIFEST_FILENAME).unlink(missing_ok=True)
            (path / "labels.json").unlink(missing_ok=True)
            (path / "metadata.json").unlink(missing_ok=True)
            self._fsync_directory(path)
            self._remove_unreferenced_content_files(path)
        return str(array_target)

    def _publish_labels_artifact(
        self,
        path: Path,
        *,
        labels_payload: bytes,
        metadata: dict,
        labels_contract: dict,
    ) -> str:
        labels_sha256 = hashlib.sha256(labels_payload).hexdigest()
        labels_manifest = JSONArtifactManifest(
            filename=f"labels-v2-{labels_sha256}.json",
            size_bytes=len(labels_payload),
            sha256=labels_sha256,
            role="labels",
        )
        committed_metadata = dict(metadata)
        committed_metadata["artifact_path"] = str(path / labels_manifest.filename)
        committed_metadata.update(labels_contract)
        metadata_payload = self._serialize_metadata(committed_metadata)
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
        with self._artifact_lock(path, shared=False):
            self._write_bytes_atomic(path / labels_manifest.filename, labels_payload)
            self._write_bytes_atomic(path / metadata_manifest.filename, metadata_payload)
            self._write_artifact_manifest(path, manifest)
            (path / ARRAY_MANIFEST_FILENAME).unlink(missing_ok=True)
            (path / "labels.json").unlink(missing_ok=True)
            (path / "metadata.json").unlink(missing_ok=True)
            self._fsync_directory(path)
            self._remove_unreferenced_content_files(path)
        return str(path / labels_manifest.filename)

    def _resolve_manifest_target(
        self,
        path: Path,
    ) -> tuple[ArrayArtifactManifest, Optional[Path]]:
        """Resolve a manifest target, retrying one concurrent manifest switch."""

        manifest = self._read_array_manifest(path)
        target = path / manifest.filename
        if target.exists():
            return manifest, target
        replacement = self._read_array_manifest(path)
        replacement_target = path / replacement.filename
        if replacement_target.exists():
            return replacement, replacement_target
        return replacement, None

    def _remove_unreferenced_content_files(self, path: Path) -> None:
        """Remove unreferenced digests while readers are excluded by the writer lock."""

        referenced: set[str] = set()
        if (path / ARRAY_MANIFEST_FILENAME).exists():
            referenced.add(self._read_array_manifest(path).filename)
        if (path / ARTIFACT_MANIFEST_FILENAME).exists():
            artifact = self._read_artifact_manifest(path)
            if isinstance(artifact, ArtifactManifest):
                referenced.add(artifact.array.filename)
            else:
                referenced.add(artifact.labels.filename)
            referenced.add(artifact.metadata.filename)
        for candidate in path.glob("embeddings-v2-*"):
            if candidate.name not in referenced and candidate.suffix in {".npy", ".npz"}:
                candidate.unlink(missing_ok=True)
        for candidate in path.glob("metadata-v2-*.json"):
            if candidate.name not in referenced:
                candidate.unlink(missing_ok=True)
        for candidate in path.glob("labels-v2-*.json"):
            if candidate.name not in referenced:
                candidate.unlink(missing_ok=True)
        self._fsync_directory(path)

    @staticmethod
    @contextmanager
    def _artifact_lock(path: Path, *, shared: bool):
        """Coordinate shared readers with exclusive commit switches and cleanup."""

        if shared and not path.exists():
            raise FileNotFoundError(f"No artifact directory found under {path}.")
        if not shared:
            path.mkdir(parents=True, exist_ok=True)
        lock_path = path / ".artifact-publish.lock"
        try:
            lock_file = lock_path.open("rb" if shared else "a+b")
        except FileNotFoundError:
            # Directories created by pre-lock store versions are immutable until a
            # current writer installs the permanent lock file. Reading them without
            # creating filesystem state preserves read-only cache support.
            if shared:
                yield
                return
            raise
        with lock_file:
            try:
                import fcntl

                mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(lock_file.fileno(), mode)
            except ImportError:
                pass
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    pass

    @staticmethod
    def _serialize_metadata(metadata: dict) -> bytes:
        if not isinstance(metadata, dict):
            raise TypeError("Artifact metadata must be a dictionary.")
        return json_dumps_strict(metadata, indent=2, sort_keys=True).encode("utf-8")

    @staticmethod
    def _validate_metadata(metadata: dict) -> None:
        if not isinstance(metadata, dict):
            raise TypeError("Artifact metadata must be a dictionary.")
        candidate = dict(metadata)
        candidate["artifact_path"] = ""
        json_dumps_strict(candidate, indent=2, sort_keys=True)

    @staticmethod
    def _prepare_labels_artifact(
        labels: Any,
        *,
        label_names: Optional[Iterable[Any]],
        target_type: str,
        target_names: Optional[Iterable[str]],
    ) -> tuple[bytes, dict]:
        """Serialize labels and derive the authoritative persisted row contract."""

        normalized, target_metadata = normalize_targets(
            labels,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        resolved_target_type = str(target_metadata["target_type"])
        resolved_label_names = target_metadata.get("label_names")
        resolved_target_names = target_metadata.get("target_names")
        payload = json_dumps_strict(
            labels_to_artifact_jsonable(
                normalized,
                label_names=resolved_label_names,
                target_type=resolved_target_type,
                target_names=resolved_target_names,
            ),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        classification = resolved_target_type != "regression"
        catalog_values: Iterable[Any]
        if resolved_target_type == "multi_label":
            catalog_values = tuple(resolved_label_names or ())
        elif classification:
            catalog_values = normalized.tolist()
        else:
            catalog_values = ()
        contract = {
            "target_type": resolved_target_type,
            "label_names": (
                None if resolved_label_names is None else semantic_label_keys(resolved_label_names)
            ),
            "target_names": (
                None if resolved_target_names is None else list(resolved_target_names)
            ),
            "n_samples": int(len(normalized)),
            "shape": list(normalized.shape),
            "dtype": str(normalized.dtype),
            "label_encoding": LABEL_ENCODING if classification else None,
            "label_catalog": (semantic_label_catalog(catalog_values) if classification else None),
        }
        return payload, contract

    @staticmethod
    def _preserve_v2_label_catalog(labels_contract: dict, metadata: dict) -> dict:
        """Carry a validated original catalog through a v2-to-v2 label copy."""

        if (
            labels_contract.get("label_encoding") != LABEL_ENCODING
            or metadata.get("label_encoding") != LABEL_ENCODING
            or metadata.get("target_type") != labels_contract.get("target_type")
        ):
            return labels_contract
        catalog = validate_label_catalog(metadata.get("label_catalog"))
        expected_keys = {item["key"] for item in labels_contract.get("label_catalog") or []}
        catalog_keys = {item["key"] for item in catalog}
        if catalog_keys != expected_keys:
            raise ValueError("v2 label_catalog keys must exactly cover the semantic label payload.")
        preserved = dict(labels_contract)
        preserved["label_catalog"] = catalog
        return preserved

    @staticmethod
    def _metadata_with_artifact_path(metadata: dict, artifact_path: str) -> dict:
        committed = dict(metadata)
        committed["artifact_path"] = artifact_path
        return committed

    @staticmethod
    def _metadata_with_array_contract(
        metadata: dict,
        *,
        artifact_path: str,
        manifest: ArrayArtifactManifest,
    ) -> dict:
        committed = LocalArtifactStore._metadata_with_artifact_path(metadata, artifact_path)
        sparse_value = manifest.storage_format == "npz"
        committed.update(
            {
                "shape": list(manifest.shape),
                "dtype": manifest.dtype,
                "sparse": sparse_value,
                "storage_format": manifest.sparse_format if sparse_value else "dense",
                "artifact_storage_format": manifest.storage_format,
                "nnz": manifest.nnz,
            }
        )
        if len(manifest.shape) == 2:
            committed["n_samples"] = manifest.shape[0]
            committed["embedding_dim"] = manifest.shape[1]
        return committed

    @staticmethod
    def _finalize_array_metadata(
        metadata: dict,
        *,
        manifest: ArrayArtifactManifest,
        stat: ArtifactStat,
        metadata_finalizer: Optional[Callable[[dict, ArrayArtifactManifest, ArtifactStat], dict]],
    ) -> dict:
        candidate = dict(metadata)
        if metadata_finalizer is not None:
            candidate = metadata_finalizer(candidate, manifest, stat)
            if not isinstance(candidate, dict):
                raise TypeError("Artifact metadata_finalizer must return a dictionary.")
        return LocalArtifactStore._metadata_with_array_contract(
            candidate,
            artifact_path=stat.uri,
            manifest=manifest,
        )

    def _load_array(self, target: Path, manifest: ArrayArtifactManifest) -> Any:
        self._validate_array_file(target, manifest)
        if manifest.storage_format == "npz":
            from scipy import sparse

            value = sparse.load_npz(target)
        else:
            value = np.load(target, allow_pickle=False)
        self._validate_loaded_array(value, manifest)
        return value

    def _load_json(self, target: Path, manifest: JSONArtifactManifest) -> dict:
        value = self._load_json_value(target, manifest)
        if not isinstance(value, dict):
            raise ValueError(f"JSON artifact {target} must contain a JSON object.")
        return value

    def _load_json_value(self, target: Path, manifest: JSONArtifactManifest) -> Any:
        self._validate_json_file(target, manifest)
        try:
            value = json.loads(target.read_text(encoding=manifest.encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"JSON artifact {target} is not valid UTF-8 JSON.") from exc
        return value

    def _validate_json_file(self, target: Path, manifest: JSONArtifactManifest) -> None:
        actual_size = int(target.stat().st_size)
        if actual_size != manifest.size_bytes:
            raise ValueError(
                f"JSON artifact {target} has size {actual_size}, expected "
                f"{manifest.size_bytes} from its manifest."
            )
        actual_checksum = self._sha256_file(target)
        if actual_checksum != manifest.sha256:
            raise ValueError(
                f"JSON artifact {target} failed SHA-256 verification; expected "
                f"{manifest.sha256}, got {actual_checksum}."
            )

    def _validate_array_file(self, target: Path, manifest: ArrayArtifactManifest) -> None:
        actual_size = int(target.stat().st_size)
        if actual_size != manifest.size_bytes:
            raise ValueError(
                f"Array artifact {target} has size {actual_size}, expected "
                f"{manifest.size_bytes} from its manifest."
            )
        actual_checksum = self._sha256_file(target)
        if actual_checksum != manifest.sha256:
            raise ValueError(
                f"Array artifact {target} failed SHA-256 verification; expected "
                f"{manifest.sha256}, got {actual_checksum}."
            )

    def _validate_loaded_array(self, value: Any, manifest: ArrayArtifactManifest) -> None:
        if tuple(int(size) for size in value.shape) != manifest.shape:
            raise ValueError("Loaded array shape does not match its committed manifest.")
        if str(value.dtype) != manifest.dtype:
            raise ValueError("Loaded array dtype does not match its committed manifest.")
        if manifest.storage_format == "npz" and value.getformat() != manifest.sparse_format:
            raise ValueError("Loaded sparse format does not match its committed manifest.")
        if manifest.storage_format == "npz" and int(value.nnz) != manifest.nnz:
            raise ValueError("Loaded sparse nnz does not match its committed manifest.")

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_text_atomic(target: Path, payload: str) -> None:
        LocalArtifactStore._write_bytes_atomic(target, payload.encode("utf-8"))

    @staticmethod
    def _write_bytes_atomic(target: Path, payload: bytes) -> None:
        descriptor = None
        temporary_name = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            with os.fdopen(descriptor, "wb") as file:
                descriptor = None
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, target)
            temporary_name = None
            LocalArtifactStore._fsync_directory(target.parent)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    def _write_array_manifest(self, path: Path, manifest: ArrayArtifactManifest) -> None:
        self._write_bytes_atomic(
            path / ARRAY_MANIFEST_FILENAME,
            json_dumps_strict(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )

    def _write_artifact_manifest(
        self, path: Path, manifest: Union[ArtifactManifest, LabelsArtifactManifest]
    ) -> None:
        self._write_bytes_atomic(
            path / ARTIFACT_MANIFEST_FILENAME,
            json_dumps_strict(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as file:
            os.fsync(file.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _put_dense_batches(
        self,
        path: Path,
        first_indices: np.ndarray,
        first_batch: Any,
        remaining: Iterable[Tuple[np.ndarray, Any]],
        n_samples: int,
        require_complete: bool,
    ) -> str:
        first = np.asarray(first_batch)
        if first.ndim != 2:
            raise ValueError(f"Embedding batches must be 2D; got shape {first.shape}.")
        target = path / "embeddings.npy"
        written = np.zeros(n_samples, dtype=bool)
        mmap = np.lib.format.open_memmap(
            target,
            mode="w+",
            dtype=first.dtype,
            shape=(n_samples, first.shape[1]),
        )
        self._write_dense_batch(mmap, written, first_indices, first)
        for indices, batch in remaining:
            if is_sparse_matrix(batch):
                raise ValueError("Cannot mix dense and sparse embedding batches.")
            arr = np.asarray(batch)
            self._write_dense_batch(mmap, written, indices, arr)
        mmap.flush()
        if require_complete and not bool(np.all(written)):
            missing = np.flatnonzero(~written)
            raise ValueError(
                f"Embedding batches did not cover all samples; missing {missing[:10]}."
            )
        return str(target)

    def _write_dense_batch(
        self,
        mmap: np.ndarray,
        written: np.ndarray,
        indices: np.ndarray,
        batch: np.ndarray,
    ) -> None:
        if batch.ndim != 2:
            raise ValueError(f"Embedding batches must be 2D; got shape {batch.shape}.")
        if batch.shape[1] != mmap.shape[1]:
            raise ValueError(
                "Dense embedding batches must have a consistent column count; "
                f"expected {mmap.shape[1]}, got {batch.shape[1]}."
            )
        if batch.dtype != mmap.dtype:
            raise ValueError(
                "Dense embedding batches must have a consistent dtype; "
                f"expected {mmap.dtype}, got {batch.dtype}."
            )
        indices = self._validate_batch_indices(
            indices,
            n_rows=int(batch.shape[0]),
            written=written,
        )
        mmap[indices] = batch
        written[indices] = True

    def _put_sparse_batches(
        self,
        path: Path,
        batches: Iterable[Tuple[np.ndarray, Any]],
        n_samples: int,
        require_complete: bool,
    ) -> tuple[str, int]:
        from scipy import sparse

        spool: list[tuple[Path, Path]] = []
        n_features = None
        written = np.zeros(n_samples, dtype=bool)
        row_counts = np.zeros(n_samples, dtype=np.int64)
        expected_dtype = None
        total_nnz = 0
        for batch_index, (indices, batch) in enumerate(batches):
            if not is_sparse_matrix(batch):
                raise ValueError("Cannot mix sparse and dense embedding batches.")
            if getattr(batch, "ndim", None) != 2:
                raise ValueError(f"Embedding batches must be 2D; got shape {batch.shape}.")
            if n_features is None:
                n_features = int(batch.shape[1])
                expected_dtype = batch.dtype
            elif int(batch.shape[1]) != n_features:
                raise ValueError("Sparse embedding batches must have a consistent column count.")
            if batch.dtype != expected_dtype:
                raise ValueError(
                    "Sparse embedding batches must have a consistent dtype; "
                    f"expected {expected_dtype}, got {batch.dtype}."
                )
            indices = self._validate_batch_indices(
                indices,
                n_rows=int(batch.shape[0]),
                written=written,
            )
            csr = batch.tocsr(copy=True)
            csr.sum_duplicates()
            row_counts[indices] = np.diff(csr.indptr).astype(np.int64, copy=False)
            total_nnz += int(csr.nnz)
            matrix_path = path / f"sparse-batch-{batch_index:08d}.npz"
            indices_path = path / f"sparse-indices-{batch_index:08d}.npy"
            sparse.save_npz(matrix_path, csr)
            np.save(indices_path, indices, allow_pickle=False)
            spool.append((matrix_path, indices_path))
            written[indices] = True
        if require_complete and not bool(np.all(written)):
            missing = np.flatnonzero(~written)
            raise ValueError(
                f"Embedding batches did not cover all samples; missing {missing[:10]}."
            )
        if n_features is None:
            raise ValueError("At least one sparse embedding batch is required.")
        if expected_dtype is None:
            raise RuntimeError("Sparse batch dtype accounting is inconsistent.")
        indptr = np.empty(n_samples + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(row_counts, out=indptr[1:])
        if int(indptr[-1]) != total_nnz:
            raise RuntimeError("Sparse batch nnz accounting is inconsistent.")

        data: np.ndarray
        columns: np.ndarray
        if total_nnz:
            data = np.memmap(
                path / "sparse-data.bin",
                mode="w+",
                dtype=expected_dtype,
                shape=(total_nnz,),
            )
            index_dtype = (
                np.int32
                if max(n_samples, n_features, total_nnz) <= np.iinfo(np.int32).max
                else np.int64
            )
            columns = np.memmap(
                path / "sparse-columns.bin",
                mode="w+",
                dtype=index_dtype,
                shape=(total_nnz,),
            )
            cursors = indptr[:-1].copy()
            for matrix_path, indices_path in spool:
                csr = sparse.load_npz(matrix_path).tocsr(copy=False)
                indices = np.load(indices_path, allow_pickle=False)
                for local_row, global_row in enumerate(indices):
                    source_start = int(csr.indptr[local_row])
                    source_end = int(csr.indptr[local_row + 1])
                    count = source_end - source_start
                    destination_start = int(cursors[global_row])
                    destination_end = destination_start + count
                    data[destination_start:destination_end] = csr.data[source_start:source_end]
                    columns[destination_start:destination_end] = csr.indices[
                        source_start:source_end
                    ]
                    cursors[global_row] = destination_end
            if not np.array_equal(cursors, indptr[1:]):
                raise RuntimeError("Sparse batch row assembly is inconsistent.")
            data.flush()
            columns.flush()
        else:
            data = np.asarray([], dtype=expected_dtype)
            columns = np.asarray([], dtype=np.int32)
        matrix = sparse.csr_matrix(
            (data, columns, indptr),
            shape=(n_samples, n_features),
            copy=False,
        )
        target = path / "embeddings.npz"
        sparse.save_npz(target, matrix)
        final_nnz = int(matrix.nnz)
        del matrix
        del data
        del columns
        return str(target), final_nnz

    def _validate_batch_indices(
        self,
        indices: Any,
        *,
        n_rows: int,
        written: np.ndarray,
    ) -> np.ndarray:
        raw = np.asarray(indices)
        if raw.ndim != 1:
            raise ValueError(f"Batch indices must be 1D; got shape {raw.shape}.")
        if raw.dtype.kind not in {"i", "u"}:
            raise ValueError("Batch indices must contain integers, not booleans or floats.")
        if len(raw) != n_rows:
            raise ValueError("Batch index count must match embedding row count.")
        if np.any(raw < 0) or np.any(raw >= len(written)):
            invalid = raw[(raw < 0) | (raw >= len(written))]
            raise ValueError(
                "Batch indices must be between 0 and n_samples - 1; "
                f"invalid indices {invalid[:10]}."
            )
        normalized = raw.astype(np.intp, copy=False)
        unique, counts = np.unique(normalized, return_counts=True)
        within_batch = unique[counts > 1]
        if within_batch.size:
            raise ValueError(
                "Duplicate embedding rows within one batch for sample indices "
                f"{within_batch[:10]}."
            )
        across_batches = normalized[written[normalized]]
        if across_batches.size:
            raise ValueError(
                "Duplicate embedding rows across batches for sample indices "
                f"{across_batches[:10]}."
            )
        return normalized
