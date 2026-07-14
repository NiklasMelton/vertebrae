"""Local filesystem artifact store."""

import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np

from vertebrae.cache.artifact_store import (
    ARRAY_MANIFEST_FILENAME,
    ArrayArtifactManifest,
    ArtifactStat,
    ArtifactStoreConfig,
)
from vertebrae.utils.labels import labels_from_jsonable, labels_to_jsonable
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
        manifest_path = path / ARRAY_MANIFEST_FILENAME
        if not manifest_path.exists():
            return False
        manifest = self._read_array_manifest(path)
        return (path / manifest.filename).exists()

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
            else:
                array = np.asarray(arr)
                with prepared.open("wb") as file:
                    np.save(file, array)
                    file.flush()
                    os.fsync(file.fileno())
                shape = tuple(int(size) for size in array.shape)
                dtype = str(array.dtype)
                sparse_format = None
            return self._publish_prepared_array(
                path,
                prepared,
                shape=shape,
                dtype=dtype,
                sparse_format=sparse_format,
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
                prepared = Path(
                    self._put_sparse_batches(
                        staging,
                        [(first_indices, first_batch), *list(iterator)],
                        n_samples=n_samples,
                        require_complete=require_complete,
                    )
                )
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
            return self._publish_prepared_array(
                path,
                prepared,
                shape=shape,
                dtype=dtype,
                sparse_format=sparse_format,
            )

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse embedding matrix.

        Args:
            key: Artifact key.

        Returns:
            Dense NumPy array or scipy sparse matrix.
        """

        path = self._path(key)
        manifest = self._read_array_manifest(path)
        target = path / manifest.filename
        if not target.exists():
            raise FileNotFoundError(
                f"Array manifest for key {key} references missing file {manifest.filename}."
            )
        if manifest.storage_format == "npz":
            from scipy import sparse

            return sparse.load_npz(target)
        return np.load(target, allow_pickle=False)

    def stat_array(self, key: str) -> ArtifactStat:
        """Return the persisted array file size without loading it."""

        path = self._path(key)
        manifest = self._read_array_manifest(path)
        target = path / manifest.filename
        if not target.exists():
            raise FileNotFoundError(
                f"Array manifest for key {key} references missing file {manifest.filename}."
            )
        return ArtifactStat(
            uri=str(target),
            size_bytes=int(target.stat().st_size),
            storage_format=manifest.storage_format,
        )

    def put_labels(self, key: str, labels: Any) -> str:
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
        payload = json_dumps_strict(labels_to_jsonable(labels), indent=2, sort_keys=True)
        with target.open("w", encoding="utf-8") as f:
            f.write(payload)
        return str(target)

    def get_labels(self, key: str) -> np.ndarray:
        """Load labels from a JSON artifact.

        Args:
            key: Artifact key.

        Returns:
            One-dimensional label array.
        """

        path = self._path(key)
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
        with target.open("w", encoding="utf-8") as f:
            f.write(payload)
        return str(target)

    def get_json(self, key: str) -> dict:
        """Load JSON metadata for an artifact key.

        Args:
            key: Artifact key.

        Returns:
            Metadata dictionary.
        """

        with (self._path(key) / "metadata.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def delete_prefix(self, prefix: str) -> None:
        """Delete every local artifact beneath a key prefix."""

        path = self._path(prefix)
        if path == self.root:
            raise ValueError("Refusing to delete the artifact-store root.")
        if path.exists():
            shutil.rmtree(path)

    def _path(self, key: str) -> Path:
        clean_key = key.strip("/").replace("..", "__")
        return self.root / clean_key

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

    def _publish_prepared_array(
        self,
        path: Path,
        prepared: Path,
        *,
        shape: tuple[int, ...],
        dtype: str,
        sparse_format: Any,
    ) -> str:
        path.mkdir(parents=True, exist_ok=True)
        target = path / prepared.name
        with prepared.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(prepared, target)
        manifest = ArrayArtifactManifest(
            filename=target.name,
            storage_format="npz" if target.suffix == ".npz" else "npy",
            shape=shape,
            dtype=dtype,
            size_bytes=int(target.stat().st_size),
            sparse_format=sparse_format,
        )
        self._write_array_manifest(path, manifest)
        stale = path / ("embeddings.npy" if manifest.storage_format == "npz" else "embeddings.npz")
        if stale.exists():
            try:
                stale.unlink()
            except OSError as exc:
                warnings.warn(
                    f"Committed array {target} but could not remove stale {stale}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return str(target)

    def _write_array_manifest(self, path: Path, manifest: ArrayArtifactManifest) -> None:
        target = path / ARRAY_MANIFEST_FILENAME
        descriptor = None
        temporary_name = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{ARRAY_MANIFEST_FILENAME}.", suffix=".tmp", dir=path
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                descriptor = None
                json.dump(manifest.to_dict(), file, indent=2, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, target)
            temporary_name = None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

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
        batches: list[Tuple[np.ndarray, Any]],
        n_samples: int,
        require_complete: bool,
    ) -> str:
        from scipy import sparse

        row_parts = []
        col_parts = []
        data_parts = []
        n_features = None
        written = np.zeros(n_samples, dtype=bool)
        expected_dtype = None
        for indices, batch in batches:
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
            coo = batch.tocoo()
            row_parts.append(indices[coo.row])
            col_parts.append(coo.col)
            data_parts.append(coo.data)
            written[indices] = True
        if require_complete and not bool(np.all(written)):
            missing = np.flatnonzero(~written)
            raise ValueError(
                f"Embedding batches did not cover all samples; missing {missing[:10]}."
            )
        if n_features is None:
            raise ValueError("At least one sparse embedding batch is required.")
        if row_parts:
            row = np.concatenate(row_parts)
            col = np.concatenate(col_parts)
            data = np.concatenate(data_parts)
        else:
            row = np.array([], dtype=int)
            col = np.array([], dtype=int)
            data = np.array([], dtype=float)
        matrix = sparse.csr_matrix((data, (row, col)), shape=(n_samples, n_features))
        target = path / "embeddings.npz"
        sparse.save_npz(target, matrix)
        return str(target)

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
