"""Local filesystem artifact store."""

import json
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np

from vertebrae.utils.validation import is_sparse_matrix


class LocalArtifactStore:
    """Store dense/sparse arrays and JSON metadata under a local directory.

    Args:
        root: Root cache directory.
    """

    def __init__(self, root: str = ".vertebrae_cache") -> None:
        self.root = Path(root)

    def exists(self, key: str) -> bool:
        """Return whether an embedding artifact exists for `key`.

        Args:
            key: Artifact key.

        Returns:
            Whether a dense `.npy` or sparse `.npz` embedding file exists.
        """

        path = self._path(key)
        return (path / "embeddings.npy").exists() or (path / "embeddings.npz").exists()

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
        if is_sparse_matrix(arr):
            from scipy import sparse

            target = path / "embeddings.npz"
            sparse.save_npz(target, arr)
            return str(target)
        target = path / "embeddings.npy"
        np.save(target, np.asarray(arr))
        return str(target)

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

        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        iterator = iter(batches)
        try:
            first_indices, first_batch = next(iterator)
        except StopIteration as exc:
            raise ValueError("At least one embedding batch is required.") from exc

        if is_sparse_matrix(first_batch):
            return self._put_sparse_batches(
                path,
                [(first_indices, first_batch), *list(iterator)],
                n_samples=n_samples,
                require_complete=require_complete,
            )
        return self._put_dense_batches(
            path,
            first_indices=first_indices,
            first_batch=first_batch,
            remaining=iterator,
            n_samples=n_samples,
            require_complete=require_complete,
        )

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse embedding matrix.

        Args:
            key: Artifact key.

        Returns:
            Dense NumPy array or scipy sparse matrix.
        """

        path = self._path(key)
        sparse_target = path / "embeddings.npz"
        if sparse_target.exists():
            from scipy import sparse

            return sparse.load_npz(sparse_target)
        return np.load(path / "embeddings.npy", allow_pickle=False)

    def put_json(self, key: str, obj: dict) -> str:
        """Store JSON metadata for an artifact key.

        Args:
            key: Artifact key.
            obj: JSON-serializable metadata.

        Returns:
            Filesystem path to the saved metadata file.
        """

        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "metadata.json"
        with target.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
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

    def _path(self, key: str) -> Path:
        clean_key = key.strip("/").replace("..", "__")
        return self.root / clean_key

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
        indices = np.asarray(indices, dtype=int)
        if batch.ndim != 2:
            raise ValueError(f"Embedding batches must be 2D; got shape {batch.shape}.")
        if len(indices) != batch.shape[0]:
            raise ValueError("Batch index count must match embedding row count.")
        if np.any(written[indices]):
            duplicates = indices[written[indices]]
            raise ValueError(f"Duplicate embedding rows for sample indices {duplicates[:10]}.")
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
        for indices, batch in batches:
            if not is_sparse_matrix(batch):
                raise ValueError("Cannot mix sparse and dense embedding batches.")
            indices = np.asarray(indices, dtype=int)
            if len(indices) != batch.shape[0]:
                raise ValueError("Batch index count must match embedding row count.")
            if n_features is None:
                n_features = int(batch.shape[1])
            elif int(batch.shape[1]) != n_features:
                raise ValueError("Sparse embedding batches must have a consistent column count.")
            if np.any(written[indices]):
                duplicates = indices[written[indices]]
                raise ValueError(f"Duplicate embedding rows for sample indices {duplicates[:10]}.")
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
