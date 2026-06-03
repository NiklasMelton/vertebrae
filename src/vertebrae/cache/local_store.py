"""Local filesystem artifact store."""

import json
from pathlib import Path
from typing import Any

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
