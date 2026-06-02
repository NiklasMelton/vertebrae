"""Local filesystem artifact store."""

import json
from pathlib import Path
from typing import Any

import numpy as np


class LocalArtifactStore:
    """Store arrays and JSON metadata under a local cache directory."""

    def __init__(self, root: str = ".vertebrae_cache") -> None:
        self.root = Path(root)

    def exists(self, key: str) -> bool:
        return (self._path(key) / "embeddings.npy").exists()

    def put_array(self, key: str, arr: Any) -> str:
        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "embeddings.npy"
        np.save(target, np.asarray(arr))
        return str(target)

    def get_array(self, key: str) -> Any:
        return np.load(self._path(key) / "embeddings.npy", allow_pickle=False)

    def put_json(self, key: str, obj: dict) -> str:
        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "metadata.json"
        with target.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
        return str(target)

    def get_json(self, key: str) -> dict:
        with (self._path(key) / "metadata.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def _path(self, key: str) -> Path:
        clean_key = key.strip("/").replace("..", "__")
        return self.root / clean_key
