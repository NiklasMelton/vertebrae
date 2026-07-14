"""Artifact store protocols and configuration."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, Tuple

import numpy as np

ARRAY_MANIFEST_FILENAME = "array-manifest.json"
ARRAY_MANIFEST_SCHEMA_VERSION = 1
_ARRAY_FILENAMES = {"npy": "embeddings.npy", "npz": "embeddings.npz"}


@dataclass(frozen=True)
class ArtifactStoreConfig:
    """Serializable artifact-store configuration.

    Attributes:
        uri: Artifact store URI or local path.
        options: Provider-specific options used to reconstruct the store.
    """

    uri: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactStat:
    """Physical storage information for one persisted array artifact."""

    uri: str
    size_bytes: int
    storage_format: str

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError("ArtifactStat.size_bytes must be >= 0.")
        if not self.storage_format:
            raise ValueError("ArtifactStat.storage_format must be non-empty.")


@dataclass(frozen=True)
class ArrayArtifactManifest:
    """Validated commit record for one dense or sparse array artifact."""

    filename: str
    storage_format: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    sparse_format: Optional[str] = None
    schema_version: int = ARRAY_MANIFEST_SCHEMA_VERSION
    kind: str = "array"

    def __post_init__(self) -> None:
        if self.schema_version != ARRAY_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported array manifest schema version {self.schema_version}.")
        if self.kind != "array":
            raise ValueError("Array manifest kind must be 'array'.")
        if self.storage_format not in _ARRAY_FILENAMES:
            raise ValueError("Array manifest storage_format must be 'npy' or 'npz'.")
        if self.filename != _ARRAY_FILENAMES[self.storage_format]:
            raise ValueError("Array manifest filename does not match its storage format.")
        if not self.shape or any(int(size) < 0 for size in self.shape):
            raise ValueError("Array manifest shape must contain non-negative dimensions.")
        if not self.dtype:
            raise ValueError("Array manifest dtype must be non-empty.")
        if self.size_bytes < 0:
            raise ValueError("Array manifest size_bytes must be >= 0.")
        if self.storage_format == "npy" and self.sparse_format is not None:
            raise ValueError("Dense array manifests cannot declare sparse_format.")
        if self.storage_format == "npz" and not self.sparse_format:
            raise ValueError("Sparse array manifests must declare sparse_format.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "filename": self.filename,
            "storage_format": self.storage_format,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "size_bytes": self.size_bytes,
            "sparse_format": self.sparse_format,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ArrayArtifactManifest":
        if not isinstance(value, dict):
            raise ValueError("Array manifest must be a JSON object.")
        required = {
            "schema_version",
            "kind",
            "filename",
            "storage_format",
            "shape",
            "dtype",
            "size_bytes",
            "sparse_format",
        }
        if set(value) != required:
            raise ValueError(
                "Array manifest fields do not match the supported schema; "
                f"expected {sorted(required)}."
            )
        try:
            return cls(
                schema_version=int(value["schema_version"]),
                kind=str(value["kind"]),
                filename=str(value["filename"]),
                storage_format=str(value["storage_format"]),
                shape=tuple(int(size) for size in value["shape"]),
                dtype=str(value["dtype"]),
                size_bytes=int(value["size_bytes"]),
                sparse_format=(
                    None if value["sparse_format"] is None else str(value["sparse_format"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("Array manifest"):
                raise
            raise ValueError("Array manifest contains invalid field values.") from exc


class ArtifactStore(Protocol):
    """Protocol for embedding and metadata artifact stores."""

    def config(self) -> ArtifactStoreConfig:
        """Return a serializable configuration for this store."""

        ...

    def exists(self, key: str) -> bool:
        """Return whether an artifact exists for a key."""

        ...

    def put_array(self, key: str, arr: Any) -> str:
        """Store a dense or sparse array artifact and return its URI/path."""

        ...

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse array artifact by key."""

        ...

    def stat_array(self, key: str) -> ArtifactStat:
        """Return physical storage metadata without loading the array."""

        ...

    def put_array_batches(
        self,
        key: str,
        batches: Iterable[Tuple[np.ndarray, Any]],
        n_samples: int,
        require_complete: bool = True,
    ) -> str:
        """Store arrays from deterministic batches and return their URI/path."""

        ...

    def put_labels(self, key: str, labels: Any) -> str:
        """Store labels and return their URI/path."""

        ...

    def get_labels(self, key: str) -> np.ndarray:
        """Load labels by key."""

        ...

    def put_json(self, key: str, obj: dict) -> str:
        """Store a JSON metadata artifact and return its URI/path."""

        ...

    def get_json(self, key: str) -> dict:
        """Load a JSON metadata artifact by key."""

        ...

    def delete_prefix(self, prefix: str) -> None:
        """Delete every artifact stored beneath a key prefix."""

        ...
