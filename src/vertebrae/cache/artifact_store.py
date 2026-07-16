"""Artifact store protocols and configuration."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol, Tuple

import numpy as np

ARRAY_MANIFEST_FILENAME = "array-manifest.json"
ARRAY_MANIFEST_SCHEMA_VERSION = 2
ARTIFACT_MANIFEST_FILENAME = "artifact-manifest.json"
ARTIFACT_MANIFEST_SCHEMA_VERSION = 2


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
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("ArtifactStat.uri must be a non-empty string.")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("ArtifactStat.size_bytes must be an integer.")
        if self.size_bytes < 0:
            raise ValueError("ArtifactStat.size_bytes must be >= 0.")
        if not isinstance(self.storage_format, str) or not self.storage_format:
            raise ValueError("ArtifactStat.storage_format must be non-empty.")


@dataclass(frozen=True)
class ArrayArtifactManifest:
    """Validated commit record for one dense or sparse array artifact."""

    filename: str
    storage_format: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    sha256: str
    sparse_format: Optional[str] = None
    nnz: Optional[int] = None
    schema_version: int = ARRAY_MANIFEST_SCHEMA_VERSION
    kind: str = "array"

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("Array manifest schema_version must be an integer.")
        if self.schema_version != ARRAY_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported array manifest schema version {self.schema_version}.")
        if not isinstance(self.kind, str):
            raise TypeError("Array manifest kind must be a string.")
        if self.kind != "array":
            raise ValueError("Array manifest kind must be 'array'.")
        if not isinstance(self.filename, str):
            raise TypeError("Array manifest filename must be a string.")
        if not isinstance(self.storage_format, str):
            raise TypeError("Array manifest storage_format must be a string.")
        if not isinstance(self.sha256, str):
            raise TypeError("Array manifest sha256 must be a string.")
        if self.storage_format not in {"npy", "npz"}:
            raise ValueError("Array manifest storage_format must be 'npy' or 'npz'.")
        expected_suffix = f".{self.storage_format}"
        if (
            not self.filename.startswith("embeddings-v2-")
            or not self.filename.endswith(expected_suffix)
            or len(self.filename) != len("embeddings-v2-") + 64 + len(expected_suffix)
        ):
            raise ValueError("Array manifest filename is not a canonical v2 content filename.")
        digest = self.filename[len("embeddings-v2-") : -len(expected_suffix)]
        if (
            digest != self.sha256
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("Array manifest sha256 must match its content filename.")
        if not isinstance(self.shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) for size in self.shape
        ):
            raise TypeError("Array manifest shape must be a tuple of integers.")
        if not self.shape or any(size < 0 for size in self.shape):
            raise ValueError("Array manifest shape must contain non-negative dimensions.")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("Array manifest dtype must be non-empty.")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("Array manifest size_bytes must be an integer.")
        if self.size_bytes < 0:
            raise ValueError("Array manifest size_bytes must be >= 0.")
        if self.storage_format == "npy":
            if self.sparse_format is not None:
                raise ValueError("Dense array manifests cannot declare sparse_format.")
            if self.nnz is not None:
                raise ValueError("Dense array manifests cannot declare nnz.")
        else:
            if not isinstance(self.sparse_format, str) or not self.sparse_format:
                raise ValueError("Sparse array manifests must declare sparse_format.")
            if isinstance(self.nnz, bool) or not isinstance(self.nnz, int) or self.nnz < 0:
                raise ValueError("Sparse array manifests must declare a non-negative integer nnz.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "filename": self.filename,
            "storage_format": self.storage_format,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "sparse_format": self.sparse_format,
            "nnz": self.nnz,
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
            "sha256",
            "sparse_format",
            "nnz",
        }
        if set(value) != required:
            raise ValueError(
                "Array manifest fields do not match the supported schema; "
                f"expected {sorted(required)}."
            )
        if isinstance(value["schema_version"], bool) or not isinstance(
            value["schema_version"], int
        ):
            raise ValueError("Array manifest schema_version must be an integer.")
        for name in ("kind", "filename", "storage_format", "dtype", "sha256"):
            if not isinstance(value[name], str):
                raise ValueError(f"Array manifest {name} must be a string.")
        if not isinstance(value["shape"], list) or any(
            isinstance(size, bool) or not isinstance(size, int) for size in value["shape"]
        ):
            raise ValueError("Array manifest shape must be an array of integers.")
        if isinstance(value["size_bytes"], bool) or not isinstance(value["size_bytes"], int):
            raise ValueError("Array manifest size_bytes must be an integer.")
        if value["sparse_format"] is not None and not isinstance(value["sparse_format"], str):
            raise ValueError("Array manifest sparse_format must be null or a string.")
        raw_nnz = value["nnz"]
        if raw_nnz is not None and (
            isinstance(raw_nnz, bool) or not isinstance(raw_nnz, int) or raw_nnz < 0
        ):
            raise ValueError("Array manifest nnz must be null or a non-negative integer.")
        try:
            return cls(
                schema_version=value["schema_version"],
                kind=value["kind"],
                filename=value["filename"],
                storage_format=value["storage_format"],
                shape=tuple(value["shape"]),
                dtype=value["dtype"],
                size_bytes=value["size_bytes"],
                sha256=value["sha256"],
                sparse_format=value["sparse_format"],
                nnz=None if raw_nnz is None else raw_nnz,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("Array manifest"):
                raise
            raise ValueError("Array manifest contains invalid field values.") from exc


@dataclass(frozen=True)
class JSONArtifactManifest:
    """Validated immutable JSON component referenced by an artifact commit."""

    filename: str
    size_bytes: int
    sha256: str
    role: str = "metadata"
    schema_version: int = ARTIFACT_MANIFEST_SCHEMA_VERSION
    kind: str = "json"
    media_type: str = "application/json"
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("JSON artifact schema_version must be an integer.")
        if self.schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported JSON artifact schema version {self.schema_version}.")
        for name in (
            "kind",
            "filename",
            "role",
            "media_type",
            "encoding",
            "sha256",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"JSON artifact {name} must be a string.")
        if self.kind != "json":
            raise ValueError("JSON artifact kind must be 'json'.")
        if self.media_type != "application/json":
            raise ValueError("JSON artifact media_type must be 'application/json'.")
        if self.encoding != "utf-8":
            raise ValueError("JSON artifact encoding must be 'utf-8'.")
        if self.role not in {"metadata", "labels"}:
            raise ValueError("JSON artifact role must be 'metadata' or 'labels'.")
        prefix = f"{self.role}-v2-"
        suffix = ".json"
        if (
            not self.filename.startswith(prefix)
            or not self.filename.endswith(suffix)
            or len(self.filename) != len(prefix) + 64 + len(suffix)
        ):
            raise ValueError("JSON artifact filename is not a canonical v2 content filename.")
        digest = self.filename[len(prefix) : -len(suffix)]
        if (
            digest != self.sha256
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("JSON artifact sha256 must match its content filename.")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("JSON artifact size_bytes must be an integer.")
        if self.size_bytes < 0:
            raise ValueError("JSON artifact size_bytes must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "filename": self.filename,
            "role": self.role,
            "media_type": self.media_type,
            "encoding": self.encoding,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "JSONArtifactManifest":
        if not isinstance(value, dict):
            raise ValueError("JSON artifact manifest must be a JSON object.")
        required = {
            "schema_version",
            "kind",
            "filename",
            "role",
            "media_type",
            "encoding",
            "size_bytes",
            "sha256",
        }
        if set(value) != required:
            raise ValueError(
                "JSON artifact manifest fields do not match the supported schema; "
                f"expected {sorted(required)}."
            )
        if isinstance(value["schema_version"], bool) or not isinstance(
            value["schema_version"], int
        ):
            raise ValueError("JSON artifact schema_version must be an integer.")
        for name in ("kind", "filename", "role", "media_type", "encoding", "sha256"):
            if not isinstance(value[name], str):
                raise ValueError(f"JSON artifact {name} must be a string.")
        if isinstance(value["size_bytes"], bool) or not isinstance(value["size_bytes"], int):
            raise ValueError("JSON artifact size_bytes must be an integer.")
        return cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            filename=value["filename"],
            role=value["role"],
            media_type=value["media_type"],
            encoding=value["encoding"],
            size_bytes=value["size_bytes"],
            sha256=value["sha256"],
        )


@dataclass(frozen=True)
class ArtifactManifest:
    """Last-written commit record binding an array and its metadata atomically."""

    array: ArrayArtifactManifest
    metadata: JSONArtifactManifest
    schema_version: int = ARTIFACT_MANIFEST_SCHEMA_VERSION
    kind: str = "array+metadata"

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("Artifact manifest schema_version must be an integer.")
        if self.schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact manifest schema version {self.schema_version}.")
        if not isinstance(self.kind, str):
            raise TypeError("Artifact manifest kind must be a string.")
        if self.kind != "array+metadata":
            raise ValueError("Artifact manifest kind must be 'array+metadata'.")
        if not isinstance(self.array, ArrayArtifactManifest):
            raise ValueError("Artifact manifest array must be an array contract.")
        if not isinstance(self.metadata, JSONArtifactManifest) or self.metadata.role != "metadata":
            raise ValueError("Artifact manifest metadata must be a JSON contract.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "array": self.array.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactManifest":
        if not isinstance(value, dict):
            raise ValueError("Artifact manifest must be a JSON object.")
        required = {"schema_version", "kind", "array", "metadata"}
        if set(value) != required:
            raise ValueError(
                "Artifact manifest fields do not match the supported schema; "
                f"expected {sorted(required)}."
            )
        if isinstance(value["schema_version"], bool) or not isinstance(
            value["schema_version"], int
        ):
            raise ValueError("Artifact manifest schema_version must be an integer.")
        if not isinstance(value["kind"], str):
            raise ValueError("Artifact manifest kind must be a string.")
        return cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            array=ArrayArtifactManifest.from_dict(value["array"]),
            metadata=JSONArtifactManifest.from_dict(value["metadata"]),
        )


@dataclass(frozen=True)
class LabelsArtifactManifest:
    """Last-written commit record binding labels and decoding metadata atomically."""

    labels: JSONArtifactManifest
    metadata: JSONArtifactManifest
    schema_version: int = ARTIFACT_MANIFEST_SCHEMA_VERSION
    kind: str = "labels+metadata"

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("Artifact manifest schema_version must be an integer.")
        if self.schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact manifest schema version {self.schema_version}.")
        if not isinstance(self.kind, str):
            raise TypeError("Artifact manifest kind must be a string.")
        if self.kind != "labels+metadata":
            raise ValueError("Labels artifact manifest kind must be 'labels+metadata'.")
        if not isinstance(self.labels, JSONArtifactManifest) or self.labels.role != "labels":
            raise ValueError("Labels artifact manifest labels must be a labels JSON contract.")
        if not isinstance(self.metadata, JSONArtifactManifest) or self.metadata.role != "metadata":
            raise ValueError("Labels artifact manifest metadata must be a metadata JSON contract.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "labels": self.labels.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LabelsArtifactManifest":
        if not isinstance(value, dict):
            raise ValueError("Artifact manifest must be a JSON object.")
        required = {"schema_version", "kind", "labels", "metadata"}
        if set(value) != required:
            raise ValueError(
                "Labels artifact manifest fields do not match the supported schema; "
                f"expected {sorted(required)}."
            )
        if isinstance(value["schema_version"], bool) or not isinstance(
            value["schema_version"], int
        ):
            raise ValueError("Artifact manifest schema_version must be an integer.")
        if not isinstance(value["kind"], str):
            raise ValueError("Artifact manifest kind must be a string.")
        return cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            labels=JSONArtifactManifest.from_dict(value["labels"]),
            metadata=JSONArtifactManifest.from_dict(value["metadata"]),
        )


class ArtifactStore(Protocol):
    """Protocol for embedding and metadata artifact stores.

    Keys are non-empty relative forward-slash paths. Implementations reject
    noncanonical or unsafe components rather than rewriting them.
    """

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
        """Atomically commit an array together with immutable JSON metadata."""

        ...

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
        """Atomically commit batched arrays together with immutable JSON metadata."""

        ...

    def get_artifact(self, key: str) -> Tuple[Any, dict]:
        """Load an array and metadata from the same committed artifact generation."""

        ...

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
        """Atomically commit labels together with their decoding metadata."""

        ...

    def get_labels_artifact(self, key: str) -> Tuple[np.ndarray, dict]:
        """Load labels and metadata from the same committed artifact generation."""

        ...

    def put_labels(
        self,
        key: str,
        labels: Any,
        *,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> str:
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
