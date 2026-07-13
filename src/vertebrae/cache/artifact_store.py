"""Artifact store protocols and configuration."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Tuple

import numpy as np


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
