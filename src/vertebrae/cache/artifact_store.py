"""Artifact store protocol."""

from typing import Any, Protocol


class ArtifactStore(Protocol):
    """Protocol for embedding and metadata artifact stores."""

    def exists(self, key: str) -> bool:
        """Return whether an artifact exists for a key."""

        ...

    def put_array(self, key: str, arr: Any) -> str:
        """Store a dense or sparse array artifact and return its URI/path."""

        ...

    def get_array(self, key: str) -> Any:
        """Load a dense or sparse array artifact by key."""

        ...

    def put_json(self, key: str, obj: dict) -> str:
        """Store a JSON metadata artifact and return its URI/path."""

        ...

    def get_json(self, key: str) -> dict:
        """Load a JSON metadata artifact by key."""

        ...
