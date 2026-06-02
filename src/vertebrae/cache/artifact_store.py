"""Artifact store protocol."""

from typing import Any, Protocol


class ArtifactStore(Protocol):
    def exists(self, key: str) -> bool:
        ...

    def put_array(self, key: str, arr: Any) -> str:
        ...

    def get_array(self, key: str) -> Any:
        ...

    def put_json(self, key: str, obj: dict) -> str:
        ...

    def get_json(self, key: str) -> dict:
        ...
