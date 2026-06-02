"""Execution backend protocol."""

from typing import Any, Callable, Iterable, List, Protocol


class ExecutionBackend(Protocol):
    def map(self, fn: Callable[[Any], Any], jobs: Iterable[Any]) -> List[Any]:
        ...
