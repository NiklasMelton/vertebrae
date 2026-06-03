"""Execution backend protocol."""

from typing import Any, Callable, Iterable, List, Protocol


class ExecutionBackend(Protocol):
    """Protocol for local or future distributed execution backends."""

    def map(self, fn: Callable[[Any], Any], jobs: Iterable[Any]) -> List[Any]:
        """Map a callable over jobs.

        Args:
            fn: Callable to execute.
            jobs: Iterable of job inputs.

        Returns:
            Results in job order.
        """

        ...
