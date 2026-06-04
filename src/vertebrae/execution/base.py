"""Execution backend protocol."""

from typing import Any, Callable, Iterable, List, Protocol


class ExecutionBackend(Protocol):
    """Protocol for local or future distributed execution backends."""

    def submit(self, fn: Callable[[Any], Any], job: Any) -> Any:
        """Submit a single job.

        Args:
            fn: Callable to execute.
            job: Job input.

        Returns:
            Backend-specific job handle.
        """

        ...

    def gather(self, handles: Iterable[Any]) -> List[Any]:
        """Collect submitted job results.

        Args:
            handles: Backend-specific job handles.

        Returns:
            Results in handle order.
        """

        ...

    def status(self, handle: Any) -> str:
        """Return a backend-specific job status string.

        Args:
            handle: Backend-specific job handle.

        Returns:
            Status string such as `"finished"` or `"running"`.
        """

        ...

    def map(self, fn: Callable[[Any], Any], jobs: Iterable[Any]) -> List[Any]:
        """Map a callable over jobs.

        Args:
            fn: Callable to execute.
            jobs: Iterable of job inputs.

        Returns:
            Results in job order.
        """

        ...
