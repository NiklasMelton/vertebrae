"""Execution backend protocol and benchmark execution errors."""

from typing import Any, Callable, Iterable, List, Protocol, runtime_checkable


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol for local or distributed execution backends."""

    def submit(self, fn: Callable[[Any], Any], job: Any) -> Any:
        """Submit a single job and return a backend-specific handle."""

        ...

    def gather(self, handles: Iterable[Any]) -> List[Any]:
        """Collect submitted job results in handle order."""

        ...

    def status(self, handle: Any) -> str:
        """Return a backend-specific job status string."""

        ...

    def map(self, fn: Callable[[Any], Any], jobs: Iterable[Any]) -> List[Any]:
        """Map a callable over jobs while preserving input order."""

        ...


class BenchmarkExecutionError(RuntimeError):
    """Failure raised by an artifact-backed benchmark stage."""

    def __init__(self, backend: Any, stage: str, job_identity: str, cause: Exception) -> None:
        self.backend = type(backend).__name__
        self.stage = stage
        self.job_identity = job_identity
        super().__init__(
            f"{self.backend} failed during benchmark stage {stage!r} for "
            f"{job_identity}: {cause}"
        )
