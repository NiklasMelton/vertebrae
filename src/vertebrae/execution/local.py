"""Local execution backend."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List


@dataclass
class LocalJobHandle:
    """Immediate local job handle.

    Attributes:
        result: Completed job result.
        state: Job state.
    """

    result: Any
    state: str = "finished"


class LocalBackend:
    """Local execution backend.

    Args:
        n_jobs: Number of local jobs. Values other than one use joblib.
        joblib_backend: Optional joblib backend such as `"loky"` or `"threading"`.
    """

    def __init__(self, n_jobs: int = 1, joblib_backend: str = "loky") -> None:
        self.n_jobs = n_jobs
        self.joblib_backend = joblib_backend

    def submit(self, fn: Callable[[Any], Any], job: Any) -> LocalJobHandle:
        """Execute a single local job immediately.

        Args:
            fn: Callable to execute.
            job: Job input.

        Returns:
            Completed local job handle.
        """

        return LocalJobHandle(result=fn(job))

    def gather(self, handles: Iterable[LocalJobHandle]) -> List[Any]:
        """Collect local job results.

        Args:
            handles: Local job handles.

        Returns:
            Results in handle order.
        """

        return [handle.result for handle in handles]

    def status(self, handle: LocalJobHandle) -> str:
        """Return local job status.

        Args:
            handle: Local job handle.

        Returns:
            Job status string.
        """

        return handle.state

    def map(self, fn: Callable[[Any], Any], jobs: Iterable[Any]) -> List[Any]:
        """Map a callable over jobs locally.

        Args:
            fn: Callable to execute.
            jobs: Iterable of job inputs.

        Returns:
            Results in job order.
        """

        job_list = list(jobs)
        if self.n_jobs == 1:
            return [fn(job) for job in job_list]
        from joblib import Parallel, delayed

        return Parallel(n_jobs=self.n_jobs, backend=self.joblib_backend)(
            delayed(fn)(job) for job in job_list
        )
