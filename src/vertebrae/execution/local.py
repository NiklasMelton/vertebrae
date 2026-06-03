"""Local execution backend."""

from typing import Any, Callable, Iterable, List


class LocalBackend:
    """Local execution backend.

    Args:
        n_jobs: Number of local jobs. Values other than one use joblib.
    """

    def __init__(self, n_jobs: int = 1) -> None:
        self.n_jobs = n_jobs

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

        return Parallel(n_jobs=self.n_jobs)(delayed(fn)(job) for job in job_list)
