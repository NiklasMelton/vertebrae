"""Ray execution backend."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

from vertebrae.execution.jobs import ResourceSpec


@dataclass
class RayJobHandle:
    """Handle for a submitted Ray task."""

    ref: Any


class RayBackend:
    """Ray execution backend.

    Args:
        address: Optional Ray cluster address.
        num_cpus: Optional default CPU request for tasks.
        num_gpus: Optional default GPU request for tasks.
        memory_bytes: Optional default memory request for tasks.
        runtime_env: Optional Ray runtime environment.
    """

    def __init__(
        self,
        address: Optional[str] = None,
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = None,
        memory_bytes: Optional[int] = None,
        runtime_env: Optional[dict[str, Any]] = None,
    ) -> None:
        self.address = address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.memory_bytes = memory_bytes
        self.runtime_env = runtime_env
        self._ray = None

    def submit(self, fn: Callable[[Any], Any], job: Any) -> RayJobHandle:
        """Submit a single Ray job."""

        self._ensure_ray()
        remote_fn = self._remote_function(fn, getattr(job, "resources", None))
        return RayJobHandle(ref=remote_fn.remote(job))

    def gather(self, handles: Iterable[RayJobHandle]) -> List[Any]:
        """Collect Ray job results."""

        ray = self._ensure_ray()
        refs = [handle.ref for handle in handles]
        return list(ray.get(refs))

    def status(self, handle: RayJobHandle) -> str:
        """Return Ray task status."""

        ray = self._ensure_ray()
        ready, _ = ray.wait([handle.ref], timeout=0)
        return "finished" if ready else "running"

    def map(self, fn: Callable[[Any], Any], jobs: Iterable[Any]) -> List[Any]:
        """Map a callable over Ray jobs while preserving order."""

        handles = [self.submit(fn, job) for job in list(jobs)]
        return self.gather(handles)

    def _ensure_ray(self) -> Any:
        if self._ray is not None:
            return self._ray
        try:
            import ray
        except ImportError as exc:
            raise ImportError(
                "Ray support requires the optional 'ray' extra. Install with "
                "`poetry install --extras ray`."
            ) from exc
        ray.init(address=self.address, runtime_env=self.runtime_env, ignore_reinit_error=True)
        self._ray = ray
        return ray

    def _remote_function(self, fn: Callable[[Any], Any], resources: Optional[ResourceSpec]) -> Any:
        ray = self._ensure_ray()
        options: dict[str, Any] = {}
        resource_spec = resources or ResourceSpec()
        options["num_cpus"] = self.num_cpus if self.num_cpus is not None else resource_spec.cpus
        options["num_gpus"] = self.num_gpus if self.num_gpus is not None else resource_spec.gpus
        memory_value = (
            self.memory_bytes if self.memory_bytes is not None else resource_spec.memory_bytes
        )
        if memory_value is not None:
            options["memory"] = memory_value
        return ray.remote(**options)(fn)
