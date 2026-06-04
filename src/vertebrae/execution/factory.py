"""Execution backend factory helpers."""

from typing import Any

from vertebrae.execution.local import LocalBackend


def create_execution_backend(name: str, **kwargs: Any) -> Any:
    """Create an execution backend by name.

    Args:
        name: Backend name.
        **kwargs: Backend-specific configuration.

    Returns:
        Execution backend instance.

    Raises:
        ValueError: If the backend name is unknown.
    """

    backend = name.lower()
    if backend == "local":
        local_kwargs = {
            "n_jobs": kwargs.get("n_jobs", 1),
            "joblib_backend": kwargs.get("joblib_backend", "loky"),
        }
        return LocalBackend(**local_kwargs)
    if backend == "ray":
        from vertebrae.execution.ray_backend import RayBackend

        return RayBackend(
            address=kwargs.get("ray_address") or kwargs.get("address"),
            num_cpus=kwargs.get("num_cpus"),
            num_gpus=kwargs.get("num_gpus"),
            memory_bytes=kwargs.get("memory_bytes"),
            runtime_env=kwargs.get("runtime_env"),
        )
    if backend == "dask":
        from vertebrae.execution.dask_backend import DaskBackend

        return DaskBackend(
            address=kwargs.get("dask_address") or kwargs.get("address"),
            client=kwargs.get("client"),
            timeout=kwargs.get("timeout"),
        )
    raise ValueError(f"Unknown execution backend: {name}.")
