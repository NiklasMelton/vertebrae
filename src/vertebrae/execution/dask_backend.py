"""Dask execution backend."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional


@dataclass
class DaskJobHandle:
    """Handle for a submitted Dask future."""

    future: Any


class DaskBackend:
    """Dask execution backend.

    Args:
        address: Optional scheduler address.
        client: Optional pre-existing Dask client.
        timeout: Optional client connect timeout.
    """

    def __init__(
        self,
        address: Optional[str] = None,
        client: Any = None,
        timeout: Optional[str] = None,
    ) -> None:
        self.address = address
        self.client = client
        self.timeout = timeout
        self._client = client

    def submit(self, fn: Callable[[Any], Any], job: Any) -> DaskJobHandle:
        """Submit a single Dask job."""

        client = self._ensure_client()
        return DaskJobHandle(future=client.submit(fn, job))

    def gather(self, handles: Iterable[DaskJobHandle]) -> List[Any]:
        """Collect Dask job results."""

        client = self._ensure_client()
        futures = [handle.future for handle in handles]
        return list(client.gather(futures))

    def status(self, handle: DaskJobHandle) -> str:
        """Return Dask future status."""

        return str(handle.future.status)

    def map(self, fn: Callable[[Any], Any], jobs: Iterable[Any]) -> List[Any]:
        """Map a callable over Dask jobs while preserving order."""

        handles = [self.submit(fn, job) for job in list(jobs)]
        return self.gather(handles)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from distributed import Client
        except ImportError as exc:
            raise ImportError(
                "Dask support requires the optional 'dask' extra. Install with "
                "`poetry install --extras dask`."
            ) from exc
        self._client = Client(self.address, timeout=self.timeout) if self.address else Client()
        return self._client
