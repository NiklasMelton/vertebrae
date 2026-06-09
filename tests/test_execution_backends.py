import pytest

from vertebrae.execution import LocalBackend, create_execution_backend


def test_create_execution_backend_returns_local_backend():
    backend = create_execution_backend("local", n_jobs=2, joblib_backend="threading")

    assert isinstance(backend, LocalBackend)
    assert backend.n_jobs == 2
    assert backend.joblib_backend == "threading"


def test_create_execution_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown execution backend"):
        create_execution_backend("bogus")


def test_create_execution_backend_ray_missing_dependency():
    with pytest.raises(ImportError, match="optional 'ray' extra"):
        create_execution_backend("ray").map(lambda value: value, [1])


def test_create_execution_backend_dask_missing_dependency():
    with pytest.raises(ImportError, match="optional 'dask' extra"):
        create_execution_backend("dask").map(lambda value: value, [1])
