import importlib
import os

import numpy as np
import pytest

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import (
    CacheConfig,
    EmbeddingConfig,
    OverlapScoringConfig,
    ProbeConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.extractors import TorchExtractor

pytestmark = [
    pytest.mark.devicesmoke,
    pytest.mark.skipif(
        os.environ.get("VERTABRAE_RUN_DEVICE_SMOKE") != "1",
        reason="set VERTABRAE_RUN_DEVICE_SMOKE=1 to run device smoke tests",
    ),
]


def test_torch_extractor_runs_full_benchmark_on_cpu_device(tmp_path):
    torch = _require_module("torch")
    _assert_torch_device_benchmark(torch, "cpu", tmp_path)


def test_torch_extractor_runs_full_benchmark_on_cuda_when_available(tmp_path):
    torch = _require_module("torch")
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is not available for enabled CUDA device smoke test.")
    _assert_torch_device_benchmark(torch, "cuda", tmp_path)


def test_torch_extractor_runs_full_benchmark_on_mps_when_available(tmp_path):
    torch = _require_module("torch")
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        raise AssertionError("MPS is not available for enabled MPS device smoke test.")
    _assert_torch_device_benchmark(torch, "mps", tmp_path)


def _assert_torch_device_benchmark(torch, device, tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.array(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
                [1.0, 1.1, 1.2, 1.3],
                [1.1, 1.2, 1.3, 1.4],
                [1.2, 1.3, 1.4, 1.5],
                [1.3, 1.4, 1.5, 1.6],
            ],
            dtype=np.float32,
        ),
        ["left"] * 4 + ["right"] * 4,
        modality="tabular",
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_(0.05 * (index + 1))

    extractor = TorchExtractor(
        name=f"torch_{device}",
        model=model,
        collate_fn=lambda batch: torch.as_tensor(np.asarray(batch), dtype=torch.float32),
        device=device,
        modality="tabular",
        streaming_safe=True,
        recipe_data={"device_smoke": device},
    )

    result = Benchmark(
        dataset,
        extractors=[extractor],
        scoring_config=OverlapScoringConfig(
            k=1,
            min_samples_per_cluster=1,
            kmeans_kwargs={"random_state": 11, "batch_size": 8, "n_init": 2},
        ),
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path / f"cache-{device}")),
        embedding_config=EmbeddingConfig(batch_size=2),
    ).run()

    item = result.extractor_results[0]
    assert item.name == f"torch_{device}"
    assert item.embedding_metadata["embedding_dim"] == 3
    assert item.embedding_metadata["streamed"] is True
    assert item.embedding_metadata["recipe"]["device"] == device
    assert 0.0 <= item.overlap.score <= 1.0
    assert all(parameter.device.type == device for parameter in model.parameters())


def _require_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AssertionError(
            f"Required dependency {module_name!r} is not installed for enabled "
            "device smoke tests."
        ) from exc
