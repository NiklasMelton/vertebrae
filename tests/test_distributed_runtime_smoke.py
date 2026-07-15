import importlib
import os

import numpy as np
import pytest

from vertebrae import Benchmark, BenchmarkDataset, DatasetIdentity
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.config import (
    CacheConfig,
    ExecutionConfig,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.execution import (
    DaskBackend,
    RayBackend,
    collect_score_artifacts,
    embedding_artifact_key,
    materialize_and_merge_embeddings,
    materialize_label_artifact,
    plan_scoring_jobs,
    score_embedding_artifacts,
)
from vertebrae.extractors import CallableExtractor

pytestmark = [
    pytest.mark.distributedruntime,
    pytest.mark.skipif(
        os.environ.get("VERTABRAE_RUN_DISTRIBUTED_RUNTIME") != "1",
        reason="set VERTABRAE_RUN_DISTRIBUTED_RUNTIME=1 to run Ray/Dask runtime tests",
    ),
]


def test_ray_backend_runs_artifact_backed_embedding_and_scoring_jobs(tmp_path):
    ray = _require_module("ray")
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        local_mode=False,
    )
    try:
        _assert_distributed_runtime_roundtrip(
            RayBackend(num_cpus=1),
            tmp_path,
            extractor_name="ray_runtime",
        )
    finally:
        ray.shutdown()


def test_dask_backend_runs_artifact_backed_embedding_and_scoring_jobs(tmp_path):
    distributed = _require_module("distributed")
    cluster = distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    try:
        _assert_distributed_runtime_roundtrip(
            DaskBackend(client=client),
            tmp_path,
            extractor_name="dask_runtime",
        )
    finally:
        client.close()
        cluster.close()


def _assert_distributed_runtime_roundtrip(backend, tmp_path, extractor_name):
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
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = CallableExtractor(
        extractor_name,
        transform_fn=distributed_runtime_transform,
        streaming_safe=True,
        recipe_data={"runtime_smoke": True},
    )
    store = LocalArtifactStore(str(tmp_path / extractor_name))

    merged = materialize_and_merge_embeddings(
        dataset=dataset,
        extractor=extractor,
        store=store,
        execution=backend,
        total_shards=4,
        batch_size=2,
    )
    embeddings = store.get_array(embedding_artifact_key(dataset, extractor))

    assert merged["n_shards"] == 4
    assert embeddings.shape == (8, 3)
    assert np.array_equal(embeddings, distributed_runtime_transform(dataset.X))

    labels = materialize_label_artifact(dataset, store)
    scoring_jobs = plan_scoring_jobs(
        embedding_key=merged["output_key"],
        labels_key=labels["output_key"],
        seeds=[3, 7],
        scoring_config=OverlapScoringConfig(
            k=1,
            min_samples_per_cluster=1,
            kmeans_kwargs={"batch_size": 8, "n_init": 2},
        ),
    )
    score_artifacts = score_embedding_artifacts(scoring_jobs, store, backend)
    collection = collect_score_artifacts(
        [artifact["output_key"] for artifact in score_artifacts],
        store=store,
        output_key=f"{merged['output_key']}/score-collections/runtime",
    )

    assert len(score_artifacts) == 2
    assert all(
        0.0 <= artifact["metrics"]["overlap"]["score"] <= 1.0 for artifact in score_artifacts
    )
    assert collection["repeats"] == 2
    assert len(collection["scores"]) == 2

    benchmark_result = Benchmark(
        dataset,
        [
            CallableExtractor(
                f"{extractor_name}_benchmark",
                transform_fn=distributed_runtime_transform,
                streaming_safe=True,
            )
        ],
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path / f"{extractor_name}_benchmark")),
        execution=backend,
        execution_config=ExecutionConfig(total_shards=4),
    ).run()

    assert benchmark_result.metadata["execution"]["artifact_backed"] is True
    assert len(benchmark_result.extractor_results) == 1


def distributed_runtime_transform(batch):
    values = np.asarray(batch, dtype=np.float32)
    return np.column_stack(
        [
            values[:, 0] + values[:, 1],
            values[:, 2] - values[:, 1],
            values.mean(axis=1),
        ]
    ).astype(np.float32)


def _require_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AssertionError(
            f"Required dependency {module_name!r} is not installed for enabled "
            "distributed runtime tests."
        ) from exc
