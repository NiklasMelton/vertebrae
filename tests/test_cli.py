import json
import pickle

import numpy as np
import pytest

from vertebrae import BenchmarkDataset
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.cli import main
from vertebrae.extractors import MultiOutputExtractor, PrecomputedExtractor
from vertebrae.extractors.base import EmbeddingOutputSpec


def _multi_output_transform(batch):
    values = np.asarray(batch)
    return {
        "left": values[:, :2],
        "right": values[:, 1:3],
    }


def _multimodal_multi_output_transform(batch):
    return {
        "image_branch": np.asarray([[len(item)] for item in batch["image"]], dtype=float),
        "fused": np.asarray(
            [[len(image), len(text)] for image, text in zip(batch["image"], batch["caption"])],
            dtype=float,
        ),
    }


def test_cli_plan_embed_merge_score_workflow(tmp_path, capsys, fake_overlapindex):
    dataset_path, extractor_path = _write_pickled_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    plan_path = tmp_path / "plan.json"

    assert (
        main(
            [
                "plan",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "2",
                "--batch-size",
                "2",
                "--output-json",
                str(plan_path),
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["total_shards"] == 2
    assert len(plan["shard_jobs"]) == 2

    for shard_index in range(2):
        assert (
            main(
                [
                    "embed-shard",
                    "--dataset-pickle",
                    str(dataset_path),
                    "--extractor-pickle",
                    str(extractor_path),
                    "--cache-dir",
                    str(cache_dir),
                    "--total-shards",
                    "2",
                    "--shard-index",
                    str(shard_index),
                    "--batch-size",
                    "2",
                ]
            )
            == 0
        )
        manifest = json.loads(capsys.readouterr().out)
        assert manifest["shard"]["shard_index"] == shard_index

    assert (
        main(
            [
                "merge-embeddings",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
            ]
        )
        == 0
    )
    merged_manifest = json.loads(capsys.readouterr().out)
    embeddings = LocalArtifactStore(str(cache_dir)).get_array(merged_manifest["output_key"])
    assert np.array_equal(embeddings, np.arange(24).reshape(8, 3))

    assert (
        main(
            [
                "write-labels",
                "--dataset-pickle",
                str(dataset_path),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    label_manifest = json.loads(capsys.readouterr().out)
    assert label_manifest["artifact_type"] == "labels"

    assert (
        main(
            [
                "score",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
            ]
        )
        == 0
    )
    score = json.loads(capsys.readouterr().out)
    assert score["artifact_type"] == "overlap_score"
    assert score["score"]["macro_score"] == 0.8

    repeat_plan_path = tmp_path / "score_repeats.json"
    assert (
        main(
            [
                "score-repeats",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
                "--seed",
                "3",
                "--seed",
                "5",
                "--output-json",
                str(repeat_plan_path),
            ]
        )
        == 0
    )
    repeat_plan = json.loads(repeat_plan_path.read_text(encoding="utf-8"))
    assert repeat_plan["seeds"] == [3, 5]
    assert repeat_plan["backend"] == "local"

    assert (
        main(
            [
                "collect-scores",
                "--cache-dir",
                str(cache_dir),
                "--score-plan-json",
                str(repeat_plan_path),
                "--output-key",
                f'{merged_manifest["output_key"]}/scores/stability',
            ]
        )
        == 0
    )
    collection = json.loads(capsys.readouterr().out)
    assert collection["artifact_type"] == "score_collection"

    result_json = tmp_path / "artifact_result.json"
    result_md = tmp_path / "artifact_result.md"
    assert (
        main(
            [
                "benchmark-from-artifacts",
                "--cache-dir",
                str(cache_dir),
                "--score-key",
                repeat_plan["score_keys"][0],
                "--stability-key",
                collection["output_key"],
                "--json-output",
                str(result_json),
                "--markdown-output",
                str(result_md),
            ]
        )
        == 0
    )
    benchmark_payload = json.loads(capsys.readouterr().out)
    assert benchmark_payload["metadata"]["distributed_artifacts"] is True
    assert result_json.exists()
    assert result_md.exists()


def test_cli_diagnose_complexity_and_benchmark_from_artifacts(
    tmp_path,
    capsys,
    fake_overlapindex,
):
    dataset_path, extractor_path = _write_pickled_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    plan_path = tmp_path / "plan.json"

    assert (
        main(
            [
                "plan",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "1",
                "--output-json",
                str(plan_path),
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert (
        main(
            [
                "embed-shard",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "1",
                "--shard-index",
                "0",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "merge-embeddings",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
            ]
        )
        == 0
    )
    merged = json.loads(capsys.readouterr().out)

    assert (
        main(
            [
                "write-labels",
                "--dataset-pickle",
                str(dataset_path),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "score",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
            ]
        )
        == 0
    )
    score = json.loads(capsys.readouterr().out)

    assert (
        main(
            [
                "diagnose-complexity",
                "--cache-dir",
                str(cache_dir),
                "--embedding-key",
                merged["output_key"],
                "--labels-key",
                plan["labels_key"],
                "--score-key",
                score["output_key"],
            ]
        )
        == 0
    )
    diagnostic = json.loads(capsys.readouterr().out)
    assert diagnostic["artifact_type"] == "separatix_diagnostic"
    assert diagnostic["diagnostic"]["ran"] is True

    assert (
        main(
            [
                "benchmark-from-artifacts",
                "--cache-dir",
                str(cache_dir),
                "--score-key",
                score["output_key"],
                "--separatix-key",
                diagnostic["output_key"],
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["extractor_results"][0]["separatix"]["recommendation"] == (
        "smooth_nonlinear_recommended"
    )


def test_cli_slurm_array_generates_embed_and_merge_commands(tmp_path):
    dataset_path, extractor_path = _write_pickled_inputs(tmp_path)
    script_path = tmp_path / "vertebrae_embed.sbatch"

    assert (
        main(
            [
                "slurm-array",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--total-shards",
                "3",
                "--batch-size",
                "4",
                "--script-output",
                str(script_path),
                "--job-name",
                "vt-test",
                "--time",
                "00:10:00",
                "--mem",
                "2G",
                "--cpus-per-task",
                "2",
                "--partition",
                "gpu",
                "--python-executable",
                "python",
            ]
        )
        == 0
    )

    script = script_path.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=vt-test" in script
    assert "#SBATCH --array=0-2" in script
    assert "#SBATCH --partition=gpu" in script
    assert "python -m vertebrae.cli embed-shard" in script
    assert "--shard-index ${SLURM_ARRAY_TASK_ID}" in script
    assert "merge-embeddings" in script


def test_cli_plan_embed_merge_multi_output_workflow(tmp_path, capsys, fake_overlapindex):
    dataset_path, extractor_path = _write_pickled_multi_output_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    plan_path = tmp_path / "plan.json"

    assert (
        main(
            [
                "plan",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "2",
                "--batch-size",
                "2",
                "--output-json",
                str(plan_path),
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [output["name"] for output in plan["outputs"]] == ["left", "right"]

    for shard_index in range(2):
        assert (
            main(
                [
                    "embed-shard",
                    "--dataset-pickle",
                    str(dataset_path),
                    "--extractor-pickle",
                    str(extractor_path),
                    "--cache-dir",
                    str(cache_dir),
                    "--total-shards",
                    "2",
                    "--shard-index",
                    str(shard_index),
                    "--batch-size",
                    "2",
                ]
            )
            == 0
        )
        manifest = json.loads(capsys.readouterr().out)
        assert manifest["artifact_type"] == "multi_output_embedding_shard"

    assert (
        main(
            [
                "merge-embeddings",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
            ]
        )
        == 0
    )
    merged_manifest = json.loads(capsys.readouterr().out)
    assert merged_manifest["artifact_type"] == "multi_output_embedding"
    left = LocalArtifactStore(str(cache_dir)).get_array(merged_manifest["outputs"][0]["output_key"])
    right = LocalArtifactStore(str(cache_dir)).get_array(
        merged_manifest["outputs"][1]["output_key"]
    )
    assert np.array_equal(left, np.arange(24).reshape(8, 3)[:, :2])
    assert np.array_equal(right, np.arange(24).reshape(8, 3)[:, 1:3])

    assert (
        main(
            [
                "write-labels",
                "--dataset-pickle",
                str(dataset_path),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "score",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
                "--embedding-key",
                plan["outputs"][0]["output_key"],
            ]
        )
        == 0
    )
    score = json.loads(capsys.readouterr().out)
    assert score["embedding_key"] == plan["outputs"][0]["output_key"]

    with pytest.raises(ValueError, match="multiple embedding outputs"):
        main(
            [
                "score",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
            ]
        )


def test_cli_plan_multimodal_multi_output_workflow(tmp_path, capsys):
    dataset_path, extractor_path = _write_pickled_multimodal_multi_output_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    plan_path = tmp_path / "plan.json"

    assert (
        main(
            [
                "plan",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "2",
                "--batch-size",
                "1",
                "--output-json",
                str(plan_path),
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [output["name"] for output in plan["outputs"]] == ["image_branch", "fused"]

    for shard_index in range(2):
        assert (
            main(
                [
                    "embed-shard",
                    "--dataset-pickle",
                    str(dataset_path),
                    "--extractor-pickle",
                    str(extractor_path),
                    "--cache-dir",
                    str(cache_dir),
                    "--total-shards",
                    "2",
                    "--shard-index",
                    str(shard_index),
                    "--batch-size",
                    "1",
                ]
            )
            == 0
        )
        manifest = json.loads(capsys.readouterr().out)
        assert manifest["artifact_type"] == "multi_output_embedding_shard"

    assert (
        main(
            [
                "merge-embeddings",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
            ]
        )
        == 0
    )
    merged = json.loads(capsys.readouterr().out)
    store = LocalArtifactStore(str(cache_dir))
    assert store.get_array(merged["outputs"][0]["output_key"]).tolist() == [
        [5.0],
        [5.0],
        [5.0],
        [5.0],
    ]
    assert store.get_array(merged["outputs"][1]["output_key"]).tolist() == [
        [5.0, 3.0],
        [5.0, 3.0],
        [5.0, 5.0],
        [5.0, 4.0],
    ]


def test_cli_compress_supports_prefix_truncate(tmp_path, capsys):
    dataset_path, extractor_path = _write_pickled_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    plan_path = tmp_path / "plan.json"

    assert (
        main(
            [
                "plan",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "1",
                "--output-json",
                str(plan_path),
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "embed-shard",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "1",
                "--shard-index",
                "0",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "merge-embeddings",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
            ]
        )
        == 0
    )
    merged_manifest = json.loads(capsys.readouterr().out)

    assert (
        main(
            [
                "compress",
                "--cache-dir",
                str(cache_dir),
                "--embedding-key",
                merged_manifest["output_key"],
                "--method",
                "prefix_truncate",
                "--n-components",
                "2",
                "--assume-matryoshka",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_type"] == "compressed_embedding"
    assert payload["compression_metadata"]["method"] == "prefix_truncate"
    assert payload["compression_metadata"]["compressed_dim"] == 2


def test_cli_scores_multilabel_dataset_from_artifacts(tmp_path, capsys, fake_overlapindex):
    dataset_path, extractor_path = _write_pickled_multilabel_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    plan_path = tmp_path / "plan.json"

    assert (
        main(
            [
                "plan",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "1",
                "--output-json",
                str(plan_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "embed-shard",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "1",
                "--shard-index",
                "0",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "merge-embeddings",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "write-labels",
                "--dataset-pickle",
                str(dataset_path),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    label_manifest = json.loads(capsys.readouterr().out)

    assert main(["score", "--cache-dir", str(cache_dir), "--plan-json", str(plan_path)]) == 0
    score = json.loads(capsys.readouterr().out)

    assert label_manifest["target_type"] == "multi_label"
    assert score["score"]["metadata"]["target_type"] == "multi_label"
    assert fake_overlapindex.calls[-1]["fit_y_shape"] == [12, 3]


def test_cli_slurm_score_array_generates_repeat_score_commands(tmp_path):
    dataset_path, extractor_path = _write_pickled_inputs(tmp_path)
    plan_path = tmp_path / "plan.json"
    script_path = tmp_path / "vertebrae_score.sbatch"
    cache_dir = tmp_path / "cache"

    assert (
        main(
            [
                "plan",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "2",
                "--output-json",
                str(plan_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "slurm-score-array",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
                "--repeats",
                "3",
                "--random-state",
                "7",
                "--script-output",
                str(script_path),
                "--python-executable",
                "python",
            ]
        )
        == 0
    )

    script = script_path.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2" in script
    assert "python -m vertebrae.cli score" in script
    assert "--seed ${SEED}" in script
    assert "collect-scores" in script


def test_cli_run_embedding_shards_local_backend(tmp_path, capsys):
    dataset_path, extractor_path = _write_pickled_inputs(tmp_path)
    cache_dir = tmp_path / "cache"

    assert (
        main(
            [
                "run-embedding-shards",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "2",
                "--batch-size",
                "2",
                "--backend",
                "local",
                "--n-jobs",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_type"] == "embedding_shard_plan"
    assert payload["backend"] == "local"
    assert payload["n_shards"] == 2


def test_cli_score_repeats_ray_missing_dependency(tmp_path):
    dataset_path, extractor_path = _write_pickled_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    plan_path = tmp_path / "plan.json"

    main(
        [
            "plan",
            "--dataset-pickle",
            str(dataset_path),
            "--extractor-pickle",
            str(extractor_path),
            "--cache-dir",
            str(cache_dir),
            "--total-shards",
            "2",
            "--output-json",
            str(plan_path),
        ]
    )

    with pytest.raises(ImportError, match="optional 'ray' extra"):
        main(
            [
                "score-repeats",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
                "--seed",
                "3",
                "--backend",
                "ray",
            ]
        )


def test_cli_score_repeats_dask_missing_dependency(tmp_path):
    dataset_path, extractor_path = _write_pickled_inputs(tmp_path)
    cache_dir = tmp_path / "cache"
    plan_path = tmp_path / "plan.json"

    main(
        [
            "plan",
            "--dataset-pickle",
            str(dataset_path),
            "--extractor-pickle",
            str(extractor_path),
            "--cache-dir",
            str(cache_dir),
            "--total-shards",
            "2",
            "--output-json",
            str(plan_path),
        ]
    )

    with pytest.raises(ImportError, match="optional 'dask' extra"):
        main(
            [
                "score-repeats",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
                "--seed",
                "3",
                "--backend",
                "dask",
            ]
        )


def _write_pickled_inputs(tmp_path):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
    )
    extractor = PrecomputedExtractor()
    dataset_path = tmp_path / "dataset.pkl"
    extractor_path = tmp_path / "extractor.pkl"
    with dataset_path.open("wb") as f:
        pickle.dump(dataset, f)
    with extractor_path.open("wb") as f:
        pickle.dump(extractor, f)
    return dataset_path, extractor_path


def _write_pickled_multi_output_inputs(tmp_path):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
    )
    extractor = MultiOutputExtractor(
        name="multi",
        output_specs=[EmbeddingOutputSpec("left"), EmbeddingOutputSpec("right")],
        transform_many_fn=_multi_output_transform,
        modality="tabular",
        streaming_safe=True,
    )
    dataset_path = tmp_path / "dataset.pkl"
    extractor_path = tmp_path / "extractor.pkl"
    with dataset_path.open("wb") as f:
        pickle.dump(dataset, f)
    with extractor_path.open("wb") as f:
        pickle.dump(extractor, f)
    return dataset_path, extractor_path


def _write_pickled_multilabel_inputs(tmp_path):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(36).reshape(12, 3),
        [
            ("red", "round"),
            ("red",),
            ("round",),
            ("red", "sweet"),
            ("round", "sweet"),
            ("sweet",),
            ("red", "round"),
            ("red",),
            ("round",),
            ("red", "sweet"),
            ("round", "sweet"),
            ("sweet",),
        ],
    )
    extractor = PrecomputedExtractor()
    dataset_path = tmp_path / "multilabel_dataset.pkl"
    extractor_path = tmp_path / "multilabel_extractor.pkl"
    with dataset_path.open("wb") as f:
        pickle.dump(dataset, f)
    with extractor_path.open("wb") as f:
        pickle.dump(extractor, f)
    return dataset_path, extractor_path


def _write_pickled_multimodal_multi_output_inputs(tmp_path):
    dataset = BenchmarkDataset.from_multimodal(
        inputs={
            "image": ["a.png", "b.png", "c.png", "d.png"],
            "caption": ["one", "two", "three", "four"],
        },
        labels=["a", "a", "b", "b"],
        modalities={"image": "image", "caption": "text"},
    )
    extractor = MultiOutputExtractor(
        name="fusion",
        output_specs=[EmbeddingOutputSpec("image_branch"), EmbeddingOutputSpec("fused")],
        transform_many_fn=_multimodal_multi_output_transform,
        modality="multimodal",
        streaming_safe=True,
    )
    dataset_path = tmp_path / "multimodal_dataset.pkl"
    extractor_path = tmp_path / "multimodal_extractor.pkl"
    with dataset_path.open("wb") as f:
        pickle.dump(dataset, f)
    with extractor_path.open("wb") as f:
        pickle.dump(extractor, f)
    return dataset_path, extractor_path
