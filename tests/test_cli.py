import json
import pickle

import numpy as np

from vertebrae import BenchmarkDataset
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.cli import main
from vertebrae.extractors import PrecomputedExtractor


def test_cli_plan_embed_merge_workflow(tmp_path, capsys):
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
