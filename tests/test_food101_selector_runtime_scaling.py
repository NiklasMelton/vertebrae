"""Network-free tests for the Food-101 selector runtime diagnostic."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "examples" / "food101_selector_runtime_scaling.py"


@pytest.fixture(scope="module")
def driver():
    module_name = "food101_selector_runtime_scaling_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_nested_indices_are_deterministic_and_class_balanced(driver):
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 12)
    first = driver.build_nested_indices(labels, (4, 8), seed=7)
    second = driver.build_nested_indices(labels, (4, 8), seed=7)

    assert set(first) == {4, 8}
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert set(first[4]).issubset(set(first[8]))
    for indices in first.values():
        values, counts = np.unique(labels[indices], return_counts=True)
        expected_count = int(len(indices) / 3)
        assert dict(zip(values.tolist(), counts.tolist())) == {
            "a": expected_count,
            "b": expected_count,
            "c": expected_count,
        }


def test_nested_indices_reject_insufficient_class_support(driver):
    labels = np.repeat(np.asarray(["a", "b"], dtype=object), 3)
    with pytest.raises(ValueError, match="exceeds rows"):
        driver.build_nested_indices(labels, (4,), seed=1)


def test_benchmark_rows_keep_scores_fixed_across_timing_repeats(driver, monkeypatch):
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 12)
    matrix = np.arange(len(labels) * 4, dtype=np.float32).reshape(len(labels), 4)
    calls: list[tuple[str, int]] = []

    def fake_overlap(matrix, labels, *, seed, folds, k):
        calls.append(("overlap_cross_fitted", len(labels)))
        return 0.25

    def fake_probe(matrix, labels, *, seed, folds):
        calls.append(("linear_probe_oof", len(labels)))
        return 0.75

    monkeypatch.setattr(driver, "_score_overlap", fake_overlap)
    monkeypatch.setattr(driver, "_score_probe", fake_probe)
    rows = driver.benchmark_embedding_matrix(
        matrix,
        labels,
        model="toy",
        budgets=(4, 8),
        timing_repeats=3,
        seed=3,
    )

    assert len(rows) == 12
    assert {row["score"] for row in rows if row["method"] == "overlap_cross_fitted"} == {0.25}
    assert {row["score"] for row in rows if row["method"] == "linear_probe_oof"} == {0.75}
    assert all(np.isfinite(row["ratio"]) and row["ratio"] > 0.0 for row in rows)
    # One warm-up call per method at the lowest budget, then one timed call per
    # method/budget/repeat.
    assert len(calls) == 2 + 2 * 2 * 3
    by_budget = {
        budget: {
            row["method"]: row["order_position"]
            for row in rows
            if row["repeat"] == 0 and row["budget"] == budget
        }
        for budget in (4, 8)
    }
    assert by_budget[4] != by_budget[8]


def test_paired_summary_rejects_missing_or_duplicate_cells(driver):
    rows = [
        {
            "backbone": "toy",
            "samples_per_class": 2,
            "repeat": 0,
            "method": method,
            "elapsed_seconds": 1.0,
            "process_seconds": 1.0,
            "score": 0.5,
        }
        for method in driver._METHODS
    ]
    summary, paired = driver._paired_summary(rows)
    assert len(summary) == 2
    assert len(paired) == 1
    with pytest.raises(ValueError, match="incomplete"):
        driver._paired_summary(rows[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        driver._paired_summary(rows + [dict(rows[0])])


def test_canonical_cohort_validation_and_restriction(driver, tmp_path):
    ids = [
        f"food101/train/class-{class_index}/{row:05d}"
        for class_index in range(40)
        for row in range(660)
    ] + [
        f"food101/test/class-{class_index}/{row:05d}"
        for class_index in range(40)
        for row in range(52)
    ]
    cohort = {
        "study": "food101_nonlinear_backbone_bridge",
        "extracted_train_rows": 26400,
        "extracted_test_rows": 2080,
        "extracted_sample_ids": ids,
    }
    labels, sample_ids, train_rows = driver._load_discovered_cohort(
        tmp_path / "cohort.json", cohort
    )
    assert len(labels) == 28480
    assert train_rows == 26400
    matrix = np.zeros((28480, 2), dtype=np.float32)
    manifest = driver.EmbeddingManifest(
        "toy",
        "final",
        tmp_path / "toy.npy",
        matrix,
        {
            "sample_ids_sha256": __import__("hashlib").sha256(
                json.dumps(sample_ids).encode()
            ).hexdigest(),
            "labels_sha256": driver._labels_sha256(labels.tolist()),
        },
    )
    restricted = driver._restrict_discovered_embeddings(
        {"toy": manifest},
        labels=labels,
        sample_ids=sample_ids,
        train_rows=train_rows,
    )
    assert restricted["toy"].shape == (26400, 2)


def test_manifest_relative_path_and_hash_mismatch_are_rejected(driver, tmp_path):
    matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
    array_path = tmp_path / "toy.npy"
    np.save(array_path, matrix)
    manifest_path = tmp_path / "toy.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model": "toy",
                "path": array_path.name,
                "shape": [6, 2],
                "row_count": 6,
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    assert driver.load_embedding_manifest(manifest_path).shape == (6, 2)
    bad = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        driver.load_embedding_manifest(manifest_path)


def test_manifest_identity_and_result_reader_are_strict(driver, tmp_path):
    matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
    array_path = tmp_path / "toy.npy"
    np.save(array_path, matrix)
    labels = ["a", "a", "b", "b", "c", "c"]
    manifest_path = tmp_path / "toy.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model": "toy",
                "output": "final",
                "path": str(array_path),
                "shape": [6, 2],
                "row_count": 6,
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    loaded = driver.load_embedding_manifest(manifest_path)
    assert loaded.model == "toy"
    assert loaded.shape == (6, 2)

    labels_array = np.asarray(labels, dtype=object)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(driver, "_score_overlap", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(driver, "_score_probe", lambda *args, **kwargs: 0.4)
    try:
        payload = driver.run_runtime_scaling(
            {"toy": matrix},
            labels_array,
            budgets=(2,),
            timing_repeats=1,
            configuration={"models": ["toy"], "budgets": [2], "timing_repeats": 1},
        )
    finally:
        monkeypatch.undo()
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(driver._read_results(result_path)["rows"]) == 2

    malformed = dict(payload)
    malformed["claim_supported"] = True
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="claim_supported"):
        driver._read_results(malformed_path)


def test_result_reader_rejects_tampered_configuration_and_stem(driver, tmp_path):
    labels = np.repeat(np.asarray(["a", "b"], dtype=object), 4)
    matrix = np.ones((8, 2), dtype=np.float32)
    patch = pytest.MonkeyPatch()
    patch.setattr(driver, "_score_overlap", lambda *args, **kwargs: 0.2)
    patch.setattr(driver, "_score_probe", lambda *args, **kwargs: 0.4)
    try:
        payload = driver.run_runtime_scaling(
            {"toy": matrix}, labels, budgets=(2,), timing_repeats=1
        )
    finally:
        patch.undo()
    path = tmp_path / "result.json"
    payload["configuration"]["seed"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        driver._read_results(path)
    payload["configuration"]["seed"] = 42
    payload["artifact_stem"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_stem"):
        driver._read_results(path)


def test_resume_reuses_per_model_checkpoints_and_refuses_overwrite(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_score_overlap", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(driver, "_score_probe", lambda *args, **kwargs: 0.4)
    args = driver.build_parser().parse_args(
        [
            "--synthetic-smoke",
            "--models",
            "synthetic-a",
            "--budgets",
            "8",
            "--timing-repeats",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert driver._run(args) == 0
    with pytest.raises(ValueError, match="already exists"):
        driver._run(args)
    args.resume = True
    assert driver._run(args) == 0
    assert list(tmp_path.glob("*.checkpoints/synthetic-a.json"))


def test_help_is_lazy(driver):
    parser = driver.build_parser()
    assert "--embedding-manifest" in parser.format_help()
    assert "--synthetic-smoke" in parser.format_help()
