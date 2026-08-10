"""Network-free tests for the Food-101 runtime-scaling plot reader."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLOT_PATH = ROOT / "examples" / "plot_food101_selector_runtime_scaling.py"
SUMMARY_PATH = ROOT / "examples" / "assets" / "food101_selector_runtime_scaling_summary.json"


@pytest.fixture(scope="module")
def plotter():
    module_name = "plot_food101_selector_runtime_scaling_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, PLOT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _rows():
    return [
        {
            "backbone": backbone,
            "samples_per_class": budget,
            "repeat": repeat,
            "method": method,
            "elapsed_seconds": (
                1.0 + budget / 100.0 + repeat / 10.0
                if method == "overlap_cross_fitted"
                else 2.0 + budget / 100.0 + repeat / 10.0
            ),
        }
        for backbone in ("a", "b")
        for budget in (64, 128)
        for repeat in (0, 1)
        for method in ("overlap_cross_fitted", "linear_probe_oof")
    ]


def _write_payload(tmp_path, rows):
    payload = {
        "study": "food101_selector_runtime_scaling",
        "format": "food101_selector_runtime_scaling",
        "artifact_status": "completed",
        "post_hoc_runtime_benchmark": True,
        "claim_supported": False,
        "protocol": {"post_hoc_runtime_benchmark": True, "claim_supported": False},
        "rows": rows,
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reader_requires_completed_post_hoc_false_claim(plotter, tmp_path):
    rows = _rows()
    loaded = plotter._load_rows(_write_payload(tmp_path, rows))
    assert len(loaded) == len(rows)
    bad = json.loads(_write_payload(tmp_path, rows).read_text(encoding="utf-8"))
    bad["claim_supported"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="claim_supported"):
        plotter._load_rows(path)


def test_reader_rejects_unpaired_cells(plotter, tmp_path):
    rows = _rows()[:-1]
    with pytest.raises(ValueError, match="paired"):
        plotter._load_rows(_write_payload(tmp_path, rows))


def test_speedup_uses_paired_median_and_iqr(plotter, tmp_path):
    rows = plotter._load_rows(_write_payload(tmp_path, _rows()))
    paired = plotter._paired_values(rows)
    values, lower, upper = plotter._speedup_summary(paired, (64.0, 128.0))
    expected = float(
        np.median(
            [
                (2.0 + 64.0 / 100.0 + repeat / 10.0) / (1.0 + 64.0 / 100.0 + repeat / 10.0)
                for repeat in (0, 1)
            ]
        )
    )
    assert values[0] == pytest.approx(expected)
    assert lower[0] <= values[0] <= upper[0]
    assert np.isfinite(np.concatenate([values, lower, upper])).all()


def test_tracked_summary_matches_completed_readme_results():
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert payload["artifact_status"] == "completed"
    assert payload["post_hoc_runtime_benchmark"] is True
    assert payload["claim_supported"] is False
    assert payload["source"]["configuration_hash"].startswith("4c63a49825bf")
    assert len(payload["source"]["models"]) == 10

    by_budget = {row["samples_per_class"]: row for row in payload["by_budget"]}
    assert tuple(by_budget) == (64, 128, 256, 512, 640)
    assert by_budget[64]["paired_speedup_median"] == pytest.approx(0.69831834297)
    assert by_budget[640]["paired_speedup_median"] == pytest.approx(1.99532043322)
    assert by_budget[512]["overlap_faster_fraction"] == 1.0
    assert by_budget[640]["overlap_faster_fraction"] == 1.0
    assert payload["measurement_audit"]["timed_calls"] == 500
