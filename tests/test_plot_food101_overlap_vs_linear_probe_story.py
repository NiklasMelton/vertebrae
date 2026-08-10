"""Network-free contract tests for the Food-101 story plot reader."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "plot_food101_overlap_vs_linear_probe_story.py"


@pytest.fixture(scope="module")
def plotter():
    module_name = "plot_food101_overlap_vs_linear_probe_story_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _summary_payload():
    """Build a small self-contained summary fixture for the reader tests."""

    methods = ("overlap_cross_fitted", "linear_probe_oof")
    arms = ("baseline", "nonlinearity_full", "nuisance_full")
    heads = ("linear", "knn", "quadratic", "rbf")
    quadratic_auc = {
        method: {
            arm: [
                0.50 + 0.01 * replicate + 0.02 * methods.index(method) + 0.03 * arms.index(arm)
                for replicate in range(5)
            ]
            for arm in arms
        }
        for method in methods
    }
    head_auc = {
        method: {
            head: [
                0.55 + 0.01 * replicate + 0.02 * methods.index(method) + 0.015 * heads.index(head)
                for replicate in range(5)
            ]
            for head in heads
        }
        for method in methods
    }
    quadratic_means = {
        method: {arm: float(np.mean(values)) for arm, values in arms_by_method.items()}
        for method, arms_by_method in quadratic_auc.items()
    }
    head_difference_means = {
        head: float(
            np.mean(np.asarray(head_auc[methods[0]][head]) - np.asarray(head_auc[methods[1]][head]))
        )
        for head in heads
    }
    runtime = {
        method: [1.0 + 0.2 * index + 0.1 * methods.index(method) for index in range(5)]
        for method in methods
    }
    overlap_mean = float(np.mean(runtime[methods[0]]))
    probe_mean = float(np.mean(runtime[methods[1]]))
    return {
        "format": "food101_overlap_vs_linear_probe_story_summary",
        "plot_data": {
            "quadratic_auc": quadratic_auc,
            "head_auc": head_auc,
            "quadratic_means": quadratic_means,
            "head_difference_means": head_difference_means,
            "selector_runtime_minutes": runtime,
        },
        "bootstrap": {"direct_nonlinear_oi_minus_probe": head_difference_means["quadratic"]},
        "headline_metrics": {
            "selector_runtime": {
                "overlap_cross_fitted_mean_minutes": overlap_mean,
                "linear_probe_oof_mean_minutes": probe_mean,
                "overlap_speedup": probe_mean / overlap_mean,
                "overlap_time_reduction_fraction": 1.0 - overlap_mean / probe_mean,
            }
        },
    }


def _write_summary(tmp_path, payload):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summary_fixture_loads(plotter, tmp_path):
    summary_path = _write_summary(tmp_path, _summary_payload())
    rows = plotter._load_auc_rows(summary_path)
    runtime = plotter._load_runtime_minutes(summary_path)

    assert len(rows) == 60
    assert {row["method"] for row in rows} == {
        "overlap_cross_fitted",
        "linear_probe_oof",
    }
    assert {row["arm"] for row in rows} == {
        "baseline",
        "nonlinearity_full",
        "nuisance_full",
    }
    assert {row["head"] for row in rows} == {"linear", "knn", "quadratic", "rbf"}
    assert set(runtime) == {"overlap_cross_fitted", "linear_probe_oof"}
    assert all(values.shape == (5,) for values in runtime.values())


def test_summary_has_complete_five_replicate_cells_for_both_panels(plotter, tmp_path):
    rows = plotter._load_auc_rows(_write_summary(tmp_path, _summary_payload()))

    for method in ("overlap_cross_fitted", "linear_probe_oof"):
        for arm in ("baseline", "nonlinearity_full", "nuisance_full"):
            values = plotter._values(rows, head="quadratic", arm=arm, method=method)
            assert values.shape == (5,)
            assert np.isfinite(values).all()

        for head in ("linear", "knn", "quadratic", "rbf"):
            values = plotter._values(
                rows,
                head=head,
                arm="nonlinearity_full",
                method=method,
            )
            assert values.shape == (5,)
            assert np.isfinite(values).all()


def test_plot_means_match_summary_and_headline_metrics(plotter, tmp_path):
    payload = _summary_payload()
    summary_path = _write_summary(tmp_path, payload)
    rows = plotter._load_auc_rows(summary_path)
    plot_data = payload["plot_data"]

    for method, arms in plot_data["quadratic_means"].items():
        for arm, expected in arms.items():
            values = plotter._values(rows, head="quadratic", arm=arm, method=method)
            assert float(values.mean()) == pytest.approx(expected)

    for head, expected in plot_data["head_difference_means"].items():
        overlap_values = plotter._values(
            rows,
            head=head,
            arm="nonlinearity_full",
            method="overlap_cross_fitted",
        )
        probe_values = plotter._values(
            rows,
            head=head,
            arm="nonlinearity_full",
            method="linear_probe_oof",
        )
        assert float(np.mean(overlap_values - probe_values)) == pytest.approx(expected)

    assert plot_data["head_difference_means"]["quadratic"] == pytest.approx(
        payload["bootstrap"]["direct_nonlinear_oi_minus_probe"]
    )

    runtime = plotter._load_runtime_minutes(summary_path)
    runtime_headline = payload["headline_metrics"]["selector_runtime"]
    overlap_mean = float(runtime["overlap_cross_fitted"].mean())
    probe_mean = float(runtime["linear_probe_oof"].mean())
    assert overlap_mean == pytest.approx(runtime_headline["overlap_cross_fitted_mean_minutes"])
    assert probe_mean == pytest.approx(runtime_headline["linear_probe_oof_mean_minutes"])
    assert probe_mean / overlap_mean == pytest.approx(runtime_headline["overlap_speedup"])
    assert 1.0 - overlap_mean / probe_mean == pytest.approx(
        runtime_headline["overlap_time_reduction_fraction"]
    )


def test_malformed_and_incomplete_summaries_are_rejected(plotter, tmp_path):
    malformed = {"format": "food101_overlap_vs_linear_probe_story_summary"}
    with pytest.raises(ValueError):
        plotter._load_auc_rows(_write_summary(tmp_path, malformed))

    incomplete = copy.deepcopy(_summary_payload())
    incomplete["plot_data"]["quadratic_auc"]["overlap_cross_fitted"]["baseline"].pop()
    incomplete_path = _write_summary(tmp_path, incomplete)
    with pytest.raises(ValueError):
        rows = plotter._load_auc_rows(incomplete_path)
        plotter._values(
            rows,
            head="quadratic",
            arm="baseline",
            method="overlap_cross_fitted",
        )

    missing_runtime = copy.deepcopy(_summary_payload())
    del missing_runtime["plot_data"]["selector_runtime_minutes"]
    with pytest.raises(ValueError, match="runtime"):
        plotter._load_runtime_minutes(_write_summary(tmp_path, missing_runtime))


def test_help_is_lazy_and_does_not_render(plotter):
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--results" in completed.stdout
    assert "--output-prefix" in completed.stdout
    assert "Traceback" not in completed.stderr
    assert "Wrote" not in completed.stdout
