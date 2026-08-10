"""Plot the post-hoc Food-101 selector runtime-scaling benchmark.

The input is a completed ``food101_selector_runtime_scaling`` JSON artifact.
This figure is a computational diagnostic only: it does not use, summarize, or
support the Food-101 accuracy claim.  Elapsed times are scoring-call times from
paired OverlapIndex and out-of-fold probe cells; extraction and other setup work
must not be inferred from this plot.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

_METHODS = ("overlap_cross_fitted", "linear_probe_oof")
_METHOD_LABELS = {
    "overlap_cross_fitted": "OverlapIndex (5-fold cross-fitted)",
    "linear_probe_oof": "L2 probe (5-fold OOF)",
}
_COLORS = {
    "overlap_cross_fitted": "#0072B2",
    "linear_probe_oof": "#D55E00",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Completed runtime-scaling JSON artifact (auto-discovered when omitted).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("img/visuals/food101-selector-runtime-scaling"),
        help="Output path without an extension; PNG and SVG are written.",
    )
    return parser


def _rows_from_payload(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        rows = payload.get("runtime_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The runtime-scaling artifact does not contain rows.")
    return rows


def _load_rows(path: Path) -> Sequence[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    study = payload.get("study", payload.get("format"))
    if study != "food101_selector_runtime_scaling":
        raise ValueError("The result artifact has the wrong runtime-scaling study identity.")
    if payload.get("artifact_status") != "completed":
        raise ValueError("The runtime-scaling result artifact must be completed.")
    if payload.get("post_hoc_runtime_benchmark") is not True:
        raise ValueError("The artifact is not marked as a post-hoc runtime benchmark.")
    protocol = payload.get("protocol")
    if isinstance(protocol, Mapping):
        if protocol.get("post_hoc_runtime_benchmark") is not True:
            raise ValueError("The artifact is not marked as a post-hoc runtime benchmark.")
        if protocol.get("claim_supported") is not False:
            raise ValueError("Runtime scaling artifacts must have claim_supported=false.")
    else:
        raise ValueError("The runtime-scaling artifact is missing protocol metadata.")
    if payload.get("claim_supported") is not False:
        raise ValueError("Runtime scaling artifacts must have claim_supported=false.")

    resolved: list[Dict[str, Any]] = []
    for raw in _rows_from_payload(payload):
        if not isinstance(raw, Mapping):
            raise ValueError("Runtime rows must be JSON objects.")
        method = str(raw.get("method", ""))
        if method not in _METHODS:
            raise ValueError(f"Unknown runtime method: {method!r}.")
        backbone = raw.get("backbone", raw.get("model"))
        if backbone is None or not str(backbone):
            raise ValueError("Runtime rows must identify a backbone.")
        budget = raw.get("samples_per_class", raw.get("budget"))
        if budget is None:
            budget = raw.get("n_samples_per_class")
        try:
            budget_value = int(budget)
            repeat = int(raw.get("repeat", raw.get("replicate")))
            elapsed = float(raw.get("elapsed_seconds", raw.get("seconds")))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Runtime rows need integer budget/repeat and elapsed_seconds."
            ) from exc
        if budget_value < 1 or repeat < 0 or not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(
                "Runtime budgets/repeats must be valid and elapsed times non-negative."
            )
        resolved.append(
            {
                "backbone": str(backbone),
                "samples_per_class": budget_value,
                "repeat": repeat,
                "method": method,
                "elapsed_seconds": elapsed,
            }
        )
    _validate_pairs(resolved)
    return resolved


def _validate_pairs(rows: Sequence[Mapping[str, Any]]) -> None:
    cells: Dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row["backbone"]), int(row["samples_per_class"]), int(row["repeat"]))
        method = str(row["method"])
        if method in cells[key]:
            raise ValueError(f"Duplicate runtime cell for {key!r}, method={method!r}.")
        cells[key].add(method)
    if not cells:
        raise ValueError("The runtime-scaling artifact has no paired cells.")
    incomplete = [key for key, methods in cells.items() if methods != set(_METHODS)]
    if incomplete:
        raise ValueError(f"Runtime rows are not paired for cells: {incomplete[:3]!r}.")


def _paired_values(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, np.ndarray]]:
    """Return one paired value per backbone/repeat cell for each budget."""

    grouped: Dict[int, Dict[str, list[float]]] = {
        int(budget): {method: [] for method in _METHODS}
        for budget in sorted({int(row["samples_per_class"]) for row in rows})
    }
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["samples_per_class"]),
            str(row["backbone"]),
            int(row["repeat"]),
            str(row["method"]),
        ),
    )
    by_cell: Dict[tuple[str, int, int], Dict[str, float]] = defaultdict(dict)
    for row in ordered:
        key = (str(row["backbone"]), int(row["samples_per_class"]), int(row["repeat"]))
        by_cell[key][str(row["method"])] = float(row["elapsed_seconds"])
    for (_, budget, _), values in sorted(by_cell.items(), key=str):
        for method in _METHODS:
            grouped[budget][method].append(values[method])
    return {
        budget: {method: np.asarray(values, dtype=float) for method, values in methods.items()}
        for budget, methods in grouped.items()
    }


def _mean_sd(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Runtime values must be a non-empty finite vector.")
    return float(values.mean()), float(values.std(ddof=1) if values.size > 1 else 0.0)


def _speedup_summary(
    paired: Mapping[int, Mapping[str, np.ndarray]], budgets: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median paired speedup and its 25th/75th percentile bounds."""

    cells = [
        paired[int(budget)]["linear_probe_oof"]
        / np.maximum(
            paired[int(budget)]["overlap_cross_fitted"], np.finfo(float).eps
        )
        for budget in budgets
    ]
    medians = np.asarray([float(np.median(values)) for values in cells])
    lower = np.asarray([float(np.percentile(values, 25.0)) for values in cells])
    upper = np.asarray([float(np.percentile(values, 75.0)) for values in cells])
    return medians, lower, upper


def _render(rows: Sequence[Mapping[str, Any]], output_prefix: Path) -> Sequence[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting environment
        raise ImportError("Install matplotlib to render the runtime-scaling figure.") from exc

    paired = _paired_values(rows)
    budgets = np.asarray(sorted(paired), dtype=float)
    neutral = "#4C566A"
    light_neutral = "#D8DEE9"
    figure, (runtime_axis, speedup_axis) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.1),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    figure.patch.set_facecolor("white")

    summaries: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in _METHODS:
        values = [paired[int(budget)][method] for budget in budgets]
        means = np.asarray([_mean_sd(value)[0] for value in values])
        standard_deviations = np.asarray([_mean_sd(value)[1] for value in values])
        summaries[method] = (means, standard_deviations)
        color = _COLORS[method]
        # Thin cell trajectories make the paired backbone/repeat uncertainty
        # visible without turning the plot into a forest of labeled lines.
        for index, value in enumerate(values):
            runtime_axis.scatter(
                np.full(value.size, budgets[index]),
                value,
                s=9,
                color=color,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
        runtime_axis.plot(
            budgets,
            means,
            color=color,
            marker="o",
            linewidth=2.4,
            markersize=5,
            label=_METHOD_LABELS[method],
            zorder=3,
        )
        runtime_axis.fill_between(
            budgets,
            np.maximum(0.0, means - standard_deviations),
            means + standard_deviations,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

    speedup_values, speedup_q25, speedup_q75 = _speedup_summary(paired, budgets)
    speedup_axis.axhline(1.0, color=neutral, linewidth=1.0, linestyle="--", zorder=1)
    speedup_axis.errorbar(
        budgets,
        speedup_values,
        yerr=np.vstack((speedup_values - speedup_q25, speedup_q75 - speedup_values)),
        color="#009E73",
        marker="o",
        linewidth=2.2,
        markersize=5,
        capsize=4,
        label="Probe / OverlapIndex",
        zorder=3,
    )
    for budget, value in zip(budgets, speedup_values):
        speedup_axis.annotate(
            f"{value:.2f}×",
            (budget, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#007A5E",
            fontweight="bold",
        )

    runtime_axis.set_title(
        "A  Selector scoring time scales with samples/class",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=13,
    )
    runtime_axis.set_xlabel("Nested selector samples per class", fontsize=11)
    runtime_axis.set_ylabel("Elapsed scoring time per call (s)", fontsize=11)
    runtime_axis.set_xticks(budgets)
    runtime_axis.grid(axis="y", color=light_neutral, linewidth=0.8, alpha=0.8)
    runtime_axis.legend(loc="upper left", frameon=False, fontsize=9)
    runtime_axis.text(
        0.0,
        -0.22,
        "Points: paired backbone/repeat cells; ribbons: ±1 SD across cells.",
        transform=runtime_axis.transAxes,
        fontsize=9,
        color=neutral,
    )
    speedup_axis.set_title(
        "B  Relative selector speed",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=13,
    )
    speedup_axis.set_xlabel("Nested selector samples per class", fontsize=11)
    speedup_axis.set_ylabel("Probe time / OverlapIndex time (median, IQR)", fontsize=11)
    speedup_axis.set_xticks(budgets)
    speedup_axis.grid(axis="y", color=light_neutral, linewidth=0.8, alpha=0.8)
    speedup_axis.legend(loc="upper left", frameon=False, fontsize=9)
    speedup_axis.text(
        0.0,
        -0.22,
        ">1× means OverlapIndex is faster; bars show paired-cell IQR.",
        transform=speedup_axis.transAxes,
        fontsize=9,
        color=neutral,
    )

    for axis in (runtime_axis, speedup_axis):
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(light_neutral)
        axis.spines["bottom"].set_color(light_neutral)
        axis.tick_params(colors=neutral, labelsize=10)

    figure.suptitle(
        "Food-101 selector runtime scaling (post-hoc computational benchmark)",
        x=0.5,
        y=1.02,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.965,
        "Five-fold cross-fitted OverlapIndex (k=10) vs five-fold OOF L2 probe · "
        "serial scoring only",
        ha="center",
        fontsize=10,
        color=neutral,
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.24, wspace=0.29)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    figure.savefig(png_path, dpi=200, facecolor="white", bbox_inches="tight")
    figure.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return (png_path, svg_path)


def main() -> int:
    args = _parser().parse_args()
    results = args.results
    if results is None:
        candidates = sorted(
            path
            for path in Path("examples/output").glob("food101_selector_runtime_scaling_*.json")
            if not path.name.endswith((".planned.json", ".failed.json"))
        )
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one completed runtime-scaling artifact in examples/output; "
                "pass --results to disambiguate."
            )
        results = candidates[0]
    paths = _render(_load_rows(results), args.output_prefix)
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
