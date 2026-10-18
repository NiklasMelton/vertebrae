"""Render the Food-101 overlap-vs-linear-probe story figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

_METHOD_LABELS = {
    "overlap_cross_fitted": "OverlapIndex",
    "linear_probe_oof": "Linear probe",
}
_ARM_LABELS = {
    "baseline": "Baseline",
    "nonlinearity_full": "Label-relevant\nnonlinearity",
    "nuisance_full": "Irrelevant\nnuisance",
}
_HEAD_LABELS = {
    "linear": "Linear",
    "knn": "kNN",
    "quadratic": "Quadratic*",
    "rbf": "RBF",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("examples/assets/food101_overlap_vs_linear_probe_story_summary.json"),
        help="Food-101 plot summary or completed result JSON artifact.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("img/visuals/food101-overlap-vs-linear-probe-story"),
        help="Output path without an extension; PNG and SVG are written.",
    )
    return parser


def _rows_from_summary(payload: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
    plot_data = payload.get("plot_data")
    if not isinstance(plot_data, dict):
        raise ValueError("The Food-101 plot summary does not contain plot data.")
    rows = []
    quadratic = plot_data.get("quadratic_auc")
    head_auc = plot_data.get("head_auc")
    if not isinstance(quadratic, dict) or not isinstance(head_auc, dict):
        raise ValueError("The Food-101 plot summary is missing AUC arrays.")
    for method, arms in quadratic.items():
        if not isinstance(arms, dict):
            raise ValueError("Quadratic AUC data must be grouped by arm.")
        for arm, values in arms.items():
            if not isinstance(values, list):
                raise ValueError("Quadratic AUC replicates must be lists.")
            rows.extend(
                {
                    "arm": arm,
                    "auc": value,
                    "head": "quadratic",
                    "method": method,
                    "replicate": replicate,
                }
                for replicate, value in enumerate(values)
            )
    for method, heads in head_auc.items():
        if not isinstance(heads, dict):
            raise ValueError("Head AUC data must be grouped by method.")
        for head, values in heads.items():
            if head == "quadratic":
                continue
            if not isinstance(values, list):
                raise ValueError("Head AUC replicates must be lists.")
            rows.extend(
                {
                    "arm": "nonlinearity_full",
                    "auc": value,
                    "head": head,
                    "method": method,
                    "replicate": replicate,
                }
                for replicate, value in enumerate(values)
            )
    return rows


def _load_auc_rows(path: Path) -> Sequence[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") == "food101_overlap_vs_linear_probe_story_summary":
        return _rows_from_summary(payload)
    if payload.get("artifact_status") != "completed":
        raise ValueError("The Food-101 result artifact must be completed.")
    if payload.get("study") != "food101_nonlinear_backbone_bridge":
        raise ValueError("The result artifact has the wrong study identity.")
    rows = payload.get("auc_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The result artifact does not contain AUC rows.")
    return rows


def _load_runtime_minutes(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") == "food101_overlap_vs_linear_probe_story_summary":
        plot_data = payload.get("plot_data")
        runtime = plot_data.get("selector_runtime_minutes") if isinstance(plot_data, dict) else None
        if not isinstance(runtime, dict):
            raise ValueError("The Food-101 plot summary is missing selector runtime data.")
        values_by_method = runtime
    else:
        if payload.get("artifact_status") != "completed":
            raise ValueError("The Food-101 result artifact must be completed.")
        rows = payload.get("selector_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("The result artifact does not contain selector rows.")
        totals: Dict[str, Dict[int, float]] = {method: {} for method in _METHOD_LABELS}
        counts: Dict[str, Dict[int, int]] = {method: {} for method in _METHOD_LABELS}
        for row in rows:
            method = row.get("method")
            if method not in totals:
                continue
            replicate = int(row["replicate"])
            seconds = float(row["seconds"])
            if not np.isfinite(seconds) or seconds < 0.0:
                raise ValueError("Selector runtimes must be finite and non-negative.")
            totals[method][replicate] = totals[method].get(replicate, 0.0) + seconds / 60.0
            counts[method][replicate] = counts[method].get(replicate, 0) + 1
        for method in _METHOD_LABELS:
            if counts[method] != {replicate: 120 for replicate in range(5)}:
                raise ValueError(f"Expected 120 selector timings per replicate for {method}.")
        values_by_method = {
            method: [totals[method][replicate] for replicate in range(5)]
            for method in _METHOD_LABELS
        }

    resolved: Dict[str, np.ndarray] = {}
    for method in _METHOD_LABELS:
        values = values_by_method.get(method)
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError(f"Expected five runtime replicates for {method}.")
        array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError("Selector runtimes must be finite and non-negative.")
        resolved[method] = array
    return resolved


def _values(
    rows: Sequence[Dict[str, Any]],
    *,
    head: str,
    arm: str,
    method: str,
) -> np.ndarray:
    selected = sorted(
        (
            (int(row["replicate"]), float(row["auc"]))
            for row in rows
            if row.get("head") == head and row.get("arm") == arm and row.get("method") == method
        ),
        key=lambda item: item[0],
    )
    if len(selected) != 5 or [replicate for replicate, _ in selected] != list(range(5)):
        raise ValueError(f"Expected five replicates for head={head}, arm={arm}, method={method}.")
    values = np.asarray([value for _, value in selected], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("AUC rows must be finite.")
    return values


def _render(
    rows: Sequence[Dict[str, Any]],
    runtime_minutes: Dict[str, np.ndarray],
    output_prefix: Path,
) -> Sequence[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting environment
        raise ImportError(
            "Install the backbone-selection dependencies to render this figure."
        ) from exc

    overlap_color = "#0072B2"
    probe_color = "#D55E00"
    neutral_color = "#4C566A"
    light_neutral = "#D8DEE9"

    figure, (profile_axis, family_axis, runtime_axis) = plt.subplots(
        1,
        3,
        figsize=(17.0, 5.4),
        gridspec_kw={"width_ratios": [1.12, 1.0, 0.72]},
    )
    figure.patch.set_facecolor("white")

    arms = ("baseline", "nonlinearity_full", "nuisance_full")
    methods = ("overlap_cross_fitted", "linear_probe_oof")
    colors = {
        "overlap_cross_fitted": overlap_color,
        "linear_probe_oof": probe_color,
    }
    x_positions = np.arange(len(arms), dtype=float)
    offsets = {"overlap_cross_fitted": -0.18, "linear_probe_oof": 0.18}
    bar_width = 0.34

    for method in methods:
        arm_values = [_values(rows, head="quadratic", arm=arm, method=method) for arm in arms]
        means = np.asarray([values.mean() for values in arm_values])
        standard_deviations = np.asarray([values.std(ddof=1) for values in arm_values])
        method_x = x_positions + offsets[method]
        profile_axis.bar(
            method_x,
            means,
            width=bar_width,
            color=colors[method],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
            label=_METHOD_LABELS[method],
            zorder=2,
        )
        profile_axis.errorbar(
            method_x,
            means,
            yerr=standard_deviations,
            fmt="none",
            color=colors[method],
            capsize=4,
            linewidth=1.4,
            zorder=3,
        )
        for arm_index, values in enumerate(arm_values):
            jitter = np.linspace(-0.025, 0.025, len(values))
            profile_axis.scatter(
                np.full(len(values), method_x[arm_index]) + jitter,
                values,
                s=24,
                facecolor="white",
                edgecolor=colors[method],
                linewidth=1.0,
                alpha=0.9,
                zorder=5,
            )
            profile_axis.text(
                method_x[arm_index],
                means[arm_index] - 0.065,
                f"{means[arm_index]:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
                zorder=6,
            )

    profile_axis.set_title(
        "A  Ranking quality changes with geometry",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=13,
    )
    profile_axis.set_ylabel("Spearman log-budget AUC", fontsize=11)
    profile_axis.set_xticks(x_positions, [_ARM_LABELS[arm] for arm in arms])
    profile_axis.set_ylim(0.0, 1.08)
    profile_axis.set_yticks(np.linspace(0.0, 1.0, 6))
    profile_axis.grid(axis="y", color=light_neutral, linewidth=0.8, alpha=0.8)
    profile_axis.legend(loc="lower left", frameon=False, fontsize=10)
    profile_axis.text(
        0.0,
        -0.22,
        "Quadratic-head endpoint; points are five paired replicates, error bars are ±1 SD.",
        transform=profile_axis.transAxes,
        fontsize=9,
        color=neutral_color,
    )

    heads = ("linear", "knn", "quadratic", "rbf")
    family_x = np.arange(len(heads), dtype=float)
    differences = []
    for head in heads:
        overlap_values = _values(
            rows,
            head=head,
            arm="nonlinearity_full",
            method="overlap_cross_fitted",
        )
        probe_values = _values(
            rows,
            head=head,
            arm="nonlinearity_full",
            method="linear_probe_oof",
        )
        differences.append(overlap_values - probe_values)

    difference_means = np.asarray([values.mean() for values in differences])
    difference_sd = np.asarray([values.std(ddof=1) for values in differences])
    bar_colors = [probe_color if value < 0 else overlap_color for value in difference_means]
    family_axis.axhline(0.0, color=neutral_color, linewidth=1.0, zorder=1)
    family_axis.bar(
        family_x,
        difference_means,
        color=bar_colors,
        width=0.62,
        alpha=0.88,
        zorder=2,
    )
    family_axis.errorbar(
        family_x,
        difference_means,
        yerr=difference_sd,
        fmt="none",
        ecolor=neutral_color,
        capsize=4,
        linewidth=1.3,
        zorder=3,
    )
    for index, values in enumerate(differences):
        jitter = np.linspace(-0.11, 0.11, len(values))
        family_axis.scatter(
            np.full(len(values), family_x[index]) + jitter,
            values,
            s=24,
            facecolor="white",
            edgecolor=bar_colors[index],
            linewidth=1.0,
            zorder=4,
        )
        family_axis.text(
            family_x[index],
            difference_means[index] * 0.58,
            f"{difference_means[index]:+.3f}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )

    family_axis.set_title(
        "B  The advantage is specific to nonlinear heads",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=13,
    )
    family_axis.set_ylabel("OverlapIndex − linear-probe AUC", fontsize=11)
    family_axis.set_xticks(family_x, [_HEAD_LABELS[head] for head in heads])
    family_axis.set_ylim(-0.16, 0.22)
    family_axis.grid(axis="y", color=light_neutral, linewidth=0.8, alpha=0.8)
    family_axis.text(
        0.0,
        -0.22,
        "Full-nonlinearity arm; * prespecified primary head. Positive values favor OverlapIndex.",
        transform=family_axis.transAxes,
        fontsize=9,
        color=neutral_color,
    )

    runtime_methods = ("overlap_cross_fitted", "linear_probe_oof")
    runtime_x = np.arange(len(runtime_methods), dtype=float)
    runtime_values = [runtime_minutes[method] for method in runtime_methods]
    runtime_means = np.asarray([values.mean() for values in runtime_values])
    runtime_sd = np.asarray([values.std(ddof=1) for values in runtime_values])
    runtime_colors = [colors[method] for method in runtime_methods]
    runtime_axis.bar(
        runtime_x,
        runtime_means,
        color=runtime_colors,
        width=0.62,
        alpha=0.88,
        zorder=2,
    )
    runtime_axis.errorbar(
        runtime_x,
        runtime_means,
        yerr=runtime_sd,
        fmt="none",
        ecolor=neutral_color,
        capsize=4,
        linewidth=1.3,
        zorder=3,
    )
    for index, values in enumerate(runtime_values):
        jitter = np.linspace(-0.11, 0.11, len(values))
        runtime_axis.scatter(
            np.full(len(values), runtime_x[index]) + jitter,
            values,
            s=24,
            facecolor="white",
            edgecolor=runtime_colors[index],
            linewidth=1.0,
            zorder=4,
        )
        runtime_axis.text(
            runtime_x[index],
            runtime_means[index] * 0.5,
            f"{runtime_means[index]:.2f} min",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )
    speedup = runtime_means[1] / runtime_means[0]
    runtime_axis.text(
        0.5,
        max(runtime_means + runtime_sd) * 1.12,
        f"{speedup:.2f}× faster",
        ha="center",
        va="center",
        color=overlap_color,
        fontsize=11,
        fontweight="bold",
    )
    runtime_axis.set_title(
        "C  OverlapIndex uses less selector compute",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=13,
    )
    runtime_axis.set_ylabel("Summed scoring time per replicate (min)", fontsize=11)
    runtime_axis.set_xticks(runtime_x, ["OverlapIndex", "Linear\nprobe"])
    runtime_axis.set_ylim(0.0, max(runtime_means + runtime_sd) * 1.3)
    runtime_axis.grid(axis="y", color=light_neutral, linewidth=0.8, alpha=0.8)
    runtime_axis.text(
        0.0,
        -0.22,
        "120 paired scoring calls per replicate; shared extraction and head evaluation excluded.",
        transform=runtime_axis.transAxes,
        fontsize=9,
        color=neutral_color,
    )

    for axis in (profile_axis, family_axis, runtime_axis):
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(light_neutral)
        axis.spines["bottom"].set_color(light_neutral)
        axis.tick_params(colors=neutral_color, labelsize=10)

    figure.suptitle(
        "A linear probe measures linear accessibility; "
        "OverlapIndex can reveal nonlinear transfer potential",
        x=0.5,
        y=1.02,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.965,
        "Food-101 · 10 frozen backbones · budgets 64–80 images per class",
        ha="center",
        fontsize=10,
        color=neutral_color,
    )
    figure.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.24, wspace=0.32)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    figure.savefig(png_path, dpi=200, facecolor="white", bbox_inches="tight")
    figure.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return (png_path, svg_path)


def main() -> int:
    args = _parser().parse_args()
    paths = _render(
        _load_auc_rows(args.results),
        _load_runtime_minutes(args.results),
        args.output_prefix,
    )
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
