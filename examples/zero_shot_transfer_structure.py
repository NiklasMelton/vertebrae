"""Show that prompt alignment and transfer structure are different signals.

This flagship experiment encodes a balanced CIFAR-10 image slice exactly once with
OpenCLIP.  It scores OverlapIndex on those frozen image embeddings once, then holds
them fixed while scoring several fully declared prompt protocols.  The resulting
figure makes the invariant visible: prompt wording can move zero-shot accuracy and
per-class F1 without moving sample-embedding OverlapIndex.

Install the optional dependencies and run from the repository root:

    poetry install -E openclip -E visuals
    poetry run python examples/zero_shot_transfer_structure.py

The first run downloads CIFAR-10 and the selected OpenCLIP checkpoint.  Useful
environment variables:

    VERTABRAE_CIFAR10_DIR=/path/to/cifar10
    VERTABRAE_CIFAR10_SAMPLES_PER_CLASS=50
    VERTABRAE_OPENCLIP_MODEL=ViT-B-32
    VERTABRAE_OPENCLIP_PRETRAINED=laion2b_s34b_b79k
    VERTABRAE_OPENCLIP_DEVICE=cpu
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from _common import EXAMPLES_DIR, ensure_output_dir

from vertebrae import (
    BenchmarkDataset,
    DatasetIdentity,
    OverlapScoringConfig,
    ZeroShotConfig,
    ZeroShotDataset,
)
from vertebrae.extractors import OpenCLIPExtractor
from vertebrae.scoring import OverlapMetric, ZeroShotScorer
from vertebrae.scoring.metrics import MetricResult
from vertebrae.scoring.zero_shot import ZeroShotScoreResult

CIFAR10_LABELS: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
DEFAULT_DATA_DIR = EXAMPLES_DIR / "data" / "cifar10"
DEFAULT_SAMPLES_PER_CLASS = 50
DEFAULT_RANDOM_STATE = 23


@dataclass(frozen=True)
class PromptSetEvaluation:
    """One declared prompt protocol scored against shared image embeddings."""

    name: str
    class_prompts: Dict[str, str]
    zero_shot: ZeroShotScoreResult


def main() -> None:
    output_dir = ensure_output_dir()
    data_dir = Path(os.environ.get("VERTABRAE_CIFAR10_DIR", str(DEFAULT_DATA_DIR))).expanduser()
    samples_per_class = _positive_int_from_env(
        "VERTABRAE_CIFAR10_SAMPLES_PER_CLASS", DEFAULT_SAMPLES_PER_CLASS
    )
    random_state = _positive_int_from_env("VERTABRAE_CIFAR10_RANDOM_STATE", DEFAULT_RANDOM_STATE)
    model_name = os.environ.get("VERTABRAE_OPENCLIP_MODEL", "ViT-B-32")
    pretrained = os.environ.get("VERTABRAE_OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")
    device = os.environ.get("VERTABRAE_OPENCLIP_DEVICE")

    try:
        dataset = load_cifar10_dataset(data_dir, samples_per_class, random_state)
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E openclip -E visuals")
        return
    except (OSError, ValueError) as exc:
        print(f"Could not prepare CIFAR-10: {exc}")
        return

    extractor = OpenCLIPExtractor(
        "openclip_cifar10",
        model_name=model_name,
        pretrained=pretrained,
        batch_size=32,
        image_mode="rgb",
        device=device,
    )
    print(
        f"Encoding {len(dataset.y)} CIFAR-10 images once with {model_name} ({pretrained}). "
        "Every prompt set below reuses this exact image-embedding matrix."
    )
    try:
        image_embeddings, overlap, evaluations = evaluate_prompt_sets(
            dataset,
            extractor,
            cifar10_prompt_sets(),
        )
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E openclip -E visuals")
        return
    except OSError as exc:
        print(f"Could not load the OpenCLIP checkpoint: {exc}")
        print("Use a local checkpoint cache or allow network access for the first download.")
        return

    rows = comparison_rows(overlap, evaluations, CIFAR10_LABELS)
    figure_path = output_dir / "zero_shot_transfer_structure.png"
    csv_path = output_dir / "zero_shot_transfer_structure.csv"
    plot_prompt_structure_comparison(rows, figure_path)
    write_rows_csv(rows, csv_path)

    print(f"Frozen image embedding shape: {image_embeddings.shape}")
    print(f"OverlapIndex macro score (invariant across prompts): {overlap.macro_score:.3f}")
    for evaluation in evaluations:
        print(
            f"{evaluation.name}: accuracy={evaluation.zero_shot.metrics['accuracy']:.3f}, "
            f"macro F1={evaluation.zero_shot.metrics['macro_f1']:.3f}"
        )
    print(f"\nFigure written to {figure_path}")
    print(f"Point data written to {csv_path}")


def load_cifar10_dataset(
    data_dir: Path,
    samples_per_class: int,
    random_state: int,
) -> BenchmarkDataset:
    """Download/load a deterministic balanced CIFAR-10 test subset."""

    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise ImportError(
            "This example requires torchvision. Install it with `poetry install -E visuals`."
        ) from exc

    if samples_per_class < 2:
        raise ValueError("VERTABRAE_CIFAR10_SAMPLES_PER_CLASS must be at least 2.")
    source = CIFAR10(root=str(data_dir), train=False, download=True)
    targets = np.asarray(source.targets, dtype=int)
    rng = np.random.default_rng(random_state)
    selected: List[int] = []
    for class_index, label in enumerate(CIFAR10_LABELS):
        candidates = np.flatnonzero(targets == class_index)
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"CIFAR-10 class {label!r} has {len(candidates)} test images; "
                f"need {samples_per_class}."
            )
        selected.extend(rng.permutation(candidates)[:samples_per_class].tolist())
    selected.sort(key=lambda index: (int(targets[index]), index))
    images = [source[index][0].convert("RGB") for index in selected]
    labels = np.asarray([CIFAR10_LABELS[int(targets[index])] for index in selected], dtype=object)
    return BenchmarkDataset.from_arrays(
        images,
        labels,
        modality="image",
        metadata={
            "example": "zero_shot_transfer_structure",
            "dataset_source": "CIFAR-10 test split",
            "samples_per_class": samples_per_class,
            "random_state": random_state,
            "label_rule": "CIFAR-10 semantic class names",
        },
        identity=DatasetIdentity.declared(
            "cifar10-test-balanced", f"seed-{random_state}-per-class-{samples_per_class}"
        ),
    )


def cifar10_prompt_sets() -> Dict[str, Dict[str, str]]:
    """Return prompt protocols declared before any labels are scored.

    These are deliberately ordinary alternatives, not a search space.  They make
    the effect of class naming and dataset context auditable in the output CSV.
    """

    return {
        "label_only": {label: label for label in CIFAR10_LABELS},
        "photo_of": {label: f"a photo of a {label}" for label in CIFAR10_LABELS},
        "cifar10_context": {
            label: f"a low-resolution CIFAR-10 image of a {label}" for label in CIFAR10_LABELS
        },
    }


def evaluate_prompt_sets(
    dataset: BenchmarkDataset,
    extractor: OpenCLIPExtractor,
    prompt_sets: Mapping[str, Mapping[str, str]],
) -> Tuple[np.ndarray, MetricResult, List[PromptSetEvaluation]]:
    """Encode images once, then score every declared text protocol against them."""

    if not prompt_sets:
        raise ValueError("At least one explicit prompt set is required.")
    image_embeddings = extractor.encode_retrieval(
        dataset.X, branch="image_branch", modality="image"
    )
    overlap = OverlapMetric(
        OverlapScoringConfig(k=10, min_samples_per_cluster=5, normalize_embeddings=True)
    ).score(
        image_embeddings,
        dataset.y,
        target_metadata={"target_type": "single_label"},
    )
    evaluations = []
    for name, class_prompts in prompt_sets.items():
        protocol = ZeroShotDataset.from_dataset(
            dataset,
            class_prompts,
            metadata={"prompt_set": name, "experiment": "zero_shot_transfer_structure"},
        )
        prompts, prompt_labels, template_ids = protocol.prompt_rows()
        prompt_embeddings = extractor.encode_retrieval(
            prompts, branch="text_branch", modality="text"
        )
        zero_shot = ZeroShotScorer(ZeroShotConfig()).score(
            image_embeddings,
            prompt_embeddings,
            dataset.y,
            class_labels=protocol.class_labels(),
            prompt_labels=prompt_labels,
            template_ids=template_ids,
            sample_ids=protocol.sample_ids(),
        )
        evaluations.append(PromptSetEvaluation(name, dict(class_prompts), zero_shot))
    return image_embeddings, overlap, evaluations


def comparison_rows(
    overlap: MetricResult,
    evaluations: Iterable[PromptSetEvaluation],
    class_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    """Create global and per-class points while retaining the overlap invariant."""

    rows: List[Dict[str, Any]] = []
    per_class_overlap = overlap.per_class_scores
    for evaluation in evaluations:
        rows.append(
            {
                "panel": "global",
                "prompt_set": evaluation.name,
                "class_label": "",
                "overlap_index": float(overlap.macro_score),
                "zero_shot_score": float(evaluation.zero_shot.metrics["accuracy"]),
                "score_name": "accuracy",
            }
        )
        for label in class_labels:
            rows.append(
                {
                    "panel": "per_class",
                    "prompt_set": evaluation.name,
                    "class_label": label,
                    "overlap_index": float(per_class_overlap[label]),
                    "zero_shot_score": float(evaluation.zero_shot.per_class[label]["f1"]),
                    "score_name": "f1",
                }
            )
    return rows


def plot_prompt_structure_comparison(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write an invariant-versus-variable view of structure and prompt alignment."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise ImportError(
            "This example requires matplotlib. Install it with `poetry install -E visuals`."
        ) from exc

    global_rows = [row for row in rows if row["panel"] == "global"]
    class_rows = [row for row in rows if row["panel"] == "per_class"]
    if not global_rows or not class_rows:
        raise ValueError("Comparison rows must include global and per-class panels.")
    prompt_names = list(dict.fromkeys(str(row["prompt_set"]) for row in global_rows))
    class_names = list(dict.fromkeys(str(row["class_label"]) for row in class_rows))
    markers = ("o", "s", "^", "D", "P", "X")
    marker_by_prompt = {
        name: markers[index % len(markers)] for index, name in enumerate(prompt_names)
    }
    colors = plt.get_cmap("Dark2").colors
    color_by_prompt = {name: colors[index % len(colors)] for index, name in enumerate(prompt_names)}
    overlap_values = {float(row["overlap_index"]) for row in global_rows}
    if len(overlap_values) != 1:
        raise ValueError("Global comparison rows must share one fixed Overlap Index value.")
    overlap_value = next(iter(overlap_values))
    accuracy_by_prompt = {
        str(row["prompt_set"]): float(row["zero_shot_score"]) for row in global_rows
    }
    accuracy_span = max(accuracy_by_prompt.values()) - min(accuracy_by_prompt.values())

    class_summaries = _class_prompt_summaries(class_rows, prompt_names, class_names)

    figure, (global_axis, class_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 6.8),
        gridspec_kw={"width_ratios": (0.8, 1.5)},
        constrained_layout=True,
    )

    global_positions = np.arange(len(prompt_names))
    global_bars = global_axis.barh(
        global_positions,
        [accuracy_by_prompt[name] for name in prompt_names],
        color=[color_by_prompt[name] for name in prompt_names],
        height=0.58,
    )
    global_axis.set_yticks(global_positions, labels=prompt_names)
    global_axis.invert_yaxis()
    global_axis.set_xlim(0.0, 1.0)
    global_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    global_axis.set_xlabel("Zero-shot accuracy")
    global_axis.set_title(f"Fixed structure — Overlap Index: {overlap_value:.3f}", loc="left")
    global_axis.grid(axis="x", alpha=0.22)
    global_axis.set_axisbelow(True)
    for bar, name in zip(global_bars, prompt_names):
        score = accuracy_by_prompt[name]
        global_axis.text(
            score - 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1%}",
            va="center",
            ha="right",
            color="white",
            fontweight="bold",
        )

    class_positions = np.arange(len(class_summaries))
    for position, summary in zip(class_positions, class_summaries):
        f1_by_prompt = summary["f1_by_prompt"]
        minimum = min(f1_by_prompt.values())
        maximum = max(f1_by_prompt.values())
        class_axis.hlines(position, minimum, maximum, color="0.72", linewidth=2.2, zorder=1)
        for prompt_name in prompt_names:
            class_axis.scatter(
                f1_by_prompt[prompt_name],
                position,
                s=74,
                marker=marker_by_prompt[prompt_name],
                color=color_by_prompt[prompt_name],
                edgecolor="white",
                linewidth=0.7,
                zorder=2,
                label=prompt_name if position == 0 else None,
            )
        class_axis.text(
            0.018,
            position,
            f"+{float(summary['f1_span']) * 100:.0f} pp",
            va="center",
            ha="left",
            fontsize=9,
        )
    class_axis.set_yticks(
        class_positions,
        labels=[
            f"{summary['class_name']}\nOI: {float(summary['overlap_index']):.2f}"
            for summary in class_summaries
        ],
    )
    class_axis.invert_yaxis()
    class_axis.set_xlim(0.0, 1.0)
    class_axis.set_xticks(np.linspace(0.0, 1.0, 6))
    class_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    class_axis.set_xlabel("Zero-shot F1")
    class_axis.set_title("Prompt sensitivity by class", loc="left")
    class_axis.grid(axis="x", alpha=0.22)
    class_axis.set_axisbelow(True)
    class_axis.legend(
        title="Prompt set",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(prompt_names),
    )
    figure.suptitle(
        "Same representation. Prompt wording changes accuracy by "
        f"{accuracy_span * 100:.1f} percentage points",
        fontsize=15,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _class_prompt_summaries(
    class_rows: Sequence[Mapping[str, Any]],
    prompt_names: Sequence[str],
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Summarize class prompt ranges, ordered by fixed Overlap Index."""

    summaries = []
    for class_name in class_names:
        values = {
            str(row["prompt_set"]): float(row["zero_shot_score"])
            for row in class_rows
            if str(row["class_label"]) == class_name
        }
        missing = [name for name in prompt_names if name not in values]
        if missing:
            raise ValueError(
                f"Per-class comparison rows for {class_name!r} are missing prompts {missing}."
            )
        overlap_values = {
            float(row["overlap_index"])
            for row in class_rows
            if str(row["class_label"]) == class_name
        }
        if len(overlap_values) != 1:
            raise ValueError(
                f"Per-class comparison rows for {class_name!r} must share one Overlap Index."
            )
        summaries.append(
            {
                "class_name": class_name,
                "overlap_index": next(iter(overlap_values)),
                "f1_by_prompt": values,
                "f1_span": max(values.values()) - min(values.values()),
            }
        )
    summaries.sort(key=lambda item: (-float(item["overlap_index"]), str(item["class_name"])))
    return summaries


def write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Persist plotted points so the prompt comparison remains auditable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel",
                "prompt_set",
                "class_label",
                "overlap_index",
                "zero_shot_score",
                "score_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


if __name__ == "__main__":
    main()
