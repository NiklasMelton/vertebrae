import importlib.util
import sys
from pathlib import Path

import pytest

from vertebrae import BenchmarkDataset, DatasetIdentity
from vertebrae.scoring.metrics import MetricResult
from vertebrae.scoring.zero_shot import ZeroShotScoreResult

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
EXAMPLE_PATH = EXAMPLES_DIR / "zero_shot_transfer_structure.py"


def _load_example_module():
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location("zero_shot_transfer_structure", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


experiment = _load_example_module()


def _zero_shot(accuracy, cat_f1, dog_f1):
    return ZeroShotScoreResult(
        score=accuracy,
        primary_metric="accuracy",
        metrics={"accuracy": accuracy, "macro_f1": (cat_f1 + dog_f1) / 2},
        per_class={"cat": {"f1": cat_f1}, "dog": {"f1": dog_f1}},
        confusion_matrix=[[2, 0], [0, 2]],
    )


def test_cifar10_prompt_sets_are_complete_and_distinct():
    prompt_sets = experiment.cifar10_prompt_sets()

    assert set(prompt_sets) == {"label_only", "photo_of", "cifar10_context"}
    assert all(set(prompts) == set(experiment.CIFAR10_LABELS) for prompts in prompt_sets.values())
    assert prompt_sets["label_only"]["cat"] != prompt_sets["cifar10_context"]["cat"]


def test_comparison_rows_keep_overlap_fixed_while_prompt_scores_vary():
    overlap = MetricResult(
        name="overlap",
        score=0.72,
        diagnostics={"macro_score": 0.72, "per_class_scores": {"cat": 0.6, "dog": 0.8}},
    )
    evaluations = [
        experiment.PromptSetEvaluation("label_only", {}, _zero_shot(0.4, 0.3, 0.5)),
        experiment.PromptSetEvaluation("photo_of", {}, _zero_shot(0.8, 0.7, 0.9)),
    ]

    rows = experiment.comparison_rows(overlap, evaluations, ("cat", "dog"))
    global_rows = [row for row in rows if row["panel"] == "global"]
    cat_rows = [row for row in rows if row["class_label"] == "cat"]

    assert [row["overlap_index"] for row in global_rows] == [0.72, 0.72]
    assert [row["zero_shot_score"] for row in global_rows] == [0.4, 0.8]
    assert [row["overlap_index"] for row in cat_rows] == [0.6, 0.6]
    assert [row["zero_shot_score"] for row in cat_rows] == [0.3, 0.7]


def test_class_prompt_summaries_sort_by_overlap_descending():
    overlap = MetricResult(
        name="overlap",
        score=0.72,
        diagnostics={"macro_score": 0.72, "per_class_scores": {"cat": 0.6, "dog": 0.8}},
    )
    rows = experiment.comparison_rows(
        overlap,
        [
            experiment.PromptSetEvaluation("label_only", {}, _zero_shot(0.4, 0.3, 0.5)),
            experiment.PromptSetEvaluation("photo_of", {}, _zero_shot(0.8, 0.7, 0.9)),
        ],
        ("cat", "dog"),
    )

    summaries = experiment._class_prompt_summaries(
        [row for row in rows if row["panel"] == "per_class"],
        ("label_only", "photo_of"),
        ("cat", "dog"),
    )

    assert [(item["class_name"], item["overlap_index"]) for item in summaries] == [
        ("dog", 0.8),
        ("cat", 0.6),
    ]


def test_evaluate_prompt_sets_encodes_images_once(fake_overlapindex):
    class FakeOpenCLIP:
        def __init__(self):
            self.calls = []

        def encode_retrieval(self, values, *, branch, modality):
            values = list(values)
            self.calls.append((branch, modality, values))
            if branch == "image_branch":
                return [[1.0, 0.0] if value.startswith("cat") else [0.0, 1.0] for value in values]
            return [[1.0, 0.0] if "cat" in value else [0.0, 1.0] for value in values]

    dataset = BenchmarkDataset.from_arrays(
        ["cat-0", "cat-1", "dog-0", "dog-1"],
        ["cat", "cat", "dog", "dog"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = FakeOpenCLIP()
    embeddings, overlap, evaluations = experiment.evaluate_prompt_sets(
        dataset,
        extractor,
        {
            "label_only": {"cat": "cat", "dog": "dog"},
            "photo_of": {"cat": "a photo of a cat", "dog": "a photo of a dog"},
        },
    )

    assert len(embeddings) == 4
    assert overlap.name == "overlap"
    assert [item.name for item in evaluations] == ["label_only", "photo_of"]
    assert [call[:2] for call in extractor.calls] == [
        ("image_branch", "image"),
        ("text_branch", "text"),
        ("text_branch", "text"),
    ]


def test_write_rows_csv_preserves_plot_fields(tmp_path):
    path = tmp_path / "points.csv"
    experiment.write_rows_csv(
        [
            {
                "panel": "global",
                "prompt_set": "label_only",
                "class_label": "",
                "overlap_index": 0.7,
                "zero_shot_score": 0.5,
                "score_name": "accuracy",
            }
        ],
        path,
    )

    assert path.read_text(encoding="utf-8").splitlines() == [
        "panel,prompt_set,class_label,overlap_index,zero_shot_score,score_name",
        "global,label_only,,0.7,0.5,accuracy",
    ]


def test_plot_has_global_and_per_class_panels(tmp_path):
    pytest.importorskip("matplotlib")
    overlap = MetricResult(
        name="overlap",
        score=0.72,
        diagnostics={"macro_score": 0.72, "per_class_scores": {"cat": 0.6, "dog": 0.8}},
    )
    rows = experiment.comparison_rows(
        overlap,
        [
            experiment.PromptSetEvaluation("label_only", {}, _zero_shot(0.4, 0.3, 0.5)),
            experiment.PromptSetEvaluation("photo_of", {}, _zero_shot(0.8, 0.7, 0.9)),
        ],
        ("cat", "dog"),
    )
    path = tmp_path / "comparison.png"

    experiment.plot_prompt_structure_comparison(rows, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_positive_int_from_env_rejects_non_positive(monkeypatch):
    monkeypatch.setenv("VERTABRAE_TEST_VALUE", "0")

    with pytest.raises(ValueError, match="positive"):
        experiment._positive_int_from_env("VERTABRAE_TEST_VALUE", 5)
