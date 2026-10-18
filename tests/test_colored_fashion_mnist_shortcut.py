import importlib.util
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _load_example_module():
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "colored_fashion_mnist_shortcut",
            examples_dir / "colored_fashion_mnist_shortcut.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


def test_colored_fashion_mnist_shortcut_help_needs_no_optional_dependencies():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "colored_fashion_mnist_shortcut.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--shortcut-strength" in completed.stdout
    assert "--color-opacity" in completed.stdout
    assert "--repeats" in completed.stdout
    assert "--no-download" in completed.stdout


def test_experiment_sizes_must_support_exact_class_color_balance():
    module = _load_example_module()
    parser = ArgumentParser()
    args = Namespace(
        epochs=1,
        repeats=1,
        train_size=250,
        validation_size=200,
        train_batch_size=32,
        embedding_batch_size=32,
        shortcut_strength=0.85,
        color_opacity=0.45,
    )

    with pytest.raises(SystemExit):
        module._validate_args(args, parser)


def test_color_environments_are_balanced_deterministic_and_counterfactual():
    module = _load_example_module()
    labels = np.repeat(np.arange(10), 20)

    first = module._assign_colors(
        labels,
        environment="correlated",
        shortcut_strength=0.9,
        seed=17,
    )
    second = module._assign_colors(
        labels,
        environment="correlated",
        shortcut_strength=0.9,
        seed=17,
    )
    reversed_colors = module._assign_colors(
        labels,
        environment="reversed",
        seed=999,
    )
    balanced = module._assign_colors(labels, environment="balanced", seed=23)

    assert np.array_equal(first, second)
    assert (first == labels).mean() > 0.8
    assert np.array_equal(reversed_colors, (labels + 5) % 10)
    assert np.all(reversed_colors != labels)
    for label in range(10):
        assert np.bincount(balanced[labels == label], minlength=10).tolist() == [2] * 10


def test_monitoring_dataset_has_explicit_intended_and_nuisance_target_views():
    module = _load_example_module()
    class_ids = np.repeat(np.arange(10), 3)
    color_ids = class_ids.copy()
    values = module._colorize(np.ones((len(class_ids), 2, 2), dtype=np.float32), color_ids)

    dataset = module._monitoring_dataset(
        values,
        class_ids,
        color_ids,
        color_opacity=0.45,
        seed=5,
    )

    assert dataset.target_view_names() == ["intended_class", "nuisance_color"]
    assert dataset.metadata["color_environment"] == "balanced"
    assert dataset.target_view("intended_class").active_target_view()["name"] == "intended_class"
    assert dataset.target_view("nuisance_color").y.tolist() == [
        module._COLOR_NAMES[index] for index in color_ids
    ]


def test_colored_fashion_mnist_shortcut_uses_rgb_spatial_representations():
    torch = pytest.importorskip("torch")
    module = _load_example_module()
    model = module._build_model(torch)

    model.eval()
    with torch.inference_mode():
        outputs = model(torch.zeros((2, 3 * 28 * 28), dtype=torch.float32))

    assert tuple(outputs["conv_1"].shape) == (2, 128)
    assert tuple(outputs["conv_2"].shape) == (2, 256)
    assert tuple(outputs["embedding"].shape) == (2, 128)
    assert tuple(outputs["logits"].shape) == (2, 10)


def test_paired_models_start_identically_without_sharing_parameters():
    torch = pytest.importorskip("torch")
    module = _load_example_module()

    control, shortcut = module._build_paired_models(torch, seed=31)

    for control_parameter, shortcut_parameter in zip(control.parameters(), shortcut.parameters()):
        assert torch.equal(control_parameter, shortcut_parameter)
        assert control_parameter.data_ptr() != shortcut_parameter.data_ptr()


def test_evaluation_environments_change_only_color_rendering():
    module = _load_example_module()
    labels = np.repeat(np.arange(10), 10)
    grayscale = np.linspace(0.0, 1.0, len(labels) * 4, dtype=np.float32).reshape(-1, 2, 2)

    values, colors = module._evaluation_environments(
        grayscale,
        labels,
        shortcut_strength=0.85,
        opacity=0.45,
        seed=19,
    )

    assert set(values) == {"correlated", "balanced", "reversed", "grayscale"}
    assert np.array_equal(colors["reversed"], (labels + 5) % 10)
    assert colors["grayscale"] is None
    assert all(
        environment_values.shape == (len(labels), 3 * 2 * 2)
        for environment_values in values.values()
    )
    grayscale_channels = values["grayscale"].reshape(len(labels), 3, 2, 2)
    assert np.allclose(grayscale_channels[:, 0], grayscale_channels[:, 1])
    assert np.allclose(grayscale_channels[:, 1], grayscale_channels[:, 2])


def test_cell_accuracy_summary_is_robust_to_one_failing_slice():
    module = _load_example_module()
    labels = np.repeat(np.arange(2), 4)
    colors = np.tile(np.repeat(np.arange(2), 2), 2)
    predictions = labels.copy()
    predictions[(labels == 1) & (colors == 0)] = 0

    scores = module._cell_accuracy_scores(predictions, labels, colors)
    summary = module._cell_accuracy_summary(scores)

    assert scores.tolist() == [1.0, 1.0, 0.0, 1.0]
    assert summary["cell_mean_accuracy"] == 0.75
    assert summary["cell_p10_accuracy"] > 0.0
    assert summary["bottom5_cell_mean_accuracy"] == 0.75


def test_final_metric_frame_preserves_seed_level_values_and_summarizes():
    module = _load_example_module()
    records = [
        {
            "training_seed": 3,
            "repeat_index": 0,
            "control_balanced_accuracy": 0.8,
            "shortcut_balanced_accuracy": 0.6,
        },
        {
            "training_seed": 13,
            "repeat_index": 1,
            "control_balanced_accuracy": 0.9,
            "shortcut_balanced_accuracy": 0.5,
        },
    ]

    frame = module._final_metric_frame(records)
    summary = module._final_metric_summary(frame)

    assert len(frame) == 4
    assert set(frame["training_seed"]) == {3, 13}
    assert list(summary.columns) == ["training_regime", "metric", "mean", "std", "count"]
    shortcut = summary.loc[
        (summary["training_regime"] == "shortcut") & (summary["metric"] == "balanced_accuracy")
    ].iloc[0]
    assert shortcut["mean"] == pytest.approx(0.55)
    assert shortcut["count"] == 2
    paired = summary.loc[
        (summary["training_regime"] == "shortcut_minus_control")
        & (summary["metric"] == "balanced_accuracy")
    ].iloc[0]
    assert paired["mean"] == pytest.approx(-0.30)


def test_all_color_metrics_detect_predictions_that_follow_displayed_color(monkeypatch):
    module = _load_example_module()
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    color_grid = np.broadcast_to(np.arange(10), (len(labels), 10))
    control_predictions = np.broadcast_to(labels[:, None], color_grid.shape).copy()

    def fake_all_color_predictions(model, grayscale_images, *, opacity, torch):
        return control_predictions if model == "control-model" else color_grid.copy()

    monkeypatch.setattr(module, "_all_color_predictions", fake_all_color_predictions)
    metrics = module._all_color_metrics(
        {"control": "control-model", "shortcut": "shortcut-model"},
        np.zeros((len(labels), 2, 2), dtype=np.float32),
        labels,
        opacity=0.45,
        torch=None,
    )

    assert metrics["control_all_color_accuracy"] == 1.0
    assert metrics["control_prediction_flip_rate"] == 0.0
    assert metrics["shortcut_all_color_accuracy"] == 0.1
    assert metrics["shortcut_prediction_flip_rate"] == 1.0
    assert metrics["shortcut_color_following_rate"] == 1.0
    assert metrics["shortcut_color_following_error_share"] == 1.0
    assert metrics["shortcut_reversed_color_following_rate"] == 1.0


def test_exemplar_plot_writes_documentation_ready_png_and_svg(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    module = _load_example_module()
    labels = np.asarray([0, 1, 4, 5, 8, 9], dtype=np.int64)
    images = np.linspace(0.0, 1.0, len(labels) * 16, dtype=np.float32).reshape(-1, 4, 4)

    png_path, svg_path = module._plot_environment_exemplars(
        images,
        labels,
        opacity=0.45,
        seed=11,
        figure_dir=tmp_path,
        plt=plt,
    )

    assert png_path.is_file() and png_path.stat().st_size > 0
    assert svg_path.is_file() and svg_path.stat().st_size > 0


def test_readme_shortcut_use_case_assets_are_present_and_renderable():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    stems = (
        "colored_fashion_mnist_shortcut_monitoring",
        "colored_fashion_mnist_shortcut_paired_effects",
        "colored_fashion_mnist_shortcut_exemplars",
    )

    assert "auditing shortcut learning with named target views" in readme
    assert "Prediction changes under recoloring" in readme
    for stem in stems:
        relative_png = Path("img") / "visuals" / f"{stem}.png"
        png_path = root / relative_png
        assert relative_png.as_posix() in readme
        with Image.open(png_path) as image:
            image.verify()
            assert image.width >= 1_000
            assert image.height >= 500

    assert (root / "img" / "visuals" / "colored_fashion_mnist_shortcut_exemplars.svg").is_file()


def test_loader_holds_out_validation_rows_before_coloring(tmp_path):
    module = _load_example_module()

    class FakeFashionMNIST:
        def __init__(self, root, train, download):
            row_ids = np.arange(120, dtype=np.float32)
            self.data = np.broadcast_to(row_ids[:, None, None], (120, 2, 2)).copy()
            self.targets = np.repeat(np.arange(10), 12)

    train_x, train_y, validation_x, validation_y = module._load_fashion_mnist(
        FakeFashionMNIST,
        data_dir=tmp_path,
        train_size=50,
        validation_size=30,
        seed=11,
        download=False,
    )

    assert set(np.rint(train_x[:, 0, 0] * 255).astype(int)).isdisjoint(
        set(np.rint(validation_x[:, 0, 0] * 255).astype(int))
    )
    assert np.bincount(train_y).tolist() == [5] * 10
    assert np.bincount(validation_y).tolist() == [3] * 10
