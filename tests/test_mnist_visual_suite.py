import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _load_example_module():
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "mnist_visual_suite",
            examples_dir / "mnist_visual_suite.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


def test_mnist_visual_suite_help_does_not_require_optional_dependencies():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "mnist_visual_suite.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--figure-dir" in completed.stdout
    assert "--monitor-every-batches" in completed.stdout
    assert "--no-download" in completed.stdout


def test_mnist_visual_suite_builds_nested_digit_hierarchy():
    module = _load_example_module()

    assert module._hierarchy_paths(range(10)) == [
        ("even", "even, 0-4", "0"),
        ("odd", "odd, 0-4", "1"),
        ("even", "even, 0-4", "2"),
        ("odd", "odd, 0-4", "3"),
        ("even", "even, 0-4", "4"),
        ("odd", "odd, 5-9", "5"),
        ("even", "even, 5-9", "6"),
        ("odd", "odd, 5-9", "7"),
        ("even", "even, 5-9", "8"),
        ("odd", "odd, 5-9", "9"),
    ]


def test_mnist_visual_suite_stratified_subset_and_pareto_frontier_are_deterministic():
    module = _load_example_module()
    labels = np.repeat(np.arange(10), 12)

    first = module._stratified_indices(labels, 50, seed=7)
    second = module._stratified_indices(labels, 50, seed=7)

    assert np.array_equal(first, second)
    assert np.bincount(labels[first]).tolist() == [5] * 10
    assert module._pareto_frontier_indices(
        bytes_per_sample=[8.0, 16.0, 32.0, 64.0],
        scores=[0.80, 0.78, 0.90, 0.90],
    ) == [0, 2]


def test_mnist_visual_suite_holds_validation_rows_out_before_training(tmp_path):
    module = _load_example_module()
    calls = []

    class FakeMNIST:
        def __init__(self, root, train, download):
            calls.append({"root": root, "train": train, "download": download})
            row_ids = np.arange(120, dtype=np.float32)
            self.data = np.broadcast_to(row_ids[:, None, None], (120, 2, 2)).copy()
            self.targets = np.repeat(np.arange(10), 12)

    train_x, train_y, validation_x, validation_y = module._load_mnist(
        FakeMNIST,
        data_dir=tmp_path,
        train_size=50,
        validation_size=30,
        seed=11,
        download=False,
    )

    def restore_ids(values):
        return np.rint((values[:, 0] * 0.3081 + 0.1307) * 255.0).astype(int)

    assert set(restore_ids(train_x)).isdisjoint(set(restore_ids(validation_x)))
    assert np.bincount(train_y).tolist() == [5] * 10
    assert np.bincount(validation_y).tolist() == [3] * 10
    assert calls == [{"root": str(tmp_path), "train": True, "download": False}]


def test_readme_mnist_visual_assets_are_present_and_renderable():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    stems = (
        "mnist-representation-monitoring",
        "mnist-compression-frontier",
        "mnist-hierarchy-heatmap",
    )

    for stem in stems:
        relative_png = Path("img") / "visuals" / f"{stem}.png"
        png_path = root / relative_png
        svg_path = png_path.with_suffix(".svg")
        assert relative_png.as_posix() in readme
        assert svg_path.is_file()
        with Image.open(png_path) as image:
            image.verify()
            assert image.width >= 1_000
            assert image.height >= 500
