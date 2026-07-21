import importlib.util
import subprocess
import sys
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
            "fashion_mnist_visual_suite",
            examples_dir / "fashion_mnist_visual_suite.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


def test_fashion_mnist_visual_suite_help_does_not_require_optional_dependencies():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "fashion_mnist_visual_suite.py"),
            "--help",
        ],
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


def test_fashion_mnist_visual_suite_builds_nested_product_hierarchy():
    module = _load_example_module()

    assert module._hierarchy_paths(range(10)) == [
        ("apparel", "upper garment", "T-shirt/top"),
        ("apparel", "lower/full-body", "Trouser"),
        ("apparel", "upper garment", "Pullover"),
        ("apparel", "lower/full-body", "Dress"),
        ("apparel", "upper garment", "Coat"),
        ("footwear", "open footwear", "Sandal"),
        ("apparel", "upper garment", "Shirt"),
        ("footwear", "closed footwear", "Sneaker"),
        ("accessory", "bag", "Bag"),
        ("footwear", "closed footwear", "Ankle boot"),
    ]


def test_fashion_mnist_visual_suite_uses_spatial_cnn_representations():
    torch = pytest.importorskip("torch")
    module = _load_example_module()
    model = module._build_model(torch)

    model.eval()
    with torch.inference_mode():
        outputs = model(torch.zeros((2, 28 * 28), dtype=torch.float32))

    assert tuple(outputs["conv_1"].shape) == (2, 128)
    assert tuple(outputs["conv_2"].shape) == (2, 256)
    assert tuple(outputs["embedding"].shape) == (2, 128)
    assert tuple(outputs["logits"].shape) == (2, 10)


def test_fashion_mnist_visual_suite_stratified_subset_and_pareto_are_deterministic():
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


def test_fashion_mnist_visual_suite_holds_validation_rows_out_before_training(tmp_path):
    module = _load_example_module()
    calls = []

    class FakeFashionMNIST:
        def __init__(self, root, train, download):
            calls.append({"root": root, "train": train, "download": download})
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

    def restore_ids(values):
        return np.rint(
            (values[:, 0] * module._NORMALIZATION_STD + module._NORMALIZATION_MEAN)
            * 255.0
        ).astype(int)

    assert set(restore_ids(train_x)).isdisjoint(set(restore_ids(validation_x)))
    assert np.bincount(train_y).tolist() == [5] * 10
    assert np.bincount(validation_y).tolist() == [3] * 10
    assert calls == [{"root": str(tmp_path), "train": True, "download": False}]


def test_readme_fashion_mnist_visual_assets_are_present_and_renderable():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    stems = (
        "fashion-mnist-representation-monitoring",
        "fashion-mnist-compression-frontier",
        "fashion-mnist-hierarchy-heatmap",
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
