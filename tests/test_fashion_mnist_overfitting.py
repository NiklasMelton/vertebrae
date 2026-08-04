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
            "fashion_mnist_overfitting",
            examples_dir / "fashion_mnist_overfitting.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


def test_fashion_mnist_overfitting_help_does_not_require_optional_dependencies():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "fashion_mnist_overfitting.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--label-noise" in completed.stdout
    assert "--monitor-every-epochs" in completed.stdout
    assert "--no-download" in completed.stdout


def test_label_corruption_is_stratified_deterministic_and_changes_every_selected_label():
    module = _load_example_module()
    labels = np.repeat(np.arange(10), 10)

    first, first_mask = module._corrupt_labels(
        labels,
        noise_rate=0.4,
        n_classes=10,
        seed=17,
    )
    second, second_mask = module._corrupt_labels(
        labels,
        noise_rate=0.4,
        n_classes=10,
        seed=17,
    )

    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
    assert first_mask.sum() == 40
    assert np.bincount(labels[first_mask], minlength=10).tolist() == [4] * 10
    assert np.all(first[first_mask] != labels[first_mask])
    assert np.array_equal(first[~first_mask], labels[~first_mask])


def test_fashion_mnist_overfitting_uses_three_spatial_cnn_representations():
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
    assert not any(isinstance(layer, torch.nn.Dropout) for layer in model.modules())


def test_paired_models_start_identically_without_sharing_parameters():
    torch = pytest.importorskip("torch")
    module = _load_example_module()

    clean_model, noisy_model = module._build_paired_models(torch, seed=23)

    for clean_parameter, noisy_parameter in zip(
        clean_model.parameters(),
        noisy_model.parameters(),
    ):
        assert torch.equal(clean_parameter, noisy_parameter)
        assert clean_parameter.data_ptr() != noisy_parameter.data_ptr()


def test_fashion_mnist_overfitting_holds_validation_rows_out_before_training(tmp_path):
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

    def restore_ids(values):
        return np.rint(
            (values[:, 0] * module._NORMALIZATION_STD + module._NORMALIZATION_MEAN) * 255.0
        ).astype(int)

    assert set(restore_ids(train_x)).isdisjoint(set(restore_ids(validation_x)))
    assert np.bincount(train_y).tolist() == [5] * 10
    assert np.bincount(validation_y).tolist() == [3] * 10


def test_best_validation_epoch_uses_the_first_global_minimum():
    module = _load_example_module()

    assert module._best_validation_epoch(
        epochs=[0, 2, 4, 6, 8],
        losses=[2.3, 1.4, 0.9, 0.9, 1.2],
    ) == 4

    with pytest.raises(ValueError, match="equal length"):
        module._best_validation_epoch(epochs=[0, 1], losses=[1.0])


def test_readme_paired_overfitting_asset_is_present_and_renderable():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    relative_png = (
        Path("img") / "visuals" / "fashion-mnist-paired-overfitting-monitoring.png"
    )
    png_path = root / relative_png
    svg_path = png_path.with_suffix(".svg")

    assert relative_png.as_posix() in readme
    assert svg_path.is_file()
    with Image.open(png_path) as image:
        image.verify()
        assert image.width >= 1_000
        assert image.height >= 500
