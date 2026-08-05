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
            "fashion_mnist_corruption_atlas",
            examples_dir / "fashion_mnist_corruption_atlas.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


def test_fashion_mnist_corruption_atlas_help_needs_no_optional_dependencies():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "fashion_mnist_corruption_atlas.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--checkpoint" in completed.stdout
    assert "--force-retrain" in completed.stdout
    assert "--confusion-threshold" in completed.stdout
    assert "--no-download" in completed.stdout


def test_corruptions_are_deterministic_shape_preserving_and_have_clean_identity():
    torch = pytest.importorskip("torch")
    module = _load_example_module()
    rng = np.random.default_rng(9)
    raw = rng.uniform(0.0, 1.0, size=(4, 28 * 28)).astype(np.float32)
    values = (raw - module._NORMALIZATION_MEAN) / module._NORMALIZATION_STD

    for corruption in module._CORRUPTION_ORDER:
        clean = module._corrupt_images(values, corruption, 0, seed=17, torch=torch)
        first = module._corrupt_images(values, corruption, 3, seed=17, torch=torch)
        second = module._corrupt_images(values, corruption, 3, seed=17, torch=torch)

        assert np.array_equal(clean, values)
        assert first.shape == values.shape
        assert first.dtype == np.float32
        assert np.isfinite(first).all()
        assert np.array_equal(first, second)
        assert not np.array_equal(first, values)


def test_pairwise_onset_reports_first_threshold_crossing_and_trigger():
    module = _load_example_module()
    size = len(module._FASHION_CLASS_NAMES)
    clean = np.eye(size)
    conditions = {}
    for corruption in module._CORRUPTION_ORDER:
        for severity in range(5):
            conditions[(corruption, severity)] = clean.copy()
    conditions[("noise", 2)][0, 1] = 0.12
    conditions[("noise", 3)][0, 1] = 0.20
    conditions[("rotation", 3)][0, 1] = 0.18
    conditions[("blur", 1)][2, 3] = 0.14

    result = module._pairwise_confusion_onset(conditions, clean, threshold=0.05)
    first_pair = result.loc[
        (result["class_a"] == module._FASHION_CLASS_NAMES[0])
        & (result["class_b"] == module._FASHION_CLASS_NAMES[1])
    ].iloc[0]
    second_pair = result.loc[
        (result["class_a"] == module._FASHION_CLASS_NAMES[2])
        & (result["class_b"] == module._FASHION_CLASS_NAMES[3])
    ].iloc[0]

    assert first_pair["onset_severity_index"] == 2
    assert first_pair["onset_corruption"] == "noise"
    assert second_pair["onset_severity_index"] == 1
    assert second_pair["onset_corruption"] == "blur"


def test_readme_corruption_atlas_asset_is_present_and_renderable():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    relative_png = Path("img") / "visuals" / "fashion-mnist-corruption-atlas.png"
    png_path = root / relative_png

    assert relative_png.as_posix() in readme
    assert png_path.with_suffix(".svg").is_file()
    with Image.open(png_path) as image:
        image.verify()
        assert image.width >= 1_500
        assert image.height >= 900
