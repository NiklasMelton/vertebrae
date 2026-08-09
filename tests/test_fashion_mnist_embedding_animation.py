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
            "fashion_mnist_embedding_animation",
            examples_dir / "fashion_mnist_embedding_animation.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # ``from __future__ import annotations`` leaves dataclass annotations as
        # strings; dataclasses resolves those through ``sys.modules`` while the
        # module is being executed.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            return module
        finally:
            sys.modules.pop(spec.name, None)
    finally:
        sys.path.remove(str(examples_dir))


def test_fashion_mnist_embedding_animation_help_needs_no_optional_dependencies():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "fashion_mnist_embedding_animation.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--snapshot-every-batches" in completed.stdout
    assert "--no-align" in completed.stdout
    assert "--no-download" in completed.stdout


def test_model_has_true_two_dimensional_linear_bottleneck_without_post_activation():
    torch = pytest.importorskip("torch")
    module = _load_example_module()
    model = module._build_model(torch)

    assert isinstance(model.bottleneck, torch.nn.Linear)
    assert model.bottleneck.out_features == 2
    assert model.classifier.in_features == 2

    # A negative bottleneck bias must survive the forward pass if no ReLU (or
    # other post-bottleneck activation) is applied.
    with torch.no_grad():
        model.bottleneck.weight.zero_()
        model.bottleneck.bias.copy_(torch.tensor([-1.25, -2.5]))
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((3, 28 * 28), dtype=torch.float32))

    assert tuple(output["embedding"].shape) == (3, 2)
    assert tuple(output["logits"].shape) == (3, 10)
    assert torch.all(output["embedding"] < 0)


def test_fashion_mnist_split_is_deterministic_stratified_and_disjoint(tmp_path):
    module = _load_example_module()

    class FakeFashionMNIST:
        def __init__(self, root, train, download):
            row_ids = np.arange(200, dtype=np.float32)
            self.data = np.broadcast_to(row_ids[:, None, None], (200, 2, 2)).copy()
            self.targets = np.repeat(np.arange(10), 20)

    first = module._load_fashion_mnist(
        FakeFashionMNIST,
        data_dir=tmp_path,
        train_size=50,
        validation_size=30,
        seed=11,
        download=False,
    )
    second = module._load_fashion_mnist(
        FakeFashionMNIST,
        data_dir=tmp_path,
        train_size=50,
        validation_size=30,
        seed=11,
        download=False,
    )

    for first_value, second_value in zip(first, second):
        assert np.array_equal(first_value, second_value)

    train_x, train_y, validation_x, validation_y = first

    def restore_ids(values):
        return np.rint(
            (values[:, 0] * module._NORMALIZATION_STD + module._NORMALIZATION_MEAN)
            * 255.0
        ).astype(int)

    train_ids = restore_ids(train_x)
    validation_ids = restore_ids(validation_x)
    assert set(train_ids).isdisjoint(set(validation_ids))
    assert np.bincount(train_y, minlength=10).tolist() == [5] * 10
    assert np.bincount(validation_y, minlength=10).tolist() == [3] * 10


def test_display_alignment_reduces_rigid_coordinate_mismatch_and_preserves_geometry_and_logits():
    module = _load_example_module()

    points = np.array(
        [[-3.0, -1.0], [-2.0, 2.0], [-0.5, -2.0], [0.5, 3.0], [2.0, -0.5], [4.0, 1.0]],
        dtype=np.float64,
    )
    angle = 0.73
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    translation = np.array([4.0, -2.0])
    transformed_points = points @ rotation + translation
    weights = np.array([[1.2, -0.4], [-0.7, 0.9], [0.1, 0.8]], dtype=np.float64)
    bias = np.array([0.3, -0.2, 0.6], dtype=np.float64)
    transformed_weights = weights @ rotation
    transformed_bias = bias - transformed_weights @ translation

    snapshot = module.EmbeddingSnapshot(
        epoch=0,
        global_step=0,
        batch_in_epoch=0,
        training_loss=1.0,
        validation_accuracy=0.5,
        overlap_index=0.1,
        embeddings=points,
        classifier_weight=weights,
        classifier_bias=bias,
    )
    transformed_snapshot = module.EmbeddingSnapshot(
        epoch=1,
        global_step=1,
        batch_in_epoch=1,
        training_loss=0.8,
        validation_accuracy=0.6,
        overlap_index=0.2,
        embeddings=transformed_points,
        classifier_weight=transformed_weights,
        classifier_bias=transformed_bias,
    )

    raw = module._display_snapshots([snapshot, transformed_snapshot], align=False)
    aligned = module._display_snapshots([snapshot, transformed_snapshot], align=True)

    raw_mismatch = np.linalg.norm(
        (raw[1].embeddings - raw[1].embeddings.mean(axis=0))
        - (raw[0].embeddings - raw[0].embeddings.mean(axis=0)),
        axis=1,
    ).mean()
    aligned_mismatch = np.linalg.norm(
        aligned[1].embeddings - aligned[0].embeddings,
        axis=1,
    ).mean()
    assert aligned_mismatch < raw_mismatch
    assert aligned_mismatch < 1e-10

    for source, displayed in zip((snapshot, transformed_snapshot), aligned):
        source_distances = np.linalg.norm(
            source.embeddings[:, None, :] - source.embeddings[None, :, :],
            axis=2,
        )
        display_distances = np.linalg.norm(
            displayed.embeddings[:, None, :] - displayed.embeddings[None, :, :],
            axis=2,
        )
        assert np.allclose(display_distances, source_distances)
        source_logits = source.embeddings @ source.classifier_weight.T + source.classifier_bias
        display_logits = (
            displayed.embeddings @ displayed.classifier_weight.T
            + displayed.classifier_bias
        )
        assert np.allclose(display_logits, source_logits)


def test_save_looping_gif_writes_animated_gif_with_infinite_loop(tmp_path):
    module = _load_example_module()
    output_path = tmp_path / "animation.gif"
    frames = [
        Image.new("P", (8, 6), color=17),
        Image.new("P", (8, 6), color=203),
    ]

    module._save_looping_gif(
        frames,
        output_path,
        frame_duration_ms=40,
        final_hold_frames=1,
    )

    with Image.open(output_path) as image:
        assert image.format == "GIF"
        assert image.is_animated
        # Pillow may coalesce an identical final hold frame into the previous
        # frame while retaining the longer duration.
        assert image.n_frames >= 2
        assert image.info.get("loop") == 0
        assert image.info.get("duration") == 40


def test_committed_embedding_animation_gif_is_referenced_and_renderable():
    root = Path(__file__).resolve().parents[1]
    relative_gif = Path("img") / "visuals" / "fashion-mnist-embedding-evolution.gif"
    gif_path = root / relative_gif
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert relative_gif.as_posix() in readme
    assert gif_path.is_file()
    with Image.open(gif_path) as image:
        assert image.format == "GIF"
        assert image.is_animated
        assert image.width >= 600
        assert image.height >= 500
        assert image.n_frames >= 10
        assert image.info.get("loop") == 0
        image.seek(image.n_frames - 1)
        image.load()
