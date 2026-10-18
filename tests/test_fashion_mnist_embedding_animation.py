import importlib.util
import re
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
    assert "--epochs" in completed.stdout
    assert "--train-size" in completed.stdout
    assert "--validation-size" in completed.stdout
    assert "--learning-rate" in completed.stdout
    assert "--snapshot-every-batches" in completed.stdout
    assert "--interpolation-frames" in completed.stdout
    assert "--fps" in completed.stdout
    assert "--final-hold-seconds" in completed.stdout
    assert "--no-align" in completed.stdout
    assert "--no-download" in completed.stdout


def test_model_has_true_two_dimensional_linear_bottleneck_without_post_activation():
    torch = pytest.importorskip("torch")
    module = _load_example_module()
    model = module._build_model(torch)

    assert hasattr(model, "pre_bottleneck")
    assert isinstance(model.pre_bottleneck, torch.nn.Sequential)
    assert isinstance(model.pre_bottleneck[0], torch.nn.Linear)
    assert model.pre_bottleneck[0].in_features == 64 * 7 * 7
    assert model.pre_bottleneck[0].out_features == 256
    assert isinstance(model.pre_bottleneck[1], torch.nn.ReLU)
    assert isinstance(model.pre_bottleneck[2], torch.nn.Linear)
    assert model.pre_bottleneck[2].in_features == 256
    assert model.pre_bottleneck[2].out_features == 128
    assert isinstance(model.pre_bottleneck[3], torch.nn.ReLU)
    assert isinstance(model.bottleneck, torch.nn.Linear)
    assert model.bottleneck.in_features == 128
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
            (values[:, 0] * module._NORMALIZATION_STD + module._NORMALIZATION_MEAN) * 255.0
        ).astype(int)

    train_ids = restore_ids(train_x)
    validation_ids = restore_ids(validation_x)
    assert set(train_ids).isdisjoint(set(validation_ids))
    assert np.bincount(train_y, minlength=10).tolist() == [5] * 10
    assert np.bincount(validation_y, minlength=10).tolist() == [3] * 10


def test_fashion_mnist_split_handles_promoted_large_sizes_and_capacity_guard(tmp_path):
    module = _load_example_module()

    class LargeFakeFashionMNIST:
        def __init__(self, root, train, download):
            sample_ids = np.arange(60_000, dtype=np.int64)
            low_byte = (sample_ids % 256).astype(np.uint8)
            high_byte = (sample_ids // 256).astype(np.uint8)
            self.data = np.stack([low_byte, high_byte], axis=1)[:, None, :]
            self.targets = np.repeat(np.arange(10, dtype=np.int64), 6_000)

    train_x, train_y, validation_x, validation_y = module._load_fashion_mnist(
        LargeFakeFashionMNIST,
        data_dir=tmp_path,
        train_size=50_000,
        validation_size=1_000,
        seed=37,
        download=False,
    )

    def restore_ids(values):
        raw_pixels = np.rint(
            (values[:, :2] * module._NORMALIZATION_STD + module._NORMALIZATION_MEAN) * 255.0
        ).astype(np.int64)
        return raw_pixels[:, 0] + 256 * raw_pixels[:, 1]

    train_ids = restore_ids(train_x)
    validation_ids = restore_ids(validation_x)
    assert train_x.shape == (50_000, 2)
    assert validation_x.shape == (1_000, 2)
    assert set(train_ids).isdisjoint(set(validation_ids))
    assert np.bincount(train_y, minlength=10).tolist() == [5_000] * 10
    assert np.bincount(validation_y, minlength=10).tolist() == [100] * 10

    with pytest.raises(ValueError, match="exceed"):
        module._load_fashion_mnist(
            LargeFakeFashionMNIST,
            data_dir=tmp_path,
            train_size=59_001,
            validation_size=1_000,
            seed=37,
            download=False,
        )


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
            displayed.embeddings @ displayed.classifier_weight.T + displayed.classifier_bias
        )
        assert np.allclose(display_logits, source_logits)


def test_display_interpolation_preserves_endpoints_and_linearly_interpolates_model_state(
    monkeypatch,
):
    module = _load_example_module()
    labels = np.array([0, 1, 0], dtype=np.int64)

    def make_snapshot(step, embeddings, classifier_weight, classifier_bias):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        classifier_weight = np.asarray(classifier_weight, dtype=np.float64)
        classifier_bias = np.asarray(classifier_bias, dtype=np.float64)
        logits = embeddings @ classifier_weight.T + classifier_bias
        return module.EmbeddingSnapshot(
            epoch=step,
            global_step=step,
            batch_in_epoch=step,
            training_loss=float(step),
            validation_accuracy=float(np.mean(logits.argmax(axis=1) == labels)),
            overlap_index=float(step),
            embeddings=embeddings,
            classifier_weight=classifier_weight,
            classifier_bias=classifier_bias,
        )

    snapshots = [
        make_snapshot(
            0,
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [0.0, 0.0],
        ),
        make_snapshot(
            4,
            [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [0.0, 0.0],
        ),
        make_snapshot(
            8,
            [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [0.25, -0.25],
        ),
    ]
    displayed = module._display_snapshots(snapshots, align=False)
    scored_embeddings = []

    def fake_score(embeddings, labels_arg, scoring_config):
        scored_embeddings.append((embeddings.copy(), labels_arg.copy(), scoring_config))
        return 0.42

    monkeypatch.setattr(module, "_score_embeddings", fake_score)
    frames = module._interpolate_display_snapshots(
        displayed,
        labels,
        frames_between=2,
        scoring_config=object(),
    )

    assert len(frames) == 7
    assert frames[0] is displayed[0]
    assert frames[3] is displayed[1]
    assert frames[6] is displayed[2]
    assert [frame.source.global_step for frame in frames] == [0, -1, -1, 4, -1, -1, 8]
    assert all(frame.interpolation is None for frame in (frames[0], frames[3], frames[6]))
    assert len({len(module._frame_overlay_text(frame)) for frame in frames}) == 1

    first_tween = frames[1]
    assert first_tween.interpolation is not None
    assert first_tween.source.epoch == -1
    assert first_tween.source.global_step == -1
    assert first_tween.source.batch_in_epoch == -1
    assert np.isnan(first_tween.source.training_loss)
    assert first_tween.interpolation.start_step == 0
    assert first_tween.interpolation.end_step == 4
    assert first_tween.interpolation.fraction == pytest.approx(1 / 3)
    assert np.allclose(
        first_tween.embeddings,
        (2 * displayed[0].embeddings + displayed[1].embeddings) / 3,
    )
    assert np.allclose(
        first_tween.classifier_weight,
        (2 * displayed[0].classifier_weight + displayed[1].classifier_weight) / 3,
    )
    assert np.allclose(
        first_tween.classifier_bias,
        (2 * displayed[0].classifier_bias + displayed[1].classifier_bias) / 3,
    )
    first_tween_logits = (
        first_tween.embeddings @ first_tween.classifier_weight.T + first_tween.classifier_bias
    )
    assert np.array_equal(first_tween_logits.argmax(axis=1), np.array([0, 1, 1]))
    assert first_tween.source.validation_accuracy == pytest.approx(2 / 3)
    assert first_tween.source.overlap_index == pytest.approx(0.42)

    second_tween = frames[2]
    assert second_tween.interpolation is not None
    assert second_tween.interpolation.start_step == 0
    assert second_tween.interpolation.end_step == 4
    assert second_tween.interpolation.fraction == pytest.approx(2 / 3)
    assert np.allclose(
        second_tween.embeddings,
        (displayed[0].embeddings + 2 * displayed[1].embeddings) / 3,
    )

    assert len(scored_embeddings) == 4
    assert all(np.array_equal(labels_arg, labels) for _, labels_arg, _ in scored_embeddings)

    third_tween = frames[4]
    assert third_tween.interpolation is not None
    assert third_tween.source.epoch == -1
    assert third_tween.source.global_step == -1
    assert third_tween.source.batch_in_epoch == -1
    assert np.isnan(third_tween.source.training_loss)
    assert third_tween.interpolation.start_step == 4
    assert third_tween.interpolation.end_step == 8
    assert third_tween.interpolation.fraction == pytest.approx(1 / 3)

    fourth_tween = frames[5]
    assert fourth_tween.interpolation is not None
    assert fourth_tween.interpolation.start_step == 4
    assert fourth_tween.interpolation.end_step == 8
    assert fourth_tween.interpolation.fraction == pytest.approx(2 / 3)


def test_display_interpolation_zero_frames_is_identity_and_invalid_cli_is_rejected():
    module = _load_example_module()
    snapshot = module.EmbeddingSnapshot(
        epoch=0,
        global_step=0,
        batch_in_epoch=0,
        training_loss=1.0,
        validation_accuracy=0.5,
        overlap_index=0.1,
        embeddings=np.zeros((2, 2), dtype=np.float64),
        classifier_weight=np.eye(2, dtype=np.float64),
        classifier_bias=np.zeros(2, dtype=np.float64),
    )
    displayed = module._display_snapshots([snapshot], align=False)
    result = module._interpolate_display_snapshots(
        displayed,
        np.array([0, 1], dtype=np.int64),
        frames_between=0,
        scoring_config=object(),
    )
    assert len(result) == 1
    assert result[0] is displayed[0]

    root = Path(__file__).resolve().parents[1]
    for invalid_value in ("-1", "5"):
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "examples" / "fashion_mnist_embedding_animation.py"),
                "--interpolation-frames",
                invalid_value,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode != 0
        assert "--interpolation-frames must be between 0 and 4" in completed.stderr


def test_frame_overlay_text_uses_fixed_template_and_interpolates_tween_step():
    module = _load_example_module()
    zero_embeddings = np.zeros((2, 2), dtype=np.float64)
    zero_weights = np.zeros((2, 2), dtype=np.float64)
    zero_bias = np.zeros(2, dtype=np.float64)
    checkpoint_source = module.EmbeddingSnapshot(
        epoch=2,
        global_step=8,
        batch_in_epoch=4,
        training_loss=0.42,
        validation_accuracy=0.75,
        overlap_index=0.123,
        embeddings=zero_embeddings,
        classifier_weight=zero_weights,
        classifier_bias=zero_bias,
    )
    checkpoint = module.DisplaySnapshot(
        source=checkpoint_source,
        embeddings=zero_embeddings,
        classifier_weight=zero_weights,
        classifier_bias=zero_bias,
    )
    tween_source = module.EmbeddingSnapshot(
        epoch=-1,
        global_step=-1,
        batch_in_epoch=-1,
        training_loss=float("nan"),
        validation_accuracy=0.625,
        overlap_index=0.456,
        embeddings=zero_embeddings,
        classifier_weight=zero_weights,
        classifier_bias=zero_bias,
    )
    tween = module.DisplaySnapshot(
        source=tween_source,
        embeddings=zero_embeddings,
        classifier_weight=zero_weights,
        classifier_bias=zero_bias,
        interpolation=module.InterpolationInfo(
            start_step=8,
            end_step=24,
            fraction=0.25,
        ),
    )

    checkpoint_text = module._frame_overlay_text(checkpoint)
    tween_text = module._frame_overlay_text(tween)

    assert len(checkpoint_text) == len(tween_text)
    for marker in ("STEP", "OVERLAPINDEX", "ACCURACY"):
        assert marker in checkpoint_text
        assert marker in tween_text
    for text_value in (checkpoint_text, tween_text):
        lowered = text_value.lower()
        assert "display interpolation" not in lowered
        assert "epoch" not in lowered
        assert "training loss" not in lowered

    step_pattern = re.compile(r"STEP\s+([0-9]+(?:\.[0-9])?)", re.IGNORECASE)
    checkpoint_step = step_pattern.search(checkpoint_text)
    tween_step = step_pattern.search(tween_text)
    assert checkpoint_step is not None
    assert tween_step is not None
    assert float(checkpoint_step.group(1)) == pytest.approx(8.0)
    assert float(tween_step.group(1)) == pytest.approx(12.0)


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
        assert image.n_frames >= 350
        assert image.info.get("loop") == 0
        duration = image.info.get("duration")
        assert duration is not None
        assert 0 < duration <= 50
        image.seek(image.n_frames - 1)
        image.load()
