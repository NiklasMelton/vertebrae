import sys
import types

import numpy as np
import pytest

from vertebrae.extractors import HFVisionExtractor
from vertebrae.extractors.huggingface_vision import _coerce_image


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.device = "cpu"

    @property
    def shape(self):
        return self.data.shape

    def to(self, device):
        self.device = device
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def mean(self, dim=None):
        return FakeTensor(np.mean(self.data, axis=dim))

    def flatten(self, start_dim=0):
        leading = self.data.shape[:start_dim]
        flattened = int(np.prod(self.data.shape[start_dim:]))
        return FakeTensor(self.data.reshape(*leading, flattened))

    def __getitem__(self, key):
        return FakeTensor(self.data[key])


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False

    @staticmethod
    def no_grad():
        return FakeNoGrad()


class FakeProcessor:
    def __call__(self, images, **kwargs):
        return {"pixel_values": FakeTensor(np.zeros((len(images), 3, 2, 2)))}


class FakeVisionModel:
    last_call_kwargs = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        self.__class__.last_call_kwargs = encoded
        batch = encoded["pixel_values"].shape[0]
        hidden = np.arange(batch * 5 * 6, dtype=float).reshape(batch, 5, 6)
        hidden_states = tuple(FakeTensor(hidden + layer * 100) for layer in range(4))
        return types.SimpleNamespace(
            last_hidden_state=FakeTensor(hidden),
            pooler_output=FakeTensor(np.ones((batch, 6))),
            hidden_states=hidden_states if encoded.get("output_hidden_states") else None,
        )


class FakeSpatialPoolerVisionModel(FakeVisionModel):
    def __call__(self, **encoded):
        batch = encoded["pixel_values"].shape[0]
        hidden = np.arange(batch * 5 * 6, dtype=float).reshape(batch, 5, 6)
        return types.SimpleNamespace(
            last_hidden_state=FakeTensor(hidden),
            pooler_output=FakeTensor(np.ones((batch, 6, 1, 1))),
        )


class FakeAutoImageProcessor:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeProcessor()


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeVisionModel()


class FakeSpatialPoolerAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeSpatialPoolerVisionModel()


class FakeImageModule:
    Image = object

    @staticmethod
    def fromarray(value):
        return value

    @staticmethod
    def open(value):
        return types.SimpleNamespace(convert=lambda mode: value)


@pytest.fixture
def fake_vision_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=FakeImageModule))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoImageProcessor=FakeAutoImageProcessor,
            AutoModel=FakeAutoModel,
        ),
    )


@pytest.mark.parametrize("pooling", ["cls", "mean", "pooler"])
def test_hf_vision_pooling_modes(fake_vision_modules, pooling):
    extractor = HFVisionExtractor("vit", "fake-vision", pooling=pooling, batch_size=2)

    output = extractor.transform([np.zeros((4, 4, 3), dtype=np.uint8)] * 3)

    assert output.shape == (3, 6)
    assert output.dtype == np.float32
    assert extractor.recipe()["modality"] == "image"


def test_hf_vision_recipe_includes_image_conversion_options():
    extractor = HFVisionExtractor(
        "vit",
        "fake-vision",
        processor_id="fake-processor",
        hidden_layer=2,
        image_mode="grayscale",
        alpha_mode="white_background",
    )

    recipe = extractor.recipe()

    assert recipe["processor_id"] == "fake-processor"
    assert recipe["hidden_layer"] == 2
    assert recipe["image_mode"] == "grayscale"
    assert recipe["alpha_mode"] == "white_background"


def test_hf_vision_flattens_spatial_pooler_output(fake_vision_modules, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoImageProcessor=FakeAutoImageProcessor,
            AutoModel=FakeSpatialPoolerAutoModel,
        ),
    )
    extractor = HFVisionExtractor("resnet", "fake-resnet", pooling="pooler", batch_size=2)

    output = extractor.transform([np.zeros((4, 4, 3), dtype=np.uint8)] * 3)

    assert output.shape == (3, 6)
    assert output.dtype == np.float32


def test_hf_vision_selects_hidden_layer(fake_vision_modules):
    extractor = HFVisionExtractor(
        "vit",
        "fake-vision",
        pooling="cls",
        hidden_layer=2,
        batch_size=2,
    )

    output = extractor.transform([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)

    assert output.tolist() == [
        [200.0, 201.0, 202.0, 203.0, 204.0, 205.0],
        [230.0, 231.0, 232.0, 233.0, 234.0, 235.0],
    ]
    assert FakeVisionModel.last_call_kwargs["output_hidden_states"] is True


def test_hf_vision_rejects_pooler_with_hidden_layer(fake_vision_modules):
    extractor = HFVisionExtractor(
        "vit",
        "fake-vision",
        pooling="pooler",
        hidden_layer=1,
    )

    with pytest.raises(ValueError, match="pooler"):
        extractor.transform([np.zeros((4, 4, 3), dtype=np.uint8)])


def test_hf_vision_rejects_out_of_range_hidden_layer(fake_vision_modules):
    extractor = HFVisionExtractor(
        "vit",
        "fake-vision",
        pooling="mean",
        hidden_layer=99,
    )

    with pytest.raises(ValueError, match="out of range"):
        extractor.transform([np.zeros((4, 4, 3), dtype=np.uint8)])


def test_hf_vision_rgb_mode_converts_supported_numpy_shapes():
    image_module = pytest.importorskip("PIL.Image")
    gray = np.arange(4, dtype=np.uint8).reshape(2, 2)
    single_channel = gray[:, :, np.newaxis]
    rgb = np.dstack([gray, gray, gray])
    rgba = np.dstack([gray, gray, gray, np.full_like(gray, 255)])

    outputs = [
        _coerce_image(value, image_module, image_mode="rgb", alpha_mode="drop")
        for value in [gray, single_channel, rgb, rgba]
    ]

    assert [output.mode for output in outputs] == ["RGB", "RGB", "RGB", "RGB"]
    assert [output.size for output in outputs] == [(2, 2), (2, 2), (2, 2), (2, 2)]


def test_hf_vision_grayscale_mode_converts_supported_numpy_shapes():
    image_module = pytest.importorskip("PIL.Image")
    gray = np.arange(4, dtype=np.uint8).reshape(2, 2)
    single_channel = gray[:, :, np.newaxis]
    rgb = np.dstack([gray, gray, gray])
    rgba = np.dstack([gray, gray, gray, np.full_like(gray, 255)])

    outputs = [
        _coerce_image(value, image_module, image_mode="grayscale", alpha_mode="drop")
        for value in [gray, single_channel, rgb, rgba]
    ]

    assert [output.shape for output in outputs] == [(2, 2, 1)] * 4
    assert all(output.dtype == np.uint8 for output in outputs)


def test_hf_vision_alpha_modes_are_deterministic():
    image_module = pytest.importorskip("PIL.Image")
    transparent_red = np.asarray([[[255, 0, 0, 0]]], dtype=np.uint8)

    dropped = _coerce_image(transparent_red, image_module, image_mode="rgb", alpha_mode="drop")
    white = _coerce_image(
        transparent_red,
        image_module,
        image_mode="rgb",
        alpha_mode="white_background",
    )
    black = _coerce_image(
        transparent_red,
        image_module,
        image_mode="rgb",
        alpha_mode="black_background",
    )

    assert np.asarray(dropped).tolist() == [[[255, 0, 0]]]
    assert np.asarray(white).tolist() == [[[255, 255, 255]]]
    assert np.asarray(black).tolist() == [[[0, 0, 0]]]


def test_hf_vision_rejects_invalid_image_conversion_options():
    with pytest.raises(ValueError, match="image_mode"):
        HFVisionExtractor("vit", "fake-vision", image_mode="cmyk")
    with pytest.raises(ValueError, match="alpha_mode"):
        HFVisionExtractor("vit", "fake-vision", alpha_mode="checkerboard")


def test_hf_vision_rejects_unsupported_numpy_image_shapes():
    image_module = pytest.importorskip("PIL.Image")
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        _coerce_image(
            np.zeros((2, 2, 2), dtype=np.uint8),
            image_module,
            image_mode="rgb",
            alpha_mode="drop",
        )
    with pytest.raises(ValueError, match="shape"):
        _coerce_image(
            np.zeros((2, 2, 2, 1), dtype=np.uint8),
            image_module,
            image_mode="rgb",
            alpha_mode="drop",
        )


def test_hf_vision_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "PIL", None)
    extractor = HFVisionExtractor("vit", "fake-vision")

    with pytest.raises(ImportError, match="optional Hugging Face vision"):
        extractor.transform([np.zeros((4, 4, 3), dtype=np.uint8)])
