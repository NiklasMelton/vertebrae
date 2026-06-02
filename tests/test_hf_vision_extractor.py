import sys
import types

import numpy as np
import pytest

from vertebrae.extractors import HFVisionExtractor


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
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        batch = encoded["pixel_values"].shape[0]
        hidden = np.arange(batch * 5 * 6, dtype=float).reshape(batch, 5, 6)
        return types.SimpleNamespace(
            last_hidden_state=FakeTensor(hidden),
            pooler_output=FakeTensor(np.ones((batch, 6))),
        )


class FakeAutoImageProcessor:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeProcessor()


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeVisionModel()


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
        types.SimpleNamespace(AutoImageProcessor=FakeAutoImageProcessor, AutoModel=FakeAutoModel),
    )


@pytest.mark.parametrize("pooling", ["cls", "mean", "pooler"])
def test_hf_vision_pooling_modes(fake_vision_modules, pooling):
    extractor = HFVisionExtractor("vit", "fake-vision", pooling=pooling, batch_size=2)

    output = extractor.transform([np.zeros((4, 4, 3), dtype=np.uint8)] * 3)

    assert output.shape == (3, 6)
    assert output.dtype == np.float32
    assert extractor.recipe()["modality"] == "image"


def test_hf_vision_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "PIL", None)
    extractor = HFVisionExtractor("vit", "fake-vision")

    with pytest.raises(ImportError, match="optional Hugging Face vision"):
        extractor.transform([np.zeros((4, 4, 3), dtype=np.uint8)])
