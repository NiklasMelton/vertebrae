import sys
import types

import numpy as np
import pytest

from vertebrae.extractors import HFTextExtractor


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.device = "cpu"

    @property
    def shape(self):
        return self.data.shape

    def size(self):
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

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.data, axis))

    def expand(self, shape):
        return FakeTensor(np.broadcast_to(self.data, shape))

    def float(self):
        return FakeTensor(self.data.astype(float))

    def sum(self, dim=None):
        return FakeTensor(np.sum(self.data, axis=dim))

    def clamp(self, min):
        return FakeTensor(np.maximum(self.data, min))

    def mean(self, dim=None):
        return FakeTensor(np.mean(self.data, axis=dim))

    def __mul__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data * other_data)

    def __truediv__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data / other_data)

    def __sub__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data - other_data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = tuple(_index_array(item) for item in key)
        return FakeTensor(self.data[key])


def _index_array(value):
    if isinstance(value, FakeTensor):
        return value.data.astype(int)
    return value


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

    @staticmethod
    def arange(n, device=None):
        return FakeTensor(np.arange(n))


class FakeTokenizer:
    calls = []

    def __call__(self, batch, **kwargs):
        self.calls.append(kwargs)
        seq_len = 4
        masks = []
        for idx, _text in enumerate(batch):
            active = min(seq_len, idx + 2)
            masks.append([1] * active + [0] * (seq_len - active))
        return {
            "input_ids": FakeTensor(np.zeros((len(batch), seq_len), dtype=int)),
            "attention_mask": FakeTensor(np.asarray(masks, dtype=int)),
        }


class FakeModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.did_eval = True
        return self

    def __call__(self, **encoded):
        batch, seq_len = encoded["input_ids"].shape
        hidden = np.arange(batch * seq_len * 3, dtype=float).reshape(batch, seq_len, 3)
        return types.SimpleNamespace(last_hidden_state=FakeTensor(hidden))


class FakeAutoTokenizer:
    kwargs = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.kwargs = kwargs
        return FakeTokenizer()


class FakeAutoModel:
    kwargs = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.kwargs = kwargs
        return FakeModel()


@pytest.fixture
def fake_hf_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=FakeAutoModel, AutoTokenizer=FakeAutoTokenizer),
    )


@pytest.mark.parametrize("pooling", ["mean", "cls", "last_token"])
def test_hf_text_pooling_modes(fake_hf_modules, pooling):
    extractor = HFTextExtractor(
        name=f"hf_{pooling}",
        model_id="fake-model",
        pooling=pooling,
        batch_size=2,
        tokenizer_kwargs={"clean_up_tokenization_spaces": False},
        model_kwargs={"attn_implementation": "eager"},
    )

    output = extractor.transform(["one", "two", "three"])

    assert output.shape == (3, 3)
    assert output.dtype == np.float32
    assert extractor.recipe()["pooling"] == pooling
    assert FakeAutoTokenizer.kwargs["trust_remote_code"] is False
    assert FakeAutoModel.kwargs["attn_implementation"] == "eager"


def test_hf_text_rejects_non_string_input(fake_hf_modules):
    extractor = HFTextExtractor("hf", "fake-model")

    with pytest.raises(ValueError, match="string"):
        extractor.transform(["ok", 123])


def test_hf_text_missing_optional_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    extractor = HFTextExtractor("hf", "fake-model")

    with pytest.raises(ImportError, match="optional Hugging Face"):
        extractor.transform(["hello"])
