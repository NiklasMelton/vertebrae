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
    last_call_kwargs = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.did_eval = True
        return self

    def __call__(self, **encoded):
        self.__class__.last_call_kwargs = encoded
        batch, seq_len = encoded["input_ids"].shape
        hidden = np.arange(batch * seq_len * 3, dtype=float).reshape(batch, seq_len, 3)
        hidden_states = tuple(FakeTensor(hidden + layer * 100) for layer in range(4))
        return types.SimpleNamespace(
            last_hidden_state=FakeTensor(hidden),
            hidden_states=hidden_states if encoded.get("output_hidden_states") else None,
        )


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


def test_hf_text_selects_hidden_layer(fake_hf_modules):
    extractor = HFTextExtractor(
        name="hf_layer",
        model_id="fake-model",
        pooling="cls",
        hidden_layer=2,
        batch_size=2,
    )

    output = extractor.transform(["one", "two"])

    assert output.tolist() == [[200.0, 201.0, 202.0], [212.0, 213.0, 214.0]]
    assert FakeModel.last_call_kwargs["output_hidden_states"] is True
    assert extractor.recipe()["hidden_layer"] == 2


def test_hf_text_rejects_out_of_range_hidden_layer(fake_hf_modules):
    extractor = HFTextExtractor(
        name="hf_layer",
        model_id="fake-model",
        pooling="mean",
        hidden_layer=99,
    )

    with pytest.raises(ValueError, match="out of range"):
        extractor.transform(["one"])


def test_hf_text_missing_optional_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    extractor = HFTextExtractor("hf", "fake-model")

    with pytest.raises(ImportError, match="optional Hugging Face"):
        extractor.transform(["hello"])


def test_hf_text_supports_structured_token_outputs(fake_hf_modules):
    extractor = HFTextExtractor(
        "hf",
        "fake-model",
        structured_outputs=[{"name": "tokens", "hidden_layer": 2}],
        batch_size=2,
    )

    output = extractor.transform_structured(["one", "two"])[0]

    assert output.name == "tokens"
    assert output.unit_type == "token"
    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (0, 3)
    assert output.embeddings[1].shape == (1, 3)
