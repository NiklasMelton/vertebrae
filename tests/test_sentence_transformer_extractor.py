import sys
import types

import numpy as np
import pytest

from vertebrae.extractors import SentenceTransformerExtractor


class FakeSentenceTransformer:
    init_kwargs = None
    encode_kwargs = None

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.__class__.init_kwargs = kwargs

    def encode(self, texts, **kwargs):
        self.__class__.encode_kwargs = kwargs
        return np.arange(len(texts) * 4, dtype=np.float64).reshape(len(texts), 4)


def test_sentence_transformer_extractor_uses_encode_kwargs(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    extractor = SentenceTransformerExtractor(
        name="minilm",
        model_id="fake-model",
        batch_size=7,
        normalize_embeddings=True,
        device="cpu",
        show_progress_bar=True,
        model_kwargs={"revision": "main"},
        encode_kwargs={"prompt_name": "query"},
    )

    output = extractor.transform(["alpha", "beta"])

    assert output.shape == (2, 4)
    assert output.dtype == np.float32
    assert FakeSentenceTransformer.init_kwargs["device"] == "cpu"
    assert FakeSentenceTransformer.init_kwargs["revision"] == "main"
    assert FakeSentenceTransformer.encode_kwargs["batch_size"] == 7
    assert FakeSentenceTransformer.encode_kwargs["normalize_embeddings"] is True
    assert FakeSentenceTransformer.encode_kwargs["prompt_name"] == "query"
    assert extractor.recipe()["extractor_type"] == "frozen_pretrained"


def test_sentence_transformer_rejects_non_string_input(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    extractor = SentenceTransformerExtractor("minilm", "fake-model")

    with pytest.raises(ValueError, match="string"):
        extractor.transform(["ok", object()])


def test_sentence_transformer_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    extractor = SentenceTransformerExtractor("minilm", "fake-model")

    with pytest.raises(ImportError, match="optional Hugging Face"):
        extractor.transform(["hello"])
