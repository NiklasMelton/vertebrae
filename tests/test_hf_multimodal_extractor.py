import sys
import types

import numpy as np
import pytest

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, ProbeConfig, SeparatixConfig, StabilityConfig
from vertebrae.extractors import HFMultimodalExtractor


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
    last_kwargs = None

    def __call__(self, **kwargs):
        self.__class__.last_kwargs = kwargs
        batch = len(kwargs["text"])
        return {
            "input_ids": FakeTensor(np.arange(batch * 4).reshape(batch, 4)),
            "pixel_values": FakeTensor(np.zeros((batch, 3, 2, 2))),
        }


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        batch = encoded["input_ids"].shape[0]
        hidden = np.arange(batch * 3 * 4, dtype=float).reshape(batch, 3, 4)
        return types.SimpleNamespace(
            image_embeds=FakeTensor(np.full((batch, 4), 2.0)),
            text_embeds=FakeTensor(np.full((batch, 4), 3.0)),
            pooler_output=FakeTensor(np.full((batch, 4), 5.0)),
            hidden_states=tuple(FakeTensor(hidden + index * 100.0) for index in range(4)),
        )


class FakeAutoProcessor:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeProcessor()


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeModel()


class FakeImageModule:
    @staticmethod
    def fromarray(value):
        return value

    @staticmethod
    def open(value):
        return types.SimpleNamespace(convert=lambda mode: value)


@pytest.fixture
def fake_multimodal_modules(monkeypatch):
    FakeProcessor.last_kwargs = None
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=FakeImageModule))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=FakeAutoModel,
            AutoProcessor=FakeAutoProcessor,
        ),
    )


def _dataset():
    return BenchmarkDataset.from_multimodal(
        inputs={
            "image": [np.zeros((2, 2, 3), dtype=np.uint8)] * 4,
            "caption": ["one", "two", "three", "four"],
        },
        labels=["left", "left", "right", "right"],
        modalities={"image": "image", "caption": "text"},
    )


def test_hf_multimodal_transform_many_uses_default_text_image_mapping(fake_multimodal_modules):
    extractor = HFMultimodalExtractor(
        name="clip",
        model_id="fake-clip",
        input_modalities={"image": "image", "caption": "text"},
        outputs=[
            {"name": "image_branch", "source": "image", "model_output": "image_embeds"},
            {"name": "text_branch", "source": "text", "model_output": "text_embeds"},
            {"name": "fused", "source": "fused", "model_output": "pooler_output"},
        ],
        batch_size=2,
    )

    outputs = extractor.transform_many(_dataset().X)

    assert [output.name for output in outputs] == ["image_branch", "text_branch", "fused"]
    assert all(output.embeddings.shape == (4, 4) for output in outputs)
    assert FakeProcessor.last_kwargs["text"] == ["three", "four"]
    assert len(FakeProcessor.last_kwargs["images"]) == 2


def test_hf_multimodal_supports_input_and_output_adapters(fake_multimodal_modules):
    extractor = HFMultimodalExtractor(
        name="clip",
        model_id="fake-clip",
        input_modalities={"image": "image", "caption": "text"},
        outputs=[{"name": "fused", "source": "fused", "model_output": "custom"}],
        input_fn=lambda batch: {
            "text": [item.upper() for item in batch["caption"]],
            "images": batch["image"],
        },
        output_fn=lambda model_output: {"fused": model_output.pooler_output},
        batch_size=2,
    )

    output = extractor.transform(_dataset().X)

    assert output.shape == (4, 4)
    assert FakeProcessor.last_kwargs["text"] == ["THREE", "FOUR"]


def test_hf_multimodal_selects_hidden_state_and_pooling(fake_multimodal_modules):
    extractor = HFMultimodalExtractor(
        name="clip",
        model_id="fake-clip",
        input_modalities={"image": "image", "caption": "text"},
        outputs=[
            {
                "name": "hidden_cls",
                "source": "fused",
                "model_output": "hidden_states",
                "hidden_layer": 2,
                "pooling": "cls",
            }
        ],
        batch_size=2,
    )

    output = extractor.transform(_dataset().X)

    assert output.tolist() == [
        [200.0, 201.0, 202.0, 203.0],
        [212.0, 213.0, 214.0, 215.0],
        [200.0, 201.0, 202.0, 203.0],
        [212.0, 213.0, 214.0, 215.0],
    ]


def test_hf_multimodal_recipe_and_report_metadata(
    fake_multimodal_modules,
    fake_overlapindex,
    tmp_path,
):
    extractor = HFMultimodalExtractor(
        name="clip",
        model_id="fake-clip",
        input_modalities={"image": "image", "caption": "text"},
        outputs=[{"name": "fused", "source": "fused", "model_output": "pooler_output"}],
        batch_size=2,
    )

    recipe = extractor.recipe()
    result = Evaluator(
        dataset=_dataset(),
        extractor=extractor,
        stability_config=StabilityConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()
    markdown_path = tmp_path / "multimodal.md"
    result.save_markdown(str(markdown_path))
    report = markdown_path.read_text(encoding="utf-8")

    assert recipe["modality"] == "multimodal"
    assert recipe["input_modalities"] == {"image": "image", "caption": "text"}
    assert result.extractor_results[0].embedding_metadata["output_metadata"]["source"] == "fused"
    assert "Modalities: {'image': 'image', 'caption': 'text'}" in report
    assert "Output source: fused" in report
    assert len(fake_overlapindex.calls) == 1


def test_hf_multimodal_rejects_invalid_output_resolution(fake_multimodal_modules):
    extractor = HFMultimodalExtractor(
        name="clip",
        model_id="fake-clip",
        input_modalities={"image": "image", "caption": "text"},
        outputs=[{"name": "missing", "source": "fused", "model_output": "does_not_exist"}],
    )

    with pytest.raises(ValueError, match="could not resolve"):
        extractor.transform(_dataset().X)


def test_hf_multimodal_rejects_invalid_output_shape(fake_multimodal_modules):
    extractor = HFMultimodalExtractor(
        name="clip",
        model_id="fake-clip",
        input_modalities={"image": "image", "caption": "text"},
        outputs=[{"name": "bad", "source": "fused", "model_output": "bad"}],
        output_fn=lambda model_output: {"bad": np.array([1.0, 2.0, 3.0])},
    )

    with pytest.raises(ValueError, match="1D vector"):
        extractor.transform(_dataset().X)


def test_hf_multimodal_lazy_import_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    original_import = __import__

    def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"torch", "PIL", "transformers"}:
            raise ImportError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", raising_import)
    extractor = HFMultimodalExtractor(
        name="clip",
        model_id="fake-clip",
        input_modalities={"image": "image", "caption": "text"},
        outputs=[{"name": "fused", "source": "fused", "model_output": "pooler_output"}],
    )

    with pytest.raises(ImportError, match="optional Hugging Face multi-modal dependencies"):
        extractor.transform_many(
            {
                "image": [np.zeros((2, 2, 3), dtype=np.uint8)],
                "caption": ["one"],
            }
        )
