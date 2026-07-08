import sys
import types

import numpy as np
import pytest
from scipy import sparse

from vertebrae import BenchmarkDataset, CacheConfig, Evaluator
from vertebrae.config import OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import (
    GraphModelExtractor,
    HostedEmbeddingExtractor,
    JAXFlaxExtractor,
    OpenCLIPExtractor,
    SigLIPExtractor,
    TFHubExtractor,
    TimmVisionExtractor,
    TorchvisionVisionExtractor,
    TreeLeafEmbeddingExtractor,
)


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

    def __getitem__(self, key):
        return FakeTensor(self.data[key])


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    Tensor = FakeTensor

    class cuda:
        @staticmethod
        def is_available():
            return False

    @staticmethod
    def no_grad():
        return FakeNoGrad()

    @staticmethod
    def stack(values):
        return FakeTensor(np.stack([value.data for value in values], axis=0))


class FakeImageModule:
    @staticmethod
    def fromarray(value):
        return value

    @staticmethod
    def open(value):
        return types.SimpleNamespace(convert=lambda mode: value)


@pytest.fixture
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=FakeImageModule))
    return FakeTorch


def test_timm_vision_extractor_uses_default_transform(fake_torch, monkeypatch):
    class FakeTimmModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, batch):
            size = batch.shape[0]
            return {"features": FakeTensor(np.arange(size * 3, dtype=float).reshape(size, 3))}

    class FakeTimmData:
        @staticmethod
        def resolve_model_data_config(model):
            return {"size": 4}

        @staticmethod
        def create_transform(**kwargs):
            return lambda image: FakeTensor(np.asarray(image, dtype=float))

    monkeypatch.setitem(
        sys.modules,
        "timm",
        types.SimpleNamespace(
            create_model=lambda model_name, pretrained, **kwargs: FakeTimmModel(),
            data=FakeTimmData,
        ),
    )
    extractor = TimmVisionExtractor(
        "timm_vit",
        "fake-vit",
        outputs=[{"name": "features", "selector": "features"}],
        batch_size=2,
    )

    output = extractor.transform([np.zeros((2, 2, 3), dtype=np.uint8)] * 3)

    assert output.shape == (3, 3)
    assert extractor.recipe()["preprocess_fn"] == "<timm-default>"


def test_torchvision_vision_extractor_uses_weight_transforms(fake_torch, monkeypatch):
    class FakeWeights:
        def transforms(self):
            return lambda image: FakeTensor(np.asarray(image, dtype=float) + 1.0)

    class FakeVisionModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, batch):
            size = batch.shape[0]
            return FakeTensor(np.full((size, 4), 5.0))

    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        types.SimpleNamespace(
            models=types.SimpleNamespace(resnet18=lambda weights, **kwargs: FakeVisionModel())
        ),
    )
    extractor = TorchvisionVisionExtractor("tv", "resnet18", weights=FakeWeights())

    output = extractor.transform([np.zeros((2, 2, 3), dtype=np.uint8)] * 2)

    assert output.shape == (2, 4)
    assert "torchvision-default" in extractor.recipe()["preprocess_fn"]


def test_openclip_extractor_emits_image_and_text_branches(fake_torch, monkeypatch):
    class FakeOpenCLIPModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def encode_image(self, batch):
            return FakeTensor(np.full((batch.shape[0], 3), 2.0))

        def encode_text(self, batch):
            return FakeTensor(np.full((batch.shape[0], 3), 7.0))

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        types.SimpleNamespace(
            create_model_and_transforms=(
                lambda model_name, pretrained, **kwargs: (
                    FakeOpenCLIPModel(),
                    None,
                    lambda image: FakeTensor(np.asarray(image, dtype=float)),
                )
            ),
            get_tokenizer=lambda model_name: (
                lambda texts: FakeTensor(np.arange(len(texts) * 2).reshape(len(texts), 2))
            ),
        ),
    )
    dataset = {
        "image": [np.zeros((2, 2, 3), dtype=np.uint8)] * 4,
        "text": ["a", "b", "c", "d"],
    }
    extractor = OpenCLIPExtractor(
        "clip",
        "fake-clip",
        outputs=[
            {"name": "image_branch", "source": "image"},
            {"name": "text_branch", "source": "text"},
        ],
        batch_size=2,
    )

    outputs = extractor.transform_many(dataset)

    assert [output.name for output in outputs] == ["image_branch", "text_branch"]
    assert outputs[0].embeddings.shape == (4, 3)
    assert outputs[1].embeddings.shape == (4, 3)


def test_siglip_extractor_uses_hf_multimodal_delegate(fake_torch, monkeypatch):
    class FakeProcessor:
        def __call__(self, **kwargs):
            size = len(kwargs["text"])
            return {
                "input_ids": FakeTensor(np.arange(size * 2).reshape(size, 2)),
                "pixel_values": FakeTensor(np.zeros((size, 3, 2, 2))),
            }

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            size = kwargs["input_ids"].shape[0]
            return types.SimpleNamespace(
                image_embeds=FakeTensor(np.full((size, 4), 11.0)),
                text_embeds=FakeTensor(np.full((size, 4), 13.0)),
            )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel()),
            AutoProcessor=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeProcessor()
            ),
        ),
    )
    extractor = SigLIPExtractor("siglip", "fake-siglip")
    output = extractor.transform_many(
        {
            "image": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
            "text": ["left", "right"],
        }
    )

    assert extractor.recipe()["extractor_type"] == "siglip"
    assert [item.name for item in output] == ["image_branch", "text_branch"]


def test_tfhub_extractor_supports_output_adapters(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tensorflow_hub",
        types.SimpleNamespace(
            load=lambda handle, **kwargs: (
                lambda batch, **call_kwargs: {"embeddings": FakeTensor(batch)}
            )
        ),
    )
    extractor = TFHubExtractor(
        "hub",
        "fake://hub",
        input_fn=lambda value: np.asarray(value, dtype=float),
        output_fn=lambda output: output["embeddings"],
    )

    output = extractor.transform([[1.0, 2.0], [3.0, 4.0]])

    assert output.shape == (2, 2)


def test_jax_flax_extractor_jits_apply(monkeypatch):
    jit_calls = []

    class FakeJax:
        @staticmethod
        def jit(fn):
            def wrapped(inputs):
                jit_calls.append("called")
                return fn(inputs)

            return wrapped

    monkeypatch.setitem(sys.modules, "jax", FakeJax)
    monkeypatch.setitem(sys.modules, "flax", types.SimpleNamespace())
    extractor = JAXFlaxExtractor(
        "jax_model",
        input_fn=lambda value: np.asarray(value, dtype=float),
        apply_fn=lambda inputs: {"embeddings": np.asarray(inputs, dtype=float)},
        output_fn=lambda output: output["embeddings"],
    )

    output = extractor.transform([[1.0, 2.0], [3.0, 4.0]])

    assert output.shape == (2, 2)
    assert jit_calls == ["called"]


def test_tree_leaf_extractor_supports_dense_and_sparse_outputs():
    class FakeXGBoostModel:
        __module__ = "xgboost.sklearn"

        def apply(self, X):
            return np.asarray([[1, 3], [2, 3], [1, 4]])

    model = FakeXGBoostModel()
    dense = TreeLeafEmbeddingExtractor("trees", model)
    dense_output = dense.transform(np.arange(6).reshape(3, 2))
    sparse_extractor = TreeLeafEmbeddingExtractor("trees_sparse", model, encoding="one_hot")
    sparse_output = sparse_extractor.fit_transform(np.arange(6).reshape(3, 2))

    assert dense_output.shape == (3, 2)
    assert sparse.issparse(sparse_output)
    assert sparse_output.shape[0] == 3


def test_graph_model_extractor_respects_framework_dependencies(fake_torch, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_geometric", types.SimpleNamespace())

    class FakeGraphModel:
        def __init__(self):
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

        def eval(self):
            return self

        def __call__(self, x):
            return FakeTensor(np.asarray(x.data)[:, :2])

    model = FakeGraphModel()
    extractor = GraphModelExtractor(
        "graph",
        model=model,
        collate_fn=lambda batch: FakeTensor(np.asarray(batch, dtype=float)),
        device="cpu",
        framework="pyg",
    )

    output = extractor.transform([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert output.shape == (2, 2)
    assert model.to_calls == ["cpu"]


def test_hosted_embedding_extractor_retries_and_skips_cache(tmp_path, fake_overlapindex):
    calls = []

    def embed_fn(batch):
        calls.append(list(batch))
        if len(calls) == 1:
            raise RuntimeError("try again")
        return np.arange(len(batch) * 2, dtype=float).reshape(len(batch), 2)

    dataset = BenchmarkDataset.from_arrays(
        np.asarray(["one", "two", "three", "four"], dtype=object),
        ["a", "a", "b", "b"],
        modality="text",
    )
    extractor = HostedEmbeddingExtractor(
        "hosted",
        provider="fake",
        model="embedding-test",
        embed_fn=embed_fn,
        batch_size=2,
        cache_embeddings=False,
    )
    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=1),
        stability_config=StabilityConfig(enabled=False),
        cache_config=CacheConfig(cache_dir=str(tmp_path), enabled=True),
    ).run()

    metadata = result.extractor_results[0].embedding_metadata
    assert len(calls) >= 2
    assert metadata["cache_key"]
    assert not (tmp_path / metadata["cache_key"]).exists()
