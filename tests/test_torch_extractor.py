import sys
import types

import numpy as np
import pytest

from vertebrae import BenchmarkDataset, EmbeddingConfig, Evaluator
from vertebrae.config import CacheConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import TorchExtractor


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


class FakeNoGrad:
    def __enter__(self):
        raise AssertionError("TorchExtractor must not enter no_grad automatically.")

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


class TrackingModel:
    def __init__(self, return_fn):
        self.return_fn = return_fn
        self.calls = []
        self.eval_called = False
        self.to_calls = []

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_fn(args, kwargs)


@pytest.fixture
def fake_torch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(Tensor=FakeTensor, cuda=FakeTorch.cuda, no_grad=FakeTorch.no_grad),
    )
    return sys.modules["torch"]


def _embeddings_from_first_tensor(args, kwargs):
    value = kwargs["x"] if kwargs else args[0]
    batch_size = value.shape[0]
    return FakeTensor(np.arange(batch_size * 2, dtype=float).reshape(batch_size, 2))


@pytest.mark.parametrize(
    "collate_fn, expected",
    [
        (lambda batch: {"x": FakeTensor(batch)}, "dict"),
        (lambda batch: (FakeTensor(batch), FakeTensor(batch)), "tuple"),
        (lambda batch: FakeTensor(batch), "single"),
    ],
)
def test_torch_extractor_dispatches_inputs(fake_torch, collate_fn, expected):
    model = TrackingModel(_embeddings_from_first_tensor)
    extractor = TorchExtractor("local", model=model, collate_fn=collate_fn)

    output = extractor.transform(np.ones((3, 4), dtype=float))

    assert output.shape == (3, 2)
    assert model.eval_called is False
    assert len(model.calls) == 1
    args, kwargs = model.calls[0]
    if expected == "dict":
        assert kwargs.keys() == {"x"}
        assert len(args) == 0
    elif expected == "tuple":
        assert len(args) == 2
        assert kwargs == {}
    else:
        assert len(args) == 1
        assert kwargs == {}


def test_torch_extractor_supports_output_fn_and_recipe(fake_torch):
    def collate_fn(batch):
        return {"x": FakeTensor(batch)}

    def output_fn(raw_output):
        return raw_output["embeddings"]

    class OutputModel(TrackingModel):
        pass

    model = OutputModel(
        lambda args, kwargs: {
            "embeddings": FakeTensor(np.arange(8, dtype=float).reshape(4, 2)),
            "logits": FakeTensor(np.zeros((4, 3))),
        }
    )
    extractor = TorchExtractor(
        "local",
        model=model,
        collate_fn=collate_fn,
        output_fn=output_fn,
        recipe_data={"checkpoint": "/tmp/model.pt", "revision": "abc123"},
    )

    output = extractor.transform(np.ones((4, 2), dtype=float))
    recipe = extractor.recipe()

    assert output.shape == (4, 2)
    assert recipe["model_class"].endswith(".OutputModel")
    assert recipe["recipe_data"] == {"checkpoint": "/tmp/model.pt", "revision": "abc123"}
    assert recipe["output_fn"].endswith(".output_fn")
    assert recipe["collate_fn"].endswith(".collate_fn")


def test_torch_extractor_moves_nested_batches_and_model(fake_torch):
    class NestedModel(TrackingModel):
        pass

    def return_fn(args, kwargs):
        batch = kwargs["inputs"] if kwargs else args[0]
        return FakeTensor(batch["primary"].data[:, :2])

    model = NestedModel(return_fn)
    nested_batch = {
        "inputs": {
            "primary": FakeTensor(np.arange(12, dtype=float).reshape(3, 4)),
            "auxiliary": [FakeTensor(np.ones((3, 1))), (FakeTensor(np.ones((3, 1))),)],
        }
    }
    extractor = TorchExtractor(
        "local",
        model=model,
        collate_fn=lambda batch: nested_batch,
        device="cuda:0",
        move_batch_to_device=True,
        move_model_to_device=True,
    )

    output = extractor.transform(np.ones((3, 4), dtype=float))

    assert output.shape == (3, 2)
    assert model.to_calls == ["cuda:0"]
    assert nested_batch["inputs"]["primary"].device == "cuda:0"
    assert nested_batch["inputs"]["auxiliary"][0].device == "cuda:0"
    assert nested_batch["inputs"]["auxiliary"][1][0].device == "cuda:0"


def test_torch_extractor_rejects_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    extractor = TorchExtractor(
        "local",
        model=object(),
        collate_fn=lambda batch: batch,
    )

    with pytest.raises(ImportError, match="poetry install -E torch"):
        extractor.transform(np.ones((2, 2), dtype=float))


@pytest.mark.parametrize(
    "output_value, message",
    [
        (FakeTensor(np.ones(3)), "2D"),
        (FakeTensor(np.array([[1.0, np.nan]])), "finite"),
        (np.array([["a", "b"]], dtype=object), "numeric"),
    ],
)
def test_torch_extractor_rejects_invalid_outputs(fake_torch, output_value, message):
    model = TrackingModel(lambda args, kwargs: output_value)
    extractor = TorchExtractor("local", model=model, collate_fn=lambda batch: FakeTensor(batch))

    with pytest.raises(ValueError, match=message):
        extractor.transform(np.ones((1, 2), dtype=float))


def test_torch_extractor_works_in_streaming_evaluator(fake_torch, fake_overlapindex):
    model = TrackingModel(
        lambda args, kwargs: FakeTensor(np.asarray(kwargs["x"].data[:, :2], dtype=float))
    )
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24, dtype=float).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
    )
    extractor = TorchExtractor(
        "streaming_local",
        model=model,
        collate_fn=lambda batch: {"x": FakeTensor(np.asarray(batch, dtype=float))},
        streaming_safe=True,
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        cache_config=CacheConfig(enabled=False),
        probe_config=ProbeConfig(enabled=False),
        stability_config=StabilityConfig(enabled=False),
        embedding_config=EmbeddingConfig(batch_size=3),
    ).run()

    metadata = result.extractor_results[0].embedding_metadata
    assert metadata["streamed"] is True
    assert metadata["stream_batch_size"] == 3
    assert len(model.calls) == 3
