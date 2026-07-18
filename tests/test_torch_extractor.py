import sys
import types
from collections import UserDict

import numpy as np
import pytest

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    DatasetIdentity,
    EmbeddingConfig,
    Evaluator,
    SpatialLayout,
    SpatialOutputSpec,
    StructuredOutputSpec,
)
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.config import CacheConfig, StabilityConfig
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
    entered = 0

    def __enter__(self):
        self.__class__.entered += 1
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeInferenceMode:
    entered = 0
    active = False

    def __enter__(self):
        self.__class__.entered += 1
        self.__class__.active = True
        return None

    def __exit__(self, exc_type, exc, tb):
        self.__class__.active = False
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
    def inference_mode():
        return FakeInferenceMode()


class TrackingModel:
    def __init__(self, return_fn):
        self.return_fn = return_fn
        self.calls = []
        self.eval_called = False
        self.to_calls = []
        self.training = True

    def eval(self):
        self.eval_called = True
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_fn(args, kwargs)


class TrackingChildModule:
    def __init__(self, training):
        self.training = training

    def train(self, mode=True):
        self.training = bool(mode)
        return self


class MixedModeTrackingModel(TrackingModel):
    def __init__(self, return_fn):
        super().__init__(return_fn)
        self.children = [TrackingChildModule(True), TrackingChildModule(False)]

    def modules(self):
        return [self, *self.children]

    def eval(self):
        self.eval_called = True
        return self.train(False)

    def train(self, mode=True):
        self.training = bool(mode)
        for child in self.children:
            child.train(mode)
        return self


@pytest.fixture
def fake_torch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            Tensor=FakeTensor,
            cuda=FakeTorch.cuda,
            no_grad=FakeTorch.no_grad,
            inference_mode=FakeTorch.inference_mode,
        ),
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
    assert model.eval_called is True
    assert model.training is True
    assert FakeInferenceMode.entered >= 1
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


def test_torch_extractor_supports_named_ordinary_outputs_in_one_call(fake_torch):
    output_fn_calls = []
    model = TrackingModel(
        lambda args, kwargs: {
            "hidden_states": [
                FakeTensor(np.arange(12, dtype=float).reshape(3, 2, 2)),
                FakeTensor(np.arange(12, 24, dtype=float).reshape(3, 2, 2)),
            ],
            "pooled": FakeTensor(np.arange(6, dtype=float).reshape(3, 2)),
        }
    )

    def output_fn(raw_output):
        output_fn_calls.append(raw_output)
        return raw_output

    extractor = TorchExtractor(
        "layers",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
        output_fn=output_fn,
        outputs=[
            {
                "name": "middle",
                "selector": "hidden_states.0",
                "hidden_layer": 1,
                "pooling": "flatten",
            },
            {
                "name": "final",
                "selector": "pooled",
                "flatten": False,
                "hidden_layer": 2,
                "pooling": "mean",
                "metadata": {"branch": "encoder"},
            },
        ],
    )

    outputs = extractor.transform_many(np.ones((3, 4), dtype=float))
    recipe = extractor.recipe()

    assert [output.name for output in outputs] == ["middle", "final"]
    assert [output.embeddings.shape for output in outputs] == [(3, 4), (3, 2)]
    assert len(model.calls) == 1
    assert len(output_fn_calls) == 1
    assert model.training is True
    assert recipe["outputs"][0]["selector"] == "hidden_states.0"
    assert recipe["outputs"][0]["hidden_layer"] == 1
    assert recipe["outputs"][1]["pooling"] == "mean"
    with pytest.raises(ValueError, match="transform_many"):
        extractor.transform(np.ones((3, 4), dtype=float))
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "values, message",
    [
        (
            {
                "middle": FakeTensor(np.ones((2, 2))),
                "final": FakeTensor(np.ones((2, 2))),
                "extra": FakeTensor(np.ones((2, 2))),
            },
            "extra=.*extra",
        ),
        (
            {"middle": FakeTensor(np.ones((2, 2)))},
            "missing=.*final",
        ),
    ],
)
def test_torch_extractor_requires_exact_direct_named_outputs(
    fake_torch,
    values,
    message,
):
    extractor = TorchExtractor(
        "layers",
        model=TrackingModel(lambda args, kwargs: values),
        collate_fn=lambda batch: FakeTensor(batch),
        outputs=[{"name": "middle"}, {"name": "final"}],
    )

    with pytest.raises(ValueError, match=message):
        extractor.transform_many(np.ones((2, 2)))


def test_torch_extractor_rejects_non_mapping_selector_free_outputs(fake_torch):
    extractor = TorchExtractor(
        "layers",
        model=TrackingModel(lambda args, kwargs: FakeTensor(np.ones((2, 2)))),
        collate_fn=lambda batch: FakeTensor(batch),
        outputs=[{"name": "middle"}, {"name": "final"}],
    )

    with pytest.raises(ValueError, match="must return a mapping"):
        extractor.transform_many(np.ones((2, 2)))


def test_torch_extractor_accepts_mapping_and_mixed_selectors(fake_torch):
    values = UserDict(
        {
            "middle": FakeTensor(np.ones((2, 2))),
            "nested": {"final": FakeTensor(np.full((2, 2), 2.0))},
        }
    )
    extractor = TorchExtractor(
        "layers",
        model=TrackingModel(lambda args, kwargs: values),
        collate_fn=lambda batch: FakeTensor(batch),
        outputs=[
            {"name": "middle"},
            {"name": "final", "selector": "nested.final"},
        ],
    )

    outputs = extractor.transform_many(np.ones((2, 2)))

    assert [output.name for output in outputs] == ["middle", "final"]
    assert np.array_equal(outputs[0].embeddings, np.ones((2, 2)))
    assert np.array_equal(outputs[1].embeddings, np.full((2, 2), 2.0))


def test_torch_extractor_resolves_selectors_from_custom_mappings(fake_torch):
    values = UserDict(
        {
            "nested": UserDict(
                {
                    "middle": FakeTensor(np.ones((2, 2))),
                    "final": FakeTensor(np.full((2, 2), 2.0)),
                }
            )
        }
    )
    extractor = TorchExtractor(
        "layers",
        model=TrackingModel(lambda args, kwargs: values),
        collate_fn=lambda batch: FakeTensor(batch),
        outputs=[
            {"name": "middle", "selector": "nested.middle"},
            {"name": "final", "selector": "nested.final"},
        ],
    )

    outputs = extractor.transform_many(np.ones((2, 2)))

    assert np.array_equal(outputs[0].embeddings, np.ones((2, 2)))
    assert np.array_equal(outputs[1].embeddings, np.full((2, 2), 2.0))


def test_torch_implicit_output_is_2d_but_explicit_output_may_flatten(fake_torch):
    model = TrackingModel(
        lambda args, kwargs: FakeTensor(np.arange(24, dtype=float).reshape(2, 3, 4))
    )
    implicit = TorchExtractor("implicit", model=model, collate_fn=lambda batch: FakeTensor(batch))
    explicit = TorchExtractor(
        "explicit",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
        outputs=[{"name": "tokens"}],
    )
    strict = TorchExtractor(
        "strict",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
        outputs=[{"name": "tokens", "flatten": False}],
    )

    with pytest.raises(ValueError, match="2D"):
        implicit.transform(np.ones((2, 2)))
    assert explicit.transform(np.ones((2, 2))).shape == (2, 12)
    with pytest.raises(ValueError, match="2D"):
        strict.transform(np.ones((2, 2)))
    assert implicit.recipe()["outputs"][0]["flatten"] is False
    assert explicit.recipe()["outputs"][0]["flatten"] is True
    assert fingerprint_extractor_recipe(implicit.recipe()) != fingerprint_extractor_recipe(
        explicit.recipe()
    )


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
        identity=DatasetIdentity.ephemeral(),
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
        stability_config=StabilityConfig(enabled=False),
        embedding_config=EmbeddingConfig(batch_size=3),
    ).run()

    metadata = result.extractor_results[0].embedding_metadata
    assert metadata["streamed"] is True
    assert metadata["stream_batch_size"] == 3
    assert len(model.calls) == 3


def test_torch_multi_output_streaming_calls_model_once_per_batch(
    fake_torch,
    fake_overlapindex,
):
    model = TrackingModel(
        lambda args, kwargs: {
            "middle": FakeTensor(np.asarray(kwargs["x"].data[:, :2], dtype=float)),
            "final": FakeTensor(np.asarray(kwargs["x"].data, dtype=float)),
        }
    )
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24, dtype=float).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = TorchExtractor(
        "streaming_layers",
        model=model,
        collate_fn=lambda batch: {"x": FakeTensor(np.asarray(batch, dtype=float))},
        outputs=[
            {"name": "middle", "hidden_layer": 1},
            {"name": "final", "hidden_layer": 2},
        ],
        streaming_safe=True,
    )

    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        cache_config=CacheConfig(enabled=False),
        stability_config=StabilityConfig(enabled=False),
        embedding_config=EmbeddingConfig(batch_size=3),
    ).run()

    assert len(result.extractor_results) == 2
    assert len(model.calls) == 3
    assert all(item.embedding_metadata["streamed"] is True for item in result.extractor_results)


def test_torch_extractor_supports_explicit_spatial_outputs(fake_torch):
    model = TrackingModel(
        lambda args, kwargs: {"features": FakeTensor(np.ones((2, 2, 2, 3), dtype=float))}
    )
    extractor = TorchExtractor(
        "spatial",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
        spatial_output_fn=lambda output: output["features"],
        spatial_output_specs=[SpatialOutputSpec("layer", SpatialLayout(2, 2))],
    )

    output = extractor.transform_spatial(np.ones((2, 4), dtype=float))[0]

    assert output.name == "layer"
    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (2, 2, 3)


def test_torch_extractor_supports_explicit_structured_outputs(fake_torch):
    model = TrackingModel(
        lambda args, kwargs: {"tokens": FakeTensor(np.ones((2, 3, 4), dtype=float))}
    )
    extractor = TorchExtractor(
        "structured",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
        structured_output_fn=lambda output: output["tokens"],
        structured_output_specs=[StructuredOutputSpec("tokens", unit_type="token")],
    )

    output = extractor.transform_structured(np.ones((2, 4), dtype=float))[0]

    assert output.name == "tokens"
    assert output.unit_type == "token"
    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (3, 4)


def test_torch_extractor_rejects_undeclared_named_adapter_outputs(fake_torch):
    model = TrackingModel(lambda args, kwargs: {"features": FakeTensor(np.ones((2, 2, 2, 3)))})
    extractor = TorchExtractor(
        "spatial",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
        spatial_output_fn=lambda output: {
            "layer": output["features"],
            "extra": output["features"],
        },
        spatial_output_specs=[SpatialOutputSpec("layer", SpatialLayout(2, 2))],
    )

    with pytest.raises(ValueError, match="extra=.*extra"):
        extractor.transform_spatial(np.ones((2, 4)))


def test_torch_extractor_can_opt_out_of_eval_and_inference_mode(fake_torch):
    model = TrackingModel(_embeddings_from_first_tensor)
    before = FakeNoGrad.entered
    before_inference = FakeInferenceMode.entered
    extractor = TorchExtractor(
        "local",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
        eval_mode=False,
        inference_mode=False,
    )

    extractor.transform(np.ones((2, 3), dtype=float))

    assert model.eval_called is False
    assert model.training is True
    assert FakeNoGrad.entered == before
    assert FakeInferenceMode.entered == before_inference


def test_torch_extractor_uses_inference_mode_and_restores_state(fake_torch):
    def return_embeddings(args, kwargs):
        assert FakeInferenceMode.active is True
        return _embeddings_from_first_tensor(args, kwargs)

    model = TrackingModel(return_embeddings)
    before_inference = FakeInferenceMode.entered
    before_no_grad = FakeNoGrad.entered

    TorchExtractor("local", model=model, collate_fn=lambda batch: FakeTensor(batch)).transform(
        np.ones((2, 3), dtype=float)
    )

    assert FakeInferenceMode.entered == before_inference + 1
    assert FakeNoGrad.entered == before_no_grad
    assert FakeInferenceMode.active is False
    assert model.training is True


def test_torch_extractor_restores_state_when_inference_raises(fake_torch):
    def fail(args, kwargs):
        assert FakeInferenceMode.active is True
        raise RuntimeError("model failed")

    model = TrackingModel(fail)
    extractor = TorchExtractor("local", model=model, collate_fn=lambda batch: FakeTensor(batch))

    with pytest.raises(RuntimeError, match="model failed"):
        extractor.transform(np.ones((2, 3), dtype=float))

    assert FakeInferenceMode.active is False
    assert model.training is True


@pytest.mark.parametrize("fail", [False, True])
def test_torch_extractor_restores_heterogeneous_submodule_modes(fake_torch, fail):
    def return_or_fail(args, kwargs):
        if fail:
            raise RuntimeError("model failed")
        return _embeddings_from_first_tensor(args, kwargs)

    model = MixedModeTrackingModel(return_or_fail)
    extractor = TorchExtractor(
        "local",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
    )

    if fail:
        with pytest.raises(RuntimeError, match="model failed"):
            extractor.transform(np.ones((2, 3), dtype=float))
    else:
        extractor.transform(np.ones((2, 3), dtype=float))

    assert model.training is True
    assert [child.training for child in model.children] == [True, False]


def test_torch_extractor_marks_live_state_unsafe_without_identity(fake_torch, tmp_path):
    model = TrackingModel(_embeddings_from_first_tensor)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"first weights")
    unsafe = TorchExtractor(
        "local",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
        checkpoint_paths=[str(checkpoint)],
    )
    explicit = TorchExtractor(
        "local",
        model=model,
        collate_fn=lambda batch: FakeTensor(batch),
        cache_identity="checkpoint-v1",
        checkpoint_paths=[str(checkpoint)],
    )

    assert unsafe.recipe()["cache_safe"] is False
    assert explicit.recipe()["cache_safe"] is True
    first_digest = explicit.recipe()["path_identities"][0]["sha256"]
    checkpoint.write_bytes(b"second weights")
    assert explicit.recipe()["path_identities"][0]["sha256"] != first_digest
