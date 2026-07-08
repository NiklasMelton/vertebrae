import sys
import types

import numpy as np
import pytest

from vertebrae import (
    BenchmarkDataset,
    EmbeddingConfig,
    Evaluator,
    SpatialLayout,
    SpatialOutputSpec,
    StructuredOutputSpec,
)
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.extractors import KerasExtractor


class FakeKerasModel:
    def __init__(self, call_fn=None, predict_fn=None):
        self.call_fn = call_fn
        self.predict_fn = predict_fn
        self.call_args = []
        self.predict_args = []

    def __call__(self, batch, **kwargs):
        self.call_args.append((batch, kwargs))
        if self.call_fn is None:
            raise AssertionError("call_fn was not configured.")
        return self.call_fn(batch, kwargs)

    def predict(self, batch, **kwargs):
        self.predict_args.append((batch, kwargs))
        if self.predict_fn is None:
            raise AssertionError("predict_fn was not configured.")
        return self.predict_fn(batch, kwargs)


def _install_fake_keras(monkeypatch):
    monkeypatch.setitem(sys.modules, "keras", types.ModuleType("keras"))


def _install_fake_tensorflow_keras(monkeypatch):
    tensorflow_module = types.ModuleType("tensorflow")
    tensorflow_keras_module = types.ModuleType("tensorflow.keras")
    tensorflow_module.keras = tensorflow_keras_module
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", tensorflow_keras_module)


def test_keras_extractor_uses_direct_call_and_recipe(monkeypatch):
    _install_fake_keras(monkeypatch)

    def collate_fn(batch):
        return np.asarray(batch, dtype=np.float32)

    def call_fn(batch, kwargs):
        assert kwargs == {"training": False}
        return np.arange(batch.shape[0] * 3, dtype=np.float64).reshape(batch.shape[0], 3)

    def output_fn(raw_output):
        return raw_output[:, :2]

    model = FakeKerasModel(call_fn=call_fn)
    extractor = KerasExtractor(
        name="local_keras",
        model=model,
        collate_fn=collate_fn,
        output_fn=output_fn,
        recipe_data={"checkpoint": "/tmp/model.keras"},
    )

    output = extractor.transform([[1, 2], [3, 4], [5, 6]])
    recipe = extractor.recipe()

    assert output.shape == (3, 2)
    assert len(model.call_args) == 1
    assert model.predict_args == []
    assert recipe["model_class"].endswith(".FakeKerasModel")
    assert recipe["call_method"] == "call"
    assert recipe["call_kwargs"] == {"training": False}
    assert recipe["predict_kwargs"] == {"verbose": 0}
    assert recipe["recipe_data"] == {"checkpoint": "/tmp/model.keras"}
    assert recipe["collate_fn"].endswith(".collate_fn")
    assert recipe["output_fn"].endswith(".output_fn")


def test_keras_extractor_uses_predict_mode(monkeypatch):
    _install_fake_keras(monkeypatch)

    def predict_fn(batch, kwargs):
        assert kwargs["verbose"] == 0
        assert kwargs["batch_size"] == 2
        return np.arange(batch.shape[0] * 4, dtype=np.float64).reshape(batch.shape[0], 4)

    model = FakeKerasModel(predict_fn=predict_fn)
    extractor = KerasExtractor(
        "predict_mode",
        model=model,
        call_method="predict",
        predict_kwargs={"batch_size": 2},
    )

    output = extractor.transform(np.ones((2, 3), dtype=np.float32))

    assert output.shape == (2, 4)
    assert len(model.predict_args) == 1
    assert model.call_args == []
    assert model.predict_args[0][1]["batch_size"] == 2
    assert model.predict_args[0][1]["verbose"] == 0


def test_keras_extractor_falls_back_to_tensorflow_keras(monkeypatch):
    monkeypatch.setitem(sys.modules, "keras", None)
    _install_fake_tensorflow_keras(monkeypatch)

    model = FakeKerasModel(
        call_fn=lambda batch, kwargs: np.arange(batch.shape[0] * 2, dtype=float).reshape(
            batch.shape[0], 2
        )
    )
    extractor = KerasExtractor("tf_keras", model=model)

    output = extractor.transform(np.ones((4, 2), dtype=float))

    assert output.shape == (4, 2)


def test_keras_extractor_rejects_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "keras", None)
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", None)

    extractor = KerasExtractor(
        "local_keras",
        model=FakeKerasModel(call_fn=lambda batch, kwargs: batch),
    )

    with pytest.raises(ImportError, match="poetry install -E keras"):
        extractor.transform(np.ones((2, 2), dtype=float))


@pytest.mark.parametrize(
    "output_value, message",
    [
        (np.ones(3), "2D"),
        (np.array([[1.0, np.nan]]), "finite"),
        (np.array([["a", "b"]], dtype=object), "numeric"),
    ],
)
def test_keras_extractor_rejects_invalid_outputs(monkeypatch, output_value, message):
    _install_fake_keras(monkeypatch)
    extractor = KerasExtractor(
        "local_keras",
        model=FakeKerasModel(call_fn=lambda batch, kwargs: output_value),
    )

    with pytest.raises(ValueError, match=message):
        extractor.transform(np.ones((1, 2), dtype=float))


def test_keras_extractor_works_in_streaming_evaluator(monkeypatch, fake_overlapindex):
    _install_fake_keras(monkeypatch)
    model = FakeKerasModel(
        call_fn=lambda batch, kwargs: np.asarray(batch[:, :2], dtype=float),
    )
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24, dtype=float).reshape(8, 3),
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
    )
    extractor = KerasExtractor(
        "streaming_local",
        model=model,
        collate_fn=lambda batch: np.asarray(batch, dtype=float),
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
    assert len(model.call_args) == 3


def test_keras_extractor_supports_explicit_spatial_outputs(monkeypatch):
    _install_fake_keras(monkeypatch)
    model = FakeKerasModel(
        call_fn=lambda batch, kwargs: {"features": np.ones((2, 2, 2, 3), dtype=float)}
    )
    extractor = KerasExtractor(
        "spatial",
        model=model,
        spatial_output_fn=lambda output: output["features"],
        spatial_output_specs=[SpatialOutputSpec("layer", SpatialLayout(2, 2))],
    )

    output = extractor.transform_spatial(np.ones((2, 4), dtype=float))[0]

    assert output.name == "layer"
    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (2, 2, 3)


def test_keras_extractor_supports_explicit_structured_outputs(monkeypatch):
    _install_fake_keras(monkeypatch)
    model = FakeKerasModel(
        call_fn=lambda batch, kwargs: {"tokens": np.ones((2, 3, 4), dtype=float)}
    )
    extractor = KerasExtractor(
        "structured",
        model=model,
        structured_output_fn=lambda output: output["tokens"],
        structured_output_specs=[StructuredOutputSpec("tokens", unit_type="token")],
    )

    output = extractor.transform_structured(np.ones((2, 4), dtype=float))[0]

    assert output.name == "tokens"
    assert output.unit_type == "token"
    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (3, 4)
