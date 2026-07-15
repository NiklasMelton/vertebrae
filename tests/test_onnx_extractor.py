import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vertebrae import BenchmarkDataset, DatasetIdentity, EmbeddingConfig, Evaluator
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.extractors import ONNXExtractor


class FakeInput:
    def __init__(self, name):
        self.name = name


class FakeOutput:
    def __init__(self, name):
        self.name = name


class FakeInferenceSession:
    instances = []

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.run_calls = []
        self.inputs = [FakeInput("input_0")]
        self.outputs = [FakeOutput("embedding")]
        if "multi_input" in str(model_path):
            self.inputs = [FakeInput("tokens"), FakeInput("mask")]
        if "multi_output" in str(model_path):
            self.outputs = [FakeOutput("embedding"), FakeOutput("logits")]
        self.__class__.instances.append(self)

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, output_names, inputs):
        self.run_calls.append((output_names, inputs))
        batch_size = len(next(iter(inputs.values())))
        if output_names == ["embedding"]:
            return [np.arange(batch_size * 3, dtype=np.int64).reshape(batch_size, 3)]
        return [
            np.arange(batch_size * 3, dtype=np.int64).reshape(batch_size, 3),
            np.zeros((batch_size, 2), dtype=np.int64),
        ]


@pytest.fixture
def fake_onnxruntime(monkeypatch):
    FakeInferenceSession.instances = []
    module = types.SimpleNamespace(InferenceSession=FakeInferenceSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return FakeInferenceSession


def test_onnx_extractor_uses_default_single_input_and_output(fake_onnxruntime):
    extractor = ONNXExtractor("onnx", model_path=Path("/tmp/model.onnx"))

    output = extractor.transform(np.arange(12, dtype=float).reshape(3, 4))

    assert output.shape == (3, 3)
    assert output.dtype == np.float64
    session = fake_onnxruntime.instances[0]
    assert session.model_path == "/tmp/model.onnx"
    assert session.kwargs == {}
    assert session.run_calls[0][0] == ["embedding"]
    assert list(session.run_calls[0][1].keys()) == ["input_0"]
    assert extractor.recipe()["model_path"] == "/tmp/model.onnx"


def test_onnx_extractor_supports_input_and_output_functions(fake_onnxruntime):
    def input_fn(batch):
        return {"tokens": batch + 1, "mask": np.ones((len(batch), 4), dtype=np.int64)}

    def output_fn(raw_outputs):
        return raw_outputs[0]

    extractor = ONNXExtractor(
        "onnx",
        model_path="/tmp/multi_input_multi_output.onnx",
        input_fn=input_fn,
        output_fn=output_fn,
        input_names=["tokens", "mask"],
    )

    output = extractor.transform(np.arange(8, dtype=float).reshape(2, 4))

    assert output.shape == (2, 3)
    session = fake_onnxruntime.instances[0]
    assert session.run_calls[0][0] == ["embedding", "logits"]
    assert set(session.run_calls[0][1].keys()) == {"tokens", "mask"}
    assert extractor.recipe()["input_fn"].endswith(".input_fn")
    assert extractor.recipe()["output_fn"].endswith(".output_fn")


def test_onnx_extractor_rejects_multiple_inputs_without_input_fn(fake_onnxruntime):
    extractor = ONNXExtractor("onnx", model_path="/tmp/multi_input.onnx")

    with pytest.raises(ValueError, match="input_fn"):
        extractor.transform(np.arange(8, dtype=float).reshape(2, 4))


def test_onnx_extractor_rejects_multiple_outputs_without_output_fn(fake_onnxruntime):
    extractor = ONNXExtractor(
        "onnx",
        model_path="/tmp/multi_output.onnx",
        input_fn=lambda batch: {"input_0": batch},
    )

    with pytest.raises(ValueError, match="output_fn"):
        extractor.transform(np.arange(8, dtype=float).reshape(2, 4))


def test_onnx_extractor_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    extractor = ONNXExtractor("onnx", model_path="/tmp/model.onnx")

    with pytest.raises(ImportError, match="vertebrae\\[onnx\\]"):
        extractor.transform(np.arange(8, dtype=float).reshape(2, 4))


def test_onnx_extractor_works_in_streaming_evaluator(fake_onnxruntime, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        np.arange(24, dtype=float).reshape(6, 4),
        ["a"] * 3 + ["b"] * 3,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = ONNXExtractor(
        "onnx",
        model_path="/tmp/model.onnx",
        input_fn=lambda batch: batch,
        streaming_safe=True,
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        cache_config=CacheConfig(enabled=False),
        stability_config=StabilityConfig(enabled=False),
        embedding_config=EmbeddingConfig(batch_size=2),
    ).run()

    metadata = result.extractor_results[0].embedding_metadata
    assert metadata["streamed"] is True
    assert metadata["stream_batch_size"] == 2
    # Memory admission probes a disposable clone once; the live extractor still
    # transforms exactly the three evaluation batches without being fitted twice.
    assert [len(instance.run_calls) for instance in fake_onnxruntime.instances] == [1, 3]


def test_onnx_extractor_supports_structured_outputs(fake_onnxruntime):
    extractor = ONNXExtractor(
        "onnx_structured",
        model_path="/tmp/model.onnx",
        input_fn=lambda batch: batch,
        output_fn=lambda raw_outputs: {
            "tokens": np.arange(len(raw_outputs[0]) * 2 * 3, dtype=float).reshape(
                len(raw_outputs[0]), 2, 3
            )
        },
        structured_outputs=[{"name": "tokens", "unit_type": "token"}],
    )

    output = extractor.transform_structured(np.arange(8, dtype=float).reshape(2, 4))[0]

    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (2, 3)


def test_onnx_recipe_hashes_local_model_content(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"first")
    first = ONNXExtractor("onnx", model_path=model_path).recipe()
    model_path.write_bytes(b"second")
    second = ONNXExtractor("onnx", model_path=model_path).recipe()

    assert first["cache_safe"] is True
    assert second["cache_safe"] is True
    assert first["path_identities"][0]["sha256"] != second["path_identities"][0]["sha256"]


def test_onnx_recipe_auto_discovers_and_hashes_external_data_sidecars(tmp_path):
    model_path = tmp_path / "model.onnx"
    sidecar = tmp_path / "weights.bin"
    model_path.write_bytes(b"tensor external_data location weights.bin")
    sidecar.write_bytes(b"first-weights")

    first = ONNXExtractor("onnx", model_path=model_path).recipe()
    sidecar.write_bytes(b"second-weights")
    second = ONNXExtractor("onnx", model_path=model_path).recipe()

    assert first["external_data_identity_status"] == "auto_discovered"
    assert first["external_data_paths"] == [str(sidecar)]
    assert first["cache_safe"] is True
    assert first["path_identities"][1]["sha256"] != second["path_identities"][1]["sha256"]


def test_onnx_recipe_disables_cache_when_external_data_cannot_be_discovered(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"tensor external_data location missing-weights.bin")

    recipe = ONNXExtractor("onnx", model_path=model_path).recipe()

    assert recipe["external_data_identity_status"] == "unsafe_undeclared"
    assert recipe["cache_safe"] is False
