import sys
import types

import numpy as np
import pytest
from scipy import sparse

import vertebrae.extractors._utils as extractor_utils
from vertebrae import (
    BenchmarkDataset,
    CacheConfig,
    DatasetIdentity,
    Evaluator,
    ZeroShotBenchmark,
    ZeroShotDataset,
)
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.config import OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import (
    GraphModelExtractor,
    HFAudioExtractor,
    HFMultimodalExtractor,
    HFTextExtractor,
    HFTimeSeriesExtractor,
    HFVideoExtractor,
    HFVisionExtractor,
    HostedEmbeddingExtractor,
    JAXFlaxExtractor,
    OpenCLIPExtractor,
    SigLIPExtractor,
    TFHubExtractor,
    TimmVisionExtractor,
    TorchvisionVisionExtractor,
    TreeLeafEmbeddingExtractor,
)
from vertebrae.extractors._utils import (
    resolve_output_specs,
    resolve_structured_output_specs,
    spec_to_recipe,
    structured_spec_to_recipe,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda output: HFTextExtractor(
            "hf", "remote", revision="a" * 40, outputs=[{"name": output, "pooling": "mean"}]
        ),
        lambda output: HFAudioExtractor(
            "hf", "remote", revision="a" * 40, outputs=[{"name": output, "pooling": "mean"}]
        ),
        lambda output: HFTimeSeriesExtractor(
            "hf", "remote", revision="a" * 40, outputs=[{"name": output, "pooling": "mean"}]
        ),
        lambda output: HFVisionExtractor(
            "hf", "remote", revision="a" * 40, outputs=[{"name": output, "pooling": "cls"}]
        ),
        lambda output: HFVideoExtractor(
            "hf", "remote", revision="a" * 40, outputs=[{"name": output, "pooling": "mean"}]
        ),
    ],
)
def test_single_hf_output_spec_participates_in_cache_identity(factory):
    first = factory("early").recipe()
    second = factory("late").recipe()

    assert first["cache_safe"] is True
    assert first["outputs"][0]["name"] == "early"
    assert second["outputs"][0]["name"] == "late"
    assert fingerprint_extractor_recipe(first) != fingerprint_extractor_recipe(second)


def test_complete_output_spec_metadata_participates_in_recipes_and_identity():
    output = resolve_output_specs(
        [
            {
                "name": "features",
                "selector": "hidden.states",
                "flatten": False,
                "metadata": {"custom": {"axis_order": [0, 2, 1]}},
            }
        ]
    )[0]
    structured = resolve_structured_output_specs(
        [
            {
                "name": "tokens",
                "unit_type": "token",
                "selector": "hidden.states",
                "metadata": {"custom": {"drop_prefix": 2}},
            }
        ]
    )[0]

    output_recipe = spec_to_recipe(output)
    structured_recipe = structured_spec_to_recipe(structured)

    assert output_recipe["metadata"] == {
        "custom": {"axis_order": [0, 2, 1]},
        "selector": "hidden.states",
        "flatten": False,
    }
    assert output_recipe["metadata"]["flatten"] is False
    assert structured_recipe["metadata"] == {
        "custom": {"drop_prefix": 2},
        "selector": "hidden.states",
    }

    changed = TimmVisionExtractor(
        "timm",
        "remote-model",
        outputs=[
            {
                "name": "features",
                "selector": "hidden.states",
                "flatten": False,
                "metadata": {"custom": {"axis_order": [0, 1, 2]}},
            }
        ],
    ).recipe()
    original = TimmVisionExtractor(
        "timm",
        "remote-model",
        outputs=[
            {
                "name": "features",
                "selector": "hidden.states",
                "flatten": False,
                "metadata": {"custom": {"axis_order": [0, 2, 1]}},
            }
        ],
    ).recipe()
    assert original["outputs"][0]["metadata"]["flatten"] is False
    assert fingerprint_extractor_recipe(original) != fingerprint_extractor_recipe(changed)


def test_optional_dependency_versions_change_cache_fingerprint_without_imports(monkeypatch):
    versions = {"torch": "2.4.0", "transformers": "4.50.0"}

    def fake_version(distribution):
        return versions[distribution]

    monkeypatch.setattr(extractor_utils.importlib_metadata, "version", fake_version)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    extractor = HFTextExtractor(
        "hf",
        "remote-model",
        revision="a" * 40,
    )

    first = extractor.recipe()
    versions["transformers"] = "4.51.0"
    second = extractor.recipe()

    assert first["dependency_versions"] == {
        "torch": "2.4.0",
        "transformers": "4.50.0",
    }
    assert second["dependency_versions"]["transformers"] == "4.51.0"
    assert fingerprint_extractor_recipe(first) != fingerprint_extractor_recipe(second)


def test_missing_optional_dependency_versions_are_explicit_and_deterministic(monkeypatch):
    def missing(_distribution):
        raise extractor_utils.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(extractor_utils.importlib_metadata, "version", missing)

    first = extractor_utils.optional_dependency_versions("torch", "transformers")
    second = extractor_utils.optional_dependency_versions("transformers", "torch")

    assert first == second == {"torch": None, "transformers": None}


def _hf_component_extractors(model_id, processor_id, **kwargs):
    return [
        HFAudioExtractor("audio", model_id, processor_id=processor_id, **kwargs),
        HFVisionExtractor("vision", model_id, processor_id=processor_id, **kwargs),
        HFVideoExtractor("video", model_id, processor_id=processor_id, **kwargs),
        HFMultimodalExtractor(
            "multimodal",
            model_id,
            processor_id=processor_id,
            input_modalities={"caption": "text"},
            outputs=[{"name": "embedding", "source": "fused", "model_output": "pooler_output"}],
            **kwargs,
        ),
    ]


def test_hf_model_and_processor_identities_are_component_aware(tmp_path):
    model = tmp_path / "model"
    processor = tmp_path / "processor"
    model.mkdir()
    processor.mkdir()
    (model / "weights.bin").write_bytes(b"model-v1")
    processor_config = processor / "preprocessor.json"
    processor_config.write_bytes(b"processor-v1")

    extractors = _hf_component_extractors(str(model), str(processor))
    first_identities = [item.recipe()["path_identities"] for item in extractors]
    assert all(item.recipe()["cache_safe"] is True for item in extractors)

    processor_config.write_bytes(b"processor-v2")
    assert all(
        item.recipe()["path_identities"] != first
        for item, first in zip(extractors, first_identities)
    )

    local_model_remote_processor = _hf_component_extractors(str(model), "remote-processor")
    remote_model_local_processor = _hf_component_extractors("remote-model", str(processor))
    assert all(item.recipe()["cache_safe"] is False for item in local_model_remote_processor)
    assert all(item.recipe()["cache_safe"] is False for item in remote_model_local_processor)

    pinned = _hf_component_extractors(
        "remote-model",
        "remote-processor",
        revision="a" * 40,
    )
    assert all(item.recipe()["cache_safe"] is True for item in pinned)


def test_named_optional_models_need_explicit_identity_despite_checkpoint_hint(tmp_path):
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"declared checkpoint provenance")
    checkpoint_paths = [str(checkpoint)]

    extractors = [
        TimmVisionExtractor(
            "timm",
            "remote-model",
            pretrained=True,
            checkpoint_paths=checkpoint_paths,
        ),
        TorchvisionVisionExtractor(
            "torchvision",
            "remote-model",
            weights="DEFAULT",
            checkpoint_paths=checkpoint_paths,
        ),
        OpenCLIPExtractor(
            "openclip",
            "remote-model",
            pretrained="remote-weights",
            checkpoint_paths=checkpoint_paths,
        ),
    ]

    assert all(extractor.recipe()["cache_safe"] is False for extractor in extractors)
    assert all(extractor.recipe()["path_identities"] for extractor in extractors)
    explicit = TorchvisionVisionExtractor(
        "torchvision",
        "remote-model",
        weights="DEFAULT",
        checkpoint_paths=checkpoint_paths,
        cache_identity="torchvision-weights-v1",
    )
    assert explicit.recipe()["cache_safe"] is True
    assert explicit.recipe()["path_identities"][0]["sha256"]


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

    @staticmethod
    def as_tensor(value):
        return FakeTensor(value)


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


def test_torchvision_without_weights_builds_float_chw_tensors(fake_torch, monkeypatch):
    observed = {}

    class FakeVisionModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, batch):
            observed["shape"] = batch.shape
            observed["dtype"] = batch.data.dtype
            observed["maximum"] = float(batch.data.max())
            return FakeTensor(np.ones((batch.shape[0], 2), dtype=float))

    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        types.SimpleNamespace(
            models=types.SimpleNamespace(resnet18=lambda weights, **kwargs: FakeVisionModel())
        ),
    )
    extractor = TorchvisionVisionExtractor("tv", "resnet18", weights=None)

    output = extractor.transform([np.full((2, 3, 3), 255, dtype=np.uint8)])

    assert output.shape == (1, 2)
    assert observed == {"shape": (1, 3, 2, 3), "dtype": np.dtype("float32"), "maximum": 1.0}


def test_openclip_extractor_emits_image_and_text_branches(
    fake_torch, fake_overlapindex, monkeypatch
):
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
    default = OpenCLIPExtractor("default-clip", "fake-clip", batch_size=2)
    assert default.transform_many(dataset)[0].name == "image_branch"
    text_embeddings = default.encode_retrieval(
        ["left", "right"], branch="text_branch", modality="text"
    )
    assert text_embeddings.shape == (
        2,
        3,
    )
    protocol = ZeroShotDataset.from_templates(
        BenchmarkDataset.from_arrays(
            dataset["image"],
            ["left", "left", "right", "right"],
            modality="image",
            identity=DatasetIdentity.ephemeral(),
        ),
        ["{label}"],
    )
    result = ZeroShotBenchmark(
        protocol,
        [default],
        sample_branch="image_branch",
        text_branch="text_branch",
    ).run()
    assert result.extractor_results[0].embedding_metadata["text_branch"] == "text_branch"


def test_openclip_transform_prepares_only_required_single_modality_branch(fake_torch, monkeypatch):
    calls = {
        "encode_image": 0,
        "encode_text": 0,
        "preprocess": 0,
        "tokenize": 0,
    }

    class FakeOpenCLIPModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def encode_image(self, batch):
            calls["encode_image"] += 1
            return FakeTensor(np.full((batch.shape[0], 3), 2.0))

        def encode_text(self, batch):
            calls["encode_text"] += 1
            return FakeTensor(np.full((batch.shape[0], 3), 7.0))

    def preprocess(image):
        calls["preprocess"] += 1
        return FakeTensor(np.asarray(image, dtype=float))

    def tokenize(texts):
        calls["tokenize"] += 1
        return FakeTensor(np.arange(len(texts) * 2).reshape(len(texts), 2))

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        types.SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (
                FakeOpenCLIPModel(),
                None,
                preprocess,
            ),
            get_tokenizer=lambda model_name: tokenize,
        ),
    )

    image_output = OpenCLIPExtractor(
        "image-only",
        "fake-clip",
        input_modalities={"pixels": "image"},
        outputs=[{"name": "image_branch", "source": "image"}],
    ).transform({"pixels": [np.zeros((2, 2, 3), dtype=np.uint8)]})

    assert image_output.tolist() == [[2.0, 2.0, 2.0]]
    assert calls == {
        "encode_image": 1,
        "encode_text": 0,
        "preprocess": 1,
        "tokenize": 0,
    }

    calls.update({key: 0 for key in calls})
    text_output = OpenCLIPExtractor(
        "text-only",
        "fake-clip",
        input_modalities={"caption": "text"},
        outputs=[{"name": "text_branch", "source": "text"}],
    ).transform({"caption": ["hello"]})

    assert text_output.tolist() == [[7.0, 7.0, 7.0]]
    assert calls == {
        "encode_image": 0,
        "encode_text": 1,
        "preprocess": 0,
        "tokenize": 1,
    }


def test_openclip_fused_output_prepares_both_inputs_without_branch_encodes(fake_torch, monkeypatch):
    calls = {
        "encode_image": 0,
        "encode_text": 0,
        "get_logits": 0,
        "preprocess": 0,
        "tokenize": 0,
    }

    class FakeOpenCLIPModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def encode_image(self, batch):
            calls["encode_image"] += 1
            return FakeTensor(np.ones((batch.shape[0], 3)))

        def encode_text(self, batch):
            calls["encode_text"] += 1
            return FakeTensor(np.ones((batch.shape[0], 3)))

        def get_logits(self, image_batch, text_batch):
            calls["get_logits"] += 1
            return FakeTensor(np.full((image_batch.shape[0], 2), 5.0))

    def preprocess(image):
        calls["preprocess"] += 1
        return FakeTensor(np.asarray(image, dtype=float))

    def tokenize(texts):
        calls["tokenize"] += 1
        return FakeTensor(np.arange(len(texts) * 2).reshape(len(texts), 2))

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        types.SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (
                FakeOpenCLIPModel(),
                None,
                preprocess,
            ),
            get_tokenizer=lambda model_name: tokenize,
        ),
    )
    extractor = OpenCLIPExtractor(
        "fused",
        "fake-clip",
        input_modalities={"pixels": "image", "caption": "text"},
        outputs=[{"name": "joint", "source": "fused"}],
        batch_size=2,
    )

    output = extractor.transform(
        {
            "pixels": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
            "caption": ["left", "right"],
        }
    )

    assert output.tolist() == [[5.0, 5.0], [5.0, 5.0]]
    assert calls == {
        "encode_image": 0,
        "encode_text": 0,
        "get_logits": 1,
        "preprocess": 2,
        "tokenize": 1,
    }


def test_openclip_rejects_outputs_missing_their_declared_input_modality():
    with pytest.raises(ValueError, match="missing fields.*text"):
        OpenCLIPExtractor(
            "invalid",
            "fake-clip",
            input_modalities={"pixels": "image"},
            outputs=[{"name": "joint", "source": "fused"}],
        )
    with pytest.raises(ValueError, match="output source"):
        OpenCLIPExtractor(
            "invalid",
            "fake-clip",
            outputs=[{"name": "audio", "source": "audio"}],
        )


def test_siglip_extractor_uses_hf_multimodal_delegate(fake_torch, fake_overlapindex, monkeypatch):
    class FakeProcessor:
        def __call__(self, **kwargs):
            size = len(kwargs["text"] if "text" in kwargs else kwargs["images"])
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

        def get_image_features(self, **kwargs):
            return FakeTensor(np.full((kwargs["input_ids"].shape[0], 4), 11.0))

        def get_text_features(self, **kwargs):
            return FakeTensor(np.full((kwargs["input_ids"].shape[0], 4), 13.0))

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
    protocol = ZeroShotDataset.from_templates(
        BenchmarkDataset.from_arrays(
            [np.zeros((2, 2, 3), dtype=np.uint8)] * 4,
            ["left", "left", "right", "right"],
            modality="image",
            identity=DatasetIdentity.ephemeral(),
        ),
        ["{label}"],
    )
    result = ZeroShotBenchmark(
        protocol,
        [extractor],
        sample_branch="image_branch",
        text_branch="text_branch",
    ).run()
    assert result.extractor_results[0].zero_shot.metrics["accuracy"] >= 0.0


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


@pytest.mark.parametrize("hidden_layer", [True, 1.5, "2"])
def test_shared_output_specs_require_strict_integral_hidden_layers(hidden_layer):
    with pytest.raises(TypeError, match="hidden_layer.*integer"):
        TFHubExtractor(
            "hub",
            "fake://hub",
            outputs=[{"name": "embedding", "hidden_layer": hidden_layer}],
        )
    with pytest.raises(TypeError, match="hidden_layer.*integer"):
        TFHubExtractor(
            "hub",
            "fake://hub",
            structured_outputs=[
                {
                    "name": "tokens",
                    "unit_type": "token",
                    "hidden_layer": hidden_layer,
                }
            ],
        )


def test_tfhub_extractor_honors_batch_size_for_dense_and_structured_outputs(monkeypatch):
    call_sizes = []

    def model(batch, **call_kwargs):
        call_sizes.append(len(batch))
        values = np.asarray(batch, dtype=float)
        return {
            "embeddings": values,
            "tokens": np.repeat(values[:, np.newaxis, :], 2, axis=1),
        }

    monkeypatch.setitem(
        sys.modules,
        "tensorflow_hub",
        types.SimpleNamespace(load=lambda handle, **kwargs: model),
    )
    extractor = TFHubExtractor(
        "hub",
        "fake://hub",
        outputs=[{"name": "embeddings", "selector": "embeddings"}],
        structured_outputs=[{"name": "tokens", "unit_type": "token"}],
        batch_size=2,
    )
    values = [[float(index), float(index + 1)] for index in range(5)]

    dense = extractor.transform(values)
    structured = extractor.transform_structured(values)[0]

    assert dense.shape == (5, 2)
    assert len(structured.embeddings) == 5
    assert call_sizes == [2, 2, 1, 2, 2, 1]


def test_timm_vision_extractor_supports_structured_outputs(fake_torch, monkeypatch):
    class FakeTimmModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, batch):
            size = batch.shape[0]
            return {"tokens": FakeTensor(np.arange(size * 2 * 3, dtype=float).reshape(size, 2, 3))}

    monkeypatch.setitem(
        sys.modules,
        "timm",
        types.SimpleNamespace(
            create_model=lambda model_name, pretrained, **kwargs: FakeTimmModel(),
            data=types.SimpleNamespace(
                create_transform=lambda **kwargs: (
                    lambda image: FakeTensor(np.asarray(image, dtype=float))
                )
            ),
        ),
    )
    extractor = TimmVisionExtractor(
        "timm_structured",
        "fake-vit",
        structured_outputs=[{"name": "tokens", "unit_type": "token"}],
        batch_size=2,
    )

    output = extractor.transform_structured([np.zeros((2, 2, 3), dtype=np.uint8)] * 3)[0]

    assert len(output.embeddings) == 3
    assert output.embeddings[0].shape == (2, 3)


def test_torchvision_vision_extractor_supports_structured_outputs(fake_torch, monkeypatch):
    class FakeVisionModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, batch):
            size = batch.shape[0]
            return {"tokens": FakeTensor(np.arange(size * 2 * 4, dtype=float).reshape(size, 2, 4))}

    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        types.SimpleNamespace(
            models=types.SimpleNamespace(resnet18=lambda weights, **kwargs: FakeVisionModel())
        ),
    )
    extractor = TorchvisionVisionExtractor(
        "tv_structured",
        "resnet18",
        structured_outputs=[{"name": "tokens", "unit_type": "token"}],
    )

    output = extractor.transform_structured([np.zeros((2, 2, 3), dtype=np.uint8)] * 2)[0]

    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (2, 4)


def test_tfhub_extractor_supports_structured_outputs(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tensorflow_hub",
        types.SimpleNamespace(
            load=lambda handle, **kwargs: (
                lambda batch, **call_kwargs: {
                    "tokens": np.arange(len(batch) * 2 * 3, dtype=float).reshape(len(batch), 2, 3)
                }
            )
        ),
    )
    extractor = TFHubExtractor(
        "hub_structured",
        "fake://hub",
        input_fn=lambda value: np.asarray(value, dtype=float),
        structured_outputs=[{"name": "tokens", "unit_type": "token"}],
    )

    output = extractor.transform_structured([[1.0, 2.0], [3.0, 4.0]])[0]

    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (2, 3)


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


def test_jax_flax_extractor_supports_structured_outputs(monkeypatch):
    class FakeJax:
        @staticmethod
        def jit(fn):
            return fn

    monkeypatch.setitem(sys.modules, "jax", FakeJax)
    monkeypatch.setitem(sys.modules, "flax", types.SimpleNamespace())
    extractor = JAXFlaxExtractor(
        "jax_structured",
        input_fn=lambda value: np.asarray(value, dtype=float),
        apply_fn=lambda inputs: {
            "tokens": np.arange(len(inputs) * 2 * 3, dtype=float).reshape(len(inputs), 2, 3)
        },
        structured_outputs=[{"name": "tokens", "unit_type": "token"}],
    )

    output = extractor.transform_structured([[1.0, 2.0], [3.0, 4.0]])[0]

    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (2, 3)


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


def test_tree_leaf_extractor_flattens_multiclass_leaf_axes():
    class FakeXGBoostModel:
        __module__ = "xgboost.sklearn"

        def apply(self, X):
            return np.arange(12).reshape(2, 3, 2)

    extractor = TreeLeafEmbeddingExtractor("trees", FakeXGBoostModel())

    output = extractor.transform(np.ones((2, 4)))

    assert output.shape == (2, 6)
    assert output[1].tolist() == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


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
        identity=DatasetIdentity.ephemeral(),
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"max_retries": -1}, "max_retries"),
        ({"max_retries": 1.5}, "max_retries"),
        ({"retry_backoff_seconds": float("nan")}, "retry_backoff_seconds"),
        ({"name": ""}, "name"),
    ],
)
def test_hosted_embedding_extractor_validates_constructor_options(kwargs, message):
    options = {
        "name": "hosted",
        "provider": "fake",
        "model": "model",
        "embed_fn": lambda batch: np.ones((len(batch), 2)),
        **kwargs,
    }
    with pytest.raises((TypeError, ValueError), match=message):
        HostedEmbeddingExtractor(**options)


def test_hosted_embedding_extractor_validates_batch_rows_and_widths():
    row_mismatch = HostedEmbeddingExtractor(
        "hosted",
        "fake",
        "model",
        embed_fn=lambda batch: np.ones((1, 2)),
        batch_size=2,
    )
    with pytest.raises(ValueError, match="1 rows.*batch of 2"):
        row_mismatch.transform(["a", "b"])

    calls = []

    def variable_width(batch):
        calls.append(None)
        return np.ones((len(batch), 2 if len(calls) == 1 else 3))

    width_mismatch = HostedEmbeddingExtractor(
        "hosted",
        "fake",
        "model",
        embed_fn=variable_width,
        batch_size=2,
    )
    with pytest.raises(ValueError, match="changed embedding width"):
        width_mismatch.transform(["a", "b", "c"])
