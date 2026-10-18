"""Strict, dependency-free constructor coverage for optional extractors."""

from types import MappingProxyType

import pytest

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
    KerasExtractor,
    ONNXExtractor,
    OpenCLIPExtractor,
    SentenceTransformerExtractor,
    SigLIPExtractor,
    TFHubExtractor,
    TimmVisionExtractor,
    TorchExtractor,
    TorchvisionVisionExtractor,
    TreeLeafEmbeddingExtractor,
)


def _multimodal(**kwargs):
    return HFMultimodalExtractor(
        "multimodal",
        "model",
        input_modalities={"caption": "text"},
        outputs=[
            {
                "name": "text",
                "source": "text",
                "model_output": "last_hidden_state",
            }
        ],
        **kwargs,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HFTextExtractor("text", " "),
        lambda: HFAudioExtractor("audio", "\t"),
        lambda: HFVisionExtractor("vision", "\n"),
        lambda: HFVideoExtractor("video", " "),
        lambda: HFTimeSeriesExtractor("series", " "),
        lambda: SentenceTransformerExtractor("sentence", " "),
        lambda: TimmVisionExtractor("timm", " "),
        lambda: TorchvisionVisionExtractor("torchvision", " "),
        lambda: OpenCLIPExtractor("clip", " "),
        lambda: TFHubExtractor("hub", " "),
        lambda: _multimodal(processor_id=" "),
        lambda: SigLIPExtractor("siglip", "model", image_field=" "),
    ],
)
def test_optional_extractors_reject_blank_component_identifiers(factory):
    with pytest.raises(ValueError, match="non-empty string"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HFTextExtractor("text", "model", trust_remote_code=1),
        lambda: HFAudioExtractor("audio", "model", trust_remote_code="false"),
        lambda: HFVisionExtractor("vision", "model", trust_remote_code=None),
        lambda: HFVideoExtractor("video", "model", trust_remote_code=0),
        lambda: HFTimeSeriesExtractor("series", "model", trust_remote_code=1),
        lambda: SentenceTransformerExtractor("sentence", "model", normalize_embeddings=1),
        lambda: SentenceTransformerExtractor("sentence", "model", show_progress_bar=0),
        lambda: TimmVisionExtractor("timm", "model", pretrained=1),
        lambda: JAXFlaxExtractor("jax", lambda value: value, apply_fn=lambda value: value, jit=1),
        lambda: TreeLeafEmbeddingExtractor("tree", object(), sparse_output=1),
        lambda: GraphModelExtractor("graph", object(), lambda value: value, allow_sparse=1),
        lambda: TorchExtractor("torch", object(), lambda value: value, streaming_safe=1),
        lambda: KerasExtractor("keras", object(), allow_sparse=1),
    ],
)
def test_optional_extractors_reject_non_boolean_flags(factory):
    with pytest.raises(TypeError, match="bool"):
        factory()


@pytest.mark.parametrize(
    ("factory", "option"),
    [
        (lambda: HFTextExtractor("text", "model", pooling=1), "pooling"),
        (lambda: HFVisionExtractor("vision", "model", image_mode=1), "image_mode"),
        (lambda: HFVideoExtractor("video", "model", pooling="tokens"), "pooling"),
        (lambda: TimmVisionExtractor("timm", "model", alpha_mode="transparent"), "alpha_mode"),
        (lambda: TreeLeafEmbeddingExtractor("tree", object(), backend="sklearn"), "backend"),
        (
            lambda: GraphModelExtractor("graph", object(), lambda value: value, framework=[]),
            "framework",
        ),
    ],
)
def test_optional_extractors_reject_invalid_enum_options(factory, option):
    with pytest.raises((TypeError, ValueError), match=option):
        factory()


@pytest.mark.parametrize(
    ("factory", "option"),
    [
        (lambda: HFTextExtractor("text", "model", tokenizer_kwargs=[]), "tokenizer_kwargs"),
        (lambda: HFAudioExtractor("audio", "model", processor_kwargs=[]), "processor_kwargs"),
        (
            lambda: HFVisionExtractor("vision", "model", preprocess_kwargs=[]),
            "preprocess_kwargs",
        ),
        (lambda: HFVisionExtractor("vision", "model", model_kwargs=[]), "model_kwargs"),
        (lambda: HFVideoExtractor("video", "model", processor_kwargs=[]), "processor_kwargs"),
        (lambda: HFTimeSeriesExtractor("series", "model", input_kwargs=[]), "input_kwargs"),
        (lambda: _multimodal(model_kwargs=[]), "model_kwargs"),
        (
            lambda: SentenceTransformerExtractor("sentence", "model", encode_kwargs=[]),
            "encode_kwargs",
        ),
        (lambda: TimmVisionExtractor("timm", "model", data_config=[]), "data_config"),
        (
            lambda: TorchvisionVisionExtractor("torchvision", "model", model_kwargs=[]),
            "model_kwargs",
        ),
        (lambda: OpenCLIPExtractor("clip", "model", model_kwargs=[]), "model_kwargs"),
        (lambda: TFHubExtractor("hub", "handle", call_kwargs=[]), "call_kwargs"),
        (
            lambda: JAXFlaxExtractor(
                "jax", lambda value: value, apply_fn=lambda value: value, apply_kwargs=[]
            ),
            "apply_kwargs",
        ),
        (lambda: TreeLeafEmbeddingExtractor("tree", object(), recipe_data=[]), "recipe_data"),
        (
            lambda: GraphModelExtractor("graph", object(), lambda value: value, recipe_data=[]),
            "recipe_data",
        ),
        (
            lambda: HostedEmbeddingExtractor(
                "hosted", "provider", "model", lambda value: value, request_metadata=[]
            ),
            "request_metadata",
        ),
        (lambda: ONNXExtractor("onnx", "model.onnx", recipe_data=[]), "recipe_data"),
        (
            lambda: TorchExtractor("torch", object(), lambda value: value, recipe_data=[]),
            "recipe_data",
        ),
        (lambda: KerasExtractor("keras", object(), call_kwargs=[]), "call_kwargs"),
    ],
)
def test_optional_extractors_require_mapping_options(factory, option):
    with pytest.raises(TypeError, match=option):
        factory()


@pytest.mark.parametrize(
    ("factory", "attribute"),
    [
        (lambda value: HFTextExtractor("text", "model", model_kwargs=value), "model_kwargs"),
        (
            lambda value: HFAudioExtractor("audio", "model", processor_kwargs=value),
            "processor_kwargs",
        ),
        (lambda value: TimmVisionExtractor("timm", "model", data_config=value), "data_config"),
        (lambda value: TFHubExtractor("hub", "handle", call_kwargs=value), "call_kwargs"),
        (
            lambda value: HostedEmbeddingExtractor(
                "hosted", "provider", "model", lambda batch: batch, recipe_data=value
            ),
            "recipe_data",
        ),
        (
            lambda value: TorchExtractor("torch", object(), lambda batch: batch, recipe_data=value),
            "recipe_data",
        ),
        (lambda value: KerasExtractor("keras", object(), recipe_data=value), "recipe_data"),
    ],
)
def test_optional_mapping_options_are_defensively_snapshotted(factory, attribute):
    original = {"nested": {"values": [1]}}
    extractor = factory(original)

    original["nested"]["values"].append(2)

    assert getattr(extractor, attribute) == {"nested": {"values": [1]}}


def test_optional_kwargs_accept_general_mapping_objects():
    extractor = HFTextExtractor(
        "text",
        "model",
        model_kwargs=MappingProxyType({"revision_hint": "pinned"}),
    )

    assert extractor.model_kwargs == {"revision_hint": "pinned"}


def test_optional_recipe_mappings_reject_nonserializable_values():
    with pytest.raises(TypeError, match="serializable"):
        HostedEmbeddingExtractor(
            "hosted",
            "provider",
            "model",
            lambda value: value,
            recipe_data={"opaque": object()},
        )


@pytest.mark.parametrize(
    "input_modalities",
    [
        {1: "text"},
        {"caption": 1},
        {"caption": "unknown"},
    ],
)
def test_multimodal_field_names_and_modalities_are_not_coerced(input_modalities):
    with pytest.raises((TypeError, ValueError), match="input_modalities"):
        HFMultimodalExtractor(
            "multimodal",
            "model",
            input_modalities=input_modalities,
            outputs=[
                {
                    "name": "text",
                    "source": "text",
                    "model_output": "last_hidden_state",
                }
            ],
        )


def test_multimodal_mappings_are_snapshotted_without_coercion():
    modalities = {"caption": "text"}
    input_map = {"caption": "text"}
    extractor = HFMultimodalExtractor(
        "multimodal",
        "model",
        input_modalities=modalities,
        input_map=input_map,
        outputs=[
            {
                "name": "text",
                "source": "text",
                "model_output": "last_hidden_state",
            }
        ],
    )

    modalities["image"] = "image"
    input_map["caption"] = "captions"

    assert extractor.input_modalities == {"caption": "text"}
    assert extractor.input_map == {"caption": "text"}


def test_onnx_provider_options_are_validated_and_snapshotted():
    options = [{"device_id": 0}]
    extractor = ONNXExtractor(
        "onnx",
        "model.onnx",
        providers=["CUDAExecutionProvider"],
        provider_options=options,
    )

    options[0]["device_id"] = 1

    assert extractor.provider_options == [{"device_id": 0}]
    with pytest.raises(ValueError, match="one-to-one"):
        ONNXExtractor(
            "onnx",
            "model.onnx",
            providers=["CPUExecutionProvider"],
            provider_options=[{}, {}],
        )
