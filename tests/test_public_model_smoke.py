import importlib
import json
import os

import numpy as np
import pytest

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import (
    CacheConfig,
    EmbeddingConfig,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.extractors import (
    HFAudioExtractor,
    HFMultimodalExtractor,
    HFTextExtractor,
    HFVisionExtractor,
    SentenceTransformerExtractor,
)

pytestmark = [
    pytest.mark.publicmodels,
    pytest.mark.skipif(
        os.environ.get("VERTABRAE_RUN_PUBLIC_MODELS") != "1",
        reason="set VERTABRAE_RUN_PUBLIC_MODELS=1 to run public model smoke tests",
    ),
]


def test_public_hf_text_model_smoke(tmp_path):
    _require_module("transformers")
    _require_module("torch")
    model_id = os.environ.get("VERTABRAE_PUBLIC_TEXT_MODEL", "hf-internal-testing/tiny-random-bert")
    dataset = BenchmarkDataset.from_arrays(
        [
            "invoice refund payment",
            "billing receipt charge",
            "customer account payment",
            "api timeout database",
            "server latency queue",
            "backend error logs",
        ],
        ["business"] * 3 + ["engineering"] * 3,
        modality="text",
    )

    item = _run_public_model_benchmark(
        dataset,
        HFTextExtractor("public_hf_text", model_id, pooling="mean", batch_size=2),
        tmp_path,
    )

    assert item.embedding_metadata["recipe"]["model_id"] == model_id
    assert item.embedding_metadata["recipe"]["modality"] == "text"


def test_public_sentence_transformer_model_smoke(tmp_path):
    _require_module("sentence_transformers")
    model_id = os.environ.get(
        "VERTABRAE_PUBLIC_SENTENCE_MODEL",
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
    )
    dataset = BenchmarkDataset.from_arrays(
        [
            "reset password login",
            "recover account access",
            "change authentication token",
            "invoice payment receipt",
            "refund billing status",
            "export finance report",
        ],
        ["support"] * 3 + ["billing"] * 3,
        modality="text",
    )

    item = _run_public_model_benchmark(
        dataset,
        SentenceTransformerExtractor("public_sentence_transformer", model_id, batch_size=2),
        tmp_path,
    )

    assert item.embedding_metadata["recipe"]["model_id"] == model_id


def test_public_hf_vision_model_smoke(tmp_path):
    _require_module("transformers")
    _require_module("torch")
    _require_module("PIL")
    model_id = os.environ.get(
        "VERTABRAE_PUBLIC_VISION_MODEL",
        "hf-internal-testing/tiny-random-vit",
    )
    images = [np.full((32, 32, 3), value, dtype=np.uint8) for value in [0, 12, 24, 180, 210, 240]]
    dataset = BenchmarkDataset.from_arrays(images, ["dark"] * 3 + ["bright"] * 3, modality="image")

    item = _run_public_model_benchmark(
        dataset,
        HFVisionExtractor("public_hf_vision", model_id, pooling="mean", batch_size=2),
        tmp_path,
    )

    assert item.embedding_metadata["recipe"]["model_id"] == model_id
    assert item.embedding_metadata["recipe"]["modality"] == "image"


def test_public_hf_audio_model_smoke(tmp_path):
    _require_module("transformers")
    _require_module("torch")
    model_id = os.environ.get(
        "VERTABRAE_PUBLIC_AUDIO_MODEL",
        "hf-internal-testing/tiny-random-wav2vec2",
    )
    low = [np.sin(np.linspace(0, 8 * np.pi, 4096, dtype=np.float32) + phase) for phase in (0, 1, 2)]
    high = [
        np.sin(np.linspace(0, 24 * np.pi, 4096, dtype=np.float32) + phase) for phase in (0, 1, 2)
    ]
    dataset = BenchmarkDataset.from_audio_arrays(
        low + high,
        ["low_tone"] * 3 + ["high_tone"] * 3,
        sampling_rate=16_000,
    )

    item = _run_public_model_benchmark(
        dataset,
        HFAudioExtractor(
            "public_hf_audio",
            model_id,
            pooling="mean",
            batch_size=2,
            sampling_rate=16_000,
        ),
        tmp_path,
    )

    assert item.embedding_metadata["recipe"]["model_id"] == model_id
    assert item.embedding_metadata["recipe"]["modality"] == "audio"


def test_public_hf_multimodal_model_smoke(tmp_path):
    _require_module("transformers")
    _require_module("torch")
    _require_module("PIL")
    model_id = os.environ.get(
        "VERTABRAE_PUBLIC_MULTIMODAL_MODEL",
        "hf-internal-testing/tiny-random-CLIPModel",
    )
    dataset = BenchmarkDataset.from_multimodal(
        inputs={
            "image": [
                np.full((32, 32, 3), value, dtype=np.uint8) for value in [0, 12, 24, 180, 210, 240]
            ],
            "caption": [
                "dark product photo",
                "dim catalog image",
                "low light object",
                "bright outdoor photo",
                "sunlit catalog image",
                "high key object",
            ],
        },
        labels=["catalog"] * 3 + ["lifestyle"] * 3,
        modalities={"image": "image", "caption": "text"},
    )

    item = _run_public_model_benchmark(
        dataset,
        HFMultimodalExtractor(
            "public_hf_multimodal",
            model_id,
            input_modalities={"image": "image", "caption": "text"},
            outputs=[{"name": "image_branch", "source": "image", "model_output": "image_embeds"}],
            batch_size=2,
            processor_kwargs={"padding": True, "truncation": True},
        ),
        tmp_path,
    )

    assert item.embedding_metadata["recipe"]["model_id"] == model_id
    assert item.embedding_metadata["recipe"]["modality"] == "multimodal"


def _run_public_model_benchmark(dataset, extractor, tmp_path):
    result = Benchmark(
        dataset,
        extractors=[extractor],
        scoring_config=OverlapScoringConfig(
            k=1,
            min_samples_per_cluster=1,
            kmeans_kwargs={"random_state": 101, "batch_size": 16, "n_init": 2},
        ),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path / "cache")),
        embedding_config=EmbeddingConfig(batch_size=2),
    ).run()
    assert len(result.extractor_results) == 1
    item = result.extractor_results[0]
    assert np.isfinite(item.overlap.score)
    assert 0.0 <= item.overlap.score <= 1.0
    assert item.embedding_metadata["embedding_dim"] > 0
    assert item.embedding_metadata["cache_key"]

    json_path = tmp_path / f"{item.name.replace('/', '_')}.json"
    result.save_json(str(json_path))
    assert json.loads(json_path.read_text(encoding="utf-8"))["extractor_results"]
    return item


def _require_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AssertionError(
            f"Required dependency {module_name!r} is not installed for enabled "
            "public model smoke tests."
        ) from exc
