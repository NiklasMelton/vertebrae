import importlib.util
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import overlapindex
import pytest
from sklearn.datasets import load_diabetes, load_digits
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vertebrae import Benchmark, BenchmarkDataset, DatasetIdentity, EmbeddingCompressionConfig
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.cli import main
from vertebrae.config import (
    CacheConfig,
    ContinuousOverlapScoringConfig,
    EmbeddingConfig,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.extractors import (
    HFAudioExtractor,
    HFMultimodalExtractor,
    HFTextExtractor,
    HFTimeSeriesExtractor,
    HFVideoExtractor,
    HFVisionExtractor,
    KerasExtractor,
    ONNXExtractor,
    PrecomputedExtractor,
    SentenceTransformerExtractor,
    SklearnExtractor,
    TorchExtractor,
)


def _load_test_module(filename):
    module_name = f"_vertebrae_realworld_{Path(filename).stem}"
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_hf_audio = _load_test_module("test_hf_audio_extractor.py")
_hf_multimodal = _load_test_module("test_hf_multimodal_extractor.py")
_hf_text = _load_test_module("test_hf_text_extractor.py")
_hf_time_series = _load_test_module("test_hf_time_series_extractor.py")
_hf_video = _load_test_module("test_hf_video_extractor.py")
_hf_vision = _load_test_module("test_hf_vision_extractor.py")
_keras = _load_test_module("test_keras_extractor.py")
_onnx = _load_test_module("test_onnx_extractor.py")
_sentence_transformers = _load_test_module("test_sentence_transformer_extractor.py")
_torch = _load_test_module("test_torch_extractor.py")

FakeAudioAutoModel = _hf_audio.FakeAutoModel
FakeAudioAutoProcessor = _hf_audio.FakeAutoProcessor
FakeAudioTorch = _hf_audio.FakeTorch
FakeSoundFile = _hf_audio.FakeSoundFile
FakeMultimodalAutoModel = _hf_multimodal.FakeAutoModel
FakeMultimodalAutoProcessor = _hf_multimodal.FakeAutoProcessor
FakeMultimodalImageModule = _hf_multimodal.FakeImageModule
FakeMultimodalTorch = _hf_multimodal.FakeTorch
FakeTextAutoModel = _hf_text.FakeAutoModel
FakeAutoTokenizer = _hf_text.FakeAutoTokenizer
FakeTextTorch = _hf_text.FakeTorch
FakeTimeSeriesAutoModel = _hf_time_series.FakeAutoModel
FakeTimeSeriesTorch = _hf_time_series.FakeTorch
FakeVideoAutoModel = _hf_video.FakeAutoModel
FakeAutoVideoProcessor = _hf_video.FakeAutoVideoProcessor
FakeEncodedVideo = _hf_video.FakeEncodedVideo
FakeVideoTorch = _hf_video.FakeTorch
FakeAutoImageProcessor = _hf_vision.FakeAutoImageProcessor
FakeVisionAutoModel = _hf_vision.FakeAutoModel
FakeVisionImageModule = _hf_vision.FakeImageModule
FakeVisionTorch = _hf_vision.FakeTorch
FakeKerasModel = _keras.FakeKerasModel
FakeInferenceSession = _onnx.FakeInferenceSession
FakeSentenceTransformer = _sentence_transformers.FakeSentenceTransformer
FakeTensor = _torch.FakeTensor
TrackingModel = _torch.TrackingModel

pytestmark = [
    pytest.mark.realworld,
    pytest.mark.skipif(
        os.environ.get("VERTABRAE_RUN_REALWORLD") != "1",
        reason="set VERTABRAE_RUN_REALWORLD=1 to run real-world integration tests",
    ),
]


def test_digits_benchmark_real_metrics_cache_compression_reports(tmp_path):
    X, y = _balanced_digits_subset(n_per_class=35)
    dataset = BenchmarkDataset.from_arrays(
        X, y, modality="tabular", identity=DatasetIdentity.ephemeral()
    )
    cache_dir = tmp_path / "cache"

    benchmark = Benchmark(
        dataset,
        extractors=[
            PrecomputedExtractor("pixels"),
            SklearnExtractor(
                "scaled_pca",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=24, random_state=7)),
                    ]
                ),
            ),
        ],
        scoring_config=OverlapScoringConfig(
            k=3,
            kmeans_kwargs={"random_state": 11, "batch_size": 128, "n_init": 3},
            normalize_embeddings=True,
        ),
        stability_config=StabilityConfig(repeats=3, random_state=13),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=True, cache_dir=str(cache_dir)),
        compression_configs=[
            EmbeddingCompressionConfig(),
            EmbeddingCompressionConfig(
                enabled=True,
                method="pca",
                n_components=12,
                random_state=17,
                dtype="float32",
            ),
        ],
        embedding_config=EmbeddingConfig(batch_size=64),
    )

    result = benchmark.run()
    json_path = tmp_path / "digits-result.json"
    markdown_path = tmp_path / "digits-report.md"
    result.save_json(str(json_path))
    result.save_markdown(str(markdown_path))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = result.to_dataframe()

    assert len(result.extractor_results) == 4
    assert set(rows["compression_method"]) == {"none", "pca"}
    assert all(np.isfinite(rows["overlap_score"]))
    assert all(0.0 <= score <= 1.0 for score in rows["overlap_score"])
    assert all(
        item.stability and len(item.stability["scores"]) == 3 for item in result.extractor_results
    )
    assert all("cache_key" in item.embedding_metadata for item in result.extractor_results)
    assert any(item.compression_metadata.get("cache_key") for item in result.extractor_results)
    assert payload["dataset_summary"]["n_samples"] == 350
    assert "Ranking" in markdown_path.read_text(encoding="utf-8")
    assert any(cache_dir.rglob("*.json"))


def test_diabetes_regression_real_continuous_overlap(tmp_path):
    if not hasattr(overlapindex, "ContinuousOverlapIndex"):
        raise AssertionError("Installed overlapindex does not expose ContinuousOverlapIndex.")

    data = load_diabetes()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    dataset = BenchmarkDataset.from_arrays(
        X,
        y,
        modality="tabular",
        target_type="regression",
        target_names=["target"],
        identity=DatasetIdentity.ephemeral(),
    )

    result = Benchmark(
        dataset,
        extractors=[
            SklearnExtractor(
                "scaled_regression_pca",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=6, random_state=19)),
                    ]
                ),
            )
        ],
        scoring_config=ContinuousOverlapScoringConfig(
            k=4,
            kmeans_kwargs={"random_state": 23, "batch_size": 128, "n_init": 3},
            n_projections=8,
            n_null_permutations=3,
        ),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path / "cache")),
    ).run()

    item = result.extractor_results[0]

    assert item.overlap.metadata["target_type"] == "regression"
    assert np.isfinite(item.overlap.score)
    assert 0.0 <= item.overlap.score <= 1.0
    assert item.overlap.metadata["target_names"] == ("target",)
    assert item.weakest_class is None
    assert result.to_dataframe().loc[0, "target_type"] == "regression"


def test_cli_artifact_workflow_scores_real_overlapindex(tmp_path, capsys):
    X, y = _balanced_digits_subset(n_per_class=12)
    dataset = BenchmarkDataset.from_arrays(
        X, y, modality="tabular", identity=DatasetIdentity.ephemeral()
    )
    extractor = SklearnExtractor(
        "cli_scaled_pca",
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=16, random_state=29)),
            ]
        ),
    )
    scoring_config = OverlapScoringConfig(
        k=2,
        kmeans_kwargs={"random_state": 31, "batch_size": 64, "n_init": 2},
    )
    dataset_path = tmp_path / "dataset.pkl"
    extractor_path = tmp_path / "extractor.pkl"
    scoring_path = tmp_path / "scoring.pkl"
    dataset_path.write_bytes(pickle.dumps(dataset))
    extractor_path.write_bytes(pickle.dumps(extractor))
    scoring_path.write_bytes(pickle.dumps(scoring_config))
    cache_dir = tmp_path / "cache"
    plan_path = tmp_path / "plan.json"

    assert (
        main(
            [
                "plan",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--cache-dir",
                str(cache_dir),
                "--total-shards",
                "3",
                "--batch-size",
                "40",
                "--output-json",
                str(plan_path),
            ]
        )
        == 0
    )

    for shard_index in range(3):
        assert (
            main(
                [
                    "embed-shard",
                    "--dataset-pickle",
                    str(dataset_path),
                    "--extractor-pickle",
                    str(extractor_path),
                    "--cache-dir",
                    str(cache_dir),
                    "--total-shards",
                    "3",
                    "--shard-index",
                    str(shard_index),
                    "--batch-size",
                    "40",
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert (
        main(["merge-embeddings", "--cache-dir", str(cache_dir), "--plan-json", str(plan_path)])
        == 0
    )
    merged_manifest = json.loads(capsys.readouterr().out)
    embeddings = LocalArtifactStore(str(cache_dir)).get_array(merged_manifest["output_key"])
    assert embeddings.shape == (120, 16)

    assert (
        main(
            [
                "write-labels",
                "--dataset-pickle",
                str(dataset_path),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "score",
                "--cache-dir",
                str(cache_dir),
                "--plan-json",
                str(plan_path),
                "--scoring-config-pickle",
                str(scoring_path),
            ]
        )
        == 0
    )
    score = json.loads(capsys.readouterr().out)

    assert score["artifact_type"] == "metric_evaluation"
    assert score["primary_metric"] == "overlap"
    overlap = score["metrics"]["overlap"]
    assert np.isfinite(overlap["score"])
    assert 0.0 <= overlap["score"] <= 1.0
    assert overlap["diagnostics"]["k_per_class"]


def test_hf_text_model_family_runs_full_benchmark(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", FakeTextTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=FakeTextAutoModel, AutoTokenizer=FakeAutoTokenizer),
    )
    dataset = BenchmarkDataset.from_arrays(
        [
            "refund invoice billing",
            "payment receipt billing",
            "customer chargeback invoice",
            "server latency api",
            "timeout database api",
            "backend queue outage",
        ],
        ["business"] * 3 + ["engineering"] * 3,
        modality="text",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFTextExtractor("hf_text", "fake-text", pooling="mean", batch_size=2)

    item = _run_real_model_family_benchmark(dataset, extractor, tmp_path)

    assert item.name == "hf_text"
    assert item.embedding_metadata["recipe"]["modality"] == "text"


def test_sentence_transformer_model_family_runs_full_benchmark(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    dataset = BenchmarkDataset.from_arrays(
        [
            "how to reset a password",
            "recover account login",
            "change authentication settings",
            "invoice payment status",
            "refund receipt question",
            "billing export request",
        ],
        ["support"] * 3 + ["billing"] * 3,
        modality="text",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = SentenceTransformerExtractor("sentence_text", "fake-sentence", batch_size=2)

    item = _run_real_model_family_benchmark(dataset, extractor, tmp_path)

    assert item.name == "sentence_text"
    assert item.embedding_metadata["recipe"]["extractor_type"] == "frozen_pretrained"


def test_hf_audio_model_family_runs_full_benchmark(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", FakeAudioTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=FakeAudioAutoModel,
            AutoProcessor=FakeAudioAutoProcessor,
        ),
    )
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile)
    dataset = BenchmarkDataset.from_audio_arrays(
        [
            np.linspace(0.0, 0.2, 6, dtype=np.float32),
            np.linspace(0.1, 0.3, 6, dtype=np.float32),
            np.linspace(0.2, 0.4, 6, dtype=np.float32),
            np.linspace(1.0, 1.2, 6, dtype=np.float32),
            np.linspace(1.1, 1.3, 6, dtype=np.float32),
            np.linspace(1.2, 1.4, 6, dtype=np.float32),
        ],
        ["speech"] * 3 + ["music"] * 3,
        sampling_rate=16_000,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFAudioExtractor("hf_audio", "fake-audio", pooling="mean", batch_size=2)

    item = _run_real_model_family_benchmark(dataset, extractor, tmp_path)

    assert item.name == "hf_audio"
    assert item.embedding_metadata["recipe"]["modality"] == "audio"


def test_hf_vision_model_family_runs_full_benchmark(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", FakeVisionTorch)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=FakeVisionImageModule))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoImageProcessor=FakeAutoImageProcessor,
            AutoModel=FakeVisionAutoModel,
        ),
    )
    dataset = BenchmarkDataset.from_arrays(
        [np.full((4, 4, 3), value, dtype=np.uint8) for value in [0, 8, 16, 120, 140, 160]],
        ["dark"] * 3 + ["bright"] * 3,
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFVisionExtractor("hf_vision", "fake-vision", pooling="mean", batch_size=2)

    item = _run_real_model_family_benchmark(dataset, extractor, tmp_path)

    assert item.name == "hf_vision"
    assert item.embedding_metadata["recipe"]["modality"] == "image"


def test_hf_multimodal_model_family_runs_full_benchmark(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", FakeMultimodalTorch)
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        types.SimpleNamespace(Image=FakeMultimodalImageModule),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=FakeMultimodalAutoModel,
            AutoProcessor=FakeMultimodalAutoProcessor,
        ),
    )
    dataset = BenchmarkDataset.from_multimodal(
        inputs={
            "image": [
                np.full((2, 2, 3), value, dtype=np.uint8) for value in [0, 8, 16, 120, 140, 160]
            ],
            "caption": [
                "small dark product",
                "dim product photo",
                "low light object",
                "bright outdoor scene",
                "sunlit product image",
                "high key catalog image",
            ],
        },
        labels=["catalog"] * 3 + ["lifestyle"] * 3,
        modalities={"image": "image", "caption": "text"},
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFMultimodalExtractor(
        "hf_multimodal",
        "fake-clip",
        input_modalities={"image": "image", "caption": "text"},
        outputs=[{"name": "fused", "source": "fused", "model_output": "pooler_output"}],
        batch_size=2,
    )

    item = _run_real_model_family_benchmark(dataset, extractor, tmp_path)

    assert item.name == "hf_multimodal"
    assert item.embedding_metadata["recipe"]["modality"] == "multimodal"
    assert item.embedding_metadata["output_metadata"]["source"] == "fused"


def test_hf_time_series_model_family_runs_full_benchmark(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", FakeTimeSeriesTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=FakeTimeSeriesAutoModel),
    )
    dataset = BenchmarkDataset.from_time_series(
        series=np.array(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [1.0, 1.1, 1.2, 1.3],
                [1.1, 1.2, 1.3, 1.4],
                [1.2, 1.3, 1.4, 1.5],
            ],
            dtype=np.float32,
        ),
        labels=["low"] * 3 + ["high"] * 3,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFTimeSeriesExtractor("hf_timeseries", "fake-ts", pooling="mean", batch_size=2)

    item = _run_real_model_family_benchmark(dataset, extractor, tmp_path)

    assert item.name == "hf_timeseries"
    assert item.embedding_metadata["recipe"]["modality"] == "time_series"


def test_hf_video_model_family_runs_full_benchmark(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", FakeVideoTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=FakeVideoAutoModel,
            AutoVideoProcessor=FakeAutoVideoProcessor,
        ),
    )
    monkeypatch.setitem(sys.modules, "pytorchvideo", types.ModuleType("pytorchvideo"))
    monkeypatch.setitem(sys.modules, "pytorchvideo.data", types.ModuleType("pytorchvideo.data"))
    monkeypatch.setitem(
        sys.modules,
        "pytorchvideo.data.encoded_video",
        types.SimpleNamespace(EncodedVideo=FakeEncodedVideo),
    )
    dataset = BenchmarkDataset.from_video_arrays(
        [np.full((4, 2, 2, 3), value, dtype=np.uint8) for value in [0, 8, 16, 120, 140, 160]],
        ["indoor"] * 3 + ["outdoor"] * 3,
        frame_rate=24.0,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFVideoExtractor("hf_video", "fake-video", pooling="mean", batch_size=2)

    item = _run_real_model_family_benchmark(dataset, extractor, tmp_path)

    assert item.name == "hf_video"
    assert item.embedding_metadata["recipe"]["modality"] == "video"


def test_local_torch_keras_and_onnx_model_families_run_full_benchmark(monkeypatch, tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        np.array(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [1.0, 1.1, 1.2],
                [1.1, 1.2, 1.3],
                [1.2, 1.3, 1.4],
            ],
            dtype=np.float32,
        ),
        ["left"] * 3 + ["right"] * 3,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            Tensor=FakeTensor,
            cuda=types.SimpleNamespace(is_available=lambda: False),
        ),
    )
    torch_model = TrackingModel(
        lambda args, kwargs: FakeTensor(np.asarray(args[0].data, dtype=np.float32)[:, :2])
    )
    torch_item = _run_real_model_family_benchmark(
        dataset,
        TorchExtractor(
            "local_torch",
            model=torch_model,
            collate_fn=lambda batch: FakeTensor(batch),
        ),
        tmp_path,
    )
    assert torch_item.extractor_type == "custom_torch"

    monkeypatch.setitem(sys.modules, "keras", types.ModuleType("keras"))
    keras_model = FakeKerasModel(
        call_fn=lambda batch, kwargs: np.asarray(batch, dtype=np.float32)[:, :2]
    )
    keras_item = _run_real_model_family_benchmark(
        dataset,
        KerasExtractor("local_keras", model=keras_model),
        tmp_path,
    )
    assert keras_item.extractor_type == "custom_keras"

    FakeInferenceSession.instances = []
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(InferenceSession=FakeInferenceSession),
    )
    onnx_item = _run_real_model_family_benchmark(
        dataset,
        ONNXExtractor("local_onnx", model_path="/tmp/model.onnx", streaming_safe=True),
        tmp_path,
    )
    assert onnx_item.extractor_type == "custom_onnx"


def _balanced_digits_subset(n_per_class):
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target.astype(str)
    indices = []
    for label in sorted(np.unique(y)):
        label_indices = np.flatnonzero(y == label)
        indices.extend(label_indices[:n_per_class])
    selected = np.asarray(indices)
    return X[selected], y[selected]


def _run_real_model_family_benchmark(dataset, extractor, tmp_path):
    result = Benchmark(
        dataset,
        extractors=[extractor],
        scoring_config=OverlapScoringConfig(
            k=1,
            min_samples_per_cluster=1,
            kmeans_kwargs={"random_state": 41, "batch_size": 16, "n_init": 2},
        ),
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path / "family-cache")),
        embedding_config=EmbeddingConfig(batch_size=2),
    ).run()
    json_path = tmp_path / f"{result.extractor_results[0].name.replace('/', '_')}.json"
    result.save_json(str(json_path))

    assert len(result.extractor_results) == 1
    item = result.extractor_results[0]
    assert np.isfinite(item.overlap.score)
    assert 0.0 <= item.overlap.score <= 1.0
    assert item.embedding_metadata["embedding_dim"] > 0
    assert item.embedding_metadata["cache_key"]
    assert json.loads(json_path.read_text(encoding="utf-8"))["extractor_results"]
    return item
