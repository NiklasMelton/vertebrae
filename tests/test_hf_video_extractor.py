import sys
import types

import numpy as np
import pytest

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.extractors import HFVideoExtractor


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

    def reshape(self, *shape):
        return FakeTensor(self.data.reshape(*shape))

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


class FakeVideoProcessor:
    last_videos = None

    def __call__(self, videos=None, return_tensors=None, **kwargs):
        self.__class__.last_videos = videos
        batch = len(videos)
        return {"pixel_values": FakeTensor(np.zeros((batch, 3, 4, 2, 2), dtype=np.float32))}


class FakeVideoModel:
    last_call_kwargs = None
    call_count = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        self.__class__.last_call_kwargs = encoded
        self.__class__.call_count += 1
        batch = encoded["pixel_values"].shape[0]
        hidden = np.arange(batch * 5 * 6, dtype=float).reshape(batch, 5, 6)
        hidden_states = tuple(FakeTensor(hidden + layer * 100) for layer in range(4))
        return types.SimpleNamespace(
            last_hidden_state=FakeTensor(hidden),
            pooler_output=FakeTensor(np.ones((batch, 6))),
            hidden_states=hidden_states if encoded.get("output_hidden_states") else None,
        )


class FakeSpatialPoolerVideoModel(FakeVideoModel):
    def __call__(self, **encoded):
        batch = encoded["pixel_values"].shape[0]
        hidden = np.arange(batch * 5 * 6, dtype=float).reshape(batch, 5, 6)
        return types.SimpleNamespace(
            last_hidden_state=FakeTensor(hidden),
            pooler_output=FakeTensor(np.ones((batch, 2, 3))),
        )


class FakeAutoVideoProcessor:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeVideoProcessor()


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeVideoModel()


class FakeSpatialPoolerAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeSpatialPoolerVideoModel()


class FakeEncodedVideoInstance:
    def __init__(self, path):
        self.path = path
        self.duration = 2.0

    def get_clip(self, start_sec, end_sec):
        clip = np.arange(6 * 2 * 2 * 3, dtype=np.uint8).reshape(6, 2, 2, 3)
        return {"video": clip}


class FakeEncodedVideo:
    last_path = None

    @classmethod
    def from_path(cls, path):
        cls.last_path = path
        return FakeEncodedVideoInstance(path)


@pytest.fixture
def fake_video_modules(monkeypatch):
    FakeVideoModel.call_count = 0
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=FakeAutoModel,
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


@pytest.mark.parametrize("pooling", ["mean", "cls", "pooler"])
def test_hf_video_pooling_modes(fake_video_modules, pooling):
    extractor = HFVideoExtractor("video", "fake-video", pooling=pooling, batch_size=2)

    output = extractor.transform([np.zeros((4, 2, 2, 3), dtype=np.uint8)] * 3)

    assert output.shape == (3, 6)
    assert output.dtype == np.float32
    assert extractor.recipe()["modality"] == "video"


def test_hf_video_recipe_includes_clip_configuration():
    extractor = HFVideoExtractor(
        "video",
        "fake-video",
        processor_id="fake-processor",
        hidden_layer=2,
        num_frames=8,
        clip_duration_sec=1.5,
        clip_start_sec=0.25,
    )

    recipe = extractor.recipe()

    assert recipe["processor_id"] == "fake-processor"
    assert recipe["hidden_layer"] == 2
    assert recipe["num_frames"] == 8
    assert recipe["clip_duration_sec"] == 1.5
    assert recipe["clip_start_sec"] == 0.25


def test_hf_video_path_inputs_use_decoder(fake_video_modules):
    extractor = HFVideoExtractor("video", "fake-video", batch_size=2, num_frames=4)

    output = extractor.transform({"path": np.asarray(["a.mp4", "b.mp4"], dtype=object)})

    assert output.shape == (2, 6)
    assert FakeEncodedVideo.last_path == "b.mp4"
    assert len(FakeVideoProcessor.last_videos[0]) == 4


def test_hf_video_selects_hidden_layer(fake_video_modules):
    extractor = HFVideoExtractor(
        "video",
        "fake-video",
        pooling="cls",
        hidden_layer=2,
        batch_size=2,
    )

    output = extractor.transform([np.zeros((4, 2, 2, 3), dtype=np.uint8)] * 2)

    assert output.tolist() == [
        [200.0, 201.0, 202.0, 203.0, 204.0, 205.0],
        [230.0, 231.0, 232.0, 233.0, 234.0, 235.0],
    ]
    assert FakeVideoModel.last_call_kwargs["output_hidden_states"] is True


def test_hf_video_transform_many_shares_model_forward(fake_video_modules):
    extractor = HFVideoExtractor(
        "video",
        "fake-video",
        outputs=[
            {"name": "final_cls", "pooling": "cls"},
            {"name": "mid_cls", "pooling": "cls", "hidden_layer": 2},
        ],
        batch_size=2,
    )

    outputs = extractor.transform_many([np.zeros((4, 2, 2, 3), dtype=np.uint8)] * 3)

    assert [output.name for output in outputs] == ["final_cls", "mid_cls"]
    assert all(output.embeddings.shape == (3, 6) for output in outputs)
    assert FakeVideoModel.call_count == 2
    assert FakeVideoModel.last_call_kwargs["output_hidden_states"] is True


def test_hf_video_rejects_pooler_with_hidden_layer(fake_video_modules):
    extractor = HFVideoExtractor(
        "video",
        "fake-video",
        pooling="pooler",
        hidden_layer=1,
    )

    with pytest.raises(ValueError, match="pooler"):
        extractor.transform([np.zeros((4, 2, 2, 3), dtype=np.uint8)])


def test_hf_video_rejects_out_of_range_hidden_layer(fake_video_modules):
    extractor = HFVideoExtractor(
        "video",
        "fake-video",
        pooling="mean",
        hidden_layer=99,
    )

    with pytest.raises(ValueError, match="out of range"):
        extractor.transform([np.zeros((4, 2, 2, 3), dtype=np.uint8)])


def test_hf_video_supports_structured_frame_outputs(fake_video_modules):
    extractor = HFVideoExtractor(
        "video",
        "fake-video",
        batch_size=2,
        structured_outputs=[{"name": "frames", "hidden_layer": 2, "special_tokens": 1}],
    )

    output = extractor.transform_structured([np.zeros((4, 2, 2, 3), dtype=np.uint8)] * 2)[0]

    assert output.name == "frames"
    assert output.unit_type == "frame"
    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (4, 6)


def test_hf_video_flattens_spatial_pooler_output(fake_video_modules, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=FakeSpatialPoolerAutoModel,
            AutoVideoProcessor=FakeAutoVideoProcessor,
        ),
    )
    extractor = HFVideoExtractor("video", "fake-video", pooling="pooler", batch_size=2)

    output = extractor.transform([np.zeros((4, 2, 2, 3), dtype=np.uint8)] * 3)

    assert output.shape == (3, 6)
    assert output.dtype == np.float32


def test_hf_video_accepts_dataset_frame_arrays(fake_video_modules):
    dataset = BenchmarkDataset.from_video_arrays(
        [
            np.zeros((3, 2, 2, 3), dtype=np.uint8),
            np.ones((4, 2, 2, 3), dtype=np.uint8),
            np.full((5, 2, 2, 3), 2, dtype=np.uint8),
            np.full((6, 2, 2, 3), 3, dtype=np.uint8),
        ],
        ["left", "left", "right", "right"],
        frame_rate=24.0,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFVideoExtractor("video", "fake-video", batch_size=2, num_frames=4)

    output = extractor.transform(dataset.X)

    assert output.shape == (4, 6)
    assert all(len(clip) == 4 for clip in FakeVideoProcessor.last_videos)


def test_hf_video_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    extractor = HFVideoExtractor("video", "fake-video")

    with pytest.raises(ImportError, match="optional Hugging Face video"):
        extractor.transform([np.zeros((4, 2, 2, 3), dtype=np.uint8)])


def test_hf_video_missing_decoder_dependency_for_paths(fake_video_modules, monkeypatch):
    monkeypatch.delitem(sys.modules, "pytorchvideo.data.encoded_video", raising=False)
    extractor = HFVideoExtractor("video", "fake-video")

    with pytest.raises(ImportError, match="video decoding"):
        extractor.transform({"path": np.asarray(["a.mp4"], dtype=object)})


def test_hf_video_evaluator_workflow(fake_video_modules, fake_overlapindex):
    dataset = BenchmarkDataset.from_video_arrays(
        [
            np.zeros((3, 2, 2, 3), dtype=np.uint8),
            np.ones((4, 2, 2, 3), dtype=np.uint8),
            np.full((5, 2, 2, 3), 2, dtype=np.uint8),
            np.full((6, 2, 2, 3), 3, dtype=np.uint8),
        ],
        ["left", "left", "right", "right"],
        frame_rate=24.0,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFVideoExtractor(
        name="video",
        model_id="fake-video",
        batch_size=2,
        num_frames=4,
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        stability_config=StabilityConfig(repeats=2),
        cache_config=CacheConfig(enabled=False),
    ).run()

    assert result.extractor_results[0].name == "video"
    assert len(fake_overlapindex.calls) == 3
