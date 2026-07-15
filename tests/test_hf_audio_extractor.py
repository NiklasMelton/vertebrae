import sys
import types

import numpy as np
import pytest

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.extractors import HFAudioExtractor


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.device = "cpu"

    @property
    def shape(self):
        return self.data.shape

    def size(self):
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

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.data, axis))

    def expand(self, shape):
        return FakeTensor(np.broadcast_to(self.data, shape))

    def float(self):
        return FakeTensor(self.data.astype(float))

    def sum(self, dim=None):
        return FakeTensor(np.sum(self.data, axis=dim))

    def clamp(self, min):
        return FakeTensor(np.maximum(self.data, min))

    def mean(self, dim=None):
        return FakeTensor(np.mean(self.data, axis=dim))

    def flatten(self, start_dim=0):
        leading = self.data.shape[:start_dim]
        flattened = int(np.prod(self.data.shape[start_dim:]))
        return FakeTensor(self.data.reshape(*leading, flattened))

    def __mul__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data * other_data)

    def __truediv__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data / other_data)

    def __getitem__(self, key):
        return FakeTensor(self.data[key])


def explicit_feature_mask(feature_length, attention_mask):
    assert feature_length == 2
    lengths = attention_mask.data.sum(axis=1)
    return FakeTensor(np.asarray([[1, int(length >= 3)] for length in lengths], dtype=int))


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


class FakeAudioProcessor:
    last_sampling_rate = None

    def __call__(self, batch, sampling_rate, **kwargs):
        self.__class__.last_sampling_rate = sampling_rate
        max_len = max(len(item) for item in batch)
        padded = np.zeros((len(batch), max_len), dtype=np.float32)
        attention_mask = np.zeros((len(batch), max_len), dtype=np.float32)
        for index, item in enumerate(batch):
            padded[index, : len(item)] = item
            attention_mask[index, : len(item)] = 1.0
        return {
            "input_values": FakeTensor(padded),
            "attention_mask": FakeTensor(attention_mask),
        }


class FakeAudioModel:
    last_call_kwargs = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        self.__class__.last_call_kwargs = encoded
        batch, seq_len = encoded["input_values"].shape
        hidden = np.arange(batch * seq_len * 4, dtype=float).reshape(batch, seq_len, 4)
        hidden_states = tuple(FakeTensor(hidden + layer * 100) for layer in range(4))
        return types.SimpleNamespace(
            last_hidden_state=FakeTensor(hidden),
            pooler_output=FakeTensor(np.ones((batch, 4))),
            hidden_states=hidden_states if encoded.get("output_hidden_states") else None,
        )


class FakeAutoProcessor:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeAudioProcessor()


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeAudioModel()


class FakeSoundFile:
    @staticmethod
    def read(path, always_2d=False):
        return np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32), 16_000


@pytest.fixture
def fake_audio_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=FakeAutoModel, AutoProcessor=FakeAutoProcessor),
    )
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile)


@pytest.mark.parametrize("pooling", ["mean", "cls", "pooler"])
def test_hf_audio_pooling_modes(fake_audio_modules, pooling):
    extractor = HFAudioExtractor(
        name=f"audio_{pooling}",
        model_id="fake-audio",
        pooling=pooling,
        batch_size=2,
        sampling_rate=16_000,
    )

    output = extractor.transform(
        [
            np.array([0.0, 0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4, 0.5], dtype=np.float32),
            np.array([0.6, 0.7, 0.8], dtype=np.float32),
        ]
    )

    assert output.shape == (3, 4)
    assert output.dtype == np.float32
    assert extractor.recipe()["modality"] == "audio"


def test_hf_audio_uses_dataset_sampling_rate(fake_audio_modules):
    dataset = BenchmarkDataset.from_audio_arrays(
        [
            np.array([0.0, 0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4, 0.5], dtype=np.float32),
            np.array([1.0, 1.1, 1.2], dtype=np.float32),
            np.array([1.3, 1.4, 1.5], dtype=np.float32),
        ],
        ["speech", "speech", "music", "music"],
        sampling_rate=22_050,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFAudioExtractor(name="audio", model_id="fake-audio", batch_size=2)

    output = extractor.transform(dataset.X)

    assert output.shape == (4, 4)
    assert FakeAudioProcessor.last_sampling_rate == 22_050


def test_hf_audio_accepts_audio_paths(fake_audio_modules):
    extractor = HFAudioExtractor(name="audio", model_id="fake-audio", batch_size=2)

    output = extractor.transform({"path": np.asarray(["a.wav", "b.wav"], dtype=object)})

    assert output.shape == (2, 4)
    assert FakeAudioProcessor.last_sampling_rate == 16_000


def test_hf_audio_rejects_missing_sampling_rate(fake_audio_modules):
    extractor = HFAudioExtractor(name="audio", model_id="fake-audio")

    with pytest.raises(ValueError, match="sampling rate"):
        extractor.transform([np.array([0.0, 0.1, 0.2], dtype=np.float32)])


def test_hf_audio_evaluator_workflow(fake_audio_modules, fake_overlapindex):
    dataset = BenchmarkDataset.from_audio_arrays(
        [
            np.array([0.0, 0.1, 0.2], dtype=np.float32),
            np.array([0.2, 0.3, 0.4], dtype=np.float32),
            np.array([1.0, 1.1, 1.2], dtype=np.float32),
            np.array([1.2, 1.3, 1.4], dtype=np.float32),
        ],
        ["left", "left", "right", "right"],
        sampling_rate=16_000,
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = HFAudioExtractor(
        name="audio",
        model_id="fake-audio",
        batch_size=2,
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        stability_config=StabilityConfig(repeats=2),
        cache_config=CacheConfig(enabled=False),
    ).run()

    assert result.extractor_results[0].name == "audio"


def test_hf_audio_supports_structured_frame_outputs(fake_audio_modules):
    extractor = HFAudioExtractor(
        "audio",
        "fake-audio",
        batch_size=2,
        sampling_rate=16_000,
        structured_outputs=[{"name": "frames", "hidden_layer": 2}],
    )

    output = extractor.transform_structured(
        [
            np.array([0.0, 0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
        ]
    )[0]

    assert output.name == "frames"
    assert output.unit_type == "frame"
    assert len(output.embeddings) == 2
    assert output.embeddings[0].shape == (3, 4)
    assert output.embeddings[1].shape == (2, 4)


def test_hf_audio_downsamples_attention_mask_to_feature_frames(fake_audio_modules):
    class DownsampledAudioModel(FakeAudioModel):
        def __call__(self, **encoded):
            batch = encoded["input_values"].shape[0]
            hidden = np.arange(batch * 2 * 4, dtype=float).reshape(batch, 2, 4)
            hidden_states = tuple(FakeTensor(hidden + layer * 100) for layer in range(4))
            return types.SimpleNamespace(
                last_hidden_state=FakeTensor(hidden),
                pooler_output=FakeTensor(np.ones((batch, 4))),
                hidden_states=hidden_states if encoded.get("output_hidden_states") else None,
            )

        def _get_feature_vector_attention_mask(self, feature_length, attention_mask):
            assert feature_length == 2
            raw_lengths = attention_mask.data.sum(axis=1)
            return FakeTensor(
                np.asarray([[1, int(length >= 3)] for length in raw_lengths], dtype=int)
            )

    extractor = HFAudioExtractor(
        "audio",
        "fake-audio",
        batch_size=2,
        sampling_rate=16_000,
        structured_outputs=[{"name": "frames", "hidden_layer": 2}],
    )
    extractor._processor = FakeAudioProcessor()
    extractor._model = DownsampledAudioModel()
    extractor._torch = FakeTorch

    pooled = extractor.transform(
        [
            np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
            np.asarray([0.3, 0.4], dtype=np.float32),
        ]
    )
    structured = extractor.transform_structured(
        [
            np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
            np.asarray([0.3, 0.4], dtype=np.float32),
        ]
    )[0]

    assert pooled.tolist() == [[2.0, 3.0, 4.0, 5.0], [8.0, 9.0, 10.0, 11.0]]
    assert [frames.shape[0] for frames in structured.embeddings] == [2, 1]


def test_hf_audio_supports_explicit_feature_mask_adapter(fake_audio_modules):
    class DownsampledWithoutPrivateHelper(FakeAudioModel):
        def __call__(self, **encoded):
            batch = encoded["input_values"].shape[0]
            hidden = np.arange(batch * 2 * 4, dtype=float).reshape(batch, 2, 4)
            hidden_states = tuple(FakeTensor(hidden + layer * 100) for layer in range(4))
            return types.SimpleNamespace(
                last_hidden_state=FakeTensor(hidden),
                pooler_output=FakeTensor(np.ones((batch, 4))),
                hidden_states=hidden_states if encoded.get("output_hidden_states") else None,
            )

    extractor = HFAudioExtractor(
        "audio",
        "fake-audio",
        sampling_rate=16_000,
        feature_mask_fn=explicit_feature_mask,
        structured_outputs=[{"name": "frames"}],
        cache_identity="explicit-feature-mask-v1",
    )
    extractor._processor = FakeAudioProcessor()
    extractor._model = DownsampledWithoutPrivateHelper()
    extractor._torch = FakeTorch

    output = extractor.transform_structured(
        [np.ones(3, dtype=np.float32), np.ones(2, dtype=np.float32)]
    )[0]
    recipe = extractor.recipe()

    assert [frames.shape[0] for frames in output.embeddings] == [2, 1]
    assert recipe["feature_mask_fn"].endswith(".explicit_feature_mask")
    assert recipe["callable_identities"]["feature_mask_fn"] is None
    assert recipe["cache_identity"] == "explicit-feature-mask-v1"
    assert recipe["cache_safe"] is True


def test_hf_audio_rejects_unequal_padding_without_attention_mask(fake_audio_modules):
    class ProcessorWithoutMask(FakeAudioProcessor):
        def __call__(self, batch, sampling_rate, **kwargs):
            encoded = super().__call__(batch, sampling_rate, **kwargs)
            encoded.pop("attention_mask")
            return encoded

    extractor = HFAudioExtractor(
        "audio",
        "fake-audio",
        sampling_rate=16_000,
        pooling="mean",
    )
    extractor._processor = ProcessorWithoutMask()
    extractor._model = FakeAudioModel()
    extractor._torch = FakeTorch

    with pytest.raises(ValueError, match="padded unequal-length clips"):
        extractor.transform([np.ones(3), np.ones(2)])


def test_hf_audio_rejects_equal_clips_padded_beyond_raw_length_without_mask(
    fake_audio_modules,
):
    class FixedPaddingProcessor(FakeAudioProcessor):
        def __call__(self, batch, sampling_rate, **kwargs):
            padded_length = max(len(item) for item in batch) + 2
            padded = np.zeros((len(batch), padded_length), dtype=np.float32)
            for index, item in enumerate(batch):
                padded[index, : len(item)] = item
            return {"input_values": FakeTensor(padded)}

    extractor = HFAudioExtractor(
        "audio",
        "fake-audio",
        sampling_rate=16_000,
        pooling="mean",
    )
    extractor._processor = FixedPaddingProcessor()
    extractor._model = FakeAudioModel()
    extractor._torch = FakeTorch

    with pytest.raises(ValueError, match="extended clips beyond their raw length"):
        extractor.transform([np.ones(3), np.ones(3)])


def test_hf_audio_rejects_maskless_fixed_feature_domain_padding(fake_audio_modules):
    class FixedFeatureProcessor:
        def __call__(self, batch, sampling_rate, **kwargs):
            return {
                "input_features": FakeTensor(np.zeros((len(batch), 80, 3_000), dtype=np.float32))
            }

    extractor = HFAudioExtractor(
        "audio",
        "fake-audio",
        sampling_rate=16_000,
        pooling="mean",
    )
    extractor._processor = FixedFeatureProcessor()
    extractor._model = FakeAudioModel()
    extractor._torch = FakeTorch

    with pytest.raises(ValueError, match="feature-domain.*alignment cannot be proven"):
        extractor.transform([np.ones(16_000)])


def test_hf_audio_structured_output_uses_boolean_feature_mask(fake_audio_modules):
    class NonContiguousMaskProcessor(FakeAudioProcessor):
        def __call__(self, batch, sampling_rate, **kwargs):
            encoded = super().__call__(batch, sampling_rate, **kwargs)
            encoded["attention_mask"] = FakeTensor([[1, 0, 1]])
            return encoded

    extractor = HFAudioExtractor(
        "audio",
        "fake-audio",
        sampling_rate=16_000,
        structured_outputs=[{"name": "frames"}],
    )
    extractor._processor = NonContiguousMaskProcessor()
    extractor._model = FakeAudioModel()
    extractor._torch = FakeTorch

    frames = extractor.transform_structured([np.ones(3)])[0].embeddings[0]

    assert frames.tolist() == [[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]]


def test_hf_audio_rejects_wrong_feature_mask_shape(fake_audio_modules):
    class DownsampledWithoutPrivateHelper(FakeAudioModel):
        def __call__(self, **encoded):
            batch = encoded["input_values"].shape[0]
            hidden = np.ones((batch, 2, 4), dtype=float)
            return types.SimpleNamespace(last_hidden_state=FakeTensor(hidden))

    extractor = HFAudioExtractor(
        "audio",
        "fake-audio",
        sampling_rate=16_000,
        feature_mask_fn=lambda _length, _mask: FakeTensor(np.ones((1, 2))),
    )
    extractor._processor = FakeAudioProcessor()
    extractor._model = DownsampledWithoutPrivateHelper()
    extractor._torch = FakeTorch

    with pytest.raises(ValueError, match=r"exact \(batch, frame\) shape"):
        extractor.transform([np.ones(3), np.ones(2)])
    assert extractor.recipe()["callable_identities"]["feature_mask_fn"] is None


@pytest.mark.parametrize("sampling_rate", [True, 16_000.0, "16000", 0, -1])
def test_hf_audio_requires_positive_exact_integer_sampling_rates(sampling_rate):
    with pytest.raises((TypeError, ValueError), match="sampling_rate"):
        HFAudioExtractor("audio", "fake-audio", sampling_rate=sampling_rate)


def test_hf_audio_validates_per_sample_sampling_rates(fake_audio_modules):
    extractor = HFAudioExtractor("audio", "fake-audio")

    with pytest.raises(TypeError, match="sampling_rate"):
        extractor.transform(
            {
                "array": [np.ones(2), np.ones(2)],
                "sampling_rate": [16_000, True],
            }
        )
