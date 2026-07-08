import sys
import types

import numpy as np
import pytest

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.extractors import HFTimeSeriesExtractor


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

    def flatten(self, start_dim=0):
        leading = self.data.shape[:start_dim]
        flattened = int(np.prod(self.data.shape[start_dim:]))
        return FakeTensor(self.data.reshape(*leading, flattened))

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

    @staticmethod
    def as_tensor(value):
        return FakeTensor(value)


class FakeTimeSeriesModel:
    last_call_kwargs = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        self.__class__.last_call_kwargs = encoded
        values = encoded["past_values"].data
        if values.ndim == 2:
            hidden = np.repeat(values[:, :, np.newaxis], 3, axis=2)
        else:
            hidden = values.astype(float)
        hidden_states = tuple(FakeTensor(hidden + layer * 100) for layer in range(4))
        return types.SimpleNamespace(
            last_hidden_state=FakeTensor(hidden),
            encoder_last_hidden_state=FakeTensor(hidden + 10),
            hidden_states=hidden_states if encoded.get("output_hidden_states") else None,
        )


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeTimeSeriesModel()


@pytest.fixture
def fake_time_series_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=FakeAutoModel),
    )


@pytest.mark.parametrize("pooling", ["mean", "last", "flatten"])
def test_hf_time_series_pooling_modes(fake_time_series_modules, pooling):
    extractor = HFTimeSeriesExtractor(
        name=f"ts_{pooling}",
        model_id="fake-timeseries",
        pooling=pooling,
        batch_size=2,
    )

    output = extractor.transform(np.arange(18, dtype=float).reshape(3, 6))

    assert output.shape[0] == 3
    assert output.dtype == np.float32
    assert extractor.recipe()["modality"] == "time_series"


def test_hf_time_series_preserves_optional_inputs(fake_time_series_modules):
    dataset = BenchmarkDataset.from_time_series(
        series=np.arange(24, dtype=float).reshape(4, 3, 2),
        labels=["a", "a", "b", "b"],
        observed_mask=np.ones((4, 3, 2), dtype=float),
        time_features=np.arange(24, dtype=float).reshape(4, 3, 2),
    )
    extractor = HFTimeSeriesExtractor(name="ts", model_id="fake-timeseries", batch_size=2)

    output = extractor.transform(dataset.X)

    assert output.shape == (4, 2)
    assert "past_observed_mask" in FakeTimeSeriesModel.last_call_kwargs
    assert "past_time_features" in FakeTimeSeriesModel.last_call_kwargs


def test_hf_time_series_selects_hidden_layer(fake_time_series_modules):
    extractor = HFTimeSeriesExtractor(
        name="ts",
        model_id="fake-timeseries",
        pooling="last",
        hidden_layer=2,
        batch_size=2,
    )

    output = extractor.transform(np.arange(12, dtype=float).reshape(2, 6))

    assert output.tolist() == [
        [205.0, 205.0, 205.0],
        [211.0, 211.0, 211.0],
    ]


def test_hf_time_series_rejects_invalid_shape(fake_time_series_modules):
    extractor = HFTimeSeriesExtractor(name="ts", model_id="fake-timeseries")

    with pytest.raises(ValueError, match="2D or 3D"):
        extractor.transform(np.arange(6, dtype=float))


def test_hf_time_series_evaluator_workflow(fake_time_series_modules, fake_overlapindex):
    dataset = BenchmarkDataset.from_time_series(
        series=np.array(
            [
                [0.0, 0.1, 0.2],
                [0.2, 0.3, 0.4],
                [1.0, 1.1, 1.2],
                [1.2, 1.3, 1.4],
            ],
            dtype=float,
        ),
        labels=["left", "left", "right", "right"],
    )
    extractor = HFTimeSeriesExtractor(
        name="ts",
        model_id="fake-timeseries",
        pooling="mean",
        batch_size=2,
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        stability_config=StabilityConfig(repeats=2),
        cache_config=CacheConfig(enabled=False),
    ).run()

    assert result.extractor_results[0].name == "ts"
    assert len(fake_overlapindex.calls) == 3
