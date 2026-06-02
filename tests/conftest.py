import sys
import types

import numpy as np
import pytest


class FakeOverlapIndex:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)

    def fit_offline(self, Z, y, reset_state=True):
        assert reset_state is True
        seed = (self.kwargs.get("kmeans_kwargs") or {}).get("random_state")
        jitter = 0.0 if seed is None else (int(seed) % 17) / 1_000.0
        labels = np.unique(y)
        self.index = 0.80 + jitter
        self.singleton_index = {
            str(label): 0.70 + (idx * 0.03) + jitter for idx, label in enumerate(labels)
        }
        self.pairwise_index = {}
        self.sparse_adj = {}
        self.cluster_cardinality = {}
        self.rev_map = {idx: str(label) for idx, label in enumerate(labels)}
        return self.index


@pytest.fixture
def fake_overlapindex(monkeypatch):
    FakeOverlapIndex.calls = []
    module = types.SimpleNamespace(OverlapIndex=FakeOverlapIndex)
    monkeypatch.setitem(sys.modules, "overlapindex", module)
    return FakeOverlapIndex
