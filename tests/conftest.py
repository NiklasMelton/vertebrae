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


class FakeSeparatixReport:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class FakeComplexityProfiler:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.report_ = None
        self.__class__.calls.append({"kind": "profiler_init", **kwargs})

    def fit(self, X, y):
        report = _build_fake_separatix_payload(X, y, self.kwargs)
        self.report_ = FakeSeparatixReport(report)
        self.__class__.calls.append(
            {
                "kind": "profiler_fit",
                "shape": list(X.shape),
                "is_sparse": hasattr(X, "tocsr"),
                "n_labels": len(y),
            }
        )
        return self

    def report(self):
        return self.report_


def _build_fake_separatix_payload(X, y, kwargs):
    return {
        "recommendation": "smooth_nonlinear_recommended",
        "recommendation_text": "Recommendation: smooth nonlinear boundary.",
        "confidence": "high",
        "metrics": {
            "audit": {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])},
            "geometry": {},
            "probes": {},
            "baseline": {},
            "neighborhood": {},
            "boundary": {},
            "graph": {},
            "topology": {"mode": "auto", "skipped_reason": "not requested"},
        },
        "scores": {
            "signal_score": 0.9,
            "overlap_score": 0.2,
            "linearity_score": 0.6,
            "nonlinearity_score": 0.7,
            "fragmentation_score": 0.1,
            "topology_score": None,
            "reliability_score": 0.95,
        },
        "interpretations": {"signal": "strong", "reliability": "high"},
        "decision_path": [
            "Probe signal is strong.",
            "Nonlinear probe improvement suggests a smooth boundary.",
        ],
        "warnings": ["fake separatix warning"] if kwargs.get("max_dense_mb") == 1 else [],
        "errors": [],
        "skipped_diagnostics": [{"name": "persistent_topology", "reason": "not requested"}],
        "preprocessing": {"input_type": type(X).__name__, "is_sparse": hasattr(X, "tocsr")},
        "sampling": {"probe": None, "neighbors": None, "boundary": None},
        "densification_events": [],
        "class_summary": {
            "n_classes": len(np.unique(y)),
            "classes": [str(label) for label in np.unique(y)],
            "class_counts": {
                str(label): int(np.sum(np.asarray(y) == label)) for label in np.unique(y)
            },
            "imbalance_ratio": 1.0,
            "min_class_count": int(min(np.unique(y, return_counts=True)[1])),
            "max_class_count": int(max(np.unique(y, return_counts=True)[1])),
        },
        "runtime": {"total_seconds": 0.01},
        "config": {
            "budget": kwargs.get("budget", "standard"),
            "topology": kwargs.get("topology", "auto"),
            "densify_policy": kwargs.get("densify_policy", "warn_and_sample"),
            "max_dense_mb": kwargs.get("max_dense_mb", 512),
            "max_samples": kwargs.get("max_samples"),
            "min_dense_samples": 200,
            "random_state": kwargs.get("random_state"),
            "warn_on_densify": kwargs.get("warn_on_densify", True),
            "n_jobs": kwargs.get("n_jobs"),
        },
    }


@pytest.fixture
def fake_overlapindex(monkeypatch):
    FakeOverlapIndex.calls = []
    module = types.SimpleNamespace(OverlapIndex=FakeOverlapIndex)
    monkeypatch.setitem(sys.modules, "overlapindex", module)
    return FakeOverlapIndex


@pytest.fixture
def fake_separatix(monkeypatch):
    FakeComplexityProfiler.calls = []

    def diagnose(X, y, **kwargs):
        FakeComplexityProfiler.calls.append(
            {
                "kind": "diagnose",
                "shape": list(X.shape),
                "is_sparse": hasattr(X, "tocsr"),
                "n_labels": len(y),
                **kwargs,
            }
        )
        return FakeSeparatixReport(_build_fake_separatix_payload(X, y, kwargs))

    module = types.SimpleNamespace(
        diagnose=diagnose,
        ComplexityProfiler=FakeComplexityProfiler,
        DiagnosticReport=FakeSeparatixReport,
    )
    monkeypatch.setitem(sys.modules, "separatix", module)
    return module


@pytest.fixture(autouse=True)
def _install_fake_separatix(fake_separatix):
    return fake_separatix
