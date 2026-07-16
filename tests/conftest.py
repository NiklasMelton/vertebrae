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
        y_arr = np.asarray(y)
        self.__class__.calls[-1]["fit_y_shape"] = list(y_arr.shape)
        self.__class__.calls[-1]["fit_y"] = y_arr.copy()
        seed = (self.kwargs.get("kmeans_kwargs") or {}).get("random_state")
        jitter = 0.0 if seed is None else (int(seed) % 17) / 1_000.0
        labels = np.arange(y_arr.shape[1]) if y_arr.ndim == 2 else np.unique(y_arr)
        self.index = 0.80 + jitter
        self.singleton_index = {
            str(label): 0.70 + (idx * 0.03) + jitter for idx, label in enumerate(labels)
        }
        self.pairwise_index = {}
        self.sparse_adj = {}
        self.cluster_cardinality = {}
        self.rev_map = {idx: str(label) for idx, label in enumerate(labels)}
        return self.index


class FakeContinuousOverlapIndex:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)

    def fit_offline(self, Z, y, reset_state=True):
        assert reset_state is True
        y_arr = np.asarray(y, dtype=float)
        self.__class__.calls[-1]["fit_y_shape"] = list(y_arr.shape)
        self.__class__.calls[-1]["fit_y"] = y_arr.copy()
        self.__class__.calls[-1]["fit_X_shape"] = list(np.asarray(Z).shape)
        seed = self.kwargs.get("random_state")
        jitter = 0.0 if seed is None else (int(seed) % 13) / 1_000.0
        self.index = 0.62 + jitter
        self.macro_index_ = 0.60 + jitter
        self.raw_index_ = 0.58 + jitter
        self.actual_loss_ = 0.12
        self.null_loss_ = 0.24
        self.loss_ratio_ = 0.50
        self.prototype_index_ = {0: 0.66 + jitter, 1: 0.59 + jitter}
        self.prototype_support_ = {0: 4, 1: 4}
        self.prototype_target_mean_ = {0: [0.2], 1: [0.8]}
        self.prototype_target_radius_ = {0: 0.1, 1: 0.1}
        self.prototype_target_values_ = {
            0: y_arr[: len(y_arr) // 2].tolist(),
            1: y_arr[len(y_arr) // 2 :].tolist(),
        }
        self.prototype_adjacency_normalized_ = {0: {1: 0.4}, 1: {0: 0.4}}
        return self.index

    @property
    def weighted_index(self):
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

    def fit(self, X, y, **kwargs):
        report = _build_fake_separatix_payload(X, y, self.kwargs)
        self.report_ = FakeSeparatixReport(report)
        self.__class__.calls.append(
            {
                "kind": "profiler_fit",
                "shape": list(X.shape),
                "is_sparse": hasattr(X, "tocsr"),
                "n_labels": len(y),
                "y_shape": list(np.asarray(y).shape),
                **kwargs,
            }
        )
        return self

    def report(self):
        return self.report_


def _build_fake_mlp_payload(X, kwargs):
    requested = bool(kwargs.get("mlp_probes", False))
    if not requested:
        return {
            "status": "not_requested",
            "reason": "MLP probes were disabled.",
            "trigger": {
                "status": "not_requested",
                "threshold": kwargs.get("mlp_trigger_skill_threshold", 0.75),
            },
            "backend": {
                "requested_device": kwargs.get("mlp_device", "cpu"),
                "resolved_device": None,
            },
            "architectures": [],
            "aligned_comparators": {},
            "best_architecture": None,
            "pairwise_comparisons": {},
            "required_comparators_complete": False,
            "recommendation_override": False,
            "override_reason": None,
            "sample_info": None,
        }
    return {
        "status": "executed",
        "reason": "MLP probes executed in the fake test harness.",
        "trigger": {
            "status": "triggered",
            "reason": "No simpler fake probe met the configured threshold.",
            "threshold": kwargs.get("mlp_trigger_skill_threshold", 0.75),
        },
        "backend": {
            "requested_device": kwargs.get("mlp_device", "cpu"),
            "resolved_device": "cpu",
        },
        "architectures": ["mlp_one_layer_compact", "mlp_two_layer_compact"],
        "aligned_comparators": {"linear": True, "smooth_poly": True},
        "best_architecture": "mlp_two_layer_compact",
        "pairwise_comparisons": {
            "mlp_two_layer_compact_vs_linear": {"mean_delta": 0.05, "better": True}
        },
        "required_comparators_complete": True,
        "recommendation_override": True,
        "override_reason": "Fake MLP improved beyond the configured minimum.",
        "sample_info": {"n_samples": int(X.shape[0])},
    }


def _build_fake_separatix_payload(X, y, kwargs):
    label_summary = _fake_class_summary(y)
    mlp_payload = _build_fake_mlp_payload(X, kwargs)
    target_mode = kwargs.get("target_mode", "singlelabel")
    if target_mode == "multilabel":
        probe_scores = {
            "dummy": {"micro_f1": 0.30, "macro_f1": 0.20, "sample_jaccard": 0.10},
            "linear": {"micro_f1": 0.78, "macro_f1": 0.74, "sample_jaccard": 0.68},
            "knn": {"micro_f1": 0.76, "macro_f1": 0.72, "sample_jaccard": 0.66},
            "smooth_poly": {
                "micro_f1": 0.86,
                "macro_f1": 0.82,
                "sample_jaccard": 0.77,
            },
            "kernel_approx": {
                "micro_f1": 0.83,
                "macro_f1": 0.79,
                "sample_jaccard": 0.73,
            },
        }
        best_probe_metric = "macro_f1"
    elif target_mode == "regression":
        probe_scores = {
            "dummy": {"r2": 0.0, "mae": 0.30, "rmse": 0.35},
            "linear": {"r2": 0.71, "mae": 0.16, "rmse": 0.20},
            "knn": {"r2": 0.68, "mae": 0.18, "rmse": 0.22},
            "smooth_poly": {"r2": 0.84, "mae": 0.11, "rmse": 0.15},
            "kernel_approx": {"r2": 0.80, "mae": 0.13, "rmse": 0.17},
        }
        best_probe_metric = "r2"
    else:
        probe_scores = {
            "dummy": {"accuracy": 0.50, "balanced_accuracy": 0.50, "macro_f1": 0.33},
            "linear": {"accuracy": 0.83, "balanced_accuracy": 0.82, "macro_f1": 0.82},
            "knn": {"accuracy": 0.79, "balanced_accuracy": 0.78, "macro_f1": 0.78},
            "smooth_poly": {
                "accuracy": 0.91,
                "balanced_accuracy": 0.89,
                "macro_f1": 0.89,
            },
            "kernel_approx": {
                "accuracy": 0.87,
                "balanced_accuracy": 0.86,
                "macro_f1": 0.86,
            },
        }
        best_probe_metric = None
    for probe in probe_scores.values():
        probe["evaluation_mode"] = "cross_validation"
        probe["sample_info"] = {
            "sampled": False,
            "n_original": int(X.shape[0]),
            "n_used": int(X.shape[0]),
        }
    return {
        "recommendation": "smooth_nonlinear_recommended",
        "recommendation_text": "Recommendation: smooth nonlinear boundary.",
        "confidence": "high",
        "metrics": {
            "audit": {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])},
            "geometry": {},
            "probes": probe_scores,
            "baseline": {
                "best_probe": "smooth_poly",
                "best_probe_score": probe_scores["smooth_poly"][
                    best_probe_metric or "balanced_accuracy"
                ],
                **({"best_probe_metric": best_probe_metric} if best_probe_metric else {}),
            },
            "neighborhood": {},
            "boundary": {},
            "graph": {},
            "topology": {"mode": "auto", "skipped_reason": "not requested"},
            "mlp_trigger_evidence": mlp_payload.get("trigger", {}),
            "mlp_probes": mlp_payload,
            "mlp_recommendation_evidence": {
                key: value for key, value in mlp_payload.items() if key != "trigger"
            },
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
        "class_summary": label_summary,
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
            "mlp_probes": kwargs.get("mlp_probes", False),
            "mlp_device": kwargs.get("mlp_device", "cpu"),
            "mlp_trigger_skill_threshold": kwargs.get("mlp_trigger_skill_threshold", 0.75),
            "mlp_min_improvement": kwargs.get("mlp_min_improvement", 0.02),
            "mlp_max_parameters": kwargs.get("mlp_max_parameters"),
        },
    }


def _fake_class_summary(y):
    y_arr = np.asarray(y)
    if y_arr.ndim == 2:
        counts = np.asarray(y_arr.sum(axis=0), dtype=int)
        labels = [f"label_{index}" for index in range(y_arr.shape[1])]
        return {
            "n_classes": int(y_arr.shape[1]),
            "classes": labels,
            "class_counts": {label: int(count) for label, count in zip(labels, counts.tolist())},
            "imbalance_ratio": 1.0,
            "min_class_count": int(counts.min()) if counts.size else 0,
            "max_class_count": int(counts.max()) if counts.size else 0,
        }
    labels, counts = np.unique(y_arr, return_counts=True)
    return {
        "n_classes": len(labels),
        "classes": [str(label) for label in labels],
        "class_counts": {str(label): int(count) for label, count in zip(labels, counts)},
        "imbalance_ratio": 1.0,
        "min_class_count": int(min(counts)),
        "max_class_count": int(max(counts)),
    }


@pytest.fixture
def fake_overlapindex(monkeypatch):
    FakeOverlapIndex.calls = []
    FakeContinuousOverlapIndex.calls = []
    module = types.SimpleNamespace(
        OverlapIndex=FakeOverlapIndex,
        ContinuousOverlapIndex=FakeContinuousOverlapIndex,
    )
    monkeypatch.setitem(sys.modules, "overlapindex", module)
    return types.SimpleNamespace(
        OverlapIndex=FakeOverlapIndex,
        ContinuousOverlapIndex=FakeContinuousOverlapIndex,
        calls=FakeOverlapIndex.calls,
        continuous_calls=FakeContinuousOverlapIndex.calls,
    )


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
                "y_shape": list(np.asarray(y).shape),
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
