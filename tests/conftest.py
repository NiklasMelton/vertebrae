import sys
import types

import numpy as np
import pytest
from scipy import sparse


class FakeOverlapIndex:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)

    def fit_offline(self, Z, y, reset_state=True):
        assert reset_state is True
        y_arr = y.toarray() if sparse.issparse(y) else np.asarray(y)
        self.__class__.calls[-1]["fit_X_sparse"] = sparse.issparse(Z)
        self.__class__.calls[-1]["fit_X_format"] = Z.format if sparse.issparse(Z) else "dense"
        self.__class__.calls[-1]["fit_y_sparse"] = sparse.issparse(y)
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

    def fit(self, Z, y):
        self.fit_offline(Z, y, reset_state=True)
        return self

    def score_fixed(self, Z, y):
        y_arr = np.asarray(y)
        self.__class__.calls[-1]["score_fixed_X_shape"] = list(Z.shape)
        self.__class__.calls[-1]["score_fixed_y"] = y_arr.copy()
        self.cluster_cardinality = {
            str(label): int(np.count_nonzero(y_arr == label)) for label in np.unique(y_arr)
        }
        return self.index

    @property
    def weighted_index(self):
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
        self.__class__.calls[-1]["fit_X_shape"] = list(Z.shape)
        self.__class__.calls[-1]["fit_X_sparse"] = sparse.issparse(Z)
        self.__class__.calls[-1]["fit_X_format"] = Z.format if sparse.issparse(Z) else "dense"
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
                "n_labels": int(y.shape[0]),
                "y_shape": list(y.shape),
                "y_sparse": sparse.issparse(y),
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
        "architectures": [
            {
                "probe_name": "mlp_one_layer_compact",
                "probe_recipe": _fake_probe_recipe(
                    "mlp_one_layer_compact", "mlp", "mlp_architecture"
                ),
            },
            {
                "probe_name": "mlp_two_layer_compact",
                "probe_recipe": _fake_probe_recipe(
                    "mlp_two_layer_compact", "mlp", "mlp_architecture"
                ),
            },
        ],
        "aligned_comparators": {
            "linear": {
                "probe_recipe": _fake_probe_recipe("linear", "linear", "mlp_aligned_comparator")
            },
            "smooth_poly": {
                "probe_recipe": _fake_probe_recipe(
                    "smooth_poly", "smooth_nonlinear", "mlp_aligned_comparator"
                )
            },
        },
        "best_architecture": {
            "probe_name": "mlp_two_layer_compact",
            "probe_recipe_id": "fake-mlp_two_layer_compact",
        },
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
            "dummy": {
                "r2_variance_weighted": 0.0,
                "r2_uniform_average": 0.0,
                "normalized_rmse_mean": 1.0,
            },
            "linear": {
                "r2_variance_weighted": 0.71,
                "r2_uniform_average": 0.70,
                "normalized_rmse_mean": 0.55,
            },
            "knn": {
                "r2_variance_weighted": 0.68,
                "r2_uniform_average": 0.67,
                "normalized_rmse_mean": 0.58,
            },
            "smooth_poly": {
                "r2_variance_weighted": 0.84,
                "r2_uniform_average": 0.82,
                "normalized_rmse_mean": 0.41,
            },
            "kernel_approx": {
                "r2_variance_weighted": 0.80,
                "r2_uniform_average": 0.79,
                "normalized_rmse_mean": 0.45,
            },
        }
        best_probe_metric = None
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
    for probe_name, probe in probe_scores.items():
        probe["evaluation_plan_id"] = "fake-plan"
        probe["cv_stratification_method"] = "stratified_kfold"
        probe["probe_recipe"] = _fake_probe_recipe(
            probe_name,
            "linear" if probe_name == "linear" else "smooth_nonlinear",
            "core_probe",
        )
    if target_mode == "multilabel":
        family_metrics = ("micro_f1", "macro_f1", "sample_jaccard")
        family_evidence = {
            family: {metric: {"probe": probe} for metric in family_metrics}
            for family, probe in (
                ("linear", "linear"),
                ("smooth_nonlinear", "smooth_poly"),
                ("local_kernel", "knn"),
            )
        }
    elif target_mode == "regression":
        family_metrics = ("r2_variance_weighted", "r2_uniform_average")
        family_evidence = {
            family: {metric: {"probe": probe} for metric in family_metrics}
            for family, probe in (
                ("linear", "linear"),
                ("smooth_nonlinear", "smooth_poly"),
                ("local_kernel", "knn"),
            )
        }
    else:
        family_evidence = {
            "linear": {"best_probe": "linear"},
            "smooth_nonlinear": {"best_probe": "smooth_poly"},
            "local_kernel": {"best_probe": "knn"},
        }
    target_evidence_key = {
        "singlelabel": "recommendation_evidence",
        "multilabel": "multilabel_recommendation_evidence",
        "regression": "regression_recommendation_evidence",
    }[target_mode]
    target_evidence = {
        "recommended_family": "smooth_nonlinear",
        "selected_family": "smooth_nonlinear",
        "best_probe": "smooth_poly",
        "families": family_evidence,
        "plausible_family_set": {
            "status": "available",
            "scope": "core_probe_families",
            "minimum_recommended_family": "smooth_nonlinear",
            "plausible_families": ["smooth_nonlinear"],
            "decision_method": "paired_oof_bootstrap",
            "reason": None,
        },
    }
    return {
        "recommendation": "smooth_nonlinear_recommended",
        "recommendation_text": "Recommendation: smooth nonlinear boundary.",
        "confidence": "high",
        "metrics": {
            "audit": {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])},
            "geometry": {},
            "probes": probe_scores,
            "baseline": (
                {
                    "primary_metrics": [
                        "r2_variance_weighted",
                        "r2_uniform_average",
                    ],
                    "best_by_metric": {
                        "r2_variance_weighted": {
                            "probe": "smooth_poly",
                            "score": 0.84,
                        },
                        "r2_uniform_average": {
                            "probe": "smooth_poly",
                            "score": 0.82,
                        },
                    },
                }
                if target_mode == "regression"
                else {
                    "best_probe": "smooth_poly",
                    "best_probe_score": probe_scores["smooth_poly"][
                        best_probe_metric or "balanced_accuracy"
                    ],
                    **({"best_probe_metric": best_probe_metric} if best_probe_metric else {}),
                }
            ),
            "neighborhood": {},
            "boundary": {},
            "graph": {},
            "topology": {"mode": "auto", "skipped_reason": "not requested"},
            "mlp_trigger_evidence": mlp_payload.get("trigger", {}),
            "mlp_probes": mlp_payload,
            "mlp_recommendation_evidence": {
                key: value for key, value in mlp_payload.items() if key != "trigger"
            },
            target_evidence_key: target_evidence,
            "probe_evaluation": {
                "alignment_status": "aligned",
                "evaluation_plan_id": "fake-plan",
                "cv_method": "stratified_kfold",
                "n_samples": int(X.shape[0]),
                "n_splits": 5,
                "group_aware": False,
                "effective_train_size_summary": {
                    "status": "available",
                    "basis": "held_out_folds",
                    "min": max(1, int(X.shape[0]) - 2),
                    "median": max(1, int(X.shape[0]) - 2),
                    "mean": float(max(1, int(X.shape[0]) - 2)),
                    "max": max(1, int(X.shape[0]) - 2),
                },
            },
            "paired_probe_comparisons": {
                "status": "available",
                "method": "paired_oof_bootstrap",
                "evaluation_plan_id": "fake-plan",
                "resamples_used": 50,
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


def _fake_probe_recipe(name, family, role):
    """Return a compact 0.1.1-shaped recipe for adapter tests."""

    return {
        "schema": "separatix.probe_recipe",
        "schema_version": 1,
        "recipe_id": f"fake-{name}",
        "probe": {
            "name": name,
            "family": family,
            "target_mode": "singlelabel",
            "role": role,
        },
        "implementation": {"key": "fake", "version": 1},
        "input_contract": {},
        "estimator": {"kind": "estimator", "key": "fake", "params": {}},
        "training_policy": {},
        "created_with": {
            "separatix": "0.1.1",
            "python": "3.9",
            "numpy": "2.0",
            "scipy": "1.13",
            "scikit_learn": "1.5",
            "torch": None,
        },
    }


def _fake_class_summary(y):
    y_arr = y.toarray() if sparse.issparse(y) else np.asarray(y)
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
                "n_labels": int(y.shape[0]),
                "y_shape": list(y.shape),
                "y_sparse": sparse.issparse(y),
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
