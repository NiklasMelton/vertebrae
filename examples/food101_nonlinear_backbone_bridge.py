"""Frozen Food-101 confirmatory nonlinear-backbone bridge.

This example is deliberately independent of the exploratory Oxford/CIFAR
bridge.  It asks a narrow, prespecified question on the first forty Food-101
classes (alphabetical order): does the cross-fitted OverlapIndex selector
benefit from the frozen nonlinear arm more than a fixed linear probe does?

The train split contributes 80 selector and 52 development images per class
for each of five paired replicates.  The official test split contributes 52
reference images per class.  Three paired geometry arms are evaluated at
``q=1``: baseline, full nonlinearity, and full nuisance.  The nuisance arm is
reported as a diagnostic and never supports the confirmatory claim.

Imports for torchvision, torch, scikit-learn, and the backbone providers are
lazy.  Consequently ``python examples/food101_nonlinear_backbone_bridge.py
--help`` is safe in a minimal documentation environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

try:  # psutil is a package dependency, but keep --help importable in docs builds.
    import psutil
except Exception:  # pragma: no cover - defensive optional import
    psutil = None  # type: ignore[assignment]

try:
    from vertebrae.config import OverlapScoringConfig
    from vertebrae.scoring.overlap import OverlapIndexScorer
except Exception:  # pragma: no cover - minimal environments may omit vertebrae extras
    OverlapScoringConfig = None  # type: ignore[assignment]
    OverlapIndexScorer = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Frozen scientific constants
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = (
    "dinov2-small",
    "deit-tiny",
    "convnext-tiny",
    "mobilenetv3-large",
    "openclip-vit-b-32",
    "resnet50",
    "efficientnet-b0",
    "swin-tiny",
    "vit-small-16",
    "densenet121",
)
_FINAL_OUTPUTS = frozenset({"final", "final_cls", "final_image"})
_EXTRA_TIMM_MODELS = {
    "resnet50": "resnet50",
    "efficientnet-b0": "efficientnet_b0",
    "swin-tiny": "swin_tiny_patch4_window7_224",
    "vit-small-16": "vit_small_patch16_224",
    "densenet121": "densenet121",
}

_FOOD101_CLASS_COUNT = 40
_FOOD101_CLASSES_COUNT = _FOOD101_CLASS_COUNT
_SELECTOR_PER_CLASS = 80
_DEVELOPMENT_PER_CLASS = 52
_TEST_PER_CLASS = 52
_REPLICATES = 5
_FOOD101_REPLICATES = _REPLICATES
_FOOD101_BUDGETS = (64, 68, 72, 80)
_DEFAULT_BUDGETS = _FOOD101_BUDGETS
_BUDGETS = _FOOD101_BUDGETS
_QUALITY_LEVELS = (1.0,)
_QUALITY_VALUES = _QUALITY_LEVELS
_Q_VALUES = _QUALITY_LEVELS
_Q = 1.0
_FROZEN_K = 10
_CROSS_FIT_FOLDS = 5
_PROBE_FOLDS = 5
_BOOTSTRAP_RESAMPLES = 10_000
_METHODS = ("overlap_cross_fitted", "linear_probe_oof")
_HEAD_FAMILIES = ("linear", "quadratic", "knn", "rbf")
_FOOD101_ARMS = (
    ("baseline", 0.0, 0.0),
    ("nonlinearity_full", 1.0, 0.0),
    ("nuisance_full", 0.0, 1.5),
)
_ARMS = _FOOD101_ARMS
_RESULT_SCHEMA_VERSION = 1
_PROTOCOL_VERSION = 1
_STUDY = "food101_nonlinear_backbone_bridge"
_ARTIFACT_STEM_PREFIX = "food101_nonlinear_backbone_bridge"


@dataclass(frozen=True)
class Food101Sample:
    """An image row with stable identity and semantic Food-101 class."""

    sample_id: str
    image_path: Path
    class_name: str
    source_split: str

    @property
    def label(self) -> str:
        return self.class_name

    @property
    def food_class(self) -> str:
        return self.class_name

    @property
    def split(self) -> str:
        return self.source_split

    @property
    def official_split(self) -> str:
        return self.source_split


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", default="auto", help="CPU scoring workers (default: auto)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Sequential extraction device (default: auto)",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse verified caches/checkpoints")
    parser.add_argument("--data-dir", type=Path, default=Path("examples/data"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument(
        "--models", default="", help="Comma-separated subset of the ten frozen models"
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--budgets", default="64,68,72,80", help="Per-class selector budgets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--replicates", type=int, default=_REPLICATES)
    parser.add_argument("--bootstrap-resamples", type=int, default=_BOOTSTRAP_RESAMPLES)
    return parser


def _parser() -> argparse.ArgumentParser:
    """Return the lazy, public argument parser."""

    return _build_parser()


def build_parser() -> argparse.ArgumentParser:
    """Public parser alias used by notebooks and embedding applications."""

    return _parser()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with replace-on-success semantics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_float32_memmap(path: Path, values: np.ndarray) -> Dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embedding cache values must be a two-dimensional matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=matrix.shape)
    mapped[:] = matrix
    mapped.flush()
    del mapped
    return {
        "path": str(path),
        "shape": list(matrix.shape),
        "dtype": "float32",
        "sha256": _file_sha256(path),
    }


def _read_float32_memmap(manifest: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(manifest["path"]))
    if not path.exists():
        raise FileNotFoundError(path)
    expected = manifest.get("sha256")
    if expected and _file_sha256(path) != expected:
        raise ValueError(f"Embedding cache hash mismatch for {path}")
    matrix = np.load(path, mmap_mode="r")
    expected_shape = list(manifest.get("shape", matrix.shape))
    if list(matrix.shape) != expected_shape:
        raise ValueError(f"Embedding cache shape mismatch for {path}")
    if str(matrix.dtype) != str(manifest.get("dtype", "float32")):
        raise ValueError(f"Embedding cache dtype mismatch for {path}")
    return matrix


def _l2_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


def _bridge_transform(
    x: Optional[np.ndarray] = None,
    *,
    embeddings: Optional[np.ndarray] = None,
    base: Optional[np.ndarray] = None,
    donor: Optional[np.ndarray] = None,
    donor_embeddings: Optional[np.ndarray] = None,
    noise: Optional[np.ndarray] = None,
    noise_bank: Optional[np.ndarray] = None,
    mode: Optional[np.ndarray] = None,
    mode_bank: Optional[np.ndarray] = None,
    nuisance: Optional[np.ndarray] = None,
    nuisance_bank: Optional[np.ndarray] = None,
    quality: Optional[float] = None,
    q: Optional[float] = None,
    lambda_: float = 0.0,
    lam: Optional[float] = None,
    nonlinearity: Optional[float] = None,
    nuisance_strength: float = 0.0,
    nu: Optional[float] = None,
    seed: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply the paired geometry bridge, preserving a constant feature width."""

    values = embeddings if embeddings is not None else base if base is not None else x
    if values is None:
        raise TypeError("x/embeddings must be provided")
    matrix = _l2_rows(np.asarray(values, dtype=np.float32))
    generator = rng or np.random.default_rng(int(seed))
    donor_values = donor if donor is not None else donor_embeddings
    donor_values = donor_values if donor_values is not None else noise
    donor_values = donor_values if donor_values is not None else noise_bank
    if donor_values is None:
        donor_values = generator.normal(size=matrix.shape).astype(np.float32)
    donor_matrix = np.asarray(donor_values, dtype=np.float32)
    if donor_matrix.shape != matrix.shape:
        raise ValueError("donor/noise bank must match x shape")
    mode_values = mode if mode is not None else mode_bank
    if mode_values is None:
        mode_values = generator.choice(np.asarray([-1.0, 1.0]), size=len(matrix))
    mode_array = np.asarray(mode_values, dtype=np.float32).reshape(-1)
    if len(mode_array) != len(matrix):
        raise ValueError("mode bank must contain one value per row")
    nuisance_values = nuisance if nuisance is not None else nuisance_bank
    if nuisance_values is None:
        nuisance_values = generator.normal(size=matrix.shape).astype(np.float32)
    nuisance_matrix = np.asarray(nuisance_values, dtype=np.float32)
    if nuisance_matrix.shape != matrix.shape:
        raise ValueError("nuisance bank must match x shape")
    quality_value = float(q if q is not None else quality if quality is not None else 1.0)
    lambda_value = float(
        lam if lam is not None else nonlinearity if nonlinearity is not None else lambda_
    )
    nuisance_value = float(nu if nu is not None else nuisance_strength)
    if not 0.0 <= quality_value <= 1.0:
        raise ValueError("quality/q must be between 0 and 1")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lam/lambda_ must be between 0 and 1")
    if nuisance_value < 0.0:
        raise ValueError("nu/nuisance_strength must be non-negative")
    u = _l2_rows(
        quality_value * matrix + np.sqrt(max(0.0, 1.0 - quality_value**2)) * _l2_rows(donor_matrix)
    )
    scalar_mode = mode_array[:, None]
    joined = np.concatenate(
        (
            np.sqrt(1.0 - lambda_value) * u,
            np.sqrt(lambda_value) * scalar_mode * u,
            np.sqrt(lambda_value) * scalar_mode,
            nuisance_value * quality_value * nuisance_matrix,
        ),
        axis=1,
    )
    return _l2_rows(joined).astype(np.float32, copy=False)


_apply_bridge_transform = _bridge_transform


def _first_food101_classes(
    classes: Sequence[Any], count: int = _FOOD101_CLASS_COUNT
) -> tuple[str, ...]:
    """Return the deterministic alphabetical Food-101 class prefix."""

    ordered = tuple(sorted({str(value) for value in classes}))
    if len(ordered) < int(count):
        raise ValueError(f"Food-101 metadata exposes {len(ordered)} classes; {count} required")
    return ordered[: int(count)]


_food101_class_prefix = _first_food101_classes
_select_food101_classes = _first_food101_classes


def _food101_cohort_splits(
    samples: Sequence[Any],
    labels: Optional[Sequence[Any]] = None,
    *,
    test_samples: Optional[Sequence[Any]] = None,
    test_labels: Optional[Sequence[Any]] = None,
    replicates: int = _REPLICATES,
    repeats: Optional[int] = None,
    seed: int = 42,
    selector_per_class: int = _SELECTOR_PER_CLASS,
    development_per_class: int = _DEVELOPMENT_PER_CLASS,
    test_per_class: int = _TEST_PER_CLASS,
) -> Dict[int, Dict[str, list[Any]]]:
    """Build deterministic, disjoint train roles and a fixed test reference role.

    Replicates consume disjoint train rows when enough official rows are
    available.  Test reference rows are fixed across replicates, making the
    official-test component paired in the hierarchical bootstrap.
    """

    count = int(repeats if repeats is not None else replicates)
    if count < 1:
        raise ValueError("replicates must be >=1")
    rows = list(samples)
    labels_array = np.asarray(
        labels
        if labels is not None
        else [getattr(row, "class_name", getattr(row, "label", None)) for row in rows],
        dtype=object,
    )
    if len(labels_array) != len(rows):
        raise ValueError("samples and labels must have equal length")
    grouped: Dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels_array):
        grouped[str(label)].append(index)
    if not grouped:
        raise ValueError("Food-101 cohorts require at least one labeled row")
    required = int(selector_per_class + development_per_class)
    # A disjoint replicate panel is the frozen design.  Never reuse rows across
    # replicates: doing so would invalidate the paired replicate bootstrap.
    disjoint_required = required * count
    permutations: Dict[str, np.ndarray] = {}
    for label, values in grouped.items():
        if len(values) < disjoint_required:
            raise ValueError(
                f"Class {label!r} has {len(values)} rows; "
                f"{disjoint_required} required for {count} disjoint replicates"
            )
        stable = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
        generator = np.random.default_rng(int(seed) + stable)
        permutations[label] = generator.permutation(np.asarray(values, dtype=np.int64))
    result: Dict[int, Dict[str, list[Any]]] = {}
    for replicate in range(count):
        roles: Dict[str, list[Any]] = {"selector": [], "development": [], "test": []}
        for label in sorted(grouped):
            indices = np.asarray(grouped[label], dtype=np.int64)
            if len(indices) < required:
                raise ValueError(f"Class {label!r} has {len(indices)} rows; {required} required")
            permuted = permutations[label]
            chosen = permuted[replicate * required : (replicate + 1) * required]
            selector_end = int(selector_per_class)
            roles["selector"].extend(rows[int(index)] for index in chosen[:selector_end])
            roles["development"].extend(rows[int(index)] for index in chosen[selector_end:])

        if test_samples is None:
            test_rows = rows
            test_values = labels_array
        else:
            test_rows = list(test_samples)
            test_values = np.asarray(
                test_labels
                if test_labels is not None
                else [getattr(row, "class_name", getattr(row, "label", None)) for row in test_rows],
                dtype=object,
            )
            if len(test_values) != len(test_rows):
                raise ValueError("test_samples and test_labels must have equal length")
        test_grouped: Dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(test_values):
            test_grouped[str(label)].append(index)
        test_seed = int(seed) + 700_001
        for label in sorted(grouped):
            candidates = np.asarray(test_grouped.get(label, []), dtype=np.int64)
            if len(candidates) < int(test_per_class):
                raise ValueError(
                    f"Test class {label!r} has {len(candidates)} rows; {test_per_class} required"
                )
            stable = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
            test_rng = np.random.default_rng(test_seed + stable)
            chosen_test = test_rng.permutation(candidates)[: int(test_per_class)]
            roles["test"].extend(test_rows[int(index)] for index in chosen_test)
        result[replicate] = roles
    return result


_make_food101_cohorts = _food101_cohort_splits
_food101_splits = _food101_cohort_splits


def _nested_stratified_indices(
    labels: Sequence[Any], budgets: Sequence[int], *, seed: int = 42
) -> Dict[int, np.ndarray]:
    target = np.asarray(labels, dtype=object)
    classes = sorted(np.unique(target).tolist(), key=str)
    orders = {
        label: np.random.default_rng(
            int(seed) + int(hashlib.sha256(str(label).encode()).hexdigest()[:8], 16)
        ).permutation(np.flatnonzero(target == label))
        for label in classes
    }
    output: Dict[int, np.ndarray] = {}
    for budget in budgets:
        if any(len(orders[label]) < int(budget) for label in classes):
            raise ValueError("selector budget exceeds available rows for a class")
        output[int(budget)] = np.sort(
            np.concatenate([orders[label][: int(budget)] for label in classes]).astype(np.int64)
        )
    return output


_nested_food101_indices = _nested_stratified_indices


def _paired_split_banks(
    embeddings: np.ndarray, labels: Sequence[Any], *, seed: int
) -> Dict[str, np.ndarray]:
    """Construct donor/mode/nuisance banks from one split only."""

    matrix = _l2_rows(np.asarray(embeddings, dtype=np.float32))
    if len(matrix) == 0:
        raise ValueError("split-local banks require at least one row")
    seed_sequence = np.random.SeedSequence(int(seed))
    donor_rng, mode_rng, nuisance_rng, scale_rng = (
        np.random.default_rng(child) for child in seed_sequence.spawn(4)
    )
    donor = matrix[donor_rng.permutation(len(matrix))]
    mode = np.empty(len(matrix), dtype=np.float32)
    labels_array = np.asarray(labels, dtype=object)
    for label in sorted(np.unique(labels_array).tolist(), key=str):
        rows = np.flatnonzero(labels_array == label)
        values = np.resize(np.asarray([-1.0, 1.0], dtype=np.float32), len(rows))
        mode_rng.shuffle(values)
        mode[rows] = values
    scales = np.geomspace(1.0, 8.0, matrix.shape[1], dtype=np.float32)
    scale_rng.shuffle(scales)
    nuisance = matrix[nuisance_rng.permutation(len(matrix))] * scales[None, :]
    return {"donor": donor, "mode": mode, "nuisance": nuisance, "scales": scales}


_paired_real_bank = _paired_split_banks
_make_split_local_banks = _paired_split_banks


def _score_overlap(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    *,
    seed: int = 0,
    folds: int = _CROSS_FIT_FOLDS,
    n_splits: Optional[int] = None,
    k: int = _FROZEN_K,
) -> Dict[str, Any]:
    """Score only through vertebrae's cross-fitted OverlapIndex adapter."""

    if OverlapIndexScorer is None or OverlapScoringConfig is None:
        raise ImportError("OverlapIndexScorer requires the vertebrae scoring dependencies")
    resolved_folds = int(n_splits if n_splits is not None else folds)
    config = OverlapScoringConfig(
        k=int(k),
        min_k=int(k),
        max_k=int(k),
        min_samples_per_cluster=5,
        kmeans_kwargs={"random_state": int(seed)},
        normalize_embeddings=True,
    )
    result = OverlapIndexScorer(config).score_cross_fitted(
        np.asarray(embeddings), np.asarray(labels), n_splits=resolved_folds, seed=int(seed)
    )
    return {
        "score": float(result.macro_score),
        "macro_score": float(result.macro_score),
        "warnings": list(getattr(result, "warnings", [])),
        "k_per_class": dict(getattr(result, "k_per_class", {})),
        "metadata": dict(getattr(result, "metadata", {})),
    }


_score_cross_fitted_overlap = _score_overlap


def _score_probe(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    *,
    seed: int = 0,
    folds: int = _PROBE_FOLDS,
    n_splits: Optional[int] = None,
) -> Dict[str, Any]:
    """Return five-fold OOF accuracy for the fixed L2 logistic probe."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    matrix = np.asarray(embeddings, dtype=np.float32)
    target = np.asarray(labels, dtype=object)
    splits = int(n_splits if n_splits is not None else folds)
    counts = np.unique(target, return_counts=True)[1]
    splits = min(splits, int(np.min(counts)))
    if splits < 2:
        raise ValueError("OOF probe requires at least two rows per class")
    predictions = np.empty(len(target), dtype=object)
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=int(seed))
    for train, holdout in splitter.split(matrix, target):
        estimator = make_pipeline(
            Normalizer(norm="l2"),
            LogisticRegression(C=1.0, max_iter=2_000, random_state=int(seed), n_jobs=1),
        )
        estimator.fit(matrix[train], target[train])
        predictions[holdout] = estimator.predict(matrix[holdout])
    score = float(accuracy_score(target, predictions))
    return {"score": score, "accuracy": score, "folds": splits}


_score_linear_probe = _score_probe


def _make_head_estimator(family: str, seed: int = 0) -> Any:
    """Build one exact, untuned L2-only downstream head."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.svm import SVC

    if family == "linear":
        classifier: Any = LogisticRegression(
            C=1.0, max_iter=2_000, random_state=int(seed), n_jobs=1
        )
    elif family == "quadratic":
        classifier = SVC(kernel="poly", degree=2, C=1.0, gamma="scale", coef0=1.0)
    elif family == "knn":
        classifier = KNeighborsClassifier(
            n_neighbors=15, weights="distance", metric="cosine", n_jobs=1
        )
    elif family == "rbf":
        classifier = SVC(kernel="rbf", C=1.0, gamma="scale")
    else:
        raise ValueError(f"Unknown head family {family!r}")
    return make_pipeline(Normalizer(norm="l2"), classifier)


_make_head = _make_head_estimator


def _head_recipe(family: str) -> Dict[str, Any]:
    if family not in _HEAD_FAMILIES:
        raise ValueError(f"Unknown head family {family!r}")
    recipes = {
        "linear": {"classifier": "LogisticRegression", "C": 1.0, "max_iter": 2_000},
        "quadratic": {
            "classifier": "SVC",
            "kernel": "poly",
            "degree": 2,
            "C": 1.0,
            "gamma": "scale",
            "coef0": 1.0,
        },
        "knn": {
            "classifier": "KNeighborsClassifier",
            "n_neighbors": 15,
            "weights": "distance",
            "metric": "cosine",
        },
        "rbf": {"classifier": "SVC", "kernel": "rbf", "C": 1.0, "gamma": "scale"},
    }
    return {"preprocessing": ["Normalizer(norm='l2')"], **recipes[family]}


_recipe_for_head = _head_recipe


def _rank_metrics(
    candidates: Mapping[str, float],
    reference: Optional[Mapping[str, float]] = None,
    reference_scores: Optional[Mapping[str, float]] = None,
) -> Dict[str, Optional[float]]:
    """Return tie-safe Spearman/Kendall ranking metrics and regret."""

    target = reference if reference is not None else reference_scores
    if target is None:
        raise TypeError("reference scores are required")
    names = [
        name
        for name in sorted(set(candidates) & set(target))
        if np.isfinite(float(candidates[name])) and np.isfinite(float(target[name]))
    ]
    empty: Dict[str, Optional[float]] = {
        "spearman": None,
        "kendall": None,
        "regret": None,
        "exact_best": None,
        "within_1pct": None,
        "within_one_percent": None,
    }
    if len(names) < 2:
        return empty
    observed = np.asarray([float(candidates[name]) for name in names])
    actual = np.asarray([float(target[name]) for name in names])
    try:
        from scipy.stats import kendalltau, spearmanr

        spearman = float(spearmanr(observed, actual).statistic)
        kendall = float(kendalltau(observed, actual).statistic)
    except Exception:  # pragma: no cover - scipy is a core dependency
        spearman = kendall = float("nan")
    if np.ptp(observed) == 0.0 or np.ptp(actual) == 0.0:
        spearman = kendall = float("nan")
    best_selector = float(np.max(observed))
    selected = np.flatnonzero(np.isclose(observed, best_selector, atol=1e-12, rtol=0.0))
    selected_actual = actual[selected]
    best_actual = float(np.max(actual))
    exact = float(np.mean(np.isclose(selected_actual, best_actual, atol=1e-12, rtol=0.0)))
    within = float(np.mean(best_actual - selected_actual <= 0.01))
    return {
        "spearman": spearman if np.isfinite(spearman) else None,
        "kendall": kendall if np.isfinite(kendall) else None,
        "regret": float(best_actual - float(np.mean(selected_actual))),
        "exact_best": exact,
        "within_1pct": within,
        "within_one_percent": within,
    }


_ranking_metrics = _rank_metrics


def _normalized_log_auc(budgets: Sequence[int], values: Sequence[float]) -> Optional[float]:
    x = np.log2(np.asarray(budgets, dtype=float))
    y = np.asarray(values, dtype=float)
    if len(x) < 2 or len(x) != len(y) or not np.all(np.isfinite(y)) or x[-1] <= x[0]:
        return None
    normalized_x = (x - x[0]) / float(x[-1] - x[0])
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0
        trapezoid = np.trapz
    return float(trapezoid(y, x=normalized_x))


_log_budget_auc = _normalized_log_auc


def _parse_budget_values(value: str, default: Sequence[int] = _FOOD101_BUDGETS) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return tuple(int(item) for item in default)
    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("budgets must be comma-separated positive integers") from exc
    if not parsed or tuple(sorted(set(parsed))) != parsed or any(item < 1 for item in parsed):
        raise ValueError("budgets must be increasing positive integers")
    return parsed


def _resolve_cli_models(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not names:
        return _DEFAULT_MODELS
    unknown = sorted(set(names) - set(_DEFAULT_MODELS))
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"Unknown or duplicate models: {unknown!r}")
    return names


def _scientific_identity(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop operational fields so worker-count changes reuse scientific caches."""

    operational = {
        "jobs",
        "requested_jobs",
        "worker_count",
        "jobs_requested",
        "jobs_resolved",
        "device",
        "device_requested",
        "device_resolved",
        "output_dir",
        "cache_dir",
        "runtime",
        "progress",
    }
    return {
        str(key): _json_safe(value)
        for key, value in configuration.items()
        if key not in operational
    }


def _configuration_hash(configuration: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _scientific_identity(configuration), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _configuration(
    *, stage: str = "food101", jobs: Any = "auto", seed: int = 42, **kwargs: Any
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": str(stage),
        "jobs": jobs,
        "seed": int(seed),
        "protocol_version": _PROTOCOL_VERSION,
        "models": list(_DEFAULT_MODELS),
        "budgets": list(_FOOD101_BUDGETS),
        "replicates": _REPLICATES,
        "selector_per_class": _SELECTOR_PER_CLASS,
        "development_per_class": _DEVELOPMENT_PER_CLASS,
        "test_per_class": _TEST_PER_CLASS,
        "classes": _FOOD101_CLASS_COUNT,
        "q": _Q,
        "overlap_k": _FROZEN_K,
        "cross_fit_folds": _CROSS_FIT_FOLDS,
        "probe_folds": _PROBE_FOLDS,
        "arms": [{"name": name, "lambda": lam, "nu": nu} for name, lam, nu in _FOOD101_ARMS],
        "methods": list(_METHODS),
        "heads": list(_HEAD_FAMILIES),
        "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
    }
    payload.update({key: value for key, value in kwargs.items() if value is not None})
    return payload


_artifact_configuration = _configuration


def _is_canonical_configuration(configuration: Mapping[str, Any], **kwargs: Any) -> bool:
    values = dict(configuration)
    values.update({key: value for key, value in kwargs.items() if value is not None})
    try:
        models = tuple(values["models"])
        budgets = tuple(int(value) for value in values["budgets"])
        configured_arms = tuple(
            (
                str(item["name"]),
                float(item["lambda"]),
                float(item["nu"]),
            )
            for item in values["arms"]
        )
        expected_arms = tuple(
            (name, float(lambda_value), float(nuisance_value))
            for name, lambda_value, nuisance_value in _FOOD101_ARMS
        )
        configured_methods = tuple(values["methods"])
        configured_heads = tuple(values["heads"])
        return bool(
            str(values["stage"]) in {"food101", "all"}
            and int(values["protocol_version"]) == _PROTOCOL_VERSION
            and models == _DEFAULT_MODELS
            and budgets == _FOOD101_BUDGETS
            and int(values["replicates"]) == _REPLICATES
            and int(values["seed"]) == 42
            and int(values["bootstrap_resamples"]) == _BOOTSTRAP_RESAMPLES
            and int(values["selector_per_class"]) == _SELECTOR_PER_CLASS
            and int(values["development_per_class"]) == _DEVELOPMENT_PER_CLASS
            and int(values["test_per_class"]) == _TEST_PER_CLASS
            and int(values["classes"]) == _FOOD101_CLASS_COUNT
            and float(values["q"]) == 1.0
            and int(values["overlap_k"]) == _FROZEN_K
            and int(values["cross_fit_folds"]) == _CROSS_FIT_FOLDS
            and int(values["probe_folds"]) == _PROBE_FOLDS
            and configured_arms == expected_arms
            and configured_methods == _METHODS
            and configured_heads == _HEAD_FAMILIES
        )
    except (KeyError, TypeError, ValueError):
        return False


_canonical_configuration = _is_canonical_configuration
_is_canonical_run = _is_canonical_configuration


def _artifact_stem(
    configuration: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    *,
    stage: Optional[str] = None,
) -> str:
    values = dict(configuration if configuration is not None else config or {})
    stage_name = str(values.get("stage", stage or "food101"))
    digest = _configuration_hash(values)[:12]
    return f"{_ARTIFACT_STEM_PREFIX}_{stage_name}_k{_FROZEN_K}_{digest}"


_make_artifact_stem = _artifact_stem


def _master_protocol(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe the frozen protocol before any extraction or scoring begins."""

    return {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "protocol_version": _PROTOCOL_VERSION,
        "study": _STUDY,
        "artifact_status": "planned",
        "claim_supported": False,
        "configuration": _json_safe(configuration),
        "configuration_hash": _configuration_hash(configuration),
        "frozen": {
            "classes": "first_40_alphabetically_sorted_food101_classes",
            "class_count": _FOOD101_CLASS_COUNT,
            "roles_per_class": {
                "official_train_selector": _SELECTOR_PER_CLASS,
                "official_train_development": _DEVELOPMENT_PER_CLASS,
                "official_test_reference": _TEST_PER_CLASS,
            },
            "replicates": _REPLICATES,
            "budgets_per_class": list(_FOOD101_BUDGETS),
            "q": 1.0,
            "arms": [{"name": name, "lambda": lam, "nu": nu} for name, lam, nu in _FOOD101_ARMS],
            "overlap": {
                "adapter": "OverlapIndexScorer",
                "k": _FROZEN_K,
                "folds": _CROSS_FIT_FOLDS,
                "min_samples_per_cluster": 5,
                "normalize_embeddings": True,
                "kmeans_seed": "scoring_seed_per_cell",
            },
            "probe": {
                "estimator": "L2 LogisticRegression",
                "C": 1.0,
                "folds": _PROBE_FOLDS,
                "max_iter": 2_000,
                "random_state": "scoring_seed_per_cell",
            },
            "reference_heads": {family: _head_recipe(family) for family in _HEAD_FAMILIES},
            "primary_reference_head": "quadratic",
            "bootstrap": {
                "resamples": _BOOTSTRAP_RESAMPLES,
                "hierarchy": ["replicate", "backbone", "official_test_rows_within_class"],
                "support_lower_bounds": [
                    "direct_nonlinear_oi_minus_probe",
                    "nonlinear_baseline_interaction",
                ],
            },
        },
    }


master_protocol = _master_protocol


def _validate_factorial_grid(
    rows: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
    expected_keys: Optional[Sequence[Sequence[Any]]] = None,
) -> Dict[str, Any]:
    seen: set[tuple[Any, ...]] = set()
    duplicates: list[tuple[Any, ...]] = []
    nonfinite: list[tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        for value in row.values():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                nonfinite.append(key)
                break
    expected = {tuple(key) for key in expected_keys} if expected_keys is not None else set()
    missing = sorted(expected - seen, key=str)
    if duplicates or missing or nonfinite:
        raise ValueError(
            "Invalid factorial grid: "
            f"duplicates={duplicates[:3]!r}, missing={missing[:3]!r}, "
            f"nonfinite={nonfinite[:3]!r}"
        )
    return {
        "rows": len(rows),
        "unique": len(seen),
        "duplicates": duplicates,
        "missing": missing,
        "nonfinite": nonfinite,
    }


def _checkpoint_path(output_dir: Path, stem: str, key: str) -> Path:
    digest = hashlib.sha256(str(key).encode()).hexdigest()[:20]
    return Path(output_dir) / f"{stem}.checkpoints" / f"{digest}.json"


def _read_checkpoint(
    path: Path, configuration_hash: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "complete":
        return None
    if configuration_hash is not None and payload.get("configuration_hash") != configuration_hash:
        return None
    return payload


def _read_artifact(path: Path, source: Optional[Path] = None) -> Dict[str, Any]:
    """Read a completed Food-101 artifact and verify its scientific identity."""

    target = Path(source or path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read artifact {target}: {exc}") from exc
    if payload.get("schema_version") != _RESULT_SCHEMA_VERSION:
        raise ValueError("Artifact has unsupported schema version")
    if payload.get("protocol_version") != _PROTOCOL_VERSION:
        raise ValueError("Artifact has unsupported protocol version")
    if payload.get("artifact_status") != "completed":
        raise ValueError(f"Artifact status {payload.get('artifact_status')!r} is not completed")
    if payload.get("study") != _STUDY:
        raise ValueError("Artifact has the wrong study identity")
    if payload.get("claim_supported") is not False:
        raise ValueError("Universal claim_supported must remain false")
    embedded_protocol = payload.get("protocol")
    if not isinstance(embedded_protocol, Mapping):
        raise ValueError("Artifact is missing embedded completed protocol metadata")
    if embedded_protocol.get("schema_version") != _RESULT_SCHEMA_VERSION:
        raise ValueError("Embedded protocol has unsupported schema version")
    if embedded_protocol.get("protocol_version") != _PROTOCOL_VERSION:
        raise ValueError("Embedded protocol has unsupported protocol version")
    if embedded_protocol.get("study") != _STUDY:
        raise ValueError("Embedded protocol has the wrong study identity")
    if embedded_protocol.get("artifact_status") != "completed":
        raise ValueError("Embedded protocol is not completed")
    configuration = payload.get("configuration")
    expected = payload.get("configuration_hash")
    if not isinstance(configuration, Mapping) or not isinstance(expected, str):
        raise ValueError("Artifact is missing configuration/configuration_hash")
    if _configuration_hash(configuration) != expected:
        raise ValueError("Artifact configuration hash does not match configuration")
    embedded_configuration = embedded_protocol.get("configuration")
    embedded_hash = embedded_protocol.get("configuration_hash")
    if not isinstance(embedded_configuration, Mapping) or not isinstance(embedded_hash, str):
        raise ValueError("Embedded protocol is missing configuration/configuration_hash")
    if embedded_hash != expected:
        raise ValueError("Embedded protocol configuration hash does not match artifact")
    if _configuration_hash(embedded_configuration) != expected:
        raise ValueError("Embedded protocol configuration hash is invalid")
    if _scientific_identity(embedded_configuration) != _scientific_identity(configuration):
        raise ValueError("Embedded protocol configuration does not match artifact")
    narrow = payload.get("food101_nonlinearity_supported")
    bootstrap = payload.get("bootstrap")
    if not isinstance(narrow, bool) or not isinstance(bootstrap, Mapping):
        raise ValueError("Artifact is missing narrow Food-101 support metadata")
    bootstrap_narrow = bootstrap.get("food101_nonlinearity_supported")
    if not isinstance(bootstrap_narrow, bool) or bool(narrow) != bool(bootstrap_narrow):
        raise ValueError("Artifact narrow support flag does not match bootstrap")
    protocol_narrow = embedded_protocol.get("food101_nonlinearity_supported")
    if protocol_narrow is not None and bool(protocol_narrow) != bool(narrow):
        raise ValueError("Embedded protocol narrow support flag does not match artifact")
    return payload


_validate_artifact = _read_artifact


def _resolve_jobs(
    jobs: Any = "auto",
    requested_jobs: Any = None,
    *,
    physical_cores: Optional[int] = None,
    **_: Any,
) -> int:
    requested = jobs if requested_jobs is None else requested_jobs
    cores = int(physical_cores or 0)
    if cores < 1:
        try:
            cores = int(psutil.cpu_count(logical=False)) if psutil is not None else 0
        except Exception:
            cores = 0
    if cores < 1:
        cores = int(os.cpu_count() or 1)
    if str(requested).lower() == "auto":
        return max(1, min(6, cores - 1 if cores > 1 else 1))
    try:
        resolved = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError("jobs must be 'auto' or a positive integer") from exc
    if resolved < 1:
        raise ValueError("jobs must be >=1")
    return max(1, min(resolved, cores))


def _thread_limited_worker(worker: Callable[[Any], Any], item: Any) -> Any:
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            return worker(item)
    except ImportError:  # pragma: no cover
        return worker(item)


def _spawn_worker_call(payload: tuple[Callable[[Any], Any], Any]) -> Any:
    worker, item = payload
    return _thread_limited_worker(worker, item)


def _run_parallel_blocks(
    blocks: Optional[Sequence[Any]] = None,
    tasks: Optional[Sequence[Any]] = None,
    worker: Optional[Callable[[Any], Any]] = None,
    *,
    jobs: Any = 1,
    output_dir: Optional[Path] = None,
    stem: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    configuration_hash: Optional[str] = None,
    resume: bool = False,
) -> list[Any]:
    """Run deterministic blocks with spawn workers and incremental checkpoints."""

    values = list(blocks if blocks is not None else tasks or [])
    if worker is None:
        raise TypeError("worker is required")
    values.sort(
        key=lambda item: str(item.get("key", "")) if isinstance(item, Mapping) else repr(item)
    )
    checkpoint_root = Path(
        checkpoint_dir
        or (Path(output_dir) / f"{stem}.checkpoints" if output_dir and stem else "checkpoints")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    pending: list[Any] = []
    results: list[Any] = []
    for item in values:
        key = str(item.get("key", repr(item))) if isinstance(item, Mapping) else repr(item)
        checkpoint = checkpoint_root / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.json"
        if resume:
            cached = _read_checkpoint(checkpoint, configuration_hash)
            if cached is not None:
                results.append(cached)
                continue
        pending.append(item)

    def persist(result: Any, index: int) -> None:
        key = str(result.get("key", repr(result))) if isinstance(result, Mapping) else repr(result)
        payload = dict(result) if isinstance(result, Mapping) else {"key": key, "result": result}
        payload.setdefault("status", "complete")
        if configuration_hash is not None:
            payload.setdefault("configuration_hash", configuration_hash)
        checkpoint = checkpoint_root / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.json"
        _atomic_json(checkpoint, payload)
        results.append(payload)
        print(f"Completed block {key} ({index}/{len(pending)})", flush=True)

    if pending:
        resolved = _resolve_jobs(jobs)
        # A caller-provided closure or pytest fixture is not spawn-picklable;
        # keep that API path deterministic and local.  The production worker
        # is a module-level function and still uses the spawn pool below.
        spawn_safe = getattr(worker, "__module__", __name__) == __name__
        if resolved == 1 or len(pending) < 2 or not spawn_safe:
            for index, item in enumerate(pending, 1):
                persist(_thread_limited_worker(worker, item), index)
        else:
            # ``maxtasksperchild=1`` is supported by multiprocessing.Pool on
            # Python 3.9+, and avoids the Loky/threadpool warning seen in long
            # sklearn scoring runs.  A spawn context keeps worker imports safe.
            try:
                context = mp.get_context("spawn")
                pool = context.Pool(processes=resolved, maxtasksperchild=1)
            except (AttributeError, OSError, PermissionError):
                # Backend construction can fail in restricted notebook
                # environments; scientific worker exceptions must propagate.
                for index, item in enumerate(pending, 1):
                    persist(_thread_limited_worker(worker, item), index)
            else:
                with pool:
                    iterator = pool.imap_unordered(
                        _spawn_worker_call, [(worker, item) for item in pending]
                    )
                    for index, result in enumerate(iterator, 1):
                        persist(result, index)
    results.sort(
        key=lambda item: str(item.get("key", repr(item)))
        if isinstance(item, Mapping)
        else repr(item)
    )
    return results


_parallel_map_blocks = _run_parallel_blocks
_run_blocks = _run_parallel_blocks


def _reference_head_predictions(
    development: np.ndarray,
    development_labels: Sequence[Any],
    test: np.ndarray,
    test_labels: Sequence[Any],
    *,
    seed: int,
) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    from sklearn.metrics import accuracy_score

    scores: Dict[str, float] = {}
    predictions: Dict[str, np.ndarray] = {}
    for index, family in enumerate(_HEAD_FAMILIES):
        estimator = _make_head_estimator(family, int(seed) + index)
        estimator.fit(development, development_labels)
        predicted = np.asarray(estimator.predict(test), dtype=object)
        predictions[family] = predicted
        scores[family] = float(accuracy_score(test_labels, predicted))
    return scores, predictions


def _score_food101_block(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Score one model/replicate block; safe as a spawn worker target."""

    matrix = (
        _read_float32_memmap(task["embedding_manifest"])
        if "embedding_manifest" in task
        else np.asarray(task["embeddings"], dtype=np.float32)
    )
    labels = np.asarray(task["labels"], dtype=object)
    roles = task.get("roles") or {
        "selector": task["selector_indices"],
        "development": task["development_indices"],
        "test": task["test_indices"],
    }
    selector_indices = np.asarray(roles["selector"], dtype=np.int64)
    development_indices = np.asarray(roles["development"], dtype=np.int64)
    test_indices = np.asarray(roles["test"], dtype=np.int64)
    selector_raw = np.asarray(matrix[selector_indices], dtype=np.float32)
    development_raw = np.asarray(matrix[development_indices], dtype=np.float32)
    test_raw = np.asarray(matrix[test_indices], dtype=np.float32)
    selector_labels = labels[selector_indices]
    development_labels = labels[development_indices]
    test_labels = labels[test_indices]
    seed = int(task.get("seed", 0))
    banks = task.get("bank_manifests")
    if banks:
        selector_bank = {
            key: _read_float32_memmap(value)
            for key, value in banks["selector"].items()
            if key in {"donor", "mode", "nuisance"}
        }
        development_bank = {
            key: _read_float32_memmap(value)
            for key, value in banks["development"].items()
            if key in {"donor", "mode", "nuisance"}
        }
        test_bank = {
            key: _read_float32_memmap(value)
            for key, value in banks["test"].items()
            if key in {"donor", "mode", "nuisance"}
        }
    else:
        # Each call sees only one role's rows: no donor, mode, or nuisance bank
        # can leak from another split.
        selector_bank = _paired_split_banks(selector_raw, selector_labels, seed=seed + 11)
        development_bank = _paired_split_banks(development_raw, development_labels, seed=seed + 13)
        test_bank = _paired_split_banks(test_raw, test_labels, seed=seed + 17)
    budgets = tuple(int(value) for value in task.get("budgets", _FOOD101_BUDGETS))
    nested = _nested_stratified_indices(selector_labels, budgets, seed=seed + 23)
    reference_rows: list[Dict[str, Any]] = []
    selector_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    backbone = str(task.get("backbone", "backbone"))
    replicate = int(task.get("replicate", 0))
    for arm_name, lam, nu in _FOOD101_ARMS:
        selector_transformed = _bridge_transform(
            selector_raw,
            donor=selector_bank["donor"],
            mode=selector_bank["mode"],
            nuisance=selector_bank["nuisance"],
            q=1.0,
            lam=lam,
            nu=nu,
        )
        development_transformed = _bridge_transform(
            development_raw,
            donor=development_bank["donor"],
            mode=development_bank["mode"],
            nuisance=development_bank["nuisance"],
            q=1.0,
            lam=lam,
            nu=nu,
        )
        test_transformed = _bridge_transform(
            test_raw,
            donor=test_bank["donor"],
            mode=test_bank["mode"],
            nuisance=test_bank["nuisance"],
            q=1.0,
            lam=lam,
            nu=nu,
        )
        reference_scores, predictions = _reference_head_predictions(
            development_transformed, development_labels, test_transformed, test_labels, seed=seed
        )
        for family in _HEAD_FAMILIES:
            reference_rows.append(
                {
                    "backbone": backbone,
                    "replicate": replicate,
                    "arm": arm_name,
                    "lambda": lam,
                    "nu": nu,
                    "q": 1.0,
                    "head": family,
                    "test_accuracy": reference_scores[family],
                }
            )
            if family == "quadratic":
                # The hierarchical bootstrap needs only the primary head;
                # secondary-head reference accuracies stay in compact rows.
                prediction_rows.append(
                    {
                        "backbone": backbone,
                        "replicate": replicate,
                        "arm": arm_name,
                        "head": family,
                        "labels": test_labels.tolist(),
                        "predictions": predictions[family].tolist(),
                        "class_labels": test_labels.tolist(),
                    }
                )
        for budget in budgets:
            selected = nested[int(budget)]
            selected_labels = selector_labels[selected]
            for method in _METHODS:
                started = perf_counter()
                outcome = (
                    _score_overlap(
                        selector_transformed[selected],
                        selected_labels,
                        seed=seed,
                        folds=_CROSS_FIT_FOLDS,
                        k=_FROZEN_K,
                    )
                    if method == "overlap_cross_fitted"
                    else _score_probe(
                        selector_transformed[selected],
                        selected_labels,
                        seed=seed,
                        folds=_PROBE_FOLDS,
                    )
                )
                selector_rows.append(
                    {
                        "backbone": backbone,
                        "replicate": replicate,
                        "arm": arm_name,
                        "lambda": lam,
                        "nu": nu,
                        "q": 1.0,
                        "budget": int(budget),
                        "method": method,
                        "score": float(outcome["score"]),
                        "seconds": float(perf_counter() - started),
                    }
                )
    return {
        "key": task.get("key", f"food101/{backbone}/{replicate}"),
        "status": "complete",
        "configuration_hash": task.get("configuration_hash"),
        "reference_rows": reference_rows,
        "selector_rows": selector_rows,
        "test_prediction_rows": prediction_rows,
    }


_score_block = _score_food101_block


def _cross_backbone_rows(
    selector_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int] = _FOOD101_BUDGETS,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Compute fixed-budget cross-backbone ranks and normalized log-budget AUC."""

    ranking_rows: list[Dict[str, Any]] = []
    for replicate in sorted({int(row["replicate"]) for row in selector_rows}):
        for arm_name, lam, nu in _FOOD101_ARMS:
            for method in _METHODS:
                for head in _HEAD_FAMILIES:
                    for budget in budgets:
                        observed = {
                            str(row["backbone"]): float(row["score"])
                            for row in selector_rows
                            if int(row["replicate"]) == replicate
                            and str(row["arm"]) == arm_name
                            and str(row["method"]) == method
                            and int(row["budget"]) == int(budget)
                        }
                        reference = {
                            str(row["backbone"]): float(row["test_accuracy"])
                            for row in reference_rows
                            if int(row["replicate"]) == replicate
                            and str(row["arm"]) == arm_name
                            and str(row["head"]) == head
                        }
                        metrics = _rank_metrics(observed, reference)
                        ranking_rows.append(
                            {
                                "replicate": replicate,
                                "arm": arm_name,
                                "lambda": lam,
                                "nu": nu,
                                "method": method,
                                "head": head,
                                "budget": int(budget),
                                **metrics,
                            }
                        )
    groups: Dict[tuple[int, str, str, str], Dict[int, float]] = defaultdict(dict)
    for row in ranking_rows:
        value = row.get("spearman")
        if value is not None and np.isfinite(float(value)):
            groups[(int(row["replicate"]), str(row["arm"]), str(row["method"]), str(row["head"]))][
                int(row["budget"])
            ] = float(value)
    auc_rows: list[Dict[str, Any]] = []
    for (replicate, arm, method, head), values in sorted(groups.items(), key=str):
        ordered = sorted(values)
        auc_rows.append(
            {
                "replicate": replicate,
                "arm": arm,
                "method": method,
                "head": head,
                "auc": _normalized_log_auc(ordered, [values[key] for key in ordered]),
                "valid_budgets": len(ordered),
            }
        )
    return ranking_rows, auc_rows


def _cross_backbone_auc(*args: Any, **kwargs: Any) -> list[Dict[str, Any]]:
    return _cross_backbone_rows(*args, **kwargs)[1]


def _reference_lookup_from_predictions(
    prediction_rows: Sequence[Mapping[str, Any]],
    sampled_rows: Optional[np.ndarray] = None,
) -> Dict[tuple[int, str, str, str], float]:
    lookup: Dict[tuple[int, str, str, str], float] = {}
    for row in prediction_rows:
        try:
            labels = np.asarray(row["labels"], dtype=object)
            predicted = np.asarray(row["predictions"], dtype=object)
            if sampled_rows is not None:
                labels = labels[sampled_rows]
                predicted = predicted[sampled_rows]
            score = float(np.mean(labels == predicted))
            key = (int(row["replicate"]), str(row["backbone"]), str(row["arm"]), str(row["head"]))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if np.isfinite(score):
            lookup[key] = score
    return lookup


def _bootstrap_food101(
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
    auc_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    selector_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    ranking_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    reference_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    test_prediction_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    test_predictions: Optional[Sequence[Mapping[str, Any]]] = None,
    test_within_class_predictions: Optional[Sequence[Mapping[str, Any]]] = None,
    test_labels: Optional[Sequence[Any]] = None,
    test_fine_labels: Optional[Sequence[Any]] = None,
    replicates: int = _REPLICATES,
    backbones: int = len(_DEFAULT_MODELS),
    models: Optional[Sequence[str]] = None,
    budgets: Optional[Sequence[int]] = None,
    n_resamples: int = _BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    canonical: bool = True,
    protocol_conformant: bool = True,
) -> Dict[str, Any]:
    """Hierarchically bootstrap Food-101 effects and apply frozen support guards."""

    auc_data = list(auc_rows or rows or [])
    prediction_data = list(
        test_prediction_rows or test_predictions or test_within_class_predictions or []
    )
    primary_head = "quadratic"
    required_arms = {name for name, _, _ in _FOOD101_ARMS}
    methods = tuple(_METHODS)
    if not auc_data:
        return {
            "claim_supported": False,
            "food101_supported": False,
            "food101_nonlinearity_supported": False,
            "valid_fraction": 0.0,
        }
    # Normalize compact synthetic rows: absent head is the primary quadratic
    # endpoint, and absent method/arm fields are rejected as malformed.
    cells: Dict[tuple[int, str, str, str], float] = {}
    duplicate = False
    invalid_auc = False
    for row in auc_data:
        try:
            key = (
                int(row["replicate"]),
                str(row["arm"]),
                str(row["method"]),
                str(row.get("head", primary_head)),
            )
            value = float(row["auc"])
        except (KeyError, TypeError, ValueError):
            invalid_auc = True
            continue
        if key[3] != primary_head:
            continue
        if key in cells:
            duplicate = True
        if not np.isfinite(value):
            invalid_auc = True
        cells[key] = value
    rep_values = sorted({key[0] for key in cells})
    arm_values = {key[1] for key in cells}
    method_values = {key[2] for key in cells}
    expected_reps = list(range(int(replicates)))
    expected = {
        (replicate, arm, method, primary_head)
        for replicate in expected_reps
        for arm in required_arms
        for method in methods
    }
    observed = set(cells)
    complete_auc = bool(not duplicate and not invalid_auc and expected <= observed)
    panel = sorted(
        {
            str(row.get("backbone"))
            for row in (selector_rows or [])
            if row.get("backbone") is not None
        }
    )
    if not panel:
        panel = list(models or _DEFAULT_MODELS)
    exact_panel = len(panel) == int(backbones) and set(panel) == set(models or _DEFAULT_MODELS)
    requested_budgets = tuple(int(value) for value in (budgets or _FOOD101_BUDGETS))
    exact_budgets = requested_budgets == _FOOD101_BUDGETS
    complete_factorial = bool(
        complete_auc
        and set(arm_values) == required_arms
        and method_values == set(methods)
        and tuple(rep_values) == tuple(expected_reps)
        and exact_panel
        and exact_budgets
    )

    def static_effect(rep_sample: np.ndarray) -> tuple[float, float, float, float]:
        direct_values: list[float] = []
        interaction_values: list[float] = []
        nuisance_direct: list[float] = []
        nuisance_interaction: list[float] = []
        for index in rep_sample:
            replicate = rep_values[int(index)]
            try:
                oi_full = cells[(replicate, "nonlinearity_full", methods[0], primary_head)]
                probe_full = cells[(replicate, "nonlinearity_full", methods[1], primary_head)]
                oi_base = cells[(replicate, "baseline", methods[0], primary_head)]
                probe_base = cells[(replicate, "baseline", methods[1], primary_head)]
                oi_nuisance = cells[(replicate, "nuisance_full", methods[0], primary_head)]
                probe_nuisance = cells[(replicate, "nuisance_full", methods[1], primary_head)]
            except KeyError:
                continue
            direct_values.append(oi_full - probe_full)
            interaction_values.append((oi_full - oi_base) - (probe_full - probe_base))
            nuisance_direct.append(oi_nuisance - probe_nuisance)
            nuisance_interaction.append((oi_nuisance - oi_base) - (probe_nuisance - probe_base))
        values = [direct_values, interaction_values, nuisance_direct, nuisance_interaction]
        return tuple(float(np.mean(value)) if value else float("nan") for value in values)  # type: ignore[return-value]

    # Keep the raw selector/reference panel so each bootstrap draw can resample
    # replicate, backbone, and official-test rows coherently.  The compact AUC
    # table remains a fallback for callers that only persisted aggregate rows.
    selector_lookup: Dict[tuple[int, str, str, int, str], float] = {}
    selector_duplicate = False
    selector_values = list(selector_rows or [])
    for row in selector_values:
        try:
            key = (
                int(row["replicate"]),
                str(row["backbone"]),
                str(row["arm"]),
                int(row["budget"]),
                str(row["method"]),
            )
            value = float(row["score"])
        except (KeyError, TypeError, ValueError):
            selector_duplicate = True
            continue
        if key in selector_lookup:
            selector_duplicate = True
        if not np.isfinite(value):
            selector_duplicate = True
        selector_lookup[key] = value
    if selector_values:
        panel = sorted({key[1] for key in selector_lookup})
    expected_panel = tuple(models or _DEFAULT_MODELS)
    expected_selector = {
        (replicate, backbone, arm, budget, method)
        for replicate in expected_reps
        for backbone in expected_panel
        for arm in required_arms
        for budget in requested_budgets
        for method in methods
    }
    complete_selector = bool(
        not selector_duplicate and selector_lookup and set(selector_lookup) == expected_selector
    )

    reference_static: Dict[tuple[int, str, str, str], float] = {}
    reference_duplicate = False
    for row in reference_rows or []:
        try:
            key = (
                int(row["replicate"]),
                str(row["backbone"]),
                str(row["arm"]),
                str(row["head"]),
            )
            value = float(row["test_accuracy"])
        except (KeyError, TypeError, ValueError):
            reference_duplicate = True
            continue
        if key in reference_static:
            reference_duplicate = True
        if not np.isfinite(value):
            reference_duplicate = True
        reference_static[key] = value

    # The confirmatory hierarchy needs only quadratic predictions.  Older or
    # hand-written fixtures may contain all heads; retaining those is harmless.
    prediction_data = [
        row
        for row in prediction_data
        if isinstance(row, Mapping) and str(row.get("head", primary_head)) == primary_head
    ]
    prediction_lookup: Dict[tuple[int, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    prediction_duplicate = False
    labels_array: Optional[np.ndarray] = None
    for row in prediction_data:
        try:
            row_labels = np.asarray(
                row.get("labels", test_labels),
                dtype=object,
            )
            if row_labels.ndim == 0:
                raise ValueError("missing test labels")
            predicted = np.asarray(row["predictions"], dtype=object)
            if len(row_labels) != len(predicted):
                raise ValueError("prediction length mismatch")
            key = (
                int(row["replicate"]),
                str(row["backbone"]),
                str(row["arm"]),
                primary_head,
            )
        except (KeyError, TypeError, ValueError):
            prediction_duplicate = True
            continue
        if labels_array is None:
            labels_array = row_labels
        elif not np.array_equal(labels_array, row_labels):
            prediction_duplicate = True
        if key in prediction_lookup:
            prediction_duplicate = True
        prediction_lookup[key] = (row_labels, predicted)
    class_groups: list[np.ndarray] = []
    if labels_array is not None:
        class_groups = [
            np.flatnonzero(labels_array == label)
            for label in sorted(np.unique(labels_array).tolist(), key=str)
        ]
    expected_prediction = {
        (replicate, backbone, arm, primary_head)
        for replicate in expected_reps
        for backbone in expected_panel
        for arm in required_arms
    }
    complete_test_strata = bool(
        labels_array is not None
        and class_groups
        and len(class_groups) == int(_FOOD101_CLASS_COUNT)
        and all(len(group) == int(_TEST_PER_CLASS) for group in class_groups)
    )
    complete_predictions = bool(
        bool(prediction_data)
        and not prediction_duplicate
        and set(prediction_lookup) == expected_prediction
        and complete_test_strata
    )
    expected_reference = {
        (replicate, backbone, arm, head)
        for replicate in expected_reps
        for backbone in expected_panel
        for arm in required_arms
        for head in _HEAD_FAMILIES
    }
    complete_reference = bool(
        not reference_rows
        or (not reference_duplicate and set(reference_static) == expected_reference)
    )
    complete_factorial = bool(
        complete_factorial
        and (not selector_values or complete_selector)
        and complete_reference
        and complete_predictions
    )

    def _draw_spearman(observed: Sequence[float], reference: Sequence[float]) -> float:
        if len(observed) < 2 or len(observed) != len(reference):
            return float("nan")
        # A bootstrap draw can contain one repeated backbone or a tied
        # reference endpoint.  Treat that draw as a valid zero-information
        # rank rather than dropping the whole hierarchical resample.
        if (
            np.ptp(np.asarray(observed, dtype=float)) == 0.0
            or np.ptp(np.asarray(reference, dtype=float)) == 0.0
        ):
            return 0.0
        metrics = _rank_metrics(
            {str(index): float(value) for index, value in enumerate(observed)},
            {str(index): float(value) for index, value in enumerate(reference)},
        )
        value = metrics.get("spearman")
        return float(value) if value is not None else float("nan")

    def hierarchical_effect(
        replicate_sample: np.ndarray,
        backbone_sample: np.ndarray,
        sampled_test: Optional[np.ndarray],
    ) -> tuple[float, float, float, float]:
        if not selector_lookup:
            return static_effect(replicate_sample)
        by_rep: list[tuple[float, float, float, float]] = []
        for rep_index in replicate_sample:
            replicate = rep_values[int(rep_index)]
            aucs: Dict[tuple[str, str], float] = {}
            for arm in required_arms:
                for method in methods:
                    scores: Dict[tuple[str, str], float] = {}
                    for budget in requested_budgets:
                        observed: list[float] = []
                        reference: list[float] = []
                        for backbone_index in backbone_sample:
                            backbone = expected_panel[int(backbone_index)]
                            selector_key = (replicate, backbone, arm, int(budget), method)
                            if selector_key not in selector_lookup:
                                continue
                            observed.append(selector_lookup[selector_key])
                            prediction_key = (replicate, backbone, arm, primary_head)
                            if sampled_test is not None and prediction_key in prediction_lookup:
                                labels_for_row, predictions_for_row = prediction_lookup[
                                    prediction_key
                                ]
                                reference.append(
                                    float(
                                        np.mean(
                                            labels_for_row[sampled_test]
                                            == predictions_for_row[sampled_test]
                                        )
                                    )
                                )
                            else:
                                reference.append(reference_static.get(prediction_key, float("nan")))
                        rank = _draw_spearman(observed, reference)
                        scores[(arm, method, int(budget))] = rank
                    ordered = [scores[(arm, method, int(budget))] for budget in requested_budgets]
                    auc = _normalized_log_auc(requested_budgets, ordered)
                    aucs[(arm, method)] = float(auc) if auc is not None else float("nan")
            try:
                oi_full = aucs[("nonlinearity_full", methods[0])]
                probe_full = aucs[("nonlinearity_full", methods[1])]
                oi_base = aucs[("baseline", methods[0])]
                probe_base = aucs[("baseline", methods[1])]
                oi_nuisance = aucs[("nuisance_full", methods[0])]
                probe_nuisance = aucs[("nuisance_full", methods[1])]
            except KeyError:
                continue
            by_rep.append(
                (
                    oi_full - probe_full,
                    (oi_full - oi_base) - (probe_full - probe_base),
                    oi_nuisance - probe_nuisance,
                    (oi_nuisance - oi_base) - (probe_nuisance - probe_base),
                )
            )
        if not by_rep:
            return (float("nan"),) * 4
        return tuple(float(np.mean([values[index] for values in by_rep])) for index in range(4))  # type: ignore[return-value]

    full_replicates = np.arange(len(rep_values), dtype=np.int64)
    full_backbones = np.arange(len(expected_panel), dtype=np.int64)
    full_test = np.arange(len(labels_array), dtype=np.int64) if labels_array is not None else None
    hierarchical_point = hierarchical_effect(full_replicates, full_backbones, full_test)
    static_point = static_effect(full_replicates)
    # Nuisance is a boundary diagnostic, not part of the confirmatory rank
    # hierarchy.  Keep its reported endpoint tied to the persisted arm AUCs so
    # selector/test resampling cannot turn a diagnostic into a claim gate.
    point = (
        hierarchical_point[0],
        hierarchical_point[1],
        static_point[2],
        static_point[3],
    )
    generator = np.random.default_rng(int(seed))
    draws_direct: list[float] = []
    draws_interaction: list[float] = []
    draws_nuisance_direct: list[float] = []
    draws_nuisance_interaction: list[float] = []
    valid_draws = 0
    for _ in range(max(1, int(n_resamples))):
        if not rep_values:
            break
        rep_sample = generator.integers(0, len(rep_values), size=len(rep_values))
        backbone_sample = generator.integers(0, len(expected_panel), size=len(expected_panel))
        sampled_test = None
        if class_groups:
            sampled_test = np.concatenate(
                [generator.choice(group, size=len(group), replace=True) for group in class_groups]
            )
        direct, interaction, _, _ = hierarchical_effect(rep_sample, backbone_sample, sampled_test)
        _, _, nuisance_direct, nuisance_interaction = static_effect(rep_sample)
        # Primary validity is defined by the direct and interaction effects;
        # nuisance is deliberately diagnostic-only and may be undefined when
        # every nuisance reference rank is tied.
        if np.isfinite(direct) and np.isfinite(interaction):
            valid_draws += 1
            draws_direct.append(direct)
            draws_interaction.append(interaction)
            if np.isfinite(nuisance_direct):
                draws_nuisance_direct.append(nuisance_direct)
            if np.isfinite(nuisance_interaction):
                draws_nuisance_interaction.append(nuisance_interaction)

    def interval(values: Sequence[float]) -> list[Optional[float]]:
        if not values:
            return [None, None]
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    direct_interval = interval(draws_direct)
    interaction_interval = interval(draws_interaction)
    nuisance_direct_interval = interval(draws_nuisance_direct)
    nuisance_interaction_interval = interval(draws_nuisance_interaction)
    # Replicate-level advantages are based on the prespecified quadratic AUC.
    replicate_effects: Dict[int, tuple[float, float]] = {}
    for replicate in rep_values:
        # The persisted per-replicate AUC endpoint is the prespecified unit for
        # the >=4/5 guard.  Selector rows are retained for the hierarchical
        # bootstrap, but must not silently redefine this guard when a compact
        # fixture supplies only one backbone's endpoint rows.
        direct, interaction, _, _ = static_effect(
            np.asarray([rep_values.index(replicate)], dtype=np.int64)
        )
        replicate_effects[replicate] = (direct, interaction)
    advantageous = sum(
        int(
            np.isfinite(values[0])
            and np.isfinite(values[1])
            and values[0] > 0.0
            and values[1] > 0.0
        )
        for values in replicate_effects.values()
    )

    # Quadratic regret is supplied by ranking rows when available.  Lower mean
    # regret for the nonlinear OI arm than the nonlinear probe is a frozen guard.
    regrets: Dict[str, list[float]] = defaultdict(list)
    for row in ranking_rows or []:
        if str(row.get("head")) == primary_head and str(row.get("arm")) == "nonlinearity_full":
            value = row.get("regret")
            if value is not None and np.isfinite(float(value)):
                regrets[str(row.get("method"))].append(float(value))
    regret_means = {method: float(np.mean(values)) for method, values in regrets.items() if values}
    lower_mean_quadratic_regret = bool(
        regret_means.get("overlap_cross_fitted", np.inf)
        < regret_means.get("linear_probe_oof", np.inf)
    )
    valid_fraction = float(valid_draws / max(1, int(n_resamples)))
    lower_direct = direct_interval[0]
    lower_interaction = interaction_interval[0]
    supported = bool(
        bool(protocol_conformant and canonical and complete_factorial)
        and valid_fraction >= 0.90
        and lower_direct is not None
        and float(lower_direct) > 0.0
        and lower_interaction is not None
        and float(lower_interaction) > 0.0
        and advantageous >= max(4, int(np.ceil(0.8 * int(replicates))))
        and lower_mean_quadratic_regret
    )
    return {
        "claim_supported": False,
        "food101_supported": supported,
        "food101_nonlinearity_supported": supported,
        "confirmatory_nonlinearity_supported": supported,
        "valid_fraction": valid_fraction,
        "complete_factorial": complete_factorial,
        "complete_auc_grid": complete_auc,
        "direct_nonlinear_oi_minus_probe": point[0] if np.isfinite(point[0]) else None,
        "direct_nonlinear_oi_minus_probe_interval_95": direct_interval,
        "nonlinear_baseline_interaction": point[1] if np.isfinite(point[1]) else None,
        "nonlinear_baseline_interaction_interval_95": interaction_interval,
        "nuisance_direct_oi_minus_probe": point[2] if np.isfinite(point[2]) else None,
        "nuisance_direct_interval_95": nuisance_direct_interval,
        "nuisance_baseline_interaction": point[3] if np.isfinite(point[3]) else None,
        "nuisance_baseline_interaction_interval_95": nuisance_interaction_interval,
        "nuisance_diagnostic_only": True,
        "replicate_advantages": int(advantageous),
        "replicate_advantage_fraction": float(advantageous / max(1, int(replicates))),
        "quadratic_regret_means": regret_means,
        "lower_mean_quadratic_regret": lower_mean_quadratic_regret,
        "bootstrap_resamples": int(n_resamples),
        "seed": int(seed),
    }


_bootstrap_primary = _bootstrap_food101
_bootstrap_confirmatory = _bootstrap_food101
_bootstrap = _bootstrap_food101


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_safe(row.get(field)) for field in fields})


def _load_food101_rows(
    data_dir: Path, no_download: bool
) -> tuple[list[Food101Sample], np.ndarray, list[str], list[Food101Sample], np.ndarray, list[str]]:
    """Load and alphabetically filter the official Food-101 train/test rows."""

    try:
        from torchvision.datasets import Food101
    except ImportError as exc:
        raise ImportError(
            "Food-101 extraction requires torchvision; install the backbone-selection extra."
        ) from exc
    kwargs = {"root": str(data_dir), "download": not bool(no_download)}
    train = Food101(split="train", **kwargs)
    test = Food101(split="test", **kwargs)
    classes = _first_food101_classes(getattr(train, "classes", ()))
    selected = set(classes)

    def records(source: Any, split: str) -> tuple[list[Food101Sample], np.ndarray, list[str]]:
        image_files = getattr(source, "_image_files", None)
        labels = getattr(source, "_labels", None)
        if image_files is None or labels is None:
            image_files, labels = [], []
            for index in range(len(source)):
                image, label = source[index]
                image_files.append(
                    getattr(source, "_image_files", [index])[index]
                    if getattr(source, "_image_files", None)
                    else index
                )
                labels.append(label)
        rows: list[Food101Sample] = []
        for index, (image_file, target) in enumerate(zip(image_files, labels)):
            class_name = str(source.classes[int(target)])
            if class_name not in selected:
                continue
            path = Path(str(image_file))
            rows.append(
                Food101Sample(f"food101/{split}/{class_name}/{index:05d}", path, class_name, split)
            )
        rows.sort(key=lambda row: (row.class_name, row.sample_id))
        return (
            rows,
            np.asarray([row.class_name for row in rows], dtype=object),
            [str(row.image_path) for row in rows],
        )

    return (*records(train, "train"), *records(test, "test"))


_load_food101 = _load_food101_rows
_load_food101_data = _load_food101_rows


def _resolve_torch_device(value: str) -> str:
    requested = str(value or "auto").lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested != "auto":
            raise RuntimeError("Torch is required for an explicit accelerator device") from None
        return "cpu"
    cuda_backend = getattr(torch, "cuda", None)
    cuda_available = bool(
        cuda_backend is not None
        and callable(getattr(cuda_backend, "is_available", None))
        and cuda_backend.is_available()
    )
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(
        mps_backend is not None
        and callable(getattr(mps_backend, "is_available", None))
        and mps_backend.is_available()
    )
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA is unavailable")
    if requested == "mps" and not mps_available:
        raise RuntimeError("Apple MPS is unavailable")
    if requested not in {"cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return requested


def _release_extractor(extractor: Any) -> None:
    for name in (
        "_model",
        "_processor",
        "_torch",
        "_image_module",
        "_preprocess",
        "_tokenizer",
        "_resolved_preprocess",
    ):
        if hasattr(extractor, name):
            try:
                setattr(extractor, name, None)
            except Exception:
                pass
    try:
        import torch

        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
        if bool(getattr(torch.backends, "mps", None)) and bool(torch.backends.mps.is_available()):
            torch.mps.empty_cache()
    except Exception:
        pass


def _build_final_output_extractors(
    models: Sequence[str], *, batch_size: int, device: Optional[str]
) -> list[Any]:
    """Build one lazy extractor for the declared final output of each backbone.

    The Food-101 protocol uses a frozen panel of ten explicitly configured
    backbones. Optional provider imports stay inside this factory so parser
    construction and ``--help`` remain safe without the extras.
    """

    from vertebrae.extractors.huggingface_vision import HFVisionExtractor
    from vertebrae.extractors.openclip import OpenCLIPExtractor
    from vertebrae.extractors.timm import TimmVisionExtractor

    extractors: list[Any] = []
    for model in models:
        if model == "dinov2-small":
            extractors.append(
                HFVisionExtractor(
                    name=model,
                    model_id="facebook/dinov2-small",
                    outputs=[{"name": "final_cls", "hidden_layer": -1, "pooling": "cls"}],
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                    processor_kwargs={"use_fast": False},
                )
            )
        elif model == "deit-tiny":
            extractors.append(
                HFVisionExtractor(
                    name=model,
                    model_id="facebook/deit-tiny-patch16-224",
                    outputs=[{"name": "final_cls", "hidden_layer": -1, "pooling": "cls"}],
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                    processor_kwargs={"use_fast": False},
                    model_kwargs={"add_pooling_layer": False},
                )
            )
        elif model == "convnext-tiny":
            extractors.append(
                TimmVisionExtractor(
                    name=model,
                    model_name="convnext_tiny",
                    pretrained=True,
                    outputs=[{"name": "final"}],
                    model_kwargs={"num_classes": 0},
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                )
            )
        elif model == "mobilenetv3-large":
            extractors.append(
                TimmVisionExtractor(
                    name=model,
                    model_name="mobilenetv3_large_100",
                    pretrained=True,
                    outputs=[{"name": "final"}],
                    model_kwargs={"num_classes": 0},
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                )
            )
        elif model == "openclip-vit-b-32":
            extractors.append(
                OpenCLIPExtractor(
                    name=model,
                    model_name="ViT-B-32",
                    pretrained="laion2b_s34b_b79k",
                    input_modalities={"image": "image"},
                    outputs=[{"name": "final_image", "source": "image"}],
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                )
            )
        elif model in _EXTRA_TIMM_MODELS:
            extractors.append(
                TimmVisionExtractor(
                    name=model,
                    model_name=_EXTRA_TIMM_MODELS[model],
                    pretrained=True,
                    outputs=[{"name": "final"}],
                    model_kwargs={"num_classes": 0},
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                )
            )
        else:
            raise ValueError(f"Unsupported model {model!r}.")
    return extractors


def _transform_final_outputs(extractor: Any, images: Sequence[Any]) -> list[Any]:
    """Transform image inputs using the provider-specific input contract."""

    if getattr(extractor, "extractor_type", None) == "openclip":
        return extractor.transform_many({"image": list(images)})
    return extractor.transform_many(list(images))


def _extract_final_embeddings(
    images: Sequence[Any],
    models: Sequence[str],
    *,
    batch_size: int,
    device: str,
    cache_dir: Path,
    resume: bool,
    sample_ids: Optional[Sequence[Any]] = None,
    labels: Optional[Sequence[Any]] = None,
    configuration: Optional[Mapping[str, Any]] = None,
) -> tuple[Dict[str, Dict[str, Any]], list[Dict[str, Any]]]:
    """Sequentially extract final model outputs and persist identity-checked memmaps."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(value) for value in (sample_ids if sample_ids is not None else range(len(images)))]
    label_values = [_json_safe(value) for value in labels] if labels is not None else None
    identity = {
        "sample_ids": ids,
        "labels": label_values,
        "configuration": _scientific_identity(configuration or {}),
    }
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    manifests: Dict[str, Dict[str, Any]] = {}
    timings: list[Dict[str, Any]] = []
    for model in models:
        cache_path = cache_dir / f"food101_{model}_final.npy"
        manifest_path = cache_path.with_suffix(".json")
        if resume and cache_path.exists() and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("identity_hash") != identity_hash:
                    raise ValueError("embedding cache identity hash mismatch")
                _read_float32_memmap(manifest)
                manifests[model] = manifest
                timings.append({"model": model, "seconds": 0.0, "cached": True})
                continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        started = perf_counter()
        extractor = _build_final_output_extractors(
            [model], batch_size=int(batch_size), device=device
        )[0]
        try:
            outputs = _transform_final_outputs(extractor, list(images))
            final = [output for output in outputs if str(output.name) in _FINAL_OUTPUTS]
            if len(final) != 1:
                raise ValueError(f"{model} must emit exactly one final output")
            matrix = np.asarray(final[0].embeddings, dtype=np.float32)
            if len(matrix) != len(images):
                raise ValueError(f"{model} returned {len(matrix)} rows for {len(images)} images")
            manifest = _write_float32_memmap(cache_path, matrix)
            recipe = extractor.recipe() if callable(getattr(extractor, "recipe", None)) else {}
            manifest.update(
                {
                    "model": model,
                    "output": str(final[0].name),
                    "row_count": int(len(matrix)),
                    "sample_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    "labels_sha256": hashlib.sha256(
                        json.dumps(label_values, default=str).encode()
                    ).hexdigest()
                    if label_values is not None
                    else None,
                    "identity_hash": identity_hash,
                    "configuration": _json_safe(configuration or {}),
                    "extractor_recipe": _json_safe(recipe),
                    "device": device,
                }
            )
            _atomic_json(manifest_path, manifest)
            manifests[model] = manifest
            timings.append(
                {
                    "model": model,
                    "seconds": float(perf_counter() - started),
                    "cached": False,
                    "rows": len(matrix),
                    "dimensions": int(matrix.shape[1]),
                }
            )
        finally:
            _release_extractor(extractor)
    return manifests, timings


def _run(args: argparse.Namespace) -> int:
    models = _resolve_cli_models(args.models)
    budgets = _parse_budget_values(args.budgets)
    if int(args.replicates) != _REPLICATES:
        raise ValueError("the frozen Food-101 confirmatory protocol requires --replicates 5")
    if int(args.bootstrap_resamples) < 1:
        raise ValueError("bootstrap-resamples must be positive")
    device = _resolve_torch_device(args.device)
    configuration = _configuration(
        stage="food101",
        jobs=args.jobs,
        seed=int(args.seed),
        models=list(models),
        budgets=list(budgets),
        replicates=int(args.replicates),
        bootstrap_resamples=int(args.bootstrap_resamples),
        data_dir=str(args.data_dir),
        embedding_batch_size=int(args.embedding_batch_size),
        device_requested=str(args.device),
        device_resolved=device,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir or output_dir / "cache")
    stem = _artifact_stem(configuration)
    protocol = _master_protocol(configuration)
    _atomic_json(output_dir / f"{stem}_planned_protocol.json", protocol)
    print(f"Frozen Food-101 protocol written to {output_dir / f'{stem}_planned_protocol.json'}")
    train_rows, train_labels, train_images, test_rows, test_labels, test_images = (
        _load_food101_rows(Path(args.data_dir), bool(args.no_download))
    )
    cohorts = _food101_cohort_splits(
        train_rows,
        train_labels,
        test_samples=test_rows,
        test_labels=test_labels,
        replicates=_REPLICATES,
        seed=int(args.seed),
    )
    # Extract exactly the frozen cohort union: five disjoint 80+52 train rows
    # per class and the fixed 52 official-test rows.  The complete torchvision
    # split contains 40,000 images; 28,480 rows are sufficient for this design.
    used_train_ids = {
        row.sample_id
        for roles in cohorts.values()
        for role in ("selector", "development")
        for row in roles[role]
    }
    used_test_ids = {row.sample_id for row in cohorts[0]["test"]}
    compact_train_rows = [row for row in train_rows if row.sample_id in used_train_ids]
    compact_test_rows = [row for row in test_rows if row.sample_id in used_test_ids]
    train_image_by_id = {row.sample_id: image for row, image in zip(train_rows, train_images)}
    test_image_by_id = {row.sample_id: image for row, image in zip(test_rows, test_images)}
    compact_train_images = [train_image_by_id[row.sample_id] for row in compact_train_rows]
    compact_test_images = [test_image_by_id[row.sample_id] for row in compact_test_rows]
    compact_train_labels = np.asarray([row.class_name for row in compact_train_rows], dtype=object)
    compact_test_labels = np.asarray([row.class_name for row in compact_test_rows], dtype=object)
    train_index = {row.sample_id: index for index, row in enumerate(compact_train_rows)}
    test_index = {row.sample_id: index for index, row in enumerate(compact_test_rows)}
    all_images = list(compact_train_images) + list(compact_test_images)
    all_labels = np.concatenate([compact_train_labels, compact_test_labels]).astype(object)
    all_ids = [row.sample_id for row in compact_train_rows] + [
        row.sample_id for row in compact_test_rows
    ]
    role_indices: Dict[int, Dict[str, list[int]]] = {}
    for replicate, roles in cohorts.items():
        role_indices[replicate] = {
            "selector": [train_index[row.sample_id] for row in roles["selector"]],
            "development": [train_index[row.sample_id] for row in roles["development"]],
            "test": [len(compact_train_rows) + test_index[row.sample_id] for row in roles["test"]],
        }
    compact_test_indices = [index for roles in role_indices.values() for index in roles["test"]]
    if compact_test_indices and not (
        min(compact_test_indices) >= len(compact_train_rows)
        and max(compact_test_indices) < len(all_images)
    ):
        raise RuntimeError("compact Food-101 test role indices fall outside the extracted panel")
    _atomic_json(
        output_dir / f"{stem}.cohorts.json",
        {
            "schema_version": 1,
            "study": _STUDY,
            "classes": sorted(set(train_labels.tolist())),
            "roles": role_indices,
            "extracted_sample_ids": all_ids,
            "extracted_train_rows": len(compact_train_rows),
            "extracted_test_rows": len(compact_test_rows),
            "configuration_hash": _configuration_hash(configuration),
        },
    )
    manifests, extraction = _extract_final_embeddings(
        all_images,
        models,
        batch_size=int(args.embedding_batch_size),
        device=device,
        cache_dir=cache_dir,
        resume=bool(args.resume),
        sample_ids=all_ids,
        labels=all_labels,
        configuration=configuration,
    )
    tasks: list[Dict[str, Any]] = []
    config_hash = _configuration_hash(configuration)
    for model in models:
        for replicate in range(_REPLICATES):
            tasks.append(
                {
                    "key": f"food101/{model}/{replicate}",
                    "configuration_hash": config_hash,
                    "backbone": model,
                    "replicate": replicate,
                    "seed": int(args.seed) + replicate,
                    "embedding_manifest": manifests[model],
                    "labels": all_labels.tolist(),
                    "roles": role_indices[replicate],
                    "budgets": list(budgets),
                }
            )
    jobs = _resolve_jobs(args.jobs)
    block_results = _run_parallel_blocks(
        tasks=tasks,
        worker=_score_food101_block,
        jobs=jobs,
        output_dir=output_dir,
        stem=stem,
        configuration_hash=config_hash,
        resume=bool(args.resume),
    )
    reference_rows = [row for result in block_results for row in result.get("reference_rows", [])]
    selector_rows = [row for result in block_results for row in result.get("selector_rows", [])]
    prediction_rows = [
        row for result in block_results for row in result.get("test_prediction_rows", [])
    ]
    expected_reference = [
        (model, replicate, arm, head)
        for model in models
        for replicate in range(_REPLICATES)
        for arm, _, _ in _FOOD101_ARMS
        for head in _HEAD_FAMILIES
    ]
    expected_selector = [
        (model, replicate, arm, budget, method)
        for model in models
        for replicate in range(_REPLICATES)
        for arm, _, _ in _FOOD101_ARMS
        for budget in budgets
        for method in _METHODS
    ]
    _validate_factorial_grid(
        reference_rows,
        ("backbone", "replicate", "arm", "head"),
        expected_reference,
    )
    _validate_factorial_grid(
        selector_rows,
        ("backbone", "replicate", "arm", "budget", "method"),
        expected_selector,
    )
    ranking_rows, auc_rows = _cross_backbone_rows(selector_rows, reference_rows, budgets=budgets)
    bootstrap = _bootstrap_food101(
        auc_rows=auc_rows,
        selector_rows=selector_rows,
        ranking_rows=ranking_rows,
        reference_rows=reference_rows,
        test_prediction_rows=prediction_rows,
        replicates=_REPLICATES,
        backbones=len(models),
        models=models,
        budgets=budgets,
        n_resamples=int(args.bootstrap_resamples),
        seed=int(args.seed),
        canonical=_is_canonical_configuration(configuration),
        protocol_conformant=True,
    )
    result_path = output_dir / f"{stem}.json"
    completed_protocol = dict(protocol)
    completed_protocol.update(
        {
            "artifact_status": "completed",
            "result_path": str(result_path),
            "food101_nonlinearity_supported": bool(
                bootstrap.get("food101_nonlinearity_supported", False)
            ),
        }
    )
    payload = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "protocol_version": _PROTOCOL_VERSION,
        "study": _STUDY,
        "artifact_status": "completed",
        "claim_supported": False,
        "food101_nonlinearity_supported": bool(
            bootstrap.get("food101_nonlinearity_supported", False)
        ),
        "configuration": _json_safe(configuration),
        "configuration_hash": config_hash,
        "protocol": completed_protocol,
        "reference_rows": reference_rows,
        "selector_rows": selector_rows,
        "test_prediction_rows": prediction_rows,
        "ranking_rows": ranking_rows,
        "auc_rows": auc_rows,
        "bootstrap": bootstrap,
        "runtime": {
            "device_resolved": device,
            "jobs_requested": args.jobs,
            "jobs_resolved": jobs,
            "extraction": extraction,
        },
    }
    _atomic_json(result_path, payload)
    _write_rows_csv(output_dir / f"{stem}_reference.csv", reference_rows)
    _write_rows_csv(output_dir / f"{stem}_selector.csv", selector_rows)
    _write_rows_csv(output_dir / f"{stem}_ranking.csv", ranking_rows)
    print(f"Completed Food-101 bridge: {result_path}")
    return 0


def _write_failed_artifact(args: argparse.Namespace, error: BaseException) -> None:
    """Persist a failure marker without replacing a completed result."""

    try:
        models = _resolve_cli_models(str(getattr(args, "models", "")))
    except (TypeError, ValueError):
        models = _DEFAULT_MODELS
    try:
        budgets = _parse_budget_values(str(getattr(args, "budgets", "")))
    except ValueError:
        budgets = _FOOD101_BUDGETS
    configuration = _configuration(
        stage="food101",
        jobs=getattr(args, "jobs", "auto"),
        seed=int(getattr(args, "seed", 42)),
        models=list(models),
        budgets=list(budgets),
        replicates=int(getattr(args, "replicates", _REPLICATES)),
        bootstrap_resamples=int(getattr(args, "bootstrap_resamples", _BOOTSTRAP_RESAMPLES)),
        data_dir=str(getattr(args, "data_dir", Path("examples/data"))),
        embedding_batch_size=int(getattr(args, "embedding_batch_size", 16)),
        device_requested=str(getattr(args, "device", "auto")),
    )
    output_dir = Path(getattr(args, "output_dir", Path("examples/output")))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _artifact_stem(configuration)
        completed_path = output_dir / f"{stem}.json"
        if completed_path.exists():
            return
        failed_path = output_dir / f"{stem}.failed.json"
        _atomic_json(
            failed_path,
            {
                "schema_version": _RESULT_SCHEMA_VERSION,
                "protocol_version": _PROTOCOL_VERSION,
                "study": _STUDY,
                "artifact_status": "failed",
                "claim_supported": False,
                "configuration": _json_safe(configuration),
                "configuration_hash": _configuration_hash(configuration),
                "error": f"{type(error).__name__}: {error}",
            },
        )
    except OSError:
        # Failure reporting must never mask the original experiment error.
        return


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the frozen Food-101 experiment and return a process exit code."""

    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        _write_failed_artifact(args, exc)
        print(f"Food-101 bridge could not run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
