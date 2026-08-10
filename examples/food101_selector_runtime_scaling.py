"""Measure Food-101 selector scoring time as the nested sample budget grows.

This is a post-hoc computational benchmark.  It intentionally does not reuse
the confirmatory accuracy result: every artifact records ``claim_supported`` as
false, and the timed interval contains only selector scoring calls on already
materialized embedding rows.  Model extraction, cache reads, subset selection,
imports, warmup, and downstream-head evaluation are outside the measurement.

The default run discovers the unique completed Food-101 bridge cohort and its
validated cache manifests when they are present.  Explicit embedding/label
manifests support exploratory panels.  ``--synthetic-smoke`` provides a tiny,
network-free path for checking the CLI and artifact schema without model
dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

# Set conservative BLAS/OpenMP controls before NumPy or scikit-learn can
# initialize their thread pools.  The benchmark is deliberately serial so
# process contention does not become part of the per-call measurement.
_THREAD_ENV = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}
for _thread_name, _thread_value in _THREAD_ENV.items():
    os.environ[_thread_name] = _thread_value

import numpy as np  # noqa: E402

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
_DEFAULT_BUDGETS = (64, 128, 256, 512, 640)
_TIMING_REPEATS = 5
_CROSS_FIT_FOLDS = 5
_PROBE_FOLDS = 5
_OVERLAP_K = 10
_METHODS = ("overlap_cross_fitted", "linear_probe_oof")
_STUDY = "food101_selector_runtime_scaling"
_ARTIFACT_STEM = _STUDY
_FIGURE_STEM = "food101-selector-runtime-scaling"
_SCHEMA_VERSION = 1
_PROTOCOL_VERSION = 1


def _artifact_stem(configuration: Mapping[str, Any]) -> str:
    """Return the configuration-hashed runtime artifact stem."""

    return f"{_ARTIFACT_STEM}_{_configuration_hash(configuration)[:12]}"


_make_artifact_stem = _artifact_stem


@dataclass(frozen=True)
class EmbeddingManifest:
    """Validated metadata and memory-mapped matrix for one backbone output."""

    model: str
    output: str
    path: Path
    matrix: np.ndarray
    payload: Mapping[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.matrix.shape[0]), int(self.matrix.shape[1]))


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


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), default=str)


def _configuration_hash(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(configuration).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels_sha256(labels: Sequence[Any]) -> str:
    values = [_json_safe(value) for value in labels]
    return hashlib.sha256(json.dumps(values, default=str).encode("utf-8")).hexdigest()


def _resolve_manifest_path(path: Path, cache_dir: Optional[Path]) -> Path:
    if path.exists():
        return path
    if cache_dir is not None:
        candidate = cache_dir / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def load_embedding_manifest(
    path: Path | str,
    *,
    cache_dir: Optional[Path | str] = None,
) -> EmbeddingManifest:
    """Read and integrity-check one JSON ``.npy`` embedding manifest."""

    manifest_path = Path(path)
    cache_root = Path(cache_dir) if cache_dir is not None else None
    manifest_path = _resolve_manifest_path(manifest_path, cache_root)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read embedding manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Embedding manifest must be a JSON object.")
    raw_path = payload.get("path")
    if raw_path is None:
        raise ValueError(f"Embedding manifest {manifest_path} is missing path.")
    matrix_path = Path(str(raw_path))
    if not matrix_path.is_absolute() and matrix_path.exists():
        # Bridge manifests persist repository-relative paths.  Resolve those
        # before trying the manifest directory, which would otherwise prefix
        # ``examples/output/cache`` twice.
        matrix_path = matrix_path
    elif not matrix_path.is_absolute():
        manifest_relative = manifest_path.parent / matrix_path
        cache_relative = cache_root / matrix_path if cache_root is not None else None
        if manifest_relative.exists():
            matrix_path = manifest_relative
        elif cache_relative is not None and cache_relative.exists():
            matrix_path = cache_relative
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    if payload.get("sha256") and _file_sha256(matrix_path) != str(payload["sha256"]):
        raise ValueError(f"Embedding SHA-256 mismatch for {matrix_path}.")
    try:
        matrix = np.load(matrix_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load embedding array {matrix_path}: {exc}") from exc
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Embedding arrays must be two-dimensional NumPy matrices.")
    expected_shape = payload.get("shape")
    if expected_shape is not None and list(matrix.shape) != [int(v) for v in expected_shape]:
        raise ValueError(f"Embedding shape mismatch for {matrix_path}.")
    expected_rows = payload.get("row_count")
    if expected_rows is not None and int(expected_rows) != int(matrix.shape[0]):
        raise ValueError(f"Embedding row-count mismatch for {matrix_path}.")
    expected_dtype = payload.get("dtype")
    if expected_dtype is not None and str(matrix.dtype) != str(expected_dtype):
        raise ValueError(f"Embedding dtype mismatch for {matrix_path}.")
    if not np.isfinite(np.asarray(matrix[: min(4, len(matrix))], dtype=float)).all():
        raise ValueError(f"Embedding preview contains non-finite values for {matrix_path}.")
    model = str(payload.get("model") or payload.get("name") or matrix_path.stem)
    output = str(payload.get("output") or "final")
    if not model:
        raise ValueError(f"Embedding manifest {manifest_path} has an empty model name.")
    return EmbeddingManifest(model, output, matrix_path, matrix, payload)


def load_embedding_manifests(
    paths: Sequence[Path | str],
    *,
    cache_dir: Optional[Path | str] = None,
    expected_models: Optional[Sequence[str]] = None,
) -> Dict[str, EmbeddingManifest]:
    """Load a unique model panel and validate common row identity metadata."""

    if not paths:
        raise ValueError("At least one embedding manifest is required.")
    manifests: Dict[str, EmbeddingManifest] = {}
    expected_rows: Optional[int] = None
    expected_labels_hash: Optional[str] = None
    expected_sample_hash: Optional[str] = None
    for path in paths:
        manifest = load_embedding_manifest(path, cache_dir=cache_dir)
        if manifest.model in manifests:
            raise ValueError(f"Duplicate embedding model {manifest.model!r}.")
        if expected_rows is None:
            expected_rows = manifest.shape[0]
        elif manifest.shape[0] != expected_rows:
            raise ValueError("Embedding manifests do not have a common row count.")
        labels_hash = manifest.payload.get("labels_sha256")
        sample_hash = manifest.payload.get("sample_ids_sha256")
        if labels_hash is not None:
            if expected_labels_hash is None:
                expected_labels_hash = str(labels_hash)
            elif str(labels_hash) != expected_labels_hash:
                raise ValueError("Embedding manifests disagree on labels_sha256.")
        if sample_hash is not None:
            if expected_sample_hash is None:
                expected_sample_hash = str(sample_hash)
            elif str(sample_hash) != expected_sample_hash:
                raise ValueError("Embedding manifests disagree on sample_ids_sha256.")
        manifests[manifest.model] = manifest
    if expected_models is not None:
        requested = tuple(str(value) for value in expected_models)
        if set(manifests) != set(requested):
            raise ValueError(
                "Embedding model panel mismatch: "
                f"expected {sorted(requested)!r}, got {sorted(manifests)!r}."
            )
    return manifests


def _load_json_or_array(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        loaded = np.load(path, allow_pickle=False)
        if "labels" in loaded:
            return loaded["labels"]
        if len(loaded.files) == 1:
            return loaded[loaded.files[0]]
        raise ValueError(f"Label archive {path} must contain one array or a labels array.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read labels manifest {path}: {exc}") from exc


def load_labels_manifest(path: Path | str, *, cache_dir: Optional[Path | str] = None) -> np.ndarray:
    """Load a labels manifest and return a one-dimensional object array."""

    manifest_path = _resolve_manifest_path(Path(path), Path(cache_dir) if cache_dir else None)
    payload = _load_json_or_array(manifest_path)
    if isinstance(payload, Mapping):
        nested_path = payload.get("path")
        if nested_path is not None and "labels" not in payload:
            nested = Path(str(nested_path))
            if not nested.is_absolute():
                nested = manifest_path.parent / nested
            return load_labels_manifest(nested, cache_dir=cache_dir)
        labels = payload.get("labels")
        if labels is None:
            labels = payload.get("y")
        if labels is None:
            raise ValueError("Labels manifest must contain a labels or y array.")
        expected_hash = payload.get("labels_sha256")
    else:
        labels = payload
        expected_hash = None
    values = np.asarray(labels, dtype=object)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Labels must be a non-empty one-dimensional sequence.")
    if expected_hash and _labels_sha256(values.tolist()) != str(expected_hash):
        raise ValueError("Labels manifest labels_sha256 does not match its labels.")
    return values


def _discover_food101_cohort(output_dir: Path) -> tuple[Path, Mapping[str, Any]]:
    """Find the unique completed bridge cohort used by the canonical cache."""

    candidates: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(output_dir.glob("food101_nonlinear_backbone_bridge_*.cohorts.json")):
        result_path = path.with_name(path.name.replace(".cohorts.json", ".json"))
        if not result_path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, Mapping)
            and isinstance(result, Mapping)
            and payload.get("study") == "food101_nonlinear_backbone_bridge"
            and result.get("artifact_status") == "completed"
            and result.get("configuration_hash") == payload.get("configuration_hash")
        ):
            candidates.append((path, payload))
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one completed Food-101 bridge cohort for automatic discovery; "
            f"found {len(candidates)}. Pass explicit manifests to disambiguate."
        )
    return candidates[0]


def _load_discovered_cohort(
    cohort_path: Path,
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, list[str], int]:
    sample_ids = payload.get("extracted_sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError(f"Food-101 cohort {cohort_path} has no extracted_sample_ids.")
    if len(set(str(value) for value in sample_ids)) != len(sample_ids):
        raise ValueError("Food-101 cohort sample IDs must be unique.")
    train_rows = int(payload.get("extracted_train_rows", 0))
    test_rows = int(payload.get("extracted_test_rows", 0))
    if train_rows != 26_400 or test_rows != 2_080 or train_rows + test_rows != len(sample_ids):
        raise ValueError(
            "Canonical Food-101 cohort must contain 26,400 train and 2,080 test rows."
        )
    ids = [str(value) for value in sample_ids]
    labels: list[str] = []
    for index, sample_id in enumerate(ids):
        pieces = sample_id.split("/")
        if len(pieces) < 4 or pieces[0] != "food101" or pieces[1] not in {"train", "test"}:
            raise ValueError(f"Unexpected Food-101 sample ID at row {index}: {sample_id!r}")
        if index < train_rows and pieces[1] != "train":
            raise ValueError("Canonical cohort train rows must precede test rows.")
        if index >= train_rows and pieces[1] != "test":
            raise ValueError("Canonical cohort test rows must follow train rows.")
        labels.append(pieces[2])
    class_values = sorted(set(labels[:train_rows]))
    if len(class_values) != 40 or any(
        labels[:train_rows].count(value) != 660 for value in class_values
    ):
        raise ValueError(
            "Canonical Food-101 cohort must contain 40 classes with 660 train rows each."
        )
    return np.asarray(labels, dtype=object), ids, train_rows


def _restrict_discovered_embeddings(
    manifests: Mapping[str, EmbeddingManifest],
    *,
    labels: np.ndarray,
    sample_ids: Sequence[str],
    train_rows: int,
) -> Dict[str, np.ndarray]:
    expected_sample_hash = hashlib.sha256(json.dumps(list(sample_ids)).encode()).hexdigest()
    expected_labels_hash = _labels_sha256(labels.tolist())
    restricted: Dict[str, np.ndarray] = {}
    for model, manifest in manifests.items():
        if manifest.shape[0] != len(sample_ids):
            raise ValueError(f"Embedding cache {model!r} does not match cohort row count.")
        if str(manifest.payload.get("sample_ids_sha256")) != expected_sample_hash:
            raise ValueError(f"Embedding cache {model!r} sample identity does not match cohort.")
        if str(manifest.payload.get("labels_sha256")) != expected_labels_hash:
            raise ValueError(f"Embedding cache {model!r} label identity does not match cohort.")
        restricted[model] = manifest.matrix[:train_rows]
    return restricted


def _sorted_classes(labels: np.ndarray) -> list[Any]:
    return sorted(np.unique(labels).tolist(), key=lambda value: str(value))


def validate_nested_class_balanced_samples(
    labels: Sequence[Any],
    budgets: Sequence[int],
    *,
    classes: Optional[Sequence[Any]] = None,
) -> tuple[Any, ...]:
    """Validate that every requested class supports the largest nested budget."""

    values = np.asarray(labels, dtype=object)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    resolved_budgets = tuple(int(value) for value in budgets)
    if not resolved_budgets or any(value < 1 for value in resolved_budgets):
        raise ValueError("budgets must contain positive integers")
    if len(set(resolved_budgets)) != len(resolved_budgets):
        raise ValueError("budgets must be unique")
    selected = tuple(classes) if classes is not None else tuple(_sorted_classes(values))
    if not selected:
        raise ValueError("at least one class is required")
    counts = {label: int(np.sum(values == label)) for label in selected}
    missing = [label for label, count in counts.items() if count < max(resolved_budgets)]
    if missing:
        raise ValueError(
            f"largest budget {max(resolved_budgets)} exceeds rows for classes {missing!r}"
        )
    unknown = [label for label in selected if counts[label] == 0]
    if unknown:
        raise ValueError(f"requested classes are absent from labels: {unknown!r}")
    return selected


def _label_seed(seed: int, label: Any) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:8]
    return int(seed) + int(digest, 16)


def build_nested_indices(
    labels: Sequence[Any],
    budgets: Sequence[int],
    *,
    seed: int = 42,
    classes: Optional[Sequence[Any]] = None,
) -> Dict[int, np.ndarray]:
    """Build deterministic class-balanced nested row indices shared by repeats."""

    values = np.asarray(labels, dtype=object)
    selected = validate_nested_class_balanced_samples(values, budgets, classes=classes)
    permutations = {
        label: np.random.default_rng(_label_seed(int(seed), label)).permutation(
            np.flatnonzero(values == label)
        )
        for label in selected
    }
    result: Dict[int, np.ndarray] = {}
    for budget in sorted(int(value) for value in budgets):
        result[budget] = np.sort(
            np.concatenate([permutations[label][:budget] for label in selected]).astype(np.int64)
        )
    return result


def _score_overlap(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    folds: int,
    k: int,
) -> float:
    from vertebrae.config import OverlapScoringConfig
    from vertebrae.scoring.overlap import OverlapIndexScorer

    config = OverlapScoringConfig(
        k=int(k),
        min_k=int(k),
        max_k=int(k),
        min_samples_per_cluster=5,
        kmeans_kwargs={"random_state": int(seed)},
        normalize_embeddings=True,
    )
    result = OverlapIndexScorer(config).score_cross_fitted(
        matrix,
        labels,
        n_splits=int(folds),
        seed=int(seed),
    )
    return float(result.macro_score)


def _score_probe(matrix: np.ndarray, labels: np.ndarray, *, seed: int, folds: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    target = np.asarray(labels, dtype=object)
    counts = np.unique(target, return_counts=True)[1]
    splits = min(int(folds), int(np.min(counts)))
    if splits < 2:
        raise ValueError("OOF probe requires at least two rows per class")
    predictions = np.empty(len(target), dtype=object)
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=int(seed))
    for train, holdout in splitter.split(np.zeros(len(target), dtype=np.uint8), target):
        estimator = make_pipeline(
            Normalizer(norm="l2"),
            LogisticRegression(C=1.0, max_iter=2_000, random_state=int(seed), n_jobs=1),
        )
        estimator.fit(matrix[train], target[train])
        predictions[holdout] = estimator.predict(matrix[holdout])
    return float(accuracy_score(target, predictions))


def _thread_limit():
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - sklearn normally supplies threadpoolctl
        return nullcontext()


def benchmark_embedding_matrix(
    matrix: np.ndarray,
    labels: Sequence[Any],
    *,
    model: str = "backbone",
    budgets: Sequence[int] = _DEFAULT_BUDGETS,
    timing_repeats: int = _TIMING_REPEATS,
    seed: int = 42,
    folds: int = _CROSS_FIT_FOLDS,
    k: int = _OVERLAP_K,
    nested_indices: Optional[Mapping[int, np.ndarray]] = None,
    warmup: bool = True,
    model_index: int = 0,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> list[Dict[str, Any]]:
    """Time paired selector calls for one already materialized embedding matrix."""

    values = np.asarray(labels, dtype=object)
    embeddings = np.asarray(matrix)
    if embeddings.ndim != 2 or len(embeddings) != len(values):
        raise ValueError("matrix and labels must have matching rows and a 2-D matrix")
    resolved_budgets = tuple(int(value) for value in budgets)
    indices = dict(nested_indices or build_nested_indices(values, resolved_budgets, seed=seed))
    if set(indices) != set(resolved_budgets):
        raise ValueError("nested_indices must contain exactly the requested budgets")
    if int(timing_repeats) < 1:
        raise ValueError("timing_repeats must be positive")
    for budget in resolved_budgets:
        selected = indices[int(budget)]
        if len(selected) != int(budget) * len(np.unique(values)):
            raise ValueError("nested indices are not class-balanced")

    def score(method: str, subset: np.ndarray, target: np.ndarray, call_seed: int) -> float:
        if method == "overlap_cross_fitted":
            return _score_overlap(subset, target, seed=call_seed, folds=folds, k=k)
        if method == "linear_probe_oof":
            return _score_probe(subset, target, seed=call_seed, folds=folds)
        raise ValueError(f"Unknown selector method {method!r}")

    with _thread_limit():
        if warmup:
            if progress is not None:
                progress({"event": "warmup_start", "model": str(model)})
            lowest_budget = min(resolved_budgets)
            selected = indices[int(lowest_budget)]
            subset = embeddings[selected]
            target = values[selected]
            for method in _METHODS:
                score(method, subset, target, int(seed) + int(lowest_budget))
            if progress is not None:
                progress(
                    {
                        "event": "warmup_end",
                        "model": str(model),
                        "samples_per_class": int(lowest_budget),
                    }
                )
        rows: list[Dict[str, Any]] = []
        for repeat in range(int(timing_repeats)):
            for budget in resolved_budgets:
                budget_index = resolved_budgets.index(budget)
                order = (
                    _METHODS
                    if (int(model_index) + int(budget_index) + int(repeat)) % 2 == 0
                    else tuple(reversed(_METHODS))
                )
                selected = indices[int(budget)]
                subset = embeddings[selected]
                target = values[selected]
                timed: Dict[str, Dict[str, float]] = {}
                for order_position, method in enumerate(order):
                    call_seed = int(seed) + int(budget)
                    started = perf_counter()
                    process_started = process_time()
                    score_value = score(method, subset, target, call_seed)
                    elapsed = float(perf_counter() - started)
                    cpu_elapsed = float(process_time() - process_started)
                    timed[method] = {
                        "score": float(score_value),
                        "elapsed_seconds": elapsed,
                        "process_seconds": cpu_elapsed,
                        "order_position": float(order_position),
                    }
                denominator = max(
                    timed["overlap_cross_fitted"]["elapsed_seconds"], np.finfo(float).eps
                )
                ratio = timed["linear_probe_oof"]["elapsed_seconds"] / denominator
                for method in _METHODS:
                    rows.append(
                        {
                            "backbone": str(model),
                            "samples_per_class": int(budget),
                            "budget": int(budget),
                            "repeat": int(repeat),
                            "method": method,
                            "score": timed[method]["score"],
                            "elapsed_seconds": timed[method]["elapsed_seconds"],
                            "seconds": timed[method]["elapsed_seconds"],
                            "process_seconds": timed[method]["process_seconds"],
                            "order_position": int(timed[method]["order_position"]),
                            "speedup": float(ratio),
                            "ratio": float(ratio),
                        }
                    )
                if progress is not None:
                    progress(
                        {
                            "event": "cell",
                            "model": str(model),
                            "samples_per_class": int(budget),
                            "repeat": int(repeat),
                            "overlap_elapsed_seconds": timed[
                                "overlap_cross_fitted"
                            ]["elapsed_seconds"],
                            "probe_elapsed_seconds": timed["linear_probe_oof"][
                                "elapsed_seconds"
                            ],
                            "ratio": float(ratio),
                        }
                    )
    return rows


def _paired_summary(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    grouped: Dict[tuple[str, int, int], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["backbone"]), int(row["samples_per_class"]), int(row["repeat"]))
        method = str(row["method"])
        if method in grouped.setdefault(key, {}):
            raise ValueError(f"Duplicate runtime cell {key!r}, method={method!r}")
        grouped[key][method] = row
    if not grouped or any(set(values) != set(_METHODS) for values in grouped.values()):
        raise ValueError(
            "Runtime grid is incomplete; every backbone/budget/repeat needs both methods"
        )
    paired: list[Dict[str, Any]] = []
    for (backbone, budget, repeat), values in sorted(grouped.items(), key=str):
        overlap = values["overlap_cross_fitted"]
        probe = values["linear_probe_oof"]
        overlap_elapsed = float(overlap["elapsed_seconds"])
        probe_elapsed = float(probe["elapsed_seconds"])
        paired.append(
            {
                "backbone": backbone,
                "samples_per_class": budget,
                "repeat": repeat,
                "overlap_elapsed_seconds": overlap_elapsed,
                "probe_elapsed_seconds": probe_elapsed,
                "overlap_process_seconds": float(overlap.get("process_seconds", 0.0)),
                "probe_process_seconds": float(probe.get("process_seconds", 0.0)),
                "ratio": probe_elapsed / max(overlap_elapsed, np.finfo(float).eps),
                "speedup": probe_elapsed / max(overlap_elapsed, np.finfo(float).eps),
            }
        )
    summary: list[Dict[str, Any]] = []
    budgets = sorted({int(row["samples_per_class"]) for row in rows})
    for budget in budgets:
        cells = [row for row in paired if int(row["samples_per_class"]) == budget]
        for method, field, process_field in (
            ("overlap_cross_fitted", "overlap_elapsed_seconds", "overlap_process_seconds"),
            ("linear_probe_oof", "probe_elapsed_seconds", "probe_process_seconds"),
        ):
            elapsed_values = np.asarray([float(row[field]) for row in cells], dtype=float)
            process_values = np.asarray([float(row[process_field]) for row in cells], dtype=float)
            summary.append(
                {
                    "samples_per_class": budget,
                    "method": method,
                    "n_cells": int(len(cells)),
                    "mean_elapsed_seconds": float(np.mean(elapsed_values)),
                    "sd_elapsed_seconds": float(np.std(elapsed_values, ddof=1))
                    if len(elapsed_values) > 1
                    else 0.0,
                    "mean_process_seconds": float(np.mean(process_values)),
                    "sd_process_seconds": float(np.std(process_values, ddof=1))
                    if len(process_values) > 1
                    else 0.0,
                    "mean_speedup_probe_over_overlap": float(
                        np.mean([float(row["ratio"]) for row in cells])
                    ),
                }
            )
    return summary, paired


def _validate_complete_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    budgets: Sequence[int],
    repeats: int,
) -> None:
    expected = {
        (str(model), int(budget), int(repeat), method)
        for model in models
        for budget in budgets
        for repeat in range(int(repeats))
        for method in _METHODS
    }
    observed = {
        (
            str(row["backbone"]),
            int(row["samples_per_class"]),
            int(row["repeat"]),
            str(row["method"]),
        )
        for row in rows
    }
    if observed != expected:
        raise ValueError(
            f"Runtime factorial grid mismatch: missing={len(expected - observed)}, "
            f"extra={len(observed - expected)}"
        )


def _read_results(
    path: Path | str,
    *,
    expected_configuration_hash: Optional[str] = None,
) -> Mapping[str, Any]:
    """Read a completed artifact and reject stale or partial runtime grids."""

    result_path = Path(path)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read runtime-scaling result {result_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Runtime-scaling result must be a JSON object.")
    if payload.get("study") != _STUDY or payload.get("format") != _STUDY:
        raise ValueError("Runtime-scaling result has the wrong study/format.")
    if payload.get("artifact_status") != "completed":
        raise ValueError("Runtime-scaling result is not completed.")
    if payload.get("post_hoc_runtime_benchmark") is not True:
        raise ValueError("Runtime-scaling result is missing post_hoc_runtime_benchmark=true.")
    if payload.get("claim_supported") is not False:
        raise ValueError("Runtime-scaling result must have claim_supported=false.")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Runtime-scaling result is missing protocol metadata.")
    if protocol.get("post_hoc_runtime_benchmark") is not True:
        raise ValueError("Runtime-scaling protocol is not marked post-hoc.")
    if protocol.get("claim_supported") is not False:
        raise ValueError("Runtime-scaling protocol must have claim_supported=false.")
    config_hash = str(payload.get("configuration_hash", ""))
    if len(config_hash) != 64 or any(char not in "0123456789abcdef" for char in config_hash):
        raise ValueError(
            "Runtime-scaling configuration_hash must be a 64-character SHA-256 hex digest."
        )
    if expected_configuration_hash is not None and config_hash != str(expected_configuration_hash):
        raise ValueError("Runtime-scaling configuration hash mismatch.")
    configuration = payload.get("configuration")
    rows = payload.get("rows")
    if not isinstance(configuration, Mapping) or not isinstance(rows, list) or not rows:
        raise ValueError("Runtime-scaling result is missing configuration or rows.")
    if not isinstance(payload.get("paired_rows"), list) or not isinstance(
        payload.get("summary"), list
    ):
        raise ValueError("Runtime-scaling result is missing paired_rows or summary.")
    if _configuration_hash(configuration) != config_hash:
        raise ValueError("Runtime-scaling configuration hash does not match configuration.")
    artifact_stem = payload.get("artifact_stem")
    if not isinstance(artifact_stem, str) or not artifact_stem.endswith(
        f"_{config_hash[:12]}"
    ):
        raise ValueError("Runtime-scaling artifact_stem does not match configuration hash.")
    required = {"backbone", "samples_per_class", "repeat", "method", "elapsed_seconds", "score"}
    for row in rows:
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise ValueError("Runtime rows are missing required fields.")
        if str(row["method"]) not in _METHODS:
            raise ValueError("Runtime rows contain an unknown method.")
        if not np.isfinite(float(row["elapsed_seconds"])) or float(row["elapsed_seconds"]) < 0:
            raise ValueError("Runtime rows must have finite non-negative elapsed seconds.")
        if not np.isfinite(float(row["score"])):
            raise ValueError("Runtime rows must have finite scores.")
    models = configuration.get("models")
    budgets = configuration.get("budgets")
    repeats = configuration.get("timing_repeats")
    if (
        not isinstance(models, list)
        or not isinstance(budgets, list)
        or not isinstance(repeats, int)
    ):
        raise ValueError(
            "Runtime configuration is missing models, budgets, or timing_repeats."
        )
    _validate_complete_grid(rows, models=models, budgets=budgets, repeats=repeats)
    _paired_summary(rows)
    return payload


_load_results = _read_results


def run_runtime_scaling(
    embeddings: Mapping[str, np.ndarray | EmbeddingManifest],
    labels: Sequence[Any],
    *,
    budgets: Sequence[int] = _DEFAULT_BUDGETS,
    timing_repeats: int = _TIMING_REPEATS,
    seed: int = 42,
    folds: int = _CROSS_FIT_FOLDS,
    k: int = _OVERLAP_K,
    classes: Optional[Sequence[Any]] = None,
    warmup: bool = True,
    configuration: Optional[Mapping[str, Any]] = None,
    precomputed_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the complete paired runtime grid and return a serializable payload."""

    labels_array = np.asarray(labels, dtype=object)
    resolved_models = tuple(str(model) for model in embeddings)
    if precomputed_rows is not None:
        resolved_models = tuple(
            sorted({str(row["backbone"]) for row in precomputed_rows}, key=str)
        )
    if not resolved_models:
        raise ValueError("At least one embedding model is required.")
    selected_classes = validate_nested_class_balanced_samples(
        labels_array,
        budgets,
        classes=classes,
    )
    indices = build_nested_indices(labels_array, budgets, seed=seed, classes=selected_classes)
    rows: list[Dict[str, Any]] = [dict(row) for row in (precomputed_rows or [])]
    if precomputed_rows is None:
        for model_index, model in enumerate(resolved_models):
            artifact = embeddings[model]
            matrix = (
                artifact.matrix
                if isinstance(artifact, EmbeddingManifest)
                else np.asarray(artifact)
            )
            if matrix.ndim != 2 or len(matrix) != len(labels_array):
                raise ValueError(f"Embedding model {model!r} has a row-count mismatch.")
            rows.extend(
                benchmark_embedding_matrix(
                    matrix,
                    labels_array,
                    model=model,
                    budgets=budgets,
                    timing_repeats=timing_repeats,
                    seed=seed,
                    folds=folds,
                    k=k,
                    nested_indices=indices,
                    warmup=warmup,
                    model_index=model_index,
                )
            )
    _validate_complete_grid(
        rows,
        models=resolved_models,
        budgets=budgets,
        repeats=timing_repeats,
    )
    summary, paired = _paired_summary(rows)
    scientific_configuration: Dict[str, Any] = {
        "study": _STUDY,
        "models": list(resolved_models),
        "budgets": [int(value) for value in budgets],
        "classes": [_json_safe(value) for value in selected_classes],
        "timing_repeats": int(timing_repeats),
        "cross_fit_folds": int(folds),
        "probe_folds": int(_PROBE_FOLDS),
        "overlap_k": int(k),
        "seed": int(seed),
        "warmup_excluded": bool(warmup),
        "serial_one_thread": True,
        "thread_controls": dict(_THREAD_ENV),
        "method_order": "counterbalanced by model_index+budget_index+repeat parity",
        "algorithm_seed": "fixed seed+budget across timing repeats",
    }
    if configuration:
        scientific_configuration.update(_json_safe(dict(configuration)))
    config_hash = _configuration_hash(scientific_configuration)
    protocol = {
        "study": _STUDY,
        "protocol_version": _PROTOCOL_VERSION,
        "post_hoc_runtime_benchmark": True,
        "claim_supported": False,
        "figure_stem": _FIGURE_STEM,
        "timed_stage": "selector_scoring_calls_only",
        "excluded_from_timing": [
            "embedding_cache_reads",
            "subset_materialization",
            "imports",
            "warmup",
            "feature_extraction",
            "downstream_head_evaluation",
            "plotting",
        ],
        "uncertainty_unit": "paired_backbone_repeat_cell",
        "speedup_definition": "linear_probe_oof_elapsed / overlap_cross_fitted_elapsed",
        "serial_one_thread": True,
        "thread_controls": dict(_THREAD_ENV),
        "warmup_excluded": True,
        "method_order": "counterbalanced by model_index+budget_index+repeat parity",
        "algorithm_seed": "fixed seed+budget across timing repeats",
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "protocol_version": _PROTOCOL_VERSION,
        "study": _STUDY,
        "format": _STUDY,
        "artifact_status": "completed",
        "post_hoc_runtime_benchmark": True,
        "claim_supported": False,
        "configuration": scientific_configuration,
        "configuration_hash": config_hash,
        "artifact_stem": _artifact_stem(scientific_configuration),
        "protocol": protocol,
        "rows": rows,
        "runtime_rows": rows,
        "paired_rows": paired,
        "summary": summary,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(field) for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: _json_safe(row.get(field)) for field in fields} for row in rows)


def _parse_positive_csv(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item < 1 for item in parsed) or len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must contain unique positive integers")
    return parsed


def _parse_models(value: str) -> tuple[str, ...]:
    models = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if len(set(models)) != len(models):
        raise ValueError("models must be unique")
    return models


def _parse_classes(value: str, labels: np.ndarray) -> tuple[Any, ...]:
    if not value:
        return tuple(_sorted_classes(labels))
    available = _sorted_classes(labels)
    if str(value).strip().isdigit():
        count = int(str(value).strip())
        if count < 1 or count > len(available):
            raise ValueError("classes count is outside the available class range")
        return tuple(available[:count])
    requested = tuple(item.strip() for item in str(value).split(",") if item.strip())
    available_by_string = {str(item): item for item in available}
    unknown = [item for item in requested if item not in available_by_string]
    if unknown:
        raise ValueError(f"requested classes are absent from labels: {unknown!r}")
    return tuple(available_by_string[item] for item in requested)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-manifest",
        action="append",
        default=[],
        help="JSON embedding manifest; repeat once per backbone.",
    )
    parser.add_argument("--labels-manifest", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cache-manifest-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument("--budgets", default=",".join(str(value) for value in _DEFAULT_BUDGETS))
    parser.add_argument("--models", default="", help="Comma-separated model panel subset.")
    parser.add_argument("--classes", default="", help="Class count or comma-separated labels.")
    parser.add_argument("--timing-repeats", type=int, default=_TIMING_REPEATS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Return the lazy public argument parser."""

    return _parser()


def _synthetic_inputs(seed: int = 42) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(int(seed))
    classes = np.repeat(np.asarray(["class-0", "class-1", "class-2", "class-3"], dtype=object), 48)
    matrix = rng.normal(size=(len(classes), 12)).astype(np.float32)
    return {
        "synthetic-a": matrix,
        "synthetic-b": matrix + rng.normal(0, 0.1, matrix.shape),
    }, classes


def _run(args: argparse.Namespace) -> int:
    synthetic = bool(args.synthetic_smoke)
    output_dir = Path(args.output_dir)
    if synthetic:
        budgets = _parse_positive_csv(args.budgets, name="budgets")
        if args.budgets == ",".join(str(value) for value in _DEFAULT_BUDGETS):
            budgets = (8, 16, 24, 32, 40)
        embeddings, labels = _synthetic_inputs(int(args.seed))
        models = tuple(_parse_models(args.models)) if args.models else tuple(embeddings)
        embeddings = {model: embeddings[model] for model in models if model in embeddings}
        if not embeddings:
            raise ValueError("synthetic smoke models must be synthetic-a or synthetic-b")
        classes = _parse_classes(str(args.classes), labels) if args.classes else None
        canonical_cohort = False
    else:
        budgets = _parse_positive_csv(args.budgets, name="budgets")
        if not args.embedding_manifest and args.labels_manifest is None:
            cohort_root = output_dir
            if args.cache_dir is not None and not list(
                cohort_root.glob("food101_nonlinear_backbone_bridge_*.cohorts.json")
            ):
                cohort_root = Path(args.cache_dir).parent
            cohort_path, cohort_payload = _discover_food101_cohort(cohort_root)
            full_labels, sample_ids, train_rows = _load_discovered_cohort(
                cohort_path, cohort_payload
            )
            cache_manifest_dir = (
                args.cache_manifest_dir
                or args.cache_dir
                or output_dir / "cache"
            )
            requested_models = _parse_models(args.models) if args.models else _DEFAULT_MODELS
            manifest_paths = [
                cache_manifest_dir / f"food101_{model}_final.json"
                for model in requested_models
            ]
            manifests = load_embedding_manifests(
                manifest_paths,
                cache_dir=args.cache_dir,
                expected_models=requested_models,
            )
            embeddings = _restrict_discovered_embeddings(
                manifests,
                labels=full_labels,
                sample_ids=sample_ids,
                train_rows=train_rows,
            )
            labels = full_labels[:train_rows]
            models = tuple(requested_models)
            classes = _parse_classes(str(args.classes), labels) if args.classes else None
            canonical_cohort = True
        else:
            if not args.embedding_manifest or args.labels_manifest is None:
                raise ValueError(
                    "Provide both --embedding-manifest and --labels-manifest, "
                    "or omit both for automatic canonical cohort discovery."
                )
            labels = load_labels_manifest(args.labels_manifest, cache_dir=args.cache_dir)
            manifest_paths = [Path(path) for path in args.embedding_manifest]
            if args.cache_manifest_dir is not None:
                manifest_paths = [
                    path if path.is_absolute() else args.cache_manifest_dir / path
                    for path in manifest_paths
                ]
            requested_models = _parse_models(args.models) if args.models else None
            manifests = load_embedding_manifests(
                manifest_paths,
                cache_dir=args.cache_dir,
                expected_models=requested_models,
            )
            embeddings = manifests
            models = tuple(manifests)
            classes = _parse_classes(str(args.classes), labels) if args.classes else None
            canonical_cohort = False

    if int(args.timing_repeats) < 1:
        raise ValueError("timing-repeats must be positive")
    input_identity: Dict[str, Any] = {
        "row_count": int(len(labels)),
        "labels_sha256": _labels_sha256(labels.tolist()),
        "models": {},
    }
    if not synthetic and "sample_ids" in locals():
        input_identity["sample_ids_sha256"] = hashlib.sha256(
            json.dumps(list(sample_ids)).encode()
        ).hexdigest()
    if not synthetic and "cohort_payload" in locals():
        input_identity["cohort_configuration_hash"] = cohort_payload.get("configuration_hash")
    for model in models:
        artifact = embeddings[model]
        if isinstance(artifact, EmbeddingManifest):
            array_hash = artifact.payload.get("sha256")
            if array_hash is None:
                array_hash = hashlib.sha256(np.asarray(artifact.matrix).tobytes()).hexdigest()
            input_identity["models"][model] = {
                "array_sha256": array_hash,
                "shape": list(artifact.shape),
                "dtype": str(artifact.matrix.dtype),
                "output": artifact.output,
                "identity_hash": artifact.payload.get("identity_hash"),
            }
        else:
            matrix = np.asarray(artifact)
            input_identity["models"][model] = {
                "array_sha256": hashlib.sha256(np.asarray(matrix).tobytes()).hexdigest(),
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
            }
    configuration = {
        "budgets": list(budgets),
        "models": list(models),
        "classes": [_json_safe(value) for value in (classes or _sorted_classes(labels))],
        "timing_repeats": int(args.timing_repeats),
        "seed": int(args.seed),
        "synthetic_smoke": synthetic,
        "canonical_cohort": canonical_cohort,
        "cross_fit_folds": _CROSS_FIT_FOLDS,
        "probe_folds": _PROBE_FOLDS,
        "overlap_k": _OVERLAP_K,
        "warmup_excluded": True,
        "serial_one_thread": True,
        "thread_controls": dict(_THREAD_ENV),
        "method_order": "counterbalanced by model_index+budget_index+repeat parity",
        "algorithm_seed": "fixed seed+budget across timing repeats",
        "input_identity": input_identity,
    }
    config_hash = _configuration_hash(configuration)
    stem = _artifact_stem(configuration)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / f"{stem}.json"
    planned_path = output_dir / f"{stem}.planned.json"
    if args.resume and completed_path.exists():
        _read_results(completed_path, expected_configuration_hash=config_hash)
        print(f"Reused completed runtime-scaling artifact: {completed_path}")
        return 0
    if completed_path.exists():
        raise ValueError(
            f"Completed artifact already exists at {completed_path}; pass --resume to reuse it."
        )
    planned = {
        "schema_version": _SCHEMA_VERSION,
        "protocol_version": _PROTOCOL_VERSION,
        "study": _STUDY,
        "artifact_status": "planned",
        "post_hoc_runtime_benchmark": True,
        "claim_supported": False,
        "configuration": configuration,
        "configuration_hash": config_hash,
        "protocol": {
            "post_hoc_runtime_benchmark": True,
            "claim_supported": False,
            "figure_stem": _FIGURE_STEM,
            "serial_one_thread": True,
            "thread_controls": dict(_THREAD_ENV),
        },
    }
    _atomic_json(planned_path, planned)
    checkpoint_dir = output_dir / f"{stem}.checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    nested_indices = build_nested_indices(labels, budgets, seed=int(args.seed), classes=classes)
    all_rows: list[Dict[str, Any]] = []

    def report_progress(event: Mapping[str, Any], *, model_index: int, model: str) -> None:
        prefix = f"[{model_index + 1}/{len(models)}] {model}"
        event_name = str(event.get("event"))
        if event_name == "warmup_start":
            print(f"{prefix}: warmup start", flush=True)
        elif event_name == "warmup_end":
            print(f"{prefix}: warmup complete", flush=True)
        elif event_name == "cell":
            print(
                f"{prefix}: budget={int(event['samples_per_class'])} "
                f"repeat={int(event['repeat'])} "
                f"OI={float(event['overlap_elapsed_seconds']):.3f}s "
                f"probe={float(event['probe_elapsed_seconds']):.3f}s "
                f"speedup={float(event['ratio']):.2f}x",
                flush=True,
            )

    for model_index, model in enumerate(models):
        checkpoint_path = checkpoint_dir / f"{model}.json"
        checkpoint_rows: Optional[list[Dict[str, Any]]] = None
        if args.resume and checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if (
                    checkpoint.get("artifact_status") == "completed"
                    and checkpoint.get("configuration_hash") == config_hash
                    and checkpoint.get("model") == model
                    and isinstance(checkpoint.get("rows"), list)
                ):
                    checkpoint_rows = [dict(row) for row in checkpoint["rows"]]
                    _validate_complete_grid(
                        checkpoint_rows,
                        models=[model],
                        budgets=budgets,
                        repeats=int(args.timing_repeats),
                    )
                    _paired_summary(checkpoint_rows)
                    print(
                        f"[{model_index + 1}/{len(models)}] {model}: checkpoint reused",
                        flush=True,
                    )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                checkpoint_rows = None
        if checkpoint_rows is None:
            artifact = embeddings[model]
            matrix = (
                artifact.matrix
                if isinstance(artifact, EmbeddingManifest)
                else np.asarray(artifact)
            )
            checkpoint_rows = benchmark_embedding_matrix(
                matrix,
                labels,
                model=model,
                budgets=budgets,
                timing_repeats=int(args.timing_repeats),
                seed=int(args.seed),
                folds=_CROSS_FIT_FOLDS,
                k=_OVERLAP_K,
                nested_indices=nested_indices,
                warmup=True,
                model_index=model_index,
                progress=lambda event, index=model_index, name=model: report_progress(
                    event,
                    model_index=index,
                    model=name,
                ),
            )
            _atomic_json(
                checkpoint_path,
                {
                    "schema_version": _SCHEMA_VERSION,
                    "study": _STUDY,
                    "artifact_status": "completed",
                    "configuration_hash": config_hash,
                    "model": model,
                    "rows": checkpoint_rows,
                },
            )
            print(
                f"[{model_index + 1}/{len(models)}] {model}: checkpoint saved",
                flush=True,
            )
        all_rows.extend(checkpoint_rows)
    payload = run_runtime_scaling(
        {},
        labels,
        budgets=budgets,
        timing_repeats=int(args.timing_repeats),
        seed=int(args.seed),
        classes=classes,
        configuration=configuration,
        precomputed_rows=all_rows,
    )
    payload["configuration_hash"] = config_hash
    payload["configuration"] = configuration
    payload["artifact_stem"] = stem
    _atomic_json(completed_path, payload)
    _write_csv(output_dir / f"{stem}_rows.csv", payload["rows"])
    _write_csv(output_dir / f"{stem}_paired.csv", payload["paired_rows"])
    print(f"Completed Food-101 selector runtime scaling: {completed_path}")
    return 0


def _write_failed_artifact(args: argparse.Namespace, error: BaseException) -> None:
    try:
        output_dir = Path(getattr(args, "output_dir", Path("examples/output")))
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = int(getattr(args, "seed", 42))
        configuration = {
            "budgets": str(getattr(args, "budgets", "")),
            "models": str(getattr(args, "models", "")),
            "classes": str(getattr(args, "classes", "")),
            "timing_repeats": int(getattr(args, "timing_repeats", _TIMING_REPEATS)),
            "seed": seed,
            "synthetic_smoke": bool(getattr(args, "synthetic_smoke", False)),
        }
        digest = _configuration_hash(configuration)
        stem = f"{_ARTIFACT_STEM}_{digest[:12]}"
        path = output_dir / f"{stem}.failed.json"
        if path.exists():
            return
        _atomic_json(
            path,
            {
                "schema_version": _SCHEMA_VERSION,
                "protocol_version": _PROTOCOL_VERSION,
                "study": _STUDY,
                "artifact_status": "failed",
                "post_hoc_runtime_benchmark": True,
                "claim_supported": False,
                "configuration": configuration,
                "configuration_hash": digest,
                "error": f"{type(error).__name__}: {error}",
            },
        )
    except OSError:
        return


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the serial runtime benchmark and return a process exit code."""

    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        _write_failed_artifact(args, exc)
        print(f"Food-101 runtime scaling could not run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
