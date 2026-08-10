"""Internal OverlapIndex adapters."""

import inspect
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from vertebrae.config import ContinuousOverlapScoringConfig, OverlapScoringConfig
from vertebrae.utils.labels import (
    MULTI_LABEL_TARGET,
    REGRESSION_TARGET,
    SINGLE_LABEL_TARGET,
    class_counts,
    display_label,
    multilabel_indicator,
    normalize_targets,
    target_summary,
)
from vertebrae.utils.semantic_labels import (
    semantic_label_catalog,
    semantic_label_key,
    semantic_label_keys,
)
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import (
    ensure_numeric_matrix,
    is_sparse_matrix,
    l2_normalize_rows,
)


@dataclass
class OverlapScoreResult:
    """Structured result from OverlapIndex or ContinuousOverlapIndex scoring."""

    score: float
    macro_score: float
    weighted_score: Optional[float] = None
    per_class_scores: Dict[Any, Any] = field(default_factory=dict)
    pairwise_scores: Dict[Any, Any] = field(default_factory=dict)
    sparse_adjacency: Any = None
    class_counts: Dict[Any, int] = field(default_factory=dict)
    k_per_class: Dict[Any, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    actual_loss: Optional[float] = None
    null_loss: Optional[float] = None
    loss_ratio: Optional[float] = None
    prototype_scores: Dict[Any, Any] = field(default_factory=dict)
    prototype_support: Dict[Any, Any] = field(default_factory=dict)
    prototype_target_summary: Dict[Any, Any] = field(default_factory=dict)
    prototype_adjacency: Dict[Any, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the score result to a JSON-safe dictionary."""

        return make_json_safe(self)


def auto_k_for_class(
    n_class: int,
    min_k: int = 10,
    max_k: int = 50,
    min_samples_per_cluster: int = 5,
) -> int:
    """Resolve automatic k for a single class."""

    if n_class < 2:
        raise ValueError("Each class must contain at least 2 samples.")
    upper = max(1, n_class // min_samples_per_cluster)
    return int(min(max_k, upper, max(2, min_k, int(n_class**0.5))))


def resolve_kmeans_k(
    y: Any,
    config: OverlapScoringConfig,
    return_warnings: bool = False,
    label_names: Optional[Any] = None,
) -> Union[Dict[Any, int], Tuple[Dict[Any, int], List[str]]]:
    """Resolve MiniBatchKMeans k values per class."""

    counts = class_counts(y, label_names=label_names)
    original_by_key: Dict[str, Any] = {}
    source_labels = label_names if label_names is not None else np.asarray(y, dtype=object).tolist()
    for original in source_labels:
        original_by_key.setdefault(semantic_label_key(original), original)
    warnings: List[str] = []
    k_per_class: Dict[Any, int] = {}

    for label, count in counts.items():
        if count < 1:
            warnings.append(
                "Skipped k resolution for label "
                f"{display_label(label)} because it has no samples in this scoring target."
            )
            continue
        if isinstance(config.k, Integral) and not isinstance(config.k, bool):
            requested = int(config.k)
        elif isinstance(config.k, dict):
            requested = _lookup_class_k(
                config.k,
                label,
                original_label=original_by_key.get(str(label), label),
            )
        else:
            requested = auto_k_for_class(
                count,
                min_k=config.min_k,
                max_k=config.max_k,
                min_samples_per_cluster=config.min_samples_per_cluster,
            )

        if requested < 1:
            raise ValueError(f"k for class {display_label(label)} must be >= 1.")
        max_allowed = max(1, count // config.min_samples_per_cluster)
        resolved = min(int(requested), int(config.max_k), int(max_allowed), int(count))
        if resolved < requested:
            warnings.append(
                "Reduced k for class "
                f"{display_label(label)} from {requested} to {resolved} "
                f"because the class has {count} samples."
            )
        if isinstance(config.k, str) and config.k == "auto" and resolved < config.min_k:
            warnings.append(
                "Auto k for class "
                f"{display_label(label)} resolved below min_k ({config.min_k}) "
                f"because the class has {count} samples."
            )
        k_per_class[label] = int(resolved)

    if return_warnings:
        return k_per_class, warnings
    return k_per_class


class OverlapIndexScorer:
    """Internal adapter for MiniBatchKMeans-backed overlap scoring."""

    def __init__(
        self,
        config: Optional[Union[OverlapScoringConfig, ContinuousOverlapScoringConfig]] = None,
    ) -> None:
        self.config = config or OverlapScoringConfig()

    def score(
        self,
        Z: Any,
        y: Any,
        seed: Optional[int] = None,
        label_names: Optional[Any] = None,
        target_type: str = "auto",
        target_names: Optional[Any] = None,
        label_catalog: Optional[Any] = None,
    ) -> OverlapScoreResult:
        """Score dense or sparse embeddings with OverlapIndex-family backends."""

        embeddings = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
        normalized_labels, label_metadata = normalize_targets(
            y,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        if label_metadata["target_type"] == MULTI_LABEL_TARGET:
            labels = multilabel_indicator(
                normalized_labels,
                label_metadata["label_names"],
                sparse_output=True,
            )
        elif label_metadata["target_type"] == REGRESSION_TARGET:
            labels = np.asarray(normalized_labels, dtype=float)
        else:
            labels = normalized_labels
        if label_metadata["target_type"] != REGRESSION_TARGET:
            label_metadata["label_catalog"] = list(
                label_catalog or label_metadata.get("label_catalog") or []
            )
        label_rows = int(labels.shape[0])
        if embeddings.shape[0] != label_rows:
            raise ValueError(
                "embeddings and labels must have the same length; "
                f"got {embeddings.shape[0]} and {label_rows}."
            )
        if label_metadata["target_type"] == REGRESSION_TARGET:
            return self._score_regression(embeddings, labels, label_metadata, seed=seed)
        return self._score_classification(
            embeddings,
            labels,
            normalized_labels,
            label_metadata,
            seed=seed,
        )

    def score_cross_fitted(
        self,
        Z: Any,
        y: Any,
        *,
        n_splits: int = 5,
        seed: Optional[int] = None,
        label_names: Optional[Any] = None,
        label_catalog: Optional[Any] = None,
    ) -> OverlapScoreResult:
        """Score held-out rows against fold-fitted classification prototypes.

        Each stratified fold fits a fresh MiniBatchKMeans-backed OverlapIndex on
        its training rows and evaluates overlap events on its held-out rows via
        the upstream ``score_fixed`` API. Fold macro scores and diagnostics are
        then averaged. This method intentionally supports single-label
        classification only; regression and multi-label targets require
        different fold aggregation semantics.
        """
        if isinstance(n_splits, bool) or not isinstance(n_splits, Integral):
            raise TypeError("n_splits must be an integer >= 2.")
        if int(n_splits) < 2:
            raise ValueError("n_splits must be >= 2.")

        embeddings = ensure_numeric_matrix(Z, "embeddings", allow_sparse=True)
        normalized_labels, label_metadata = normalize_targets(
            y,
            label_names=label_names,
            target_type="auto",
        )
        if label_metadata["target_type"] != SINGLE_LABEL_TARGET:
            raise ValueError(
                "score_cross_fitted supports single-label classification only."
            )
        labels = np.asarray(semantic_label_keys(normalized_labels.tolist()), dtype=object)
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                "embeddings and labels must have the same length; "
                f"got {embeddings.shape[0]} and {labels.shape[0]}."
            )
        if labels.shape[0] == 0:
            raise ValueError(
                "score_cross_fitted requires at least one sample; got empty inputs."
            )

        counts = class_counts(normalized_labels)
        smallest_class = min(counts.values())
        if smallest_class < int(n_splits):
            raise ValueError(
                "Each class must contain at least n_splits samples for stratified "
                f"cross-fitting; smallest class has {smallest_class}, "
                f"n_splits={int(n_splits)}."
            )

        config = _coerce_classification_config(self.config)
        sparse_input = is_sparse_matrix(embeddings)
        if config.normalize_embeddings:
            embeddings = l2_normalize_rows(embeddings)

        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=seed,
        )
        OverlapIndex = _load_overlap_index()
        fold_rows: List[Dict[str, Any]] = []
        fold_scores: List[float] = []
        fold_weighted_scores: List[float] = []
        per_class_values: Dict[Any, List[float]] = {}
        pairwise_values: Dict[Any, List[float]] = {}
        sparse_adjacency: Dict[Any, int] = {}
        fold_k_values: List[Dict[Any, int]] = []
        warnings: List[str] = []

        for fold, (train_indices, holdout_indices) in enumerate(
            splitter.split(np.zeros(labels.shape[0], dtype=np.uint8), labels)
        ):
            fold_seed = None if seed is None else int(seed) + int(fold)
            train_labels = labels[train_indices]
            train_original_labels = normalized_labels[train_indices]
            k_per_class, k_warnings = resolve_kmeans_k(
                train_original_labels,
                config,
                return_warnings=True,
            )
            warnings.extend(f"Fold {fold}: {message}" for message in k_warnings)
            fold_k_values.append(k_per_class)
            kmeans_kwargs = dict(config.kmeans_kwargs or {})
            if fold_seed is not None:
                kmeans_kwargs["random_state"] = fold_seed
            backend_excluded_classes = config.exclude_classes
            if backend_excluded_classes is not None:
                backend_excluded_classes = _semantic_excluded_classes(
                    backend_excluded_classes
                )
            index = _instantiate_with_supported_kwargs(
                OverlapIndex,
                {
                    "model_type": "MiniBatchKMeans",
                    "kmeans_k": k_per_class,
                    "kmeans_kwargs": kmeans_kwargs,
                    "offline_chunk_size": config.offline_chunk_size,
                    "exclude_classes": backend_excluded_classes,
                },
            )
            if not hasattr(index, "score_fixed"):
                raise RuntimeError(
                    "Cross-fitted overlap requires an OverlapIndex version exposing "
                    "score_fixed(X, y). Install an OverlapIndex version that provides "
                    "this API."
                )
            fold_warnings: List[str] = []
            with _capture_runtime_warnings(fold_warnings):
                index.fit(embeddings[train_indices], train_labels)
                raw_score = index.score_fixed(
                    embeddings[holdout_indices],
                    labels[holdout_indices],
                )
            warnings.extend(f"Fold {fold}: {message}" for message in fold_warnings)
            macro_score = _extract_macro_score(index, raw_score)
            weighted_score = _extract_optional_score(
                getattr(index, "weighted_index", None)
            )
            fold_scores.append(macro_score)
            fold_weighted_scores.append(
                macro_score if weighted_score is None else weighted_score
            )
            for label, value in getattr(index, "singleton_index", {}).items():
                per_class_values.setdefault(label, []).append(float(value))
            for pair, value in getattr(index, "pairwise_index", {}).items():
                if np.isfinite(value):
                    pairwise_values.setdefault(pair, []).append(float(value))
            for pair, value in getattr(index, "sparse_adj", {}).items():
                sparse_adjacency[pair] = sparse_adjacency.get(pair, 0) + int(value)
            fold_rows.append(
                {
                    "fold": int(fold),
                    "train_size": int(train_indices.size),
                    "holdout_size": int(holdout_indices.size),
                    "seed": fold_seed,
                    "score": macro_score,
                    "weighted_score": weighted_score,
                    "k_per_class": make_json_safe(k_per_class),
                    "warnings": fold_warnings,
                }
            )

        resolved_k = {
            label: min(int(fold_k[label]) for fold_k in fold_k_values)
            for label in fold_k_values[0]
        }
        catalog = list(label_catalog or label_metadata.get("label_catalog") or [])
        if not catalog:
            catalog = semantic_label_catalog(normalized_labels.tolist())
        excluded_class_keys = semantic_label_keys(
            _normalized_excluded_classes(config.exclude_classes)
        )
        excluded_key_set = set(excluded_class_keys)
        observed_classes = list(target_summary(normalized_labels)["class_counts"])
        included_classes = [
            label for label in observed_classes if label not in excluded_key_set
        ]
        return OverlapScoreResult(
            score=float(np.mean(fold_scores)),
            macro_score=float(np.mean(fold_scores)),
            weighted_score=float(np.mean(fold_weighted_scores)),
            per_class_scores=make_json_safe(
                {
                    label: float(np.mean(values))
                    for label, values in per_class_values.items()
                }
            ),
            pairwise_scores=make_json_safe(
                {
                    pair: float(np.mean(values))
                    for pair, values in pairwise_values.items()
                }
            ),
            sparse_adjacency=make_json_safe(sparse_adjacency),
            class_counts=counts,
            k_per_class=resolved_k,
            warnings=warnings,
            metadata={
                "backend": "MiniBatchKMeans",
                "score_kind": "classification_overlap_cross_fitted",
                "score_label": "overlap_macro",
                "normalize_embeddings": config.normalize_embeddings,
                "offline_chunk_size": config.offline_chunk_size,
                "seed": seed,
                "sparse_input": sparse_input,
                "scoring_input_format": "csr" if sparse_input else "dense",
                "target_type": label_metadata["target_type"],
                "label_catalog": catalog,
                "exclude_classes": excluded_class_keys,
                "aggregation_classes": make_json_safe(included_classes),
                "aggregate_valid": bool(included_classes),
                "cross_fit": {
                    "n_splits": int(n_splits),
                    "aggregation": "mean_fold_macro",
                    "folds": fold_rows,
                },
            },
        )

    def _score_classification(
        self,
        embeddings: Any,
        labels: Any,
        original_labels: np.ndarray,
        label_metadata: Dict[str, Any],
        seed: Optional[int],
    ) -> OverlapScoreResult:
        config = _coerce_classification_config(self.config)
        warnings: List[str] = []
        if label_metadata["target_type"] != MULTI_LABEL_TARGET:
            labels = np.asarray(semantic_label_keys(labels.tolist()), dtype=object)
        sparse_input = is_sparse_matrix(embeddings)
        if config.normalize_embeddings:
            embeddings = l2_normalize_rows(embeddings)

        k_per_class, k_warnings = resolve_kmeans_k(
            original_labels,
            config,
            return_warnings=True,
            label_names=label_metadata.get("label_names"),
        )
        warnings.extend(k_warnings)
        kmeans_kwargs = dict(config.kmeans_kwargs or {})
        if seed is not None:
            kmeans_kwargs["random_state"] = seed

        backend_k = k_per_class
        if label_metadata["target_type"] == MULTI_LABEL_TARGET:
            # OverlapIndex expands indicator targets using their column indices.
            # Keep semantic label names in vertebrae metadata, but key the backend
            # mapping by those expanded integer labels.
            label_names = tuple(label_metadata.get("label_names") or ())
            backend_k = {
                index: k_per_class[semantic_label_key(label)]
                for index, label in enumerate(label_names)
                if semantic_label_key(label) in k_per_class
            }

        backend_excluded_classes = config.exclude_classes
        if config.exclude_classes is not None:
            if label_metadata["target_type"] == MULTI_LABEL_TARGET:
                backend_excluded_classes = _multilabel_excluded_indices(
                    config.exclude_classes,
                    tuple(label_metadata.get("label_names") or ()),
                )
            else:
                backend_excluded_classes = _semantic_excluded_classes(config.exclude_classes)

        OverlapIndex = _load_overlap_index()
        index = _instantiate_with_supported_kwargs(
            OverlapIndex,
            {
                "model_type": "MiniBatchKMeans",
                "kmeans_k": backend_k,
                "kmeans_kwargs": kmeans_kwargs,
                "offline_chunk_size": config.offline_chunk_size,
                "exclude_classes": backend_excluded_classes,
            },
        )
        with _capture_runtime_warnings(warnings):
            raw_score = index.fit_offline(embeddings, labels, reset_state=True)
        macro_score = _extract_macro_score(index, raw_score)
        weighted_score = _extract_optional_score(getattr(index, "weighted_index", None))
        summary = target_summary(
            original_labels,
            label_names=label_metadata.get("label_names"),
            target_type=label_metadata["target_type"],
        )
        catalog = list(label_metadata.get("label_catalog") or [])
        if not catalog:
            if label_metadata["target_type"] == MULTI_LABEL_TARGET:
                catalog = semantic_label_catalog(label_metadata.get("label_names") or ())
            else:
                catalog = semantic_label_catalog(original_labels.tolist())
        summary["label_catalog"] = catalog
        excluded_classes = _normalized_excluded_classes(config.exclude_classes)
        excluded_class_keys = semantic_label_keys(excluded_classes)
        observed_classes = list(summary["class_counts"])
        excluded_keys = set(excluded_class_keys)
        included_classes = [label for label in observed_classes if label not in excluded_keys]
        aggregate_valid = bool(included_classes)
        per_class_scores = getattr(index, "singleton_index", {})
        pairwise_scores = getattr(index, "pairwise_index", {})
        if label_metadata["target_type"] == MULTI_LABEL_TARGET:
            per_class_scores, pairwise_scores = _restore_multilabel_names(
                per_class_scores,
                pairwise_scores,
                tuple(label_metadata.get("label_names") or ()),
            )

        metadata = {
            "backend": "MiniBatchKMeans",
            "score_kind": "classification_overlap",
            "score_label": "overlap_macro",
            "normalize_embeddings": config.normalize_embeddings,
            "offline_chunk_size": config.offline_chunk_size,
            "seed": seed,
            "kmeans_kwargs": kmeans_kwargs,
            "sparse_input": sparse_input,
            "scoring_input_format": "csr" if sparse_input else "dense",
            "target_type": label_metadata["target_type"],
            "label_names": label_metadata.get("label_names"),
            "label_catalog": catalog,
            "label_display_by_key": {
                item["key"]: item.get("report_display", item.get("display", item["key"]))
                for item in catalog
            },
            "target_summary": summary,
            "exclude_classes": excluded_class_keys,
            "aggregation_classes": make_json_safe(included_classes),
            "aggregate_valid": aggregate_valid,
        }
        return OverlapScoreResult(
            score=macro_score,
            macro_score=macro_score,
            weighted_score=weighted_score,
            per_class_scores=make_json_safe(per_class_scores),
            pairwise_scores=make_json_safe(pairwise_scores),
            sparse_adjacency=make_json_safe(getattr(index, "sparse_adj", None)),
            class_counts=summary["class_counts"],
            k_per_class=k_per_class,
            warnings=warnings,
            metadata=metadata,
        )

    def _score_regression(
        self,
        embeddings: Any,
        labels: np.ndarray,
        label_metadata: Dict[str, Any],
        seed: Optional[int],
    ) -> OverlapScoreResult:
        config = _coerce_continuous_config(self.config)
        warnings: List[str] = []
        sparse_input = is_sparse_matrix(embeddings)
        if config.normalize_embeddings:
            embeddings = l2_normalize_rows(embeddings)
        if (
            isinstance(self.config, OverlapScoringConfig)
            and self.config.exclude_classes is not None
        ):
            raise ValueError("exclude_classes is not supported for regression overlap scoring.")

        kmeans_kwargs = dict(config.kmeans_kwargs or {})
        if seed is not None:
            kmeans_kwargs["random_state"] = seed

        ContinuousOverlapIndex = _load_continuous_overlap_index()
        index = ContinuousOverlapIndex(
            model_type="MiniBatchKMeans",
            kmeans_k=config.k,
            kmeans_kwargs=kmeans_kwargs,
            offline_chunk_size=config.offline_chunk_size,
            target_cover=config.target_cover,
            n_target_cells=config.n_target_cells,
            target_cover_kwargs=config.target_cover_kwargs,
            target_distance=config.target_distance,
            target_scaling=config.target_scaling,
            n_projections=config.n_projections,
            n_null_permutations=config.n_null_permutations,
            aggregation=config.aggregation,
            random_state=seed,
        )
        with _capture_runtime_warnings(warnings):
            raw_score = index.fit_offline(embeddings, labels, reset_state=True)

        score = _extract_required_score(getattr(index, "index", raw_score))
        macro_score = _extract_optional_score(getattr(index, "macro_index_", None))
        if macro_score is None:
            macro_score = score
        weighted_score = _extract_optional_score(getattr(index, "weighted_index", None))
        if weighted_score is None:
            weighted_score = score
        summary = target_summary(
            labels,
            target_type=label_metadata["target_type"],
            target_names=label_metadata.get("target_names"),
        )

        metadata = {
            "backend": "MiniBatchKMeans",
            "score_kind": "continuous_overlap",
            "score_label": "overlap_score",
            "normalize_embeddings": config.normalize_embeddings,
            "offline_chunk_size": config.offline_chunk_size,
            "seed": seed,
            "kmeans_kwargs": kmeans_kwargs,
            "sparse_input": sparse_input,
            "scoring_input_format": "csr" if sparse_input else "dense",
            "target_type": label_metadata["target_type"],
            "target_names": label_metadata.get("target_names"),
            "target_summary": summary,
            "aggregate_valid": True,
            "continuous_null_reference": 0.0,
            "continuous_aggregation": config.aggregation,
            "target_cover": config.target_cover,
            "n_target_cells": config.n_target_cells,
            "target_distance": config.target_distance,
            "target_scaling": config.target_scaling,
            "n_projections": config.n_projections,
            "n_null_permutations": config.n_null_permutations,
        }
        return OverlapScoreResult(
            score=score,
            macro_score=macro_score,
            weighted_score=weighted_score,
            sparse_adjacency=make_json_safe(
                getattr(index, "prototype_adjacency_normalized_", None)
                or getattr(index, "prototype_adjacency_", None)
            ),
            warnings=warnings,
            metadata=metadata,
            actual_loss=_extract_optional_score(getattr(index, "actual_loss_", None)),
            null_loss=_extract_optional_score(getattr(index, "null_loss_", None)),
            loss_ratio=_extract_optional_score(getattr(index, "loss_ratio_", None)),
            prototype_scores=make_json_safe(getattr(index, "prototype_index_", {})),
            prototype_support=make_json_safe(getattr(index, "prototype_support_", {})),
            prototype_target_summary=make_json_safe(_prototype_target_summary(index)),
            prototype_adjacency=make_json_safe(
                getattr(index, "prototype_adjacency_normalized_", None)
                or getattr(index, "prototype_adjacency_", {})
            ),
        )


def _lookup_class_k(
    configured: Dict[Any, int],
    label: Any,
    *,
    original_label: Any = None,
) -> int:
    if label in configured:
        return int(configured[label])
    label_value = label.item() if hasattr(label, "item") else label
    if label_value in configured:
        return int(configured[label_value])
    label_str = str(label_value)
    if label_str in configured:
        return int(configured[label_str])
    expected_key = semantic_label_key(original_label)
    for configured_label, value in configured.items():
        if semantic_label_key(configured_label) == expected_key:
            return int(value)
    raise ValueError(f"Missing k value for class {display_label(label)}.")


def _restore_multilabel_names(
    per_class_scores: Dict[Any, Any],
    pairwise_scores: Dict[Any, Any],
    label_names: Tuple[Any, ...],
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    def restore(label: Any) -> Any:
        value = label.item() if hasattr(label, "item") else label
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, int) and 0 <= value < len(label_names):
            return semantic_label_key(label_names[value])
        return value

    restored_per_class = {restore(label): score for label, score in per_class_scores.items()}
    restored_pairwise = {}
    for pair, score in pairwise_scores.items():
        if isinstance(pair, tuple) and len(pair) == 2:
            restored_pairwise[(restore(pair[0]), restore(pair[1]))] = score
        else:
            restored_pairwise[pair] = score
    return restored_per_class, restored_pairwise


def _load_overlap_index() -> Any:
    try:
        from overlapindex import OverlapIndex
    except ImportError as exc:
        raise ImportError(
            "overlapindex>=0.1.3a4 is required for scoring. Install dependencies with "
            "Poetry or install overlapindex directly."
        ) from exc
    return OverlapIndex


def _load_continuous_overlap_index() -> Any:
    try:
        from overlapindex import ContinuousOverlapIndex
    except ImportError as exc:
        raise ImportError(
            "overlapindex>=0.1.3a4 is required for regression scoring. Install "
            "dependencies with Poetry or install overlapindex directly."
        ) from exc
    return ContinuousOverlapIndex


def _instantiate_with_supported_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return cls(**kwargs)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return cls(**supported)


def _coerce_classification_config(
    config: Union[OverlapScoringConfig, ContinuousOverlapScoringConfig],
) -> OverlapScoringConfig:
    if isinstance(config, OverlapScoringConfig):
        return config
    return OverlapScoringConfig(
        k=config.k,
        kmeans_kwargs=config.kmeans_kwargs,
        offline_chunk_size=config.offline_chunk_size,
        normalize_embeddings=config.normalize_embeddings,
        max_dense_bytes=config.max_dense_bytes,
    )


def _coerce_continuous_config(
    config: Union[OverlapScoringConfig, ContinuousOverlapScoringConfig],
) -> ContinuousOverlapScoringConfig:
    if isinstance(config, ContinuousOverlapScoringConfig):
        return config
    resolved_k = config.k if isinstance(config.k, int) else 8
    return ContinuousOverlapScoringConfig(
        k=resolved_k,
        kmeans_kwargs=config.kmeans_kwargs,
        offline_chunk_size=config.offline_chunk_size,
        normalize_embeddings=config.normalize_embeddings,
        max_dense_bytes=config.max_dense_bytes,
    )


def _prototype_target_summary(index: Any) -> Dict[Any, Any]:
    summary: Dict[Any, Any] = {}
    means = getattr(index, "prototype_target_mean_", {})
    radii = getattr(index, "prototype_target_radius_", {})
    supports = getattr(index, "prototype_support_", {})
    values = getattr(index, "prototype_target_values_", {})
    for key in set(means) | set(radii) | set(supports) | set(values):
        target_values = values.get(key)
        n_target_values = (
            int(np.asarray(target_values).shape[0]) if target_values is not None else None
        )
        summary[key] = {
            "mean": make_json_safe(means.get(key)),
            "radius": make_json_safe(radii.get(key)),
            "support": make_json_safe(supports.get(key)),
            "n_target_values": n_target_values,
        }
    return summary


def _extract_macro_score(index: Any, raw_score: Any) -> float:
    candidates = [getattr(index, "index", None), raw_score]
    for candidate in candidates:
        if isinstance(candidate, (int, float, np.number)):
            return float(candidate)
    raise ValueError("OverlapIndex did not return a numeric global score.")


def _extract_required_score(value: Any) -> float:
    if isinstance(value, (int, float, np.number)):
        return float(value)
    raise ValueError("ContinuousOverlapIndex did not return a numeric score.")


def _extract_optional_score(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return None


def _normalized_excluded_classes(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    unique: List[Any] = []
    keys = set()
    for item in values:
        key = semantic_label_key(item)
        if key not in keys:
            unique.append(item)
            keys.add(key)
    return unique


def _semantic_excluded_classes(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        return semantic_label_key(value)
    try:
        values = list(value)
    except TypeError:
        return semantic_label_key(value)
    return semantic_label_keys(values)


def _multilabel_excluded_indices(value: Any, label_names: Tuple[Any, ...]) -> List[int]:
    excluded = _normalized_excluded_classes(value)
    excluded_keys = {semantic_label_key(item) for item in excluded}
    positions = {semantic_label_key(label): index for index, label in enumerate(label_names)}
    unknown = excluded_keys - set(positions)
    if unknown:
        missing = [item for item in excluded if semantic_label_key(item) in unknown]
        raise ValueError(
            "exclude_classes contains labels absent from the multi-label target: " f"{missing!r}."
        )
    return [positions[key] for key in positions if key in excluded_keys]


class _capture_runtime_warnings:
    def __init__(self, output: List[str]) -> None:
        self.output = output
        self._context: Any = None
        self._records: Any = None

    def __enter__(self) -> None:
        import warnings

        self._context = warnings.catch_warnings(record=True)
        self._records = self._context.__enter__()
        warnings.simplefilter("always", RuntimeWarning)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._context.__exit__(exc_type, exc, traceback)
        self.output.extend(str(record.message) for record in self._records)
