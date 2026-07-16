"""Exact, frozen prompt-prototype scoring for zero-shot alignment."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from vertebrae.cache.fingerprint import hash_json_exact
from vertebrae.config import ZeroShotConfig
from vertebrae.utils.semantic_labels import (
    LABEL_ENCODING,
    portable_json,
    semantic_label_catalog,
    semantic_label_key,
)
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


@dataclass
class ZeroShotScoreResult:
    """Metrics and diagnostics for one fixed zero-shot prompt protocol."""

    score: float
    primary_metric: str
    metrics: Dict[str, float]
    per_class: Dict[Any, Dict[str, float]]
    confusion_matrix: List[List[int]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        catalog = list(
            self.metadata.get("label_catalog")
            or semantic_label_catalog(self.metadata.get("class_labels", self.per_class))
        )
        known_keys = {str(item["key"]) for item in catalog}

        def label_key(value: Any) -> str:
            if isinstance(value, str) and value in known_keys:
                return value
            return semantic_label_key(value)

        diagnostics = dict(self.diagnostics)
        coherence = diagnostics.get("prompt_coherence")
        if isinstance(coherence, dict):
            diagnostics["prompt_coherence"] = {
                label_key(label): portable_json(metrics) for label, metrics in coherence.items()
            }
        worst_samples = diagnostics.get("worst_samples")
        if isinstance(worst_samples, list):
            diagnostics["worst_samples"] = [
                {
                    **portable_json(
                        {
                            key: item
                            for key, item in sample.items()
                            if key not in {"sample_id", "label", "prediction"}
                        }
                    ),
                    "sample_id": portable_json(sample.get("sample_id")),
                    "label": label_key(sample.get("label")),
                    "prediction": label_key(sample.get("prediction")),
                }
                for sample in worst_samples
            ]
        metadata = {
            **self.metadata,
            "label_encoding": LABEL_ENCODING,
            "label_catalog": catalog,
            "class_labels": [label_key(label) for label in self.metadata.get("class_labels", [])],
        }
        return {
            "score": float(self.score),
            "primary_metric": self.primary_metric,
            "metrics": portable_json(self.metrics),
            "per_class": {
                label_key(label): portable_json(metrics)
                for label, metrics in self.per_class.items()
            },
            "confusion_matrix": portable_json(self.confusion_matrix),
            "diagnostics": portable_json(diagnostics),
            "warnings": list(self.warnings),
            "metadata": portable_json(metadata),
        }


@dataclass
class _BlockwiseScores:
    predicted_indices: np.ndarray
    correct_scores: np.ndarray
    best_incorrect_scores: np.ndarray
    top_k_hits: Dict[int, int]
    tied: int


class ZeroShotScorer:
    """Score frozen samples against fixed text prompt prototypes.

    Prompt rows belonging to one class are normalized, averaged with equal weight,
    then normalized once more.  No fitted calibration or prompt selection occurs.
    """

    def __init__(self, config: Optional[ZeroShotConfig] = None) -> None:
        self.config = config or ZeroShotConfig()

    def score(
        self,
        sample_embeddings: Any,
        prompt_embeddings: Any,
        labels: Any,
        *,
        class_labels: Sequence[Any],
        prompt_labels: Sequence[Any],
        template_ids: Optional[Sequence[str]] = None,
        sample_ids: Optional[Sequence[Any]] = None,
        labels_are_semantic_keys: bool = False,
    ) -> ZeroShotScoreResult:
        """Evaluate exact sample-to-class prototype matching."""

        sample_matrix = ensure_numeric_matrix(
            sample_embeddings, "sample embeddings", allow_sparse=True
        )
        prompt_matrix = ensure_numeric_matrix(
            prompt_embeddings, "prompt embeddings", allow_sparse=True
        )
        raw_target = np.asarray(labels, dtype=object)
        if raw_target.ndim != 1:
            raise ValueError("Zero-shot labels must be a one-dimensional single-label sequence.")
        raw_classes = tuple(class_labels)
        raw_prompt_target = tuple(prompt_labels)
        if not isinstance(labels_are_semantic_keys, bool):
            raise TypeError("labels_are_semantic_keys must be a bool.")
        if labels_are_semantic_keys:
            _validate_semantic_keys(raw_target.tolist(), "labels")
            _validate_semantic_keys(raw_classes, "class_labels")
            _validate_semantic_keys(raw_prompt_target, "prompt_labels")
            target = raw_target
            classes = raw_classes
            prompt_target = raw_prompt_target
            label_catalog: List[Dict[str, Any]] = []
        else:
            _validate_semantic_labels(raw_target.tolist(), "labels")
            _validate_semantic_labels(raw_classes, "class_labels")
            _validate_semantic_labels(raw_prompt_target, "prompt_labels")
            target = np.asarray(
                [semantic_label_key(label) for label in raw_target.tolist()], dtype=object
            )
            classes = tuple(semantic_label_key(label) for label in raw_classes)
            prompt_target = tuple(semantic_label_key(label) for label in raw_prompt_target)
            label_catalog = semantic_label_catalog(raw_classes)
        if sample_matrix.shape[0] != len(target):
            raise ValueError("sample embeddings and labels must have the same number of rows.")
        if sample_matrix.shape[1] != prompt_matrix.shape[1]:
            raise ValueError("sample and prompt embedding dimensions must match.")
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("class_labels must contain at least two unique labels.")
        if set(target.tolist()) != set(classes):
            raise ValueError("class_labels must contain exactly the observed sample labels.")
        if len(prompt_target) != prompt_matrix.shape[0] or set(prompt_target) != set(classes):
            raise ValueError("prompt_labels must align with prompts and cover every class exactly.")
        template_values: Optional[Tuple[str, ...]] = None
        if template_ids is not None:
            if len(template_ids) != prompt_matrix.shape[0]:
                raise ValueError("template_ids must align one-to-one with prompt embeddings.")
            if any(
                not isinstance(identifier, str) or not identifier.strip()
                for identifier in template_ids
            ):
                raise ValueError("template_ids must contain non-empty strings.")
            template_values = tuple(template_ids)
            for template in dict.fromkeys(template_values):
                covered = [
                    label
                    for label, identifier in zip(prompt_target, template_values)
                    if identifier == template
                ]
                expected = list(classes)
                if len(covered) != len(expected) or set(covered) != set(expected):
                    raise ValueError(
                        "Every template_id must occur exactly once for every declared class."
                    )
        if sample_ids is not None and len(sample_ids) != sample_matrix.shape[0]:
            raise ValueError("sample_ids must align one-to-one with sample embeddings.")
        required_bytes = _working_set_bytes(
            sample_matrix.shape,
            prompt_matrix.shape,
            len(classes),
            self.config.sample_batch_size,
        )
        if required_bytes > self.config.max_dense_bytes:
            raise MemoryError(
                "Zero-shot scoring requires an estimated "
                f"{required_bytes} dense working-set bytes for its float64 endpoints, "
                "prototypes, result accumulators, and one configured score block, exceeding "
                f"ZeroShotConfig.max_dense_bytes={self.config.max_dense_bytes}. Lower "
                "sample_batch_size, compress the embeddings, or increase max_dense_bytes."
            )
        samples = _owned_float64(sample_matrix)
        prompts = _owned_float64(prompt_matrix)
        _assert_nonzero_rows(prompts, "prompt embeddings")
        _normalize_rows_in_place(prompts)
        if self.config.similarity == "cosine":
            _assert_nonzero_rows(samples, "sample embeddings")
            _normalize_rows_in_place(samples)

        prototypes, coherence = _prototypes(prompts, prompt_target, classes)
        class_positions = {label: index for index, label in enumerate(classes)}
        target_indices = np.asarray([class_positions[label] for label in target], dtype=int)
        blockwise = _score_blockwise(
            samples,
            prototypes,
            target_indices,
            self.config.similarity,
            self.config.top_k,
            self.config.sample_batch_size,
        )
        predicted = np.asarray(
            [classes[index] for index in blockwise.predicted_indices], dtype=object
        )
        metrics, per_class, matrix = _classification_metrics(
            target_indices,
            blockwise.predicted_indices,
            classes,
            blockwise.top_k_hits,
            self.config.top_k,
        )
        display_target = target
        display_predicted = predicted
        display_classes: Sequence[Any] = classes
        if not labels_are_semantic_keys and len(set(raw_classes)) == len(raw_classes):
            display_classes = raw_classes
            display_target = raw_target
            display_predicted = np.asarray(
                [raw_classes[index] for index in blockwise.predicted_indices], dtype=object
            )
            per_class = {
                raw_classes[index]: per_class[label] for index, label in enumerate(classes)
            }
            coherence = {
                raw_classes[index]: coherence[label] for index, label in enumerate(classes)
            }
        warnings = []
        skipped_top_k = [k for k in self.config.top_k if k > len(classes)]
        if skipped_top_k:
            warnings.append(
                "Skipped Top-K metrics with K larger than the number of declared classes: "
                f"{skipped_top_k}."
            )
        if blockwise.tied:
            warnings.append(
                f"{blockwise.tied} sample(s) had an exact top-score tie; "
                "declared class order broke ties."
            )
        margins = blockwise.correct_scores - blockwise.best_incorrect_scores
        diagnostics: Dict[str, Any] = {
            "correct_similarity": _summary(blockwise.correct_scores),
            "best_incorrect_similarity": _summary(blockwise.best_incorrect_scores),
            "correct_class_margin": _summary(margins),
            "prompt_coherence": coherence,
            "n_top_score_ties": blockwise.tied,
            "n_samples": len(samples),
            "n_classes": len(classes),
            "n_prompts": len(prompts),
            "worst_samples": _worst_samples(
                margins,
                display_target,
                display_predicted,
                sample_ids,
                self.config.worst_samples,
            ),
        }
        del blockwise, margins, predicted, prototypes
        if template_values is not None:
            unique_templates = tuple(dict.fromkeys(template_values))
            diagnostics["per_template_metrics"] = _template_metrics(
                samples,
                prompts,
                target,
                classes,
                prompt_target,
                template_values,
                unique_templates,
                self.config,
            )
        return ZeroShotScoreResult(
            score=float(metrics[self.config.primary_metric]),
            primary_metric=self.config.primary_metric,
            metrics=metrics,
            per_class=per_class,
            confusion_matrix=matrix,
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={
                "config": asdict(self.config),
                "class_labels": list(display_classes),
                "label_encoding": LABEL_ENCODING,
                "label_catalog": label_catalog,
                "prompt_aggregation": "normalized_mean",
                "similarity": self.config.similarity,
                "exact": True,
            },
        )


def _owned_float64(matrix: Any) -> np.ndarray:
    if is_sparse_matrix(matrix):
        return matrix.astype(np.float64, copy=False).toarray()
    return np.array(matrix, dtype=np.float64, copy=True)


def _working_set_bytes(
    sample_shape: Tuple[int, int],
    prompt_shape: Tuple[int, int],
    n_classes: int,
    sample_batch_size: int,
) -> int:
    """Estimate the scorer-owned dense working set before allocating it."""

    n_samples, embedding_dim = sample_shape
    n_prompts = prompt_shape[0]
    batch_rows = min(sample_batch_size, n_samples)
    float_bytes = np.dtype(np.float64).itemsize
    endpoints = (n_samples + n_prompts) * embedding_dim * float_bytes
    prototypes = n_classes * embedding_dim * float_bytes
    # Target/prediction indices, three diagnostic vectors, margins, and one ordering buffer.
    accumulators = n_samples * 7 * float_bytes
    # One float64 score block plus conservative rank/tie comparison workspace.
    score_workspace = batch_rows * n_classes * (float_bytes + 8)
    return int(endpoints + prototypes + accumulators + score_workspace)


def _assert_nonzero_rows(value: np.ndarray, name: str) -> None:
    if np.any(np.linalg.norm(value, axis=1) == 0.0):
        raise ValueError(f"{name} must not contain zero-norm rows for zero-shot scoring.")


def _prototypes(
    prompts: np.ndarray,
    prompt_labels: Sequence[Any],
    classes: Sequence[Any],
    row_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[Any, Dict[str, float]]]:
    prototypes = []
    coherence: Dict[Any, Dict[str, float]] = {}
    for label in classes:
        indices = [
            index
            for index, prompt_label in enumerate(prompt_labels)
            if prompt_label == label and (row_mask is None or bool(row_mask[index]))
        ]
        if not indices:
            raise ValueError(f"No prompt rows are available for class {label!r}.")
        prototype = np.zeros((1, prompts.shape[1]), dtype=np.float64)
        for index in indices:
            prototype[0] += prompts[index]
        prototype /= float(len(indices))
        _assert_nonzero_rows(prototype, f"prompt prototype for class {label!r}")
        _normalize_rows_in_place(prototype)
        prototype_row = prototype[0]
        prototypes.append(prototype_row)
        similarities = np.asarray(
            [float(prompts[index] @ prototype_row) for index in indices], dtype=np.float64
        )
        coherence[label] = {
            "n_prompts": float(len(indices)),
            "mean_similarity": float(np.mean(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
        }
    return np.vstack(prototypes), coherence


def _similarities(samples: np.ndarray, prototypes: np.ndarray, similarity: str) -> np.ndarray:
    if similarity in {"cosine", "dot"}:
        return samples @ prototypes.T
    sample_norms = np.einsum("ij,ij->i", samples, samples)[:, None]
    prototype_norms = np.einsum("ij,ij->i", prototypes, prototypes)[None, :]
    scores = samples @ prototypes.T
    scores *= 2.0
    scores -= sample_norms
    scores -= prototype_norms
    return scores


def _score_blockwise(
    samples: np.ndarray,
    prototypes: np.ndarray,
    target_indices: np.ndarray,
    similarity: str,
    top_k: Sequence[int],
    sample_batch_size: int,
) -> _BlockwiseScores:
    n_samples = len(samples)
    predicted = np.empty(n_samples, dtype=np.int64)
    correct = np.empty(n_samples, dtype=np.float64)
    best_incorrect = np.empty(n_samples, dtype=np.float64)
    valid_top_k = tuple(k for k in top_k if k <= len(prototypes))
    top_k_hits = {k: 0 for k in valid_top_k}
    class_indices = np.arange(len(prototypes), dtype=np.int64)[None, :]
    tied = 0
    for start in range(0, n_samples, sample_batch_size):
        stop = min(start + sample_batch_size, n_samples)
        scores = _similarities(samples[start:stop], prototypes, similarity)
        local_rows = np.arange(stop - start)
        local_targets = target_indices[start:stop]
        maximum = np.max(scores, axis=1, keepdims=True)
        tied += int(np.count_nonzero(np.sum(scores == maximum, axis=1) > 1))
        predicted[start:stop] = np.argmax(scores, axis=1)
        local_correct = scores[local_rows, local_targets]
        correct[start:stop] = local_correct
        better = scores > local_correct[:, None]
        tied_before = (scores == local_correct[:, None]) & (class_indices < local_targets[:, None])
        ranks = 1 + np.sum(better | tied_before, axis=1)
        for k in valid_top_k:
            top_k_hits[k] += int(np.count_nonzero(ranks <= k))
        scores[local_rows, local_targets] = -np.inf
        best_incorrect[start:stop] = np.max(scores, axis=1)
    return _BlockwiseScores(predicted, correct, best_incorrect, top_k_hits, tied)


def _classification_metrics(
    target_indices: np.ndarray,
    predicted_indices: np.ndarray,
    classes: Sequence[Any],
    top_k_hits: Dict[int, int],
    top_k: Sequence[int],
) -> Tuple[Dict[str, float], Dict[Any, Dict[str, float]], List[List[int]]]:
    encoded_classes = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        target_indices,
        predicted_indices,
        labels=encoded_classes,
        average=None,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(target_indices, predicted_indices)),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(target_indices, predicted_indices)),
        "weighted_f1": float(
            precision_recall_fscore_support(
                target_indices,
                predicted_indices,
                labels=encoded_classes,
                average="weighted",
                zero_division=0,
            )[2]
        ),
    }
    for k in top_k:
        if k <= len(classes):
            metrics[f"top_k_accuracy@{k}"] = float(top_k_hits[k] / len(target_indices))
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
        }
        for index, label in enumerate(classes)
    }
    matrix = (
        confusion_matrix(target_indices, predicted_indices, labels=encoded_classes)
        .astype(int)
        .tolist()
    )
    return metrics, per_class, matrix


def _template_metrics(
    samples: np.ndarray,
    prompts: np.ndarray,
    target: np.ndarray,
    classes: Sequence[Any],
    prompt_labels: Sequence[Any],
    template_ids: Sequence[str],
    templates: Sequence[str],
    config: ZeroShotConfig,
) -> Dict[str, Dict[str, float]]:
    results = {}
    template_array = np.asarray(template_ids, dtype=object)
    for template in templates:
        mask = template_array == template
        prototypes, _ = _prototypes(prompts, prompt_labels, classes, row_mask=mask)
        class_positions = {label: index for index, label in enumerate(classes)}
        target_indices = np.asarray([class_positions[label] for label in target], dtype=int)
        blockwise = _score_blockwise(
            samples,
            prototypes,
            target_indices,
            config.similarity,
            config.top_k,
            config.sample_batch_size,
        )
        metrics, _per_class, _matrix = _classification_metrics(
            target_indices,
            blockwise.predicted_indices,
            classes,
            blockwise.top_k_hits,
            config.top_k,
        )
        results[template] = metrics
    return results


def _normalize_rows_in_place(value: np.ndarray) -> None:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    value /= norms


def _summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _worst_samples(
    margins: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sample_ids: Optional[Sequence[Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    if not limit:
        return []
    ids = list(sample_ids) if sample_ids is not None else list(range(len(margins)))
    indices = np.argsort(margins, kind="stable")[:limit]
    return [
        {
            "sample_id": ids[int(index)],
            "label": target[int(index)],
            "prediction": predicted[int(index)],
            "margin": float(margins[int(index)]),
        }
        for index in indices
    ]


def _validate_semantic_labels(values: Sequence[Any], name: str) -> None:
    for value in values:
        try:
            hash(value)
            hash_json_exact(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} entries must be hashable and have stable semantic identities."
            ) from exc


def _validate_semantic_keys(values: Sequence[Any], name: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} entries must be non-empty semantic-label key strings.")
