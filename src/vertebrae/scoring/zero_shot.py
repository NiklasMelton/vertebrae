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
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix, sparse_to_dense


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
                label_key(label): portable_json(metrics)
                for label, metrics in coherence.items()
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
    ) -> ZeroShotScoreResult:
        """Evaluate exact sample-to-class prototype matching."""

        samples = _dense(sample_embeddings, "sample embeddings", self.config.max_dense_bytes)
        prompts = _dense(prompt_embeddings, "prompt embeddings", self.config.max_dense_bytes)
        target = np.asarray(labels, dtype=object)
        if target.ndim != 1:
            raise ValueError("Zero-shot labels must be a one-dimensional single-label sequence.")
        classes = tuple(class_labels)
        prompt_target = tuple(prompt_labels)
        _validate_semantic_labels(target.tolist(), "labels")
        _validate_semantic_labels(classes, "class_labels")
        _validate_semantic_labels(prompt_target, "prompt_labels")
        if samples.shape[0] != len(target):
            raise ValueError("sample embeddings and labels must have the same number of rows.")
        if samples.shape[1] != prompts.shape[1]:
            raise ValueError("sample and prompt embedding dimensions must match.")
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("class_labels must contain at least two unique labels.")
        if set(target.tolist()) != set(classes):
            raise ValueError("class_labels must contain exactly the observed sample labels.")
        if len(prompt_target) != len(prompts) or set(prompt_target) != set(classes):
            raise ValueError("prompt_labels must align with prompts and cover every class exactly.")
        if template_ids is not None and len(template_ids) != len(prompts):
            raise ValueError("template_ids must align one-to-one with prompt embeddings.")
        if sample_ids is not None and len(sample_ids) != len(samples):
            raise ValueError("sample_ids must align one-to-one with sample embeddings.")
        _assert_nonzero_rows(samples, "sample embeddings")
        _assert_nonzero_rows(prompts, "prompt embeddings")

        prototypes, coherence = _prototypes(prompts, prompt_target, classes)
        scores = _similarities(samples, prototypes, self.config.similarity)
        predicted_indices, tied = _predictions(scores)
        predicted = np.asarray([classes[index] for index in predicted_indices], dtype=object)
        class_positions = {label: index for index, label in enumerate(classes)}
        target_indices = np.asarray([class_positions[label] for label in target], dtype=int)
        metrics, per_class, matrix = _classification_metrics(
            target_indices, predicted_indices, classes, scores, self.config.top_k
        )
        warnings = []
        skipped_top_k = [k for k in self.config.top_k if k > len(classes)]
        if skipped_top_k:
            warnings.append(
                "Skipped Top-K metrics with K larger than the number of declared classes: "
                f"{skipped_top_k}."
            )
        if tied:
            warnings.append(
                f"{tied} sample(s) had an exact top-score tie; declared class order broke ties."
            )
        correct_scores = scores[np.arange(len(scores)), target_indices]
        incorrect = scores.copy()
        incorrect[np.arange(len(scores)), target_indices] = -np.inf
        best_incorrect = np.max(incorrect, axis=1)
        margins = correct_scores - best_incorrect
        diagnostics: Dict[str, Any] = {
            "correct_similarity": _summary(correct_scores),
            "best_incorrect_similarity": _summary(best_incorrect),
            "correct_class_margin": _summary(margins),
            "prompt_coherence": coherence,
            "n_top_score_ties": tied,
            "n_samples": len(samples),
            "n_classes": len(classes),
            "n_prompts": len(prompts),
            "worst_samples": _worst_samples(
                margins,
                target,
                predicted,
                sample_ids,
                self.config.worst_samples,
            ),
        }
        if template_ids is not None:
            template_values = tuple(template_ids)
            unique_templates = tuple(dict.fromkeys(template_values))
            if all(
                template_values.count(template) == len(classes) for template in unique_templates
            ):
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
                "class_labels": list(classes),
                "label_encoding": LABEL_ENCODING,
                "label_catalog": semantic_label_catalog(classes),
                "prompt_aggregation": "normalized_mean",
                "similarity": self.config.similarity,
                "exact": True,
            },
        )


def _dense(value: Any, name: str, max_dense_bytes: int) -> np.ndarray:
    matrix = ensure_numeric_matrix(value, name, allow_sparse=True)
    if is_sparse_matrix(matrix):
        matrix = sparse_to_dense(matrix, name, max_dense_bytes)
    return np.asarray(matrix, dtype=np.float64)


def _assert_nonzero_rows(value: np.ndarray, name: str) -> None:
    if np.any(np.linalg.norm(value, axis=1) == 0.0):
        raise ValueError(f"{name} must not contain zero-norm rows for zero-shot scoring.")


def _prototypes(
    prompts: np.ndarray, prompt_labels: Sequence[Any], classes: Sequence[Any]
) -> Tuple[np.ndarray, Dict[Any, Dict[str, float]]]:
    normalized = _normalize_rows(prompts)
    prototypes = []
    coherence: Dict[Any, Dict[str, float]] = {}
    labels = np.asarray(prompt_labels, dtype=object)
    for label in classes:
        rows = normalized[labels == label]
        prototype = np.mean(rows, axis=0, keepdims=True)
        _assert_nonzero_rows(prototype, f"prompt prototype for class {label!r}")
        prototype = _normalize_rows(prototype)[0]
        prototypes.append(prototype)
        similarities = rows @ prototype
        coherence[label] = {
            "n_prompts": float(len(rows)),
            "mean_similarity": float(np.mean(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
        }
    return np.vstack(prototypes), coherence


def _similarities(samples: np.ndarray, prototypes: np.ndarray, similarity: str) -> np.ndarray:
    if similarity == "cosine":
        return _normalize_rows(samples) @ prototypes.T
    if similarity == "dot":
        return samples @ prototypes.T
    sample_norms = np.einsum("ij,ij->i", samples, samples)[:, None]
    prototype_norms = np.einsum("ij,ij->i", prototypes, prototypes)[None, :]
    return -(sample_norms + prototype_norms - 2.0 * (samples @ prototypes.T))


def _predictions(scores: np.ndarray) -> Tuple[np.ndarray, int]:
    maximum = np.max(scores, axis=1, keepdims=True)
    ties = int(np.count_nonzero(np.sum(scores == maximum, axis=1) > 1))
    return np.argmax(scores, axis=1), ties


def _classification_metrics(
    target_indices: np.ndarray,
    predicted_indices: np.ndarray,
    classes: Sequence[Any],
    scores: np.ndarray,
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
            top_indices = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            metrics[f"top_k_accuracy@{k}"] = float(
                np.mean(np.any(top_indices == target_indices[:, None], axis=1))
            )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
        }
        for index, label in enumerate(classes)
    }
    matrix = confusion_matrix(
        target_indices, predicted_indices, labels=encoded_classes
    ).astype(int).tolist()
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
    label_array = np.asarray(prompt_labels, dtype=object)
    for template in templates:
        mask = template_array == template
        prototypes, _ = _prototypes(prompts[mask], label_array[mask], classes)
        scores = _similarities(samples, prototypes, config.similarity)
        class_positions = {label: index for index, label in enumerate(classes)}
        target_indices = np.asarray([class_positions[label] for label in target], dtype=int)
        predicted_indices = np.argmax(scores, axis=1)
        metrics, _per_class, _matrix = _classification_metrics(
            target_indices, predicted_indices, classes, scores, config.top_k
        )
        results[template] = metrics
    return results


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return value / norms


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
