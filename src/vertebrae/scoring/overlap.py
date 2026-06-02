"""Internal OverlapIndex adapter."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from vertebrae.config import OverlapScoringConfig
from vertebrae.utils.labels import class_counts, display_label
from vertebrae.utils.serialization import make_json_safe
from vertebrae.utils.validation import ensure_2d_numeric_array, l2_normalize_rows


@dataclass
class OverlapScoreResult:
    macro_score: float
    per_class_scores: Dict[Any, Any] = field(default_factory=dict)
    pairwise_scores: Dict[Any, Any] = field(default_factory=dict)
    sparse_adjacency: Any = None
    class_counts: Dict[Any, int] = field(default_factory=dict)
    k_per_class: Dict[Any, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return make_json_safe(self)


def auto_k_for_class(
    n_class: int,
    min_k: int = 10,
    max_k: int = 50,
    min_samples_per_cluster: int = 5,
) -> int:
    if n_class < 2:
        raise ValueError("Each class must contain at least 2 samples.")
    upper = max(1, n_class // min_samples_per_cluster)
    return int(min(max_k, upper, max(2, min_k, int(n_class**0.5))))


def resolve_kmeans_k(
    y: Any,
    config: OverlapScoringConfig,
    return_warnings: bool = False,
) -> Union[Dict[Any, int], Tuple[Dict[Any, int], List[str]]]:
    y_arr = np.asarray(y)
    counts = class_counts(y_arr)
    warnings: List[str] = []
    k_per_class: Dict[Any, int] = {}

    for label, count in counts.items():
        if isinstance(config.k, int):
            requested = config.k
        elif isinstance(config.k, dict):
            requested = _lookup_class_k(config.k, label)
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
    """The only internal adapter allowed to instantiate overlapindex.OverlapIndex."""

    def __init__(self, config: Optional[OverlapScoringConfig] = None) -> None:
        self.config = config or OverlapScoringConfig()

    def score(self, Z: Any, y: Any, seed: Optional[int] = None) -> OverlapScoreResult:
        embeddings = ensure_2d_numeric_array(Z, "embeddings")
        labels = np.asarray(y)
        if len(embeddings) != len(labels):
            raise ValueError(
                "embeddings and labels must have the same length; "
                f"got {len(embeddings)} and {len(labels)}."
            )
        if self.config.normalize_embeddings:
            embeddings = l2_normalize_rows(embeddings)

        k_per_class, warnings = resolve_kmeans_k(labels, self.config, return_warnings=True)
        kmeans_kwargs = dict(self.config.kmeans_kwargs or {})
        if seed is not None:
            kmeans_kwargs["random_state"] = seed

        OverlapIndex = _load_overlap_index()
        index = OverlapIndex(
            model_type="MiniBatchKMeans",
            kmeans_k=k_per_class,
            kmeans_kwargs=kmeans_kwargs,
            offline_chunk_size=self.config.offline_chunk_size,
        )
        raw_score = index.fit_offline(embeddings, labels, reset_state=True)
        macro_score = _extract_macro_score(index, raw_score)

        metadata = {
            "backend": "MiniBatchKMeans",
            "normalize_embeddings": self.config.normalize_embeddings,
            "offline_chunk_size": self.config.offline_chunk_size,
            "seed": seed,
            "kmeans_kwargs": kmeans_kwargs,
        }
        return OverlapScoreResult(
            macro_score=macro_score,
            per_class_scores=make_json_safe(getattr(index, "singleton_index", {})),
            pairwise_scores=make_json_safe(getattr(index, "pairwise_index", {})),
            sparse_adjacency=make_json_safe(getattr(index, "sparse_adj", None)),
            class_counts=class_counts(labels),
            k_per_class=k_per_class,
            warnings=warnings,
            metadata=metadata,
        )


def _lookup_class_k(configured: Dict[Any, int], label: Any) -> int:
    if label in configured:
        return int(configured[label])
    label_value = label.item() if hasattr(label, "item") else label
    if label_value in configured:
        return int(configured[label_value])
    label_str = str(label_value)
    if label_str in configured:
        return int(configured[label_str])
    raise ValueError(f"Missing k value for class {display_label(label)}.")


def _load_overlap_index() -> Any:
    try:
        from overlapindex import OverlapIndex
    except ImportError as exc:
        raise ImportError(
            "overlapindex>=0.1.1 is required for scoring. Install dependencies with Poetry "
            "or install overlapindex directly."
        ) from exc
    return OverlapIndex


def _extract_macro_score(index: Any, raw_score: Any) -> float:
    candidates = [getattr(index, "index", None), raw_score]
    for candidate in candidates:
        if isinstance(candidate, (int, float, np.number)):
            return float(candidate)
    raise ValueError("OverlapIndex did not return a numeric global score.")
