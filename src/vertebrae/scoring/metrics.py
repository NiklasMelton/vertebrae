"""Generic embedding metric interfaces and adapters."""

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from vertebrae.utils.serialization import make_json_safe


@dataclass
class MetricResult:
    """A normalized, rankable result returned by an embedding metric."""

    name: str
    score: float
    higher_is_better: bool = True
    kind: str = "custom"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MetricResult.name must be a non-empty string.")
        if not np.isfinite(float(self.score)):
            raise ValueError("MetricResult.score must be a finite numeric value.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this result to JSON-compatible data."""

        return make_json_safe(self)

    @property
    def macro_score(self) -> float:
        return float(self.diagnostics.get("macro_score", self.score))

    @property
    def weighted_score(self) -> Optional[float]:
        value = self.diagnostics.get("weighted_score")
        return None if value is None else float(value)

    @property
    def per_class_scores(self) -> Dict[Any, Any]:
        return dict(self.diagnostics.get("per_class_scores", {}))


@runtime_checkable
class EmbeddingMetric(Protocol):
    """Protocol for metrics that evaluate a full batch of embeddings."""

    name: str

    def score(
        self,
        embeddings: Any,
        labels: Any,
        *,
        target_metadata: Optional[Dict[str, Any]] = None,
        groups: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> MetricResult:
        """Return one aggregate result for an embedding matrix."""

    def recipe(self) -> Dict[str, Any]:
        """Return a JSON-compatible recipe when available."""


@dataclass
class CallableMetric:
    """Adapt a callable into an :class:`EmbeddingMetric`.

    The callable receives ``embeddings`` and ``labels`` plus any supported keyword
    arguments from ``target_metadata``, ``groups``, and ``seed``. It may return a
    ``MetricResult``, a mapping containing ``score``, or a numeric score.
    """

    name: str
    metric_fn: Callable[..., Any]
    config: Dict[str, Any] = field(default_factory=dict)
    higher_is_better: bool = True
    callable_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Metric names must be non-empty strings.")
        if not callable(self.metric_fn):
            raise TypeError("metric_fn must be callable.")

    def score(
        self,
        embeddings: Any,
        labels: Any,
        *,
        target_metadata: Optional[Dict[str, Any]] = None,
        groups: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> MetricResult:
        kwargs = {
            "target_metadata": target_metadata,
            "groups": groups,
            "seed": seed,
            "config": self.config,
        }
        raw = self.metric_fn(embeddings, labels, **_supported_kwargs(self.metric_fn, kwargs))
        return _coerce_metric_result(raw, self.name, self.higher_is_better)

    def recipe(self) -> Dict[str, Any]:
        """Return enough metadata to reconstruct importable metrics on workers."""

        path = self.callable_path or _callable_path(self.metric_fn)
        return {
            "name": self.name,
            "kind": "callable",
            "callable_path": path,
            "config": make_json_safe(self.config),
            "higher_is_better": self.higher_is_better,
            "portable": path is not None,
        }

    @classmethod
    def from_import_path(
        cls,
        path: str,
        *,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        higher_is_better: bool = True,
    ) -> "CallableMetric":
        """Build a portable callable metric from ``module:attribute`` syntax."""

        metric_fn = load_metric_callable(path)
        return cls(
            name=name or path.rsplit(":", 1)[-1],
            metric_fn=metric_fn,
            config=dict(config or {}),
            higher_is_better=higher_is_better,
            callable_path=path,
        )


@dataclass
class OverlapMetric:
    """Built-in metric adapter for OverlapIndex-family scoring."""

    config: Any = None
    name: str = "overlap"

    def score(
        self,
        embeddings: Any,
        labels: Any,
        *,
        target_metadata: Optional[Dict[str, Any]] = None,
        groups: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> MetricResult:
        """Run the internal OverlapIndex adapter through the generic metric API."""

        from vertebrae.scoring.overlap import OverlapIndexScorer

        metadata = target_metadata or {}
        overlap = OverlapIndexScorer(self.config).score(
            embeddings,
            labels,
            seed=seed,
            label_names=metadata.get("label_names"),
            target_type=metadata.get("target_type", "auto"),
            target_names=metadata.get("target_names"),
        )
        return MetricResult(
            name=self.name,
            score=float(overlap.score),
            kind="overlap_index",
            diagnostics={
                "macro_score": overlap.macro_score,
                "weighted_score": overlap.weighted_score,
                "per_class_scores": overlap.per_class_scores,
                "pairwise_scores": overlap.pairwise_scores,
                "sparse_adjacency": overlap.sparse_adjacency,
                "class_counts": overlap.class_counts,
                "k_per_class": overlap.k_per_class,
                "actual_loss": overlap.actual_loss,
                "null_loss": overlap.null_loss,
                "loss_ratio": overlap.loss_ratio,
                "prototype_scores": overlap.prototype_scores,
                "prototype_support": overlap.prototype_support,
                "prototype_target_summary": overlap.prototype_target_summary,
                "prototype_adjacency": overlap.prototype_adjacency,
            },
            warnings=list(overlap.warnings),
            metadata=dict(overlap.metadata),
        )

    def recipe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": "overlap_index", "config": make_json_safe(self.config)}

def as_embedding_metric(metric: Any) -> EmbeddingMetric:
    """Normalize a metric object or callable into the embedding metric protocol."""

    if isinstance(metric, EmbeddingMetric):
        return metric
    if callable(metric):
        name = getattr(metric, "__name__", metric.__class__.__name__)
        return CallableMetric(name=name, metric_fn=metric)
    raise TypeError("Metrics must be callable or implement EmbeddingMetric.score(...).")


def load_metric_callable(path: str) -> Callable[..., Any]:
    """Load an importable metric callable from ``module:attribute`` syntax."""

    module_name, separator, attribute_path = path.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError("Metric paths must use 'module:callable' syntax.")
    module = importlib.import_module(module_name)
    value: Any = module
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise TypeError(f"Metric path {path!r} did not resolve to a callable.")
    return value


def _coerce_metric_result(raw: Any, name: str, higher_is_better: bool) -> MetricResult:
    if isinstance(raw, MetricResult):
        if raw.name != name:
            raw.metadata = {**raw.metadata, "reported_name": raw.name}
            raw.name = name
        return raw
    if isinstance(raw, dict):
        if "score" not in raw:
            raise ValueError("Metric result mappings must include a numeric 'score' field.")
        return MetricResult(
            name=str(raw.get("name", name)),
            score=float(raw["score"]),
            higher_is_better=bool(raw.get("higher_is_better", higher_is_better)),
            kind=str(raw.get("kind", "custom")),
            diagnostics=dict(raw.get("diagnostics", {})),
            warnings=list(raw.get("warnings", [])),
            metadata=dict(raw.get("metadata", {})),
        )
    if isinstance(raw, (float, int, np.floating, np.integer)):
        return MetricResult(name=name, score=float(raw), higher_is_better=higher_is_better)
    raise TypeError("Metrics must return MetricResult, a mapping with 'score', or a numeric score.")


def _supported_kwargs(callable_obj: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in signature.parameters}


def _callable_path(callable_obj: Callable[..., Any]) -> Optional[str]:
    module = getattr(callable_obj, "__module__", None)
    qualname = getattr(callable_obj, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname or "<lambda>" in qualname:
        return None
    return f"{module}:{qualname}"
