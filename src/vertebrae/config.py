"""Public configuration objects for vertebrae."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union


@dataclass
class OverlapScoringConfig:
    """Configuration for MiniBatchKMeans-backed OverlapIndex scoring."""

    k: Union[int, str, Dict[Any, int]] = "auto"
    min_k: int = 10
    max_k: int = 50
    min_samples_per_cluster: int = 5
    kmeans_kwargs: Optional[Dict[str, Any]] = None
    offline_chunk_size: Optional[int] = 10_000
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.k, str) and self.k != "auto":
            raise ValueError("OverlapScoringConfig.k must be an int, a class-to-k dict, or 'auto'.")
        if isinstance(self.k, int) and self.k < 1:
            raise ValueError("OverlapScoringConfig.k must be >= 1.")
        if self.min_k < 1:
            raise ValueError("min_k must be >= 1.")
        if self.max_k < 1:
            raise ValueError("max_k must be >= 1.")
        if self.min_k > self.max_k:
            raise ValueError("min_k must be <= max_k.")
        if self.min_samples_per_cluster < 1:
            raise ValueError("min_samples_per_cluster must be >= 1.")


@dataclass
class StabilityConfig:
    """Configuration for prototype or subsample stability analysis."""

    enabled: bool = True
    mode: str = "prototype"
    repeats: int = 20
    interval_level: float = 0.95
    subsample_fraction: float = 0.8
    stratified: bool = False
    random_state: int = 42

    def __post_init__(self) -> None:
        allowed_modes = {"prototype", "subsample", "none"}
        if self.mode not in allowed_modes:
            raise ValueError(f"StabilityConfig.mode must be one of {sorted(allowed_modes)}.")
        if self.repeats < 1:
            raise ValueError("StabilityConfig.repeats must be >= 1.")
        if not 0.0 < self.interval_level < 1.0:
            raise ValueError("interval_level must be between 0 and 1.")
        if not 0.0 < self.subsample_fraction <= 1.0:
            raise ValueError("subsample_fraction must be in (0, 1].")


@dataclass
class ProbeConfig:
    """Configuration for lightweight probe classifiers."""

    enabled: bool = True
    test_size: float = 0.2
    random_state: int = 42
    methods: Tuple[str, ...] = ("knn", "logistic_regression")

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("ProbeConfig.test_size must be between 0 and 1.")
        allowed_methods = {"knn", "logistic_regression", "nearest_centroid"}
        unknown = set(self.methods) - allowed_methods
        if unknown:
            raise ValueError(f"Unknown probe methods: {sorted(unknown)}.")


@dataclass
class CacheConfig:
    """Local embedding cache settings."""

    enabled: bool = True
    cache_dir: str = ".vertebrae_cache"
    force_recompute: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
