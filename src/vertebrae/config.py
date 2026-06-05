"""Public configuration objects for vertebrae."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

from vertebrae.execution.jobs import ShardSpec


@dataclass
class OverlapScoringConfig:
    """Configuration for MiniBatchKMeans-backed OverlapIndex scoring.

    Attributes:
        k: Global k, class-to-k mapping, or `"auto"` for per-class resolution.
        min_k: Lower target used by automatic k resolution.
        max_k: Maximum k allowed for any class.
        min_samples_per_cluster: Minimum class samples expected per prototype.
        kmeans_kwargs: Extra keyword arguments passed to MiniBatchKMeans.
        offline_chunk_size: Chunk size forwarded to `OverlapIndex.fit_offline`.
        normalize_embeddings: Whether to L2-normalize embeddings before scoring.
        max_dense_bytes: Maximum sparse-to-dense allocation allowed for scoring.
    """

    k: Union[int, str, Dict[Any, int]] = "auto"
    min_k: int = 10
    max_k: int = 50
    min_samples_per_cluster: int = 5
    kmeans_kwargs: Optional[Dict[str, Any]] = None
    offline_chunk_size: Optional[int] = 10_000
    normalize_embeddings: bool = True
    max_dense_bytes: int = 2_000_000_000

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
        if self.max_dense_bytes < 1:
            raise ValueError("max_dense_bytes must be >= 1.")


@dataclass
class StabilityConfig:
    """Configuration for prototype or subsample stability analysis.

    Attributes:
        enabled: Whether stability analysis should run.
        mode: Stability mode: `"prototype"`, `"subsample"`, or `"none"`.
        repeats: Number of repeated scoring runs.
        interval_level: Percentile interval level reported in summaries.
        subsample_fraction: Fraction sampled for subsample stability.
        stratified: Whether subsampling should preserve class membership.
        random_state: Seed used to generate repeat seeds and subsamples.
    """

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
    """Configuration for lightweight probe classifiers.

    Attributes:
        enabled: Whether probe evaluation should run.
        test_size: Fraction of samples held out for probe testing.
        random_state: Seed used for train/test splitting.
        methods: Probe classifiers to evaluate.
    """

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
    """Artifact cache settings.

    Attributes:
        enabled: Whether embedding artifacts should be cached.
        cache_dir: Local cache directory or artifact-store URI.
        force_recompute: Whether to ignore cache hits and recompute embeddings.
        storage_options: Provider-specific store options such as S3 endpoint URLs.
        metadata: Additional user metadata to preserve with cache settings.
    """

    enabled: bool = True
    cache_dir: str = ".vertebrae_cache"
    force_recompute: bool = False
    storage_options: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation and materialization.

    Attributes:
        batch_size: Number of samples to pass to streaming-safe extractors.
        streaming_enabled: Whether streaming-safe extractors should be embedded
            batch-by-batch instead of in one full transform call.
        shard: Optional deterministic shard assignment. Local benchmarking requires
            a complete shard, but this field is available for future distributed
            embedding jobs.
    """

    batch_size: int = 128
    streaming_enabled: bool = True
    shard: Optional[ShardSpec] = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("EmbeddingConfig.batch_size must be >= 1.")


@dataclass
class MemoryConfig:
    """Memory budget and admission-control settings.

    Attributes:
        max_memory_bytes: Explicit process memory budget. When `None`, the budget
            is derived from current available memory using `psutil`.
        reserve_system_bytes: Memory to leave available for the OS and other
            processes. When `None`, a conservative reserve is selected.
        max_fraction: Maximum fraction of currently available memory to use.
        allow_disk_spill: Whether large embedding artifacts may be streamed to disk.
        fail_fast: Whether to raise before expensive work when estimates exceed
            the configured budget.
        probe_batch_size: Number of samples used to infer unknown embedding shape.
        model_memory_bytes: Optional model memory hint included in admission checks.
        raw_batch_memory_bytes: Optional raw batch memory hint included in checks.
        subsample_rate: Fraction of samples to keep before embedding and scoring.
            A value of `1.0` disables user-requested subsampling.
        subsample_random_state: Seed used for deterministic stratified subsampling.
        min_subsample_samples_per_class: Minimum retained samples per class when
            class sizes allow it.
        auto_subsample_on_memory_exceeded: Whether to warn and select the largest
            fitting stratified subsample instead of raising when a full embedding
            artifact would exceed memory.
    """

    max_memory_bytes: Optional[int] = None
    reserve_system_bytes: Optional[int] = None
    max_fraction: float = 0.75
    allow_disk_spill: bool = True
    fail_fast: bool = True
    probe_batch_size: int = 4
    model_memory_bytes: int = 0
    raw_batch_memory_bytes: int = 0
    subsample_rate: float = 1.0
    subsample_random_state: int = 42
    min_subsample_samples_per_class: int = 2
    auto_subsample_on_memory_exceeded: bool = True

    def __post_init__(self) -> None:
        if self.max_memory_bytes is not None and self.max_memory_bytes < 1:
            raise ValueError("MemoryConfig.max_memory_bytes must be >= 1.")
        if self.reserve_system_bytes is not None and self.reserve_system_bytes < 0:
            raise ValueError("MemoryConfig.reserve_system_bytes must be >= 0.")
        if not 0.0 < self.max_fraction <= 1.0:
            raise ValueError("MemoryConfig.max_fraction must be in (0, 1].")
        if self.probe_batch_size < 1:
            raise ValueError("MemoryConfig.probe_batch_size must be >= 1.")
        if self.model_memory_bytes < 0:
            raise ValueError("MemoryConfig.model_memory_bytes must be >= 0.")
        if self.raw_batch_memory_bytes < 0:
            raise ValueError("MemoryConfig.raw_batch_memory_bytes must be >= 0.")
        if not 0.0 < self.subsample_rate <= 1.0:
            raise ValueError("MemoryConfig.subsample_rate must be in (0, 1].")
        if self.min_subsample_samples_per_class < 1:
            raise ValueError("MemoryConfig.min_subsample_samples_per_class must be >= 1.")
