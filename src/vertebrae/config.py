"""Public configuration objects for vertebrae."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union


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
        exclude_classes: Reporting-only classes omitted from global aggregation.
    """

    k: Union[int, str, Dict[Any, int]] = "auto"
    min_k: int = 10
    max_k: int = 50
    min_samples_per_cluster: int = 5
    kmeans_kwargs: Optional[Dict[str, Any]] = None
    offline_chunk_size: Optional[int] = 10_000
    normalize_embeddings: bool = True
    max_dense_bytes: int = 2_000_000_000
    exclude_classes: Any = None

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
        _validate_excluded_classes(self.exclude_classes)


@dataclass
class ContinuousOverlapScoringConfig:
    """Configuration for MiniBatchKMeans-backed ContinuousOverlapIndex scoring."""

    k: int = 8
    kmeans_kwargs: Optional[Dict[str, Any]] = None
    offline_chunk_size: Optional[int] = 10_000
    normalize_embeddings: bool = True
    max_dense_bytes: int = 2_000_000_000
    target_cover: str = "auto"
    n_target_cells: Union[int, str] = "auto"
    target_cover_kwargs: Optional[Dict[str, Any]] = None
    target_distance: str = "auto"
    target_scaling: str = "standard"
    n_projections: int = 64
    n_null_permutations: int = 20
    aggregation: str = "support_weighted"
    clip: bool = True

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("ContinuousOverlapScoringConfig.k must be >= 1.")
        if self.offline_chunk_size is not None and self.offline_chunk_size < 1:
            raise ValueError("offline_chunk_size must be >= 1 when provided.")
        if self.max_dense_bytes < 1:
            raise ValueError("max_dense_bytes must be >= 1.")
        allowed_target_cover = {"auto", "quantile", "kmeans"}
        if self.target_cover not in allowed_target_cover:
            raise ValueError(f"target_cover must be one of {sorted(allowed_target_cover)}.")
        if isinstance(self.n_target_cells, str) and self.n_target_cells != "auto":
            raise ValueError("n_target_cells must be an int or 'auto'.")
        if isinstance(self.n_target_cells, int) and self.n_target_cells < 1:
            raise ValueError("n_target_cells must be >= 1.")
        allowed_target_distance = {"auto", "wasserstein", "sliced_wasserstein"}
        if self.target_distance not in allowed_target_distance:
            raise ValueError(f"target_distance must be one of {sorted(allowed_target_distance)}.")
        allowed_target_scaling = {"standard", "none", "minmax", "robust"}
        if self.target_scaling not in allowed_target_scaling:
            raise ValueError(f"target_scaling must be one of {sorted(allowed_target_scaling)}.")
        if self.n_projections < 1:
            raise ValueError("n_projections must be >= 1.")
        if self.n_null_permutations < 1:
            raise ValueError("n_null_permutations must be >= 1.")
        allowed_aggregation = {"support_weighted", "macro"}
        if self.aggregation not in allowed_aggregation:
            raise ValueError(f"aggregation must be one of {sorted(allowed_aggregation)}.")


@dataclass
class RetrievalConfig:
    """Configuration for exact, training-free query--gallery retrieval scoring."""

    similarity: str = "cosine"
    ks: Tuple[int, ...] = (1, 5, 10)
    primary_metric: str = "ndcg@10"
    bidirectional: bool = False
    query_batch_size: int = 128
    gallery_batch_size: int = 10_000
    max_dense_bytes: int = 2_000_000_000
    max_pairwise_comparisons: Optional[int] = None
    worst_queries: int = 10

    def __post_init__(self) -> None:
        if self.similarity not in {"cosine", "dot", "squared_l2"}:
            raise ValueError("similarity must be one of: cosine, dot, squared_l2.")
        if not self.ks or any(not isinstance(k, int) or k < 1 for k in self.ks):
            raise ValueError("ks must contain one or more positive integers.")
        if len(set(self.ks)) != len(self.ks):
            raise ValueError("ks must not contain duplicate cutoffs.")
        if self.primary_metric not in {f"ndcg@{k}" for k in self.ks}:
            raise ValueError("primary_metric must be an ndcg@K entry present in ks.")
        if self.query_batch_size < 1 or self.gallery_batch_size < 1:
            raise ValueError("query_batch_size and gallery_batch_size must be >= 1.")
        if self.max_dense_bytes < 1:
            raise ValueError("max_dense_bytes must be >= 1.")
        if self.max_pairwise_comparisons is not None and self.max_pairwise_comparisons < 1:
            raise ValueError("max_pairwise_comparisons must be >= 1 when provided.")
        if self.worst_queries < 0:
            raise ValueError("worst_queries must be >= 0.")


@dataclass
class ZeroShotConfig:
    """Configuration for exact, training-free zero-shot semantic alignment.

    Zero-shot evaluation compares frozen sample embeddings with frozen text prompt
    prototypes.  It intentionally has no learned calibration, prompt search, or
    fitted classification head.
    """

    similarity: str = "cosine"
    top_k: Tuple[int, ...] = (1, 5)
    primary_metric: str = "accuracy"
    max_dense_bytes: int = 2_000_000_000
    worst_samples: int = 10

    def __post_init__(self) -> None:
        if self.similarity not in {"cosine", "dot", "squared_l2"}:
            raise ValueError("similarity must be one of: cosine, dot, squared_l2.")
        if not self.top_k or any(not isinstance(k, int) or k < 1 for k in self.top_k):
            raise ValueError("top_k must contain one or more positive integers.")
        if len(set(self.top_k)) != len(self.top_k):
            raise ValueError("top_k must not contain duplicate cutoffs.")
        allowed_metrics = {"accuracy", "macro_f1", "balanced_accuracy"}
        if self.primary_metric not in allowed_metrics:
            raise ValueError(f"primary_metric must be one of {sorted(allowed_metrics)}.")
        if self.max_dense_bytes < 1:
            raise ValueError("max_dense_bytes must be >= 1.")
        if self.worst_samples < 0:
            raise ValueError("worst_samples must be >= 0.")


@dataclass
class SeparatixConfig:
    """Configuration for optional Separatix complexity diagnostics.

    Attributes:
        enabled: Whether Separatix diagnostics should run.
        overlap_threshold: Minimum classification overlap score required to run.
        regression_overlap_threshold: Minimum regression overlap score required to run.
        random_state: Seed forwarded to Separatix.
        budget: Optional Separatix diagnostic budget.
        max_samples: Optional Separatix sample cap.
        max_dense_bytes: Optional sparse densification memory limit in bytes.
        n_jobs: Optional parallelism hint forwarded to Separatix.
        mlp_probes: Whether conditional Separatix MLP probes should be enabled.
        mlp_device: Requested Separatix MLP execution device.
        mlp_trigger_skill_threshold: Skill threshold that suppresses MLP execution.
        mlp_min_improvement: Minimum improvement required for an MLP override.
        mlp_max_parameters: Optional hard cap for MLP parameter count.
    """

    enabled: bool = True
    overlap_threshold: float = 0.80
    regression_overlap_threshold: float = 0.80
    random_state: Optional[int] = 42
    budget: Optional[str] = None
    max_samples: Optional[int] = None
    max_dense_bytes: Optional[int] = None
    n_jobs: Optional[int] = None
    mlp_probes: bool = False
    mlp_device: str = "cpu"
    mlp_trigger_skill_threshold: float = 0.75
    mlp_min_improvement: float = 0.02
    mlp_max_parameters: Optional[int] = None

    def __post_init__(self) -> None:
        allowed_budgets = {"fast", "standard", "extended"}
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("SeparatixConfig.overlap_threshold must be between 0 and 1.")
        if not 0.0 <= self.regression_overlap_threshold <= 1.0:
            raise ValueError(
                "SeparatixConfig.regression_overlap_threshold must be between 0 and 1."
            )
        if self.budget is not None and self.budget not in allowed_budgets:
            raise ValueError(f"SeparatixConfig.budget must be one of {sorted(allowed_budgets)}.")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("SeparatixConfig.max_samples must be >= 1 when provided.")
        if self.max_dense_bytes is not None and self.max_dense_bytes < 1:
            raise ValueError("SeparatixConfig.max_dense_bytes must be >= 1 when provided.")
        if self.n_jobs is not None and self.n_jobs < 1:
            raise ValueError("SeparatixConfig.n_jobs must be >= 1 when provided.")
        allowed_devices = {"cpu", "auto", "cuda", "mps"}
        if self.mlp_device not in allowed_devices:
            raise ValueError(
                f"SeparatixConfig.mlp_device must be one of {sorted(allowed_devices)}."
            )
        if not 0.0 <= self.mlp_trigger_skill_threshold <= 1.0:
            raise ValueError("SeparatixConfig.mlp_trigger_skill_threshold must be between 0 and 1.")
        if self.mlp_min_improvement < 0.0:
            raise ValueError("SeparatixConfig.mlp_min_improvement must be >= 0.")
        if self.mlp_max_parameters is not None and self.mlp_max_parameters < 1:
            raise ValueError("SeparatixConfig.mlp_max_parameters must be >= 1 when provided.")


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


def _validate_excluded_classes(value: Any) -> None:
    if value is None:
        return
    values = [value] if isinstance(value, (str, bytes)) or not _is_iterable(value) else value
    for item in values:
        try:
            hash(item)
        except TypeError as exc:
            raise ValueError("exclude_classes entries must be hashable.") from exc


def _is_iterable(value: Any) -> bool:
    try:
        iter(value)
    except TypeError:
        return False
    return True


@dataclass
class LabelViewConfig:
    """Configuration for optional hierarchical label-view benchmarking.

    Attributes:
        enabled: Whether hierarchy-derived label views should be evaluated.
        hierarchy_levels: Sequence of hierarchy levels to evaluate. Each level
            may be an integer index or a named level present in dataset metadata.
        output_levels: Mapping from extractor output names to hierarchy levels.
        skip_invalid_levels: Whether invalid or unavailable levels should be
            skipped with a warning instead of raising.
    """

    enabled: bool = False
    hierarchy_levels: Tuple[Union[int, str], ...] = (-1,)
    output_levels: Dict[str, Union[int, str]] = field(default_factory=dict)
    skip_invalid_levels: bool = True

    def __post_init__(self) -> None:
        if self.enabled and not self.hierarchy_levels:
            raise ValueError("LabelViewConfig.hierarchy_levels must not be empty when enabled.")
        if any(not isinstance(level, (int, str)) for level in self.hierarchy_levels):
            raise ValueError("LabelViewConfig.hierarchy_levels entries must be ints or strs.")
        if any(not isinstance(name, str) for name in self.output_levels):
            raise ValueError("LabelViewConfig.output_levels keys must be output-name strings.")
        if any(not isinstance(level, (int, str)) for level in self.output_levels.values()):
            raise ValueError("LabelViewConfig.output_levels values must be ints or strs.")


@dataclass
class TargetViewConfig:
    """Configuration for optional named target-view benchmarking.

    Attributes:
        enabled: Whether named target views should be evaluated.
        views: Explicit target-view names to evaluate. When empty and enabled,
            all registered target views are evaluated.
        output_views: Mapping from extractor output names to target-view names.
        skip_invalid_views: Whether unavailable views should be skipped with a
            warning instead of raising.
    """

    enabled: bool = False
    views: Tuple[str, ...] = ()
    output_views: Dict[str, str] = field(default_factory=dict)
    skip_invalid_views: bool = True

    def __post_init__(self) -> None:
        if any(not isinstance(name, str) or not name for name in self.views):
            raise ValueError("TargetViewConfig.views entries must be non-empty strings.")
        if any(not isinstance(name, str) or not name for name in self.output_views):
            raise ValueError("TargetViewConfig.output_views keys must be output-name strings.")
        if any(not isinstance(name, str) or not name for name in self.output_views.values()):
            raise ValueError("TargetViewConfig.output_views values must be target-view strings.")


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
    shard: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("EmbeddingConfig.batch_size must be >= 1.")


@dataclass
class SegmentationConfig:
    """Configuration for dense spatial segmentation evaluation."""

    coverage_threshold: float = 0.7
    ambiguity_margin: float = 0.2
    background_mode: str = "ignore"
    background_label: Any = "background"
    include_things: bool = True
    include_stuff: bool = True
    max_instances_per_class: Optional[int] = None
    max_tokens_per_instance: Optional[int] = None
    max_tokens_per_class: Optional[int] = None
    max_background_tokens: Optional[int] = None
    random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be between 0 and 1.")
        if not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be between 0 and 1.")
        if self.background_mode not in {"ignore", "include", "include_excluded"}:
            raise ValueError("background_mode must be one of: ignore, include, include_excluded.")
        for name in (
            "max_instances_per_class",
            "max_tokens_per_instance",
            "max_tokens_per_class",
            "max_background_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 when provided.")


@dataclass
class EmbeddingCompressionConfig:
    """Configuration for optional embedding compression.

    Attributes:
        enabled: Whether compression should run.
        method: Compression method name.
        n_components: Target dimension for projection-based compressors.
        preserve_variance: PCA variance target in `(0, 1]`.
        precision: Quantization precision such as `"float16"` or `"int8"`.
        assume_matryoshka: Whether prefix truncation should be treated as a
            matryoshka-style dimension shortening workflow.
        random_state: Seed for stochastic compressors.
        whiten: Whether PCA outputs should be whitened.
        dtype: Optional output dtype such as `"float32"`.
        algorithm_kwargs: Extra method-specific keyword arguments.
    """

    enabled: bool = False
    method: str = "none"
    n_components: Optional[int] = None
    preserve_variance: Optional[float] = None
    precision: Optional[str] = None
    assume_matryoshka: bool = False
    random_state: int = 42
    whiten: bool = False
    dtype: Optional[str] = None
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_methods = {
            "none",
            "pca",
            "incremental_pca",
            "truncated_svd",
            "gaussian_random_projection",
            "sparse_random_projection",
            "prefix_truncate",
            "quantize",
        }
        if self.method not in allowed_methods:
            raise ValueError(
                f"EmbeddingCompressionConfig.method must be one of {sorted(allowed_methods)}."
            )
        if self.n_components is not None and self.n_components < 1:
            raise ValueError("EmbeddingCompressionConfig.n_components must be >= 1.")
        if self.preserve_variance is not None and not 0.0 < self.preserve_variance <= 1.0:
            raise ValueError("EmbeddingCompressionConfig.preserve_variance must be in (0, 1].")
        if self.method == "none":
            if self.enabled and (
                self.n_components is not None
                or self.preserve_variance is not None
                or self.precision is not None
                or self.assume_matryoshka
                or self.whiten
                or self.algorithm_kwargs
            ):
                raise ValueError(
                    "EmbeddingCompressionConfig.method='none' does not accept compression "
                    "parameters."
                )
        projection_methods = {
            "truncated_svd",
            "gaussian_random_projection",
            "sparse_random_projection",
            "incremental_pca",
            "prefix_truncate",
        }
        if self.method in projection_methods and self.enabled and self.n_components is None:
            raise ValueError(
                f"EmbeddingCompressionConfig.method='{self.method}' requires n_components."
            )
        if self.method == "pca" and self.enabled:
            if self.n_components is None and self.preserve_variance is None:
                raise ValueError(
                    "EmbeddingCompressionConfig.method='pca' requires n_components or "
                    "preserve_variance."
                )
        if self.method != "pca" and self.preserve_variance is not None:
            raise ValueError("preserve_variance is only supported for method='pca'.")
        if self.method == "quantize" and self.enabled and self.precision is None:
            raise ValueError("EmbeddingCompressionConfig.method='quantize' requires precision.")
        if self.method != "quantize" and self.precision is not None:
            raise ValueError("precision is only supported for method='quantize'.")
        if self.method != "prefix_truncate" and self.assume_matryoshka:
            raise ValueError("assume_matryoshka is only supported for method='prefix_truncate'.")
        if self.method not in {"pca", "incremental_pca"} and self.whiten:
            raise ValueError("whiten is only supported for PCA-based methods.")
        if self.precision is not None and self.precision not in {"float16", "int8", "uint8"}:
            raise ValueError(
                "EmbeddingCompressionConfig.precision must be one of "
                "['float16', 'int8', 'uint8']."
            )
        if self.dtype is not None:
            try:
                import numpy as np

                np.dtype(self.dtype)
            except TypeError as exc:
                raise ValueError(f"Unsupported compression dtype: {self.dtype!r}.") from exc


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


@dataclass
class ResourceProfilingConfig:
    """Opt-in measurement of extraction cost and representation footprint.

    Resource profiles are descriptive measurements for the current benchmark
    protocol. They do not participate in quality scoring or ranking.
    """

    enabled: bool = False
    host_memory: bool = True
    device_memory: bool = True
    host_sample_interval_seconds: float = 0.01
    quality_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.host_sample_interval_seconds <= 0:
            raise ValueError(
                "ResourceProfilingConfig.host_sample_interval_seconds must be > 0."
            )
        if self.quality_tolerance < 0:
            raise ValueError("ResourceProfilingConfig.quality_tolerance must be >= 0.")
