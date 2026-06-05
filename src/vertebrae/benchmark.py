"""Benchmark runner."""

from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from vertebrae import __version__
from vertebrae.cache import ArtifactStore, create_artifact_store
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.config import (
    CacheConfig,
    EmbeddingConfig,
    MemoryConfig,
    OverlapScoringConfig,
    ProbeConfig,
    StabilityConfig,
)
from vertebrae.execution.jobs import SampleBatch
from vertebrae.execution.local import LocalBackend
from vertebrae.reports.recommendations import (
    recommendation_for_extractor,
    recommendations_for_benchmark,
)
from vertebrae.results import BenchmarkResult, ExtractorResult
from vertebrae.scoring.overlap import OverlapIndexScorer
from vertebrae.scoring.probes import run_probes
from vertebrae.scoring.stability import run_stability_analysis
from vertebrae.utils.memory import (
    EmbeddingMemoryEstimate,
    assert_within_memory,
    estimate_embedding_from_probe,
    estimate_matrix_resident_bytes,
    estimate_metadata_dense_scoring_bytes,
    estimate_metadata_resident_bytes,
    largest_fitting_subsample_rate,
)
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


class Benchmark:
    """Run one or more extractors against a labeled dataset.

    Args:
        dataset: Dataset object with inputs and labels.
        extractors: Optional iterable of extractors to evaluate.
        scoring_config: OverlapIndex scoring configuration.
        stability_config: Stability-analysis configuration.
        probe_config: Probe-classifier configuration.
        cache_config: Embedding cache configuration.
        embedding_config: Embedding batching and streaming configuration.
        execution: Local execution backend.
    """

    def __init__(
        self,
        dataset: Any,
        extractors: Optional[Iterable[Any]] = None,
        scoring_config: Optional[OverlapScoringConfig] = None,
        stability_config: Optional[StabilityConfig] = None,
        probe_config: Optional[ProbeConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        execution: Optional[Any] = None,
    ) -> None:
        self.dataset = dataset
        self.extractors = list(extractors or [])
        self.scoring_config = scoring_config or OverlapScoringConfig()
        self.stability_config = stability_config or StabilityConfig()
        self.probe_config = probe_config or ProbeConfig()
        self.cache_config = cache_config or CacheConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.memory_config = memory_config or MemoryConfig()
        self.execution = execution or LocalBackend()

    def add_extractor(self, extractor: Any) -> "Benchmark":
        """Add an extractor to this benchmark.

        Args:
            extractor: Feature extractor implementing the vertebrae protocol.

        Returns:
            This benchmark instance for fluent chaining.
        """

        self.extractors.append(extractor)
        return self

    def run(self) -> BenchmarkResult:
        """Run feature extraction, scoring, optional probes, and reporting aggregation.

        Returns:
            Aggregated benchmark result.

        Raises:
            ValueError: If no extractors are configured or dataset validation fails.
        """

        self.dataset.validate()
        if not self.extractors:
            raise ValueError("At least one extractor must be provided.")

        extractor_results = [self._run_extractor(extractor) for extractor in self.extractors]
        recommendations = recommendations_for_benchmark(extractor_results)
        return BenchmarkResult(
            dataset_summary=self.dataset.summary(),
            extractor_results=extractor_results,
            recommendations=recommendations,
            metadata={
                "vertebrae_version": __version__,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "scoring_config": asdict(self.scoring_config),
                "stability_config": asdict(self.stability_config),
                "probe_config": asdict(self.probe_config),
                "cache_config": asdict(self.cache_config),
                "embedding_config": asdict(self.embedding_config),
                "memory_config": asdict(self.memory_config),
            },
        )

    def _run_extractor(self, extractor: Any) -> ExtractorResult:
        warnings: List[str] = []
        runtime = {}
        start = perf_counter()
        dataset, subsampling_warnings, subsampling_metadata, probe_plan = (
            self._prepare_dataset_for_extractor(extractor)
        )
        warnings.extend(subsampling_warnings)
        embeddings, embedding_metadata = self._get_or_compute_embeddings(
            extractor,
            dataset,
            subsampling_metadata,
            probe_plan,
        )
        runtime["embedding_seconds"] = perf_counter() - start

        score_start = perf_counter()
        self._admit_scoring_memory(embedding_metadata)
        overlap = OverlapIndexScorer(self.scoring_config).score(embeddings, dataset.y)
        runtime["scoring_seconds"] = perf_counter() - score_start
        warnings.extend(overlap.warnings)

        stability_start = perf_counter()
        stability = run_stability_analysis(
            embeddings,
            dataset.y,
            self.scoring_config,
            self.stability_config,
        )
        runtime["stability_seconds"] = perf_counter() - stability_start
        if stability:
            warnings.extend(stability.get("warnings", []))

        probe_start = perf_counter()
        probes = run_probes(embeddings, dataset.y, self.probe_config)
        runtime["probe_seconds"] = perf_counter() - probe_start
        if probes:
            warnings.extend(probes.get("warnings", []))

        weakest_class, weakest_score = _weakest_class(overlap.per_class_scores)
        recommendation = recommendation_for_extractor(overlap.macro_score, stability, weakest_score)
        return ExtractorResult(
            name=extractor.name,
            extractor_type=getattr(extractor, "extractor_type", "unknown"),
            overlap=overlap,
            stability=stability,
            probes=probes,
            embedding_metadata=embedding_metadata,
            runtime=runtime,
            warnings=sorted(set(warnings)),
            weakest_class=weakest_class,
            weakest_class_score=weakest_score,
            recommendation=recommendation,
        )

    def _prepare_dataset_for_extractor(
        self,
        extractor: Any,
    ) -> Tuple[Any, List[str], dict, Optional[Tuple[SampleBatch, Any, EmbeddingMemoryEstimate]]]:
        dataset = self.dataset
        warnings: List[str] = []
        probe_plan: Optional[Tuple[SampleBatch, Any, EmbeddingMemoryEstimate]] = None
        metadata: dict[str, Any] = {
            "subsampled": False,
            "subsample_reason": None,
            "requested_subsample_rate": self.memory_config.subsample_rate,
            "effective_subsample_rate": 1.0,
            "parent_n_samples": int(len(dataset.y)),
        }
        if self.memory_config.subsample_rate < 1.0:
            dataset, user_metadata, warning = self._subsample_dataset(
                dataset,
                rate=self.memory_config.subsample_rate,
                reason="user_requested",
            )
            metadata.update(user_metadata)
            warnings.append(warning)

        if self._should_stream_embeddings(extractor):
            auto_rate, probe_plan = self._auto_subsample_rate_for_streaming_estimate(
                extractor,
                dataset,
            )
            if auto_rate < 1.0:
                dataset, auto_metadata, warning = self._subsample_dataset(
                    dataset,
                    rate=auto_rate,
                    reason="memory_limit",
                )
                metadata.update(auto_metadata)
                metadata["parent_n_samples"] = int(len(self.dataset.y))
                warnings.append(warning)
                probe_plan = None

        return dataset, warnings, metadata, probe_plan

    def _subsample_dataset(self, dataset: Any, rate: float, reason: str) -> Tuple[Any, dict, str]:
        indices = dataset.stratified_subsample_indices(
            rate=rate,
            random_state=self.memory_config.subsample_random_state,
            min_samples_per_class=self.memory_config.min_subsample_samples_per_class,
        )
        parent_n_samples = int(len(dataset.y))
        subset = dataset.subset(
            indices,
            metadata={
                "subsampled": True,
                "subsample_reason": reason,
                "requested_subsample_rate": rate,
                "effective_subsample_rate": len(indices) / parent_n_samples,
            },
        )
        effective_rate = len(indices) / parent_n_samples
        metadata = {
            "subsampled": True,
            "subsample_reason": reason,
            "requested_subsample_rate": rate,
            "effective_subsample_rate": effective_rate,
            "parent_n_samples": parent_n_samples,
            "sample_indices": subset.metadata.get("sample_indices", indices.tolist()),
            "n_samples_after_subsampling": int(len(indices)),
        }
        if reason == "memory_limit":
            warning = (
                "Embedding memory estimate exceeded the configured budget; using a "
                f"class-stratified subsample with effective rate {effective_rate:.3f} "
                f"({len(indices)}/{parent_n_samples} samples)."
            )
        else:
            warning = (
                "Using user-requested class-stratified subsample with effective rate "
                f"{effective_rate:.3f} ({len(indices)}/{parent_n_samples} samples)."
            )
        return subset, metadata, warning

    def _auto_subsample_rate_for_streaming_estimate(
        self,
        extractor: Any,
        dataset: Any,
    ) -> Tuple[float, Optional[Tuple[SampleBatch, Any, EmbeddingMemoryEstimate]]]:
        if not self.memory_config.auto_subsample_on_memory_exceeded:
            return 1.0, None
        extractor.fit(dataset.X, dataset.y)
        first_batch = next(
            dataset.iter_batches(
                batch_size=min(self.embedding_config.batch_size, len(dataset.y)),
                shard=self.embedding_config.shard,
            )
        )
        first_embeddings = self._embed_batch(extractor, first_batch)
        estimate = estimate_embedding_from_probe(
            first_embeddings,
            n_samples=len(dataset.y),
            batch_size=self.embedding_config.batch_size,
            memory_config=self.memory_config,
        )
        required = estimate.dense_scoring_bytes
        if estimate.strategy == "in_memory":
            required = max(required, estimate.resident_bytes)
        try:
            self._admit_embedding_plan(estimate)
        except ValueError:
            rate = largest_fitting_subsample_rate(required, self.memory_config)
            if rate <= 0.0:
                return 1.0, (first_batch, first_embeddings, estimate)
            return min(1.0, rate), None
        return 1.0, (first_batch, first_embeddings, estimate)

    def _get_or_compute_embeddings(
        self,
        extractor: Any,
        dataset: Any,
        subsampling_metadata: Optional[dict] = None,
        probe_plan: Optional[Tuple[SampleBatch, Any, EmbeddingMemoryEstimate]] = None,
    ) -> Any:
        recipe = extractor.recipe()
        dataset_key = dataset.fingerprint()
        extractor_key = fingerprint_extractor_recipe(recipe)
        cache_key = f"embeddings/{dataset_key}/{extractor_key}"
        store = create_artifact_store(
            self.cache_config.cache_dir,
            **self.cache_config.storage_options,
        )

        if (
            self.cache_config.enabled
            and not self.cache_config.force_recompute
            and store.exists(cache_key)
        ):
            metadata = store.get_json(cache_key)
            self._admit_cached_embedding_load(metadata)
            embeddings = store.get_array(cache_key)
            metadata["cache_hit"] = True
            metadata.update(subsampling_metadata or {})
            return embeddings, metadata

        if self._should_stream_embeddings(extractor):
            embeddings, metadata = self._stream_embeddings(
                extractor,
                dataset,
                store,
                cache_key,
                recipe,
                probe_plan,
            )
            metadata.update(subsampling_metadata or {})
            if self.cache_config.enabled:
                store.put_json(cache_key, metadata)
            return embeddings, metadata

        embeddings = extractor.fit_transform(dataset.X, dataset.y)
        embeddings = ensure_numeric_matrix(
            embeddings,
            f"Extractor '{extractor.name}' embeddings",
            allow_sparse=True,
        )
        self._admit_resident_embedding(embeddings)
        if embeddings.shape[0] != len(dataset.y):
            raise ValueError(
                f"Extractor '{extractor.name}' returned {embeddings.shape[0]} embeddings "
                f"for {len(dataset.y)} labels."
            )
        metadata = self._embedding_metadata(extractor, dataset, embeddings, cache_key, recipe)
        metadata.update(subsampling_metadata or {})
        if self.cache_config.enabled:
            store.put_array(cache_key, embeddings)
            store.put_json(cache_key, metadata)
        return embeddings, metadata

    def _should_stream_embeddings(self, extractor: Any) -> bool:
        if not self.embedding_config.streaming_enabled:
            return False
        shard = self.embedding_config.shard
        if shard is not None and not shard.is_complete:
            raise ValueError(
                "Benchmark.run requires a complete embedding artifact for scoring. "
                "Use BenchmarkDataset.iter_batches(..., shard=...) or a future embedding "
                "job runner to materialize distributed shards without duplicate samples."
            )
        return bool(getattr(extractor, "streaming_safe", False)) or bool(
            getattr(extractor, "already_fitted", False)
        )

    def _stream_embeddings(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        cache_key: str,
        recipe: dict,
        probe_plan: Optional[Tuple[SampleBatch, Any, EmbeddingMemoryEstimate]] = None,
    ) -> Tuple[Any, dict]:
        n_samples = len(dataset.y)
        batch_iterator = iter(
            dataset.iter_batches(
                batch_size=self.embedding_config.batch_size,
                shard=self.embedding_config.shard,
            )
        )
        if probe_plan is None:
            extractor.fit(dataset.X, dataset.y)
            try:
                first_batch = next(batch_iterator)
            except StopIteration as exc:
                raise ValueError("At least one sample is required for embedding.") from exc
            first_embeddings = self._embed_batch(extractor, first_batch)
            memory_estimate = estimate_embedding_from_probe(
                first_embeddings,
                n_samples=n_samples,
                batch_size=self.embedding_config.batch_size,
                memory_config=self.memory_config,
            )
            self._admit_embedding_plan(memory_estimate)
        else:
            first_batch, first_embeddings, memory_estimate = probe_plan
            try:
                skipped_batch = next(batch_iterator)
            except StopIteration as exc:
                raise ValueError("At least one sample is required for embedding.") from exc
            if not np.array_equal(skipped_batch.indices, first_batch.indices):
                raise ValueError("Reusable embedding probe does not match streaming batch order.")
        batch_pairs = _prepend_batch(
            first_batch.indices,
            first_embeddings,
            self._embedding_batches_from(extractor, batch_iterator),
        )
        if self.cache_config.enabled:
            store.put_array_batches(
                cache_key,
                batch_pairs,
                n_samples=n_samples,
                require_complete=True,
            )
            embeddings = store.get_array(cache_key)
        else:
            if memory_estimate.strategy == "stream_to_disk":
                raise ValueError(
                    "Embedding artifact is estimated to exceed the memory budget, but "
                    "CacheConfig.enabled=False prevents streaming it to disk. Enable "
                    "the cache or increase MemoryConfig.max_memory_bytes."
                )
            embeddings = _combine_embedding_batches(
                batch_pairs,
                n_samples=n_samples,
            )
        metadata = self._embedding_metadata(extractor, dataset, embeddings, cache_key, recipe)
        metadata["streamed"] = True
        metadata["stream_batch_size"] = self.embedding_config.batch_size
        metadata["memory_estimate"] = memory_estimate.to_dict()
        return embeddings, metadata

    def _embedding_batches_from(
        self,
        extractor: Any,
        batches: Iterator[SampleBatch],
    ) -> Iterator[Tuple[np.ndarray, Any]]:
        for batch in batches:
            yield batch.indices, self._embed_batch(extractor, batch)

    def _embed_batch(self, extractor: Any, batch: SampleBatch) -> Any:
        embeddings = extractor.transform(batch.X)
        embeddings = ensure_numeric_matrix(
            embeddings,
            f"Extractor '{extractor.name}' batch embeddings",
            allow_sparse=True,
        )
        if embeddings.shape[0] != len(batch.indices):
            raise ValueError(
                f"Extractor '{extractor.name}' returned {embeddings.shape[0]} embeddings "
                f"for a batch with {len(batch.indices)} samples."
            )
        return embeddings

    def _embedding_metadata(
        self,
        extractor: Any,
        dataset: Any,
        embeddings: Any,
        cache_key: str,
        recipe: dict,
    ) -> dict:
        sparse_embeddings = is_sparse_matrix(embeddings)
        return {
            "extractor_name": extractor.name,
            "extractor_type": getattr(extractor, "extractor_type", "unknown"),
            "modality": getattr(extractor, "modality", dataset.modality),
            "cache_hit": False,
            "cache_key": cache_key,
            "shape": list(embeddings.shape),
            "n_samples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "sparse": sparse_embeddings,
            "nnz": int(embeddings.nnz) if sparse_embeddings else None,
            "storage_format": embeddings.getformat() if sparse_embeddings else "dense",
            "streamed": False,
            "memory_estimate": None,
            "recipe": recipe,
            "extractor_recipe": recipe,
        }

    def _admit_embedding_plan(self, estimate: EmbeddingMemoryEstimate) -> None:
        batch_required = (
            estimate.batch_embedding_bytes
            + self.memory_config.model_memory_bytes
            + self.memory_config.raw_batch_memory_bytes
        )
        assert_within_memory(
            batch_required,
            self.memory_config,
            purpose="Embedding batch",
        )
        if estimate.strategy == "in_memory":
            assert_within_memory(
                estimate.resident_bytes,
                self.memory_config,
                purpose="Resident embedding artifact",
            )
        assert_within_memory(
            estimate.dense_scoring_bytes,
            self.memory_config,
            purpose="Dense scoring input",
        )

    def _admit_resident_embedding(self, embeddings: Any) -> None:
        required = estimate_matrix_resident_bytes(embeddings)
        assert_within_memory(
            required,
            self.memory_config,
            purpose="Resident embedding artifact",
        )

    def _admit_cached_embedding_load(self, metadata: dict) -> None:
        required = estimate_metadata_resident_bytes(metadata)
        if required is not None:
            assert_within_memory(
                required,
                self.memory_config,
                purpose="Cached embedding artifact load",
            )

    def _admit_scoring_memory(self, metadata: dict) -> None:
        required = estimate_metadata_dense_scoring_bytes(metadata)
        if required is not None:
            assert_within_memory(
                required,
                self.memory_config,
                purpose="Dense scoring input",
            )


def _weakest_class(per_class_scores: dict) -> Any:
    numeric_scores = {
        str(label): float(score)
        for label, score in per_class_scores.items()
        if isinstance(score, (int, float, np.number))
    }
    if not numeric_scores:
        return None, None
    label, score = min(numeric_scores.items(), key=lambda item: item[1])
    return label, score


def _combine_embedding_batches(
    batches: Iterable[Tuple[np.ndarray, Any]],
    n_samples: int,
) -> Any:
    collected = list(batches)
    if not collected:
        raise ValueError("At least one embedding batch is required.")
    first = collected[0][1]
    written = np.zeros(n_samples, dtype=bool)
    if is_sparse_matrix(first):
        from scipy import sparse

        rows = []
        for indices, batch in collected:
            if not is_sparse_matrix(batch):
                raise ValueError("Cannot mix sparse and dense embedding batches.")
            _check_batch_indices(indices, batch.shape[0], written)
            rows.append(batch)
        if not bool(np.all(written)):
            missing = np.flatnonzero(~written)
            raise ValueError(
                f"Embedding batches did not cover all samples; missing {missing[:10]}."
            )
        return sparse.vstack(rows, format="csr")

    first_arr = np.asarray(first)
    output = np.empty((n_samples, first_arr.shape[1]), dtype=first_arr.dtype)
    for indices, batch in collected:
        if is_sparse_matrix(batch):
            raise ValueError("Cannot mix sparse and dense embedding batches.")
        arr = np.asarray(batch)
        _check_batch_indices(indices, arr.shape[0], written)
        output[np.asarray(indices, dtype=int)] = arr
    if not bool(np.all(written)):
        missing = np.flatnonzero(~written)
        raise ValueError(f"Embedding batches did not cover all samples; missing {missing[:10]}.")
    return output


def _check_batch_indices(indices: np.ndarray, n_rows: int, written: np.ndarray) -> None:
    indices = np.asarray(indices, dtype=int)
    if len(indices) != n_rows:
        raise ValueError("Batch index count must match embedding row count.")
    if np.any(written[indices]):
        duplicates = indices[written[indices]]
        raise ValueError(f"Duplicate embedding rows for sample indices {duplicates[:10]}.")
    written[indices] = True


def _prepend_batch(
    indices: np.ndarray,
    embeddings: Any,
    remaining: Iterator[Tuple[np.ndarray, Any]],
) -> Iterator[Tuple[np.ndarray, Any]]:
    yield indices, embeddings
    yield from remaining
