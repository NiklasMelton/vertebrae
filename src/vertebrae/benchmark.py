"""Benchmark runner."""

from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from vertebrae import __version__
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.config import (
    CacheConfig,
    EmbeddingConfig,
    OverlapScoringConfig,
    ProbeConfig,
    StabilityConfig,
)
from vertebrae.execution.local import LocalBackend
from vertebrae.reports.recommendations import (
    recommendation_for_extractor,
    recommendations_for_benchmark,
)
from vertebrae.results import BenchmarkResult, ExtractorResult
from vertebrae.scoring.overlap import OverlapIndexScorer
from vertebrae.scoring.probes import run_probes
from vertebrae.scoring.stability import run_stability_analysis
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
        execution: Optional[Any] = None,
    ) -> None:
        self.dataset = dataset
        self.extractors = list(extractors or [])
        self.scoring_config = scoring_config or OverlapScoringConfig()
        self.stability_config = stability_config or StabilityConfig()
        self.probe_config = probe_config or ProbeConfig()
        self.cache_config = cache_config or CacheConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
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
            },
        )

    def _run_extractor(self, extractor: Any) -> ExtractorResult:
        warnings: List[str] = []
        runtime = {}
        start = perf_counter()
        embeddings, embedding_metadata = self._get_or_compute_embeddings(extractor)
        runtime["embedding_seconds"] = perf_counter() - start

        score_start = perf_counter()
        overlap = OverlapIndexScorer(self.scoring_config).score(embeddings, self.dataset.y)
        runtime["scoring_seconds"] = perf_counter() - score_start
        warnings.extend(overlap.warnings)

        stability_start = perf_counter()
        stability = run_stability_analysis(
            embeddings,
            self.dataset.y,
            self.scoring_config,
            self.stability_config,
        )
        runtime["stability_seconds"] = perf_counter() - stability_start
        if stability:
            warnings.extend(stability.get("warnings", []))

        probe_start = perf_counter()
        probes = run_probes(embeddings, self.dataset.y, self.probe_config)
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

    def _get_or_compute_embeddings(self, extractor: Any) -> Any:
        recipe = extractor.recipe()
        dataset_key = self.dataset.fingerprint()
        extractor_key = fingerprint_extractor_recipe(recipe)
        cache_key = f"embeddings/{dataset_key}/{extractor_key}"
        store = LocalArtifactStore(self.cache_config.cache_dir)

        if (
            self.cache_config.enabled
            and not self.cache_config.force_recompute
            and store.exists(cache_key)
        ):
            embeddings = store.get_array(cache_key)
            metadata = store.get_json(cache_key)
            metadata["cache_hit"] = True
            return embeddings, metadata

        if self._should_stream_embeddings(extractor):
            embeddings, metadata = self._stream_embeddings(extractor, store, cache_key, recipe)
            if self.cache_config.enabled:
                store.put_json(cache_key, metadata)
            return embeddings, metadata

        embeddings = extractor.fit_transform(self.dataset.X, self.dataset.y)
        embeddings = ensure_numeric_matrix(
            embeddings,
            f"Extractor '{extractor.name}' embeddings",
            allow_sparse=True,
        )
        if embeddings.shape[0] != len(self.dataset.y):
            raise ValueError(
                f"Extractor '{extractor.name}' returned {embeddings.shape[0]} embeddings "
                f"for {len(self.dataset.y)} labels."
            )
        metadata = self._embedding_metadata(extractor, embeddings, cache_key, recipe)
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
        store: LocalArtifactStore,
        cache_key: str,
        recipe: dict,
    ) -> Tuple[Any, dict]:
        extractor.fit(self.dataset.X, self.dataset.y)
        n_samples = len(self.dataset.y)
        if self.cache_config.enabled:
            store.put_array_batches(
                cache_key,
                self._embedding_batches(extractor),
                n_samples=n_samples,
                require_complete=True,
            )
            embeddings = store.get_array(cache_key)
        else:
            embeddings = _combine_embedding_batches(
                self._embedding_batches(extractor),
                n_samples=n_samples,
            )
        metadata = self._embedding_metadata(extractor, embeddings, cache_key, recipe)
        metadata["streamed"] = True
        metadata["stream_batch_size"] = self.embedding_config.batch_size
        return embeddings, metadata

    def _embedding_batches(self, extractor: Any) -> Iterator[Tuple[np.ndarray, Any]]:
        for batch in self.dataset.iter_batches(
            batch_size=self.embedding_config.batch_size,
            shard=self.embedding_config.shard,
        ):
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
            yield batch.indices, embeddings

    def _embedding_metadata(
        self,
        extractor: Any,
        embeddings: Any,
        cache_key: str,
        recipe: dict,
    ) -> dict:
        sparse_embeddings = is_sparse_matrix(embeddings)
        return {
            "extractor_name": extractor.name,
            "extractor_type": getattr(extractor, "extractor_type", "unknown"),
            "modality": getattr(extractor, "modality", self.dataset.modality),
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
            "recipe": recipe,
            "extractor_recipe": recipe,
        }


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
