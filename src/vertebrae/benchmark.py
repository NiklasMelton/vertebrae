"""Benchmark runner."""

from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

from vertebrae import __version__
from vertebrae.cache import ArtifactStore, create_artifact_store
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.compression import compress_embedding_artifact_key, compress_embeddings
from vertebrae.config import (
    CacheConfig,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    MemoryConfig,
    OverlapScoringConfig,
    ProbeConfig,
    StabilityConfig,
)
from vertebrae.execution.jobs import SampleBatch
from vertebrae.execution.local import LocalBackend
from vertebrae.extractors.base import EmbeddingOutput
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
        compression_config: Optional[EmbeddingCompressionConfig] = None,
        compression_configs: Optional[Iterable[EmbeddingCompressionConfig]] = None,
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
        if compression_config is not None and compression_configs is not None:
            raise ValueError("Provide compression_config or compression_configs, not both.")
        default_compressions = (
            [compression_config]
            if compression_config is not None
            else [EmbeddingCompressionConfig()]
        )
        self.compression_configs = list(compression_configs or default_compressions)
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

        extractor_results: List[ExtractorResult] = []
        for extractor in self.extractors:
            result = self._run_extractor(extractor)
            if isinstance(result, list):
                extractor_results.extend(result)
            else:
                extractor_results.append(result)
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
                "compression_configs": [asdict(config) for config in self.compression_configs],
                "embedding_config": asdict(self.embedding_config),
                "memory_config": asdict(self.memory_config),
            },
        )

    def _run_extractor(self, extractor: Any) -> Union[ExtractorResult, List[ExtractorResult]]:
        warnings: List[str] = []
        runtime = {}
        start = perf_counter()
        dataset, subsampling_warnings, subsampling_metadata, probe_plan = (
            self._prepare_dataset_for_extractor(extractor)
        )
        warnings.extend(subsampling_warnings)
        store = create_artifact_store(
            self.cache_config.cache_dir,
            **self.cache_config.storage_options,
        )
        variants = self._get_or_compute_embedding_variants(
            extractor,
            dataset,
            store,
            subsampling_metadata,
            probe_plan,
        )
        runtime["embedding_seconds"] = perf_counter() - start
        results: List[ExtractorResult] = []
        for variant in variants:
            embeddings = variant["embeddings"]
            embedding_metadata = variant["metadata"]
            for compression_config in self.compression_configs:
                variant_warnings = list(warnings)
                variant_runtime = dict(runtime)
                compression_start = perf_counter()
                compressed_embeddings, compression_metadata = (
                    self._get_or_compute_compressed_embeddings(
                        embeddings=embeddings,
                        embedding_metadata=embedding_metadata,
                        labels=dataset.y,
                        store=store,
                        config=compression_config,
                    )
                )
                variant_runtime["compression_seconds"] = perf_counter() - compression_start
                variant_warnings.extend(compression_metadata.get("warnings", []))

                score_start = perf_counter()
                scoring_metadata = dict(embedding_metadata)
                scoring_metadata["embedding_dim"] = compression_metadata.get(
                    "compressed_dim",
                    embedding_metadata.get("embedding_dim"),
                )
                scoring_metadata["shape"] = [
                    embedding_metadata.get("n_samples"),
                    scoring_metadata["embedding_dim"],
                ]
                scoring_metadata["sparse"] = compression_metadata.get(
                    "output_sparse",
                    embedding_metadata.get("sparse"),
                )
                self._admit_scoring_memory(scoring_metadata)
                overlap = OverlapIndexScorer(self.scoring_config).score(
                    compressed_embeddings,
                    dataset.y,
                )
                variant_runtime["scoring_seconds"] = perf_counter() - score_start
                variant_warnings.extend(overlap.warnings)

                stability_start = perf_counter()
                stability = run_stability_analysis(
                    compressed_embeddings,
                    dataset.y,
                    self.scoring_config,
                    self.stability_config,
                )
                variant_runtime["stability_seconds"] = perf_counter() - stability_start
                if stability:
                    variant_warnings.extend(stability.get("warnings", []))

                probe_start = perf_counter()
                probes = run_probes(compressed_embeddings, dataset.y, self.probe_config)
                variant_runtime["probe_seconds"] = perf_counter() - probe_start
                if probes:
                    variant_warnings.extend(probes.get("warnings", []))

                weakest_class, weakest_score = _weakest_class(overlap.per_class_scores)
                recommendation = recommendation_for_extractor(
                    overlap.macro_score,
                    stability,
                    weakest_score,
                )
                result_name = embedding_metadata.get("extractor_name", extractor.name)
                results.append(
                    ExtractorResult(
                        name=_variant_extractor_name(result_name, compression_metadata),
                        extractor_type=embedding_metadata.get(
                            "extractor_type",
                            getattr(extractor, "extractor_type", "unknown"),
                        ),
                        overlap=overlap,
                        stability=stability,
                        probes=probes,
                        embedding_metadata=embedding_metadata,
                        compression_metadata=compression_metadata,
                        runtime=variant_runtime,
                        warnings=sorted(set(variant_warnings)),
                        weakest_class=weakest_class,
                        weakest_class_score=weakest_score,
                        recommendation=recommendation,
                    )
                )
        if len(results) == 1:
            return results[0]
        return results

    def _prepare_dataset_for_extractor(
        self,
        extractor: Any,
    ) -> Tuple[Any, List[str], dict, Optional[Tuple[SampleBatch, Any, Any]]]:
        dataset = self.dataset
        warnings: List[str] = []
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None
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
    ) -> Tuple[float, Optional[Tuple[SampleBatch, Any, Any]]]:
        if not self.memory_config.auto_subsample_on_memory_exceeded:
            return 1.0, None
        extractor.fit(dataset.X, dataset.y)
        first_batch = next(
            dataset.iter_batches(
                batch_size=min(self.embedding_config.batch_size, len(dataset.y)),
                shard=self.embedding_config.shard,
            )
        )
        if self._supports_transform_many(extractor):
            first_outputs = self._embed_batch_many(extractor, first_batch)
            estimates, aggregate = self._estimate_multi_output_memory(
                first_outputs,
                n_samples=len(dataset.y),
            )
            required = max(
                aggregate["dense_scoring_bytes"],
                aggregate["resident_bytes"],
            )
            try:
                self._admit_multi_embedding_plan(aggregate)
            except ValueError:
                rate = largest_fitting_subsample_rate(required, self.memory_config)
                if rate <= 0.0:
                    return 1.0, (first_batch, first_outputs, {"per_output": estimates, **aggregate})
                return min(1.0, rate), None
            return 1.0, (first_batch, first_outputs, {"per_output": estimates, **aggregate})

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

    def _get_or_compute_embedding_variants(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        subsampling_metadata: Optional[dict] = None,
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
    ) -> List[dict]:
        if not self._supports_transform_many(extractor):
            embeddings, metadata = self._get_or_compute_embeddings(
                extractor,
                dataset,
                store,
                subsampling_metadata,
                probe_plan,
            )
            return [{"embeddings": embeddings, "metadata": metadata}]
        return self._get_or_compute_multi_embeddings(
            extractor,
            dataset,
            store,
            subsampling_metadata,
            probe_plan,
        )

    def _get_or_compute_embeddings(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        subsampling_metadata: Optional[dict] = None,
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
    ) -> Any:
        recipe = extractor.recipe()
        dataset_key = dataset.fingerprint()
        extractor_key = fingerprint_extractor_recipe(recipe)
        cache_key = f"embeddings/{dataset_key}/{extractor_key}"
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

    def _get_or_compute_multi_embeddings(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        subsampling_metadata: Optional[dict] = None,
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
    ) -> List[dict]:
        recipe = extractor.recipe()
        base_key = f"embeddings/{dataset.fingerprint()}/{fingerprint_extractor_recipe(recipe)}"
        specs = self._output_specs(extractor)
        cache_keys = {
            spec.name: self._output_cache_key(base_key, spec.name)
            for spec in specs
        }
        if self.cache_config.enabled and not self.cache_config.force_recompute:
            if all(store.exists(cache_key) for cache_key in cache_keys.values()):
                variants = []
                for spec in specs:
                    metadata = store.get_json(cache_keys[spec.name])
                    self._admit_cached_embedding_load(metadata)
                    embeddings = store.get_array(cache_keys[spec.name])
                    metadata["cache_hit"] = True
                    metadata.update(subsampling_metadata or {})
                    variants.append({"embeddings": embeddings, "metadata": metadata})
                return variants

        if self._should_stream_embeddings(extractor):
            variants = self._stream_multi_embeddings(
                extractor=extractor,
                dataset=dataset,
                store=store,
                base_key=base_key,
                recipe=recipe,
                probe_plan=probe_plan,
            )
            for variant in variants:
                variant["metadata"].update(subsampling_metadata or {})
            if self.cache_config.enabled:
                for variant in variants:
                    store.put_json(variant["metadata"]["cache_key"], variant["metadata"])
            return variants

        extractor.fit(dataset.X, dataset.y)
        outputs = self._validate_multi_outputs(
            extractor=extractor,
            outputs=extractor.transform_many(dataset.X),
            expected_rows=len(dataset.y),
            context="embeddings",
        )
        self._admit_multi_resident_embeddings(outputs)
        variants = []
        for output in outputs:
            cache_key = cache_keys[output.name]
            metadata = self._embedding_metadata(
                extractor=extractor,
                dataset=dataset,
                embeddings=output.embeddings,
                cache_key=cache_key,
                recipe=self._qualified_output_recipe(recipe, output),
                extractor_name=_qualified_output_name(extractor.name, output.name),
                parent_extractor_name=extractor.name,
                output_name=output.name,
                extractor_recipe=recipe,
                output_metadata=output.metadata,
            )
            metadata.update(subsampling_metadata or {})
            if self.cache_config.enabled:
                store.put_array(cache_key, output.embeddings)
                store.put_json(cache_key, metadata)
            variants.append({"embeddings": output.embeddings, "metadata": metadata})
        return variants

    def _get_or_compute_compressed_embeddings(
        self,
        embeddings: Any,
        embedding_metadata: dict,
        labels: Any,
        store: ArtifactStore,
        config: EmbeddingCompressionConfig,
    ) -> Tuple[Any, dict]:
        if not config.enabled or config.method == "none":
            compression_result = compress_embeddings(embeddings, config=config, y=labels)
            return compression_result.embeddings, compression_result.metadata

        source_key = embedding_metadata["cache_key"]
        compression_key = compress_embedding_artifact_key(source_key, config)
        if (
            self.cache_config.enabled
            and not self.cache_config.force_recompute
            and store.exists(compression_key)
        ):
            metadata = store.get_json(compression_key)
            metadata["cache_hit"] = True
            return store.get_array(compression_key), metadata

        compression_result = compress_embeddings(embeddings, config=config, y=labels)
        metadata = dict(compression_result.metadata)
        metadata["cache_key"] = compression_key
        metadata["cache_hit"] = False
        if self.cache_config.enabled:
            store.put_array(compression_key, compression_result.embeddings)
            store.put_json(compression_key, metadata)
        return compression_result.embeddings, metadata

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
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
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

    def _stream_multi_embeddings(
        self,
        extractor: Any,
        dataset: Any,
        store: ArtifactStore,
        base_key: str,
        recipe: dict,
        probe_plan: Optional[Tuple[SampleBatch, Any, Any]] = None,
    ) -> List[dict]:
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
            first_outputs = self._embed_batch_many(extractor, first_batch)
            estimates, aggregate = self._estimate_multi_output_memory(
                first_outputs,
                n_samples=n_samples,
            )
            self._admit_multi_embedding_plan(aggregate)
        else:
            first_batch, first_outputs, estimate_info = probe_plan
            estimates = estimate_info["per_output"]
            aggregate = estimate_info
            try:
                skipped_batch = next(batch_iterator)
            except StopIteration as exc:
                raise ValueError("At least one sample is required for embedding.") from exc
            if not np.array_equal(skipped_batch.indices, first_batch.indices):
                raise ValueError("Reusable embedding probe does not match streaming batch order.")

        collected = {
            output.name: [(first_batch.indices, output.embeddings)]
            for output in first_outputs
        }
        output_metadata = {output.name: dict(output.metadata) for output in first_outputs}
        output_recipe = {
            output.name: self._qualified_output_recipe(recipe, output)
            for output in first_outputs
        }
        for batch in batch_iterator:
            outputs = self._embed_batch_many(extractor, batch)
            for output in outputs:
                if output.name not in collected:
                    raise ValueError(
                        f"Extractor '{extractor.name}' returned unexpected output '{output.name}'."
                    )
                collected[output.name].append((batch.indices, output.embeddings))

        variants = []
        for output_name, batches in collected.items():
            embeddings = _combine_embedding_batches(batches, n_samples=n_samples)
            cache_key = self._output_cache_key(base_key, output_name)
            metadata = self._embedding_metadata(
                extractor=extractor,
                dataset=dataset,
                embeddings=embeddings,
                cache_key=cache_key,
                recipe=output_recipe[output_name],
                extractor_name=_qualified_output_name(extractor.name, output_name),
                parent_extractor_name=extractor.name,
                output_name=output_name,
                extractor_recipe=recipe,
                output_metadata=output_metadata[output_name],
            )
            metadata["streamed"] = True
            metadata["stream_batch_size"] = self.embedding_config.batch_size
            metadata["memory_estimate"] = estimates[output_name].to_dict()
            metadata["multi_output_memory_estimate"] = {
                key: value for key, value in aggregate.items() if key != "per_output"
            }
            if self.cache_config.enabled:
                store.put_array(cache_key, embeddings)
            variants.append({"embeddings": embeddings, "metadata": metadata})
        return variants

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

    def _embed_batch_many(self, extractor: Any, batch: SampleBatch) -> List[EmbeddingOutput]:
        return self._validate_multi_outputs(
            extractor=extractor,
            outputs=extractor.transform_many(batch.X),
            expected_rows=len(batch.indices),
            context="batch embeddings",
        )

    def _embedding_metadata(
        self,
        extractor: Any,
        dataset: Any,
        embeddings: Any,
        cache_key: str,
        recipe: dict,
        extractor_name: Optional[str] = None,
        parent_extractor_name: Optional[str] = None,
        output_name: Optional[str] = None,
        extractor_recipe: Optional[dict] = None,
        output_metadata: Optional[dict] = None,
    ) -> dict:
        sparse_embeddings = is_sparse_matrix(embeddings)
        return {
            "extractor_name": extractor_name or extractor.name,
            "parent_extractor_name": parent_extractor_name,
            "output_name": output_name,
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
            "extractor_recipe": extractor_recipe or recipe,
            "output_metadata": output_metadata or {},
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

    def _admit_multi_resident_embeddings(self, outputs: List[EmbeddingOutput]) -> None:
        required = sum(estimate_matrix_resident_bytes(output.embeddings) for output in outputs)
        assert_within_memory(
            required,
            self.memory_config,
            purpose="Resident embedding artifacts",
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

    def _supports_transform_many(self, extractor: Any) -> bool:
        if not callable(getattr(extractor, "transform_many", None)):
            return False
        if not callable(getattr(extractor, "output_specs", None)):
            return False
        return len(list(extractor.output_specs())) > 1

    def _output_specs(self, extractor: Any) -> List[Any]:
        specs = list(extractor.output_specs())
        if not specs:
            raise ValueError(f"Extractor '{extractor.name}' returned no output specs.")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Extractor '{extractor.name}' output names must be unique.")
        return specs

    def _validate_multi_outputs(
        self,
        extractor: Any,
        outputs: Any,
        expected_rows: int,
        context: str,
    ) -> List[EmbeddingOutput]:
        materialized = list(outputs)
        expected_names = [spec.name for spec in self._output_specs(extractor)]
        actual_names = [output.name for output in materialized]
        if set(actual_names) != set(expected_names):
            raise ValueError(
                f"Extractor '{extractor.name}' returned outputs {sorted(actual_names)} for "
                f"{context}, expected {sorted(expected_names)}."
            )
        validated = []
        for output in materialized:
            embeddings = ensure_numeric_matrix(
                output.embeddings,
                f"Extractor '{extractor.name}' output '{output.name}' {context}",
                allow_sparse=True,
            )
            if embeddings.shape[0] != expected_rows:
                raise ValueError(
                    f"Extractor '{extractor.name}' output '{output.name}' returned "
                    f"{embeddings.shape[0]} embeddings for {expected_rows} labels."
                )
            validated.append(
                EmbeddingOutput(
                    name=output.name,
                    embeddings=embeddings,
                    recipe=dict(output.recipe),
                    metadata=dict(output.metadata),
                )
            )
        return sorted(validated, key=lambda item: expected_names.index(item.name))

    def _estimate_multi_output_memory(
        self,
        outputs: List[EmbeddingOutput],
        n_samples: int,
    ) -> Tuple[dict[str, EmbeddingMemoryEstimate], dict[str, Any]]:
        estimates = {
            output.name: estimate_embedding_from_probe(
                output.embeddings,
                n_samples=n_samples,
                batch_size=self.embedding_config.batch_size,
                memory_config=self.memory_config,
            )
            for output in outputs
        }
        aggregate = {
            "resident_bytes": sum(estimate.resident_bytes for estimate in estimates.values()),
            "dense_scoring_bytes": max(
                estimate.dense_scoring_bytes for estimate in estimates.values()
            ),
            "batch_embedding_bytes": sum(
                estimate.batch_embedding_bytes for estimate in estimates.values()
            ),
            "strategy": (
                "stream_to_disk"
                if any(estimate.strategy == "stream_to_disk" for estimate in estimates.values())
                else "in_memory"
            ),
        }
        return estimates, aggregate

    def _admit_multi_embedding_plan(self, aggregate: dict[str, Any]) -> None:
        batch_required = (
            int(aggregate["batch_embedding_bytes"])
            + self.memory_config.model_memory_bytes
            + self.memory_config.raw_batch_memory_bytes
        )
        assert_within_memory(
            batch_required,
            self.memory_config,
            purpose="Embedding batch",
        )
        assert_within_memory(
            int(aggregate["resident_bytes"]),
            self.memory_config,
            purpose="Resident embedding artifacts",
        )
        assert_within_memory(
            int(aggregate["dense_scoring_bytes"]),
            self.memory_config,
            purpose="Dense scoring input",
        )

    def _output_cache_key(self, base_key: str, output_name: str) -> str:
        safe_name = str(output_name).replace("/", "_")
        return f"{base_key}/outputs/{safe_name}"

    def _qualified_output_recipe(self, recipe: dict, output: EmbeddingOutput) -> dict:
        qualified = dict(recipe)
        qualified.pop("outputs", None)
        qualified.update(output.recipe)
        qualified["output_name"] = output.name
        return qualified


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


def _variant_extractor_name(name: str, compression_metadata: dict) -> str:
    method = compression_metadata.get("method", "none")
    if method == "none":
        return name
    precision = compression_metadata.get("precision")
    if precision:
        return f"{name}[{method}_{precision}]"
    compressed_dim = compression_metadata.get("compressed_dim")
    if compressed_dim is None:
        return f"{name}[{method}]"
    return f"{name}[{method}_{compressed_dim}]"


def _qualified_output_name(parent_name: str, output_name: str) -> str:
    return f"{parent_name}:{output_name}"


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
