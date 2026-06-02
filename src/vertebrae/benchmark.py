"""Benchmark runner."""

from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable, List, Optional

import numpy as np

from vertebrae import __version__
from vertebrae.cache.fingerprint import fingerprint_extractor_recipe
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.execution.local import LocalBackend
from vertebrae.reports.recommendations import (
    recommendation_for_extractor,
    recommendations_for_benchmark,
)
from vertebrae.results import BenchmarkResult, ExtractorResult
from vertebrae.scoring.overlap import OverlapIndexScorer
from vertebrae.scoring.probes import run_probes
from vertebrae.scoring.stability import run_stability_analysis


class Benchmark:
    def __init__(
        self,
        dataset: Any,
        extractors: Optional[Iterable[Any]] = None,
        scoring_config: Optional[OverlapScoringConfig] = None,
        stability_config: Optional[StabilityConfig] = None,
        probe_config: Optional[ProbeConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        execution: Optional[Any] = None,
    ) -> None:
        self.dataset = dataset
        self.extractors = list(extractors or [])
        self.scoring_config = scoring_config or OverlapScoringConfig()
        self.stability_config = stability_config or StabilityConfig()
        self.probe_config = probe_config or ProbeConfig()
        self.cache_config = cache_config or CacheConfig()
        self.execution = execution or LocalBackend()

    def add_extractor(self, extractor: Any) -> "Benchmark":
        self.extractors.append(extractor)
        return self

    def run(self) -> BenchmarkResult:
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

        embeddings = extractor.fit_transform(self.dataset.X, self.dataset.y)
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Extractor '{extractor.name}' returned non-2D embeddings "
                f"with shape {embeddings.shape}."
            )
        if len(embeddings) != len(self.dataset.y):
            raise ValueError(
                f"Extractor '{extractor.name}' returned {len(embeddings)} embeddings "
                f"for {len(self.dataset.y)} labels."
            )
        metadata = {
            "extractor_name": extractor.name,
            "extractor_type": getattr(extractor, "extractor_type", "unknown"),
            "modality": getattr(extractor, "modality", self.dataset.modality),
            "cache_hit": False,
            "cache_key": cache_key,
            "shape": list(embeddings.shape),
            "n_samples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "recipe": recipe,
            "extractor_recipe": recipe,
        }
        if self.cache_config.enabled:
            store.put_array(cache_key, embeddings)
            store.put_json(cache_key, metadata)
        return embeddings, metadata


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
