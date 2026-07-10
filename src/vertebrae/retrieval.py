"""Training-free query--gallery benchmark runner."""

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from vertebrae.compression import compress_embeddings
from vertebrae.compression.base import _compression_metadata, create_embedding_compressor
from vertebrae.config import EmbeddingCompressionConfig, RetrievalConfig
from vertebrae.datasets.retrieval import RetrievalDataset
from vertebrae.scoring.retrieval import RetrievalScorer, RetrievalScoreResult
from vertebrae.utils.serialization import make_json_safe


@dataclass
class RetrievalExtractorResult:
    name: str
    extractor_type: str
    forward: RetrievalScoreResult
    reverse: Optional[RetrievalScoreResult]
    primary_score: float
    compression_metadata: Dict[str, Any]
    runtime: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    recipe: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return make_json_safe(self)


@dataclass
class RetrievalBenchmarkResult:
    dataset_summary: Dict[str, Any]
    extractor_results: List[RetrievalExtractorResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ranked_results(self) -> List[RetrievalExtractorResult]:
        return sorted(self.extractor_results, key=lambda item: item.primary_score, reverse=True)

    def to_dataframe(self) -> Any:
        import pandas as pd

        rows = []
        for rank, item in enumerate(self.ranked_results(), start=1):
            row = {
                "rank": rank,
                "extractor": item.name,
                "extractor_type": item.extractor_type,
                "primary_metric": item.forward.primary_metric,
                "primary_score": item.primary_score,
                **item.forward.metrics,
                "compression_method": item.compression_metadata.get("method", "none"),
            }
            if item.reverse:
                row.update({f"reverse_{key}": value for key, value in item.reverse.metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return make_json_safe(self)

    def save_json(self, path: str) -> None:
        from vertebrae.reports.json_report import save_json_report

        save_json_report(self, path)

    def save_markdown(self, path: str) -> None:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_retrieval_markdown_report(self), encoding="utf-8")


class RetrievalBenchmark:
    """Compare frozen extractors under an explicit exact retrieval protocol."""

    def __init__(
        self,
        dataset: RetrievalDataset,
        extractors: Iterable[Any],
        retrieval_config: Optional[RetrievalConfig] = None,
        compression_config: Optional[EmbeddingCompressionConfig] = None,
        compression_configs: Optional[Iterable[EmbeddingCompressionConfig]] = None,
        query_branch: Optional[str] = None,
        gallery_branch: Optional[str] = None,
    ) -> None:
        if compression_config is not None and compression_configs is not None:
            raise ValueError("Provide compression_config or compression_configs, not both.")
        self.dataset = dataset
        self.extractors = list(extractors)
        self.config = retrieval_config or RetrievalConfig()
        self.compression_configs = list(
            compression_configs or [compression_config or EmbeddingCompressionConfig()]
        )
        self.query_branch = query_branch
        self.gallery_branch = gallery_branch
        if (query_branch is None) != (gallery_branch is None):
            raise ValueError("query_branch and gallery_branch must be provided together.")

    def run(self) -> RetrievalBenchmarkResult:
        self.dataset.validated()
        if not self.extractors:
            raise ValueError("RetrievalBenchmark requires at least one extractor.")
        results: List[RetrievalExtractorResult] = []
        for extractor in self.extractors:
            extract_start = perf_counter()
            if self.query_branch is None:
                self._fit_standard_extractor(extractor)
            query_embeddings = self._encode(
                extractor, self.dataset.queries, self.query_branch, self.dataset.query_modality
            )
            gallery_embeddings = self._encode(
                extractor, self.dataset.gallery, self.gallery_branch, self.dataset.gallery_modality
            )
            extract_seconds = perf_counter() - extract_start
            for compression_config in self.compression_configs:
                query_compressed, gallery_compressed, compression_metadata = _compress_pair(
                    query_embeddings, gallery_embeddings, compression_config
                )
                score_start = perf_counter()
                scorer = RetrievalScorer(self.config)
                forward = scorer.score(
                    query_compressed,
                    gallery_compressed,
                    self.dataset.relevance,
                    query_ids=list(self.dataset.query_ids),
                    gallery_ids=list(self.dataset.gallery_ids),
                    exclusions=set(self.dataset.exclusions or ()),
                )
                reverse = None
                if self.config.bidirectional:
                    reverse_relevance, reverse_exclusions = _transpose_relations(
                        self.dataset.relevance,
                        set(self.dataset.exclusions or ()),
                        len(self.dataset.gallery_ids),
                    )
                    if any(not values for values in reverse_relevance.values()):
                        raise ValueError(
                            "bidirectional retrieval requires every gallery item to have an "
                            "eligible reverse relevance relation."
                        )
                    reverse = scorer.score(
                        gallery_compressed,
                        query_compressed,
                        reverse_relevance,
                        query_ids=list(self.dataset.gallery_ids),
                        gallery_ids=list(self.dataset.query_ids),
                        exclusions=reverse_exclusions,
                    )
                primary = (
                    forward.score
                    if reverse is None
                    else float((forward.score + reverse.score) / 2.0)
                )
                results.append(
                    RetrievalExtractorResult(
                        name=_variant_name(extractor.name, compression_metadata),
                        extractor_type=getattr(extractor, "extractor_type", "unknown"),
                        forward=forward,
                        reverse=reverse,
                        primary_score=primary,
                        compression_metadata=compression_metadata,
                        runtime={
                            "extraction_seconds": extract_seconds,
                            "scoring_seconds": perf_counter() - score_start,
                        },
                        warnings=sorted(
                            set(forward.warnings + (reverse.warnings if reverse else []))
                        ),
                        recipe=getattr(extractor, "recipe", lambda: {})(),
                    )
                )
        return RetrievalBenchmarkResult(
            dataset_summary=self.dataset.summary(),
            extractor_results=results,
            metadata={
                "retrieval_config": asdict(self.config),
                "compression_configs": [asdict(config) for config in self.compression_configs],
            },
        )

    def _encode(self, extractor: Any, values: Any, branch: Optional[str], modality: str) -> Any:
        if branch is not None:
            encode = getattr(extractor, "encode_retrieval", None)
            if not callable(encode):
                raise TypeError(
                    "A query_branch/gallery_branch requires an extractor implementing "
                    "encode_retrieval()."
                )
            return encode(values, branch=branch, modality=modality)
        if getattr(extractor, "modality", None) not in {modality, "multimodal", "embeddings"}:
            raise ValueError(
                f"Extractor {extractor.name!r} does not support retrieval modality {modality!r}."
            )
        return extractor.transform(values)

    def _fit_standard_extractor(self, extractor: Any) -> None:
        try:
            extractor.fit(self.dataset.gallery, y=None)
        except TypeError as exc:
            raise TypeError(
                "Standard retrieval extractors must support unsupervised fit(X, y=None) or "
                "be pre-fitted. Use a retrieval branch adapter for asymmetric models."
            ) from exc


def _compress_pair(
    query: Any, gallery: Any, config: EmbeddingCompressionConfig
) -> tuple[Any, Any, Dict[str, Any]]:
    if not config.enabled or config.method == "none":
        gallery_result = compress_embeddings(gallery, config=config)
        return (
            query,
            gallery_result.embeddings,
            {
                **gallery_result.metadata,
                "fit_side": "gallery",
            },
        )
    compressor = create_embedding_compressor(config)
    gallery_result = compressor.fit_transform(gallery)
    query_result = compressor.transform(query)
    metadata = _compression_metadata(compressor, gallery, gallery_result, warnings=[])
    return (
        query_result,
        gallery_result,
        {
            **metadata,
            "fit_side": "gallery",
        },
    )


def _transpose_relations(
    relevance: Dict[int, Dict[int, float]], exclusions: set[tuple[int, int]], n_gallery: int
) -> tuple[Dict[int, Dict[int, float]], set[tuple[int, int]]]:
    transposed: Dict[int, Dict[int, float]] = {}
    for query, values in relevance.items():
        for gallery, grade in values.items():
            transposed.setdefault(gallery, {})[query] = grade
    for index in range(n_gallery):
        transposed.setdefault(index, {})
    return transposed, {(gallery, query) for query, gallery in exclusions}


def _variant_name(name: str, compression: Dict[str, Any]) -> str:
    method = compression.get("method", "none")
    return name if method == "none" else f"{name}[{method}]"


def render_retrieval_markdown_report(result: RetrievalBenchmarkResult) -> str:
    data = result.to_dict()
    lines = ["# vertebrae retrieval report", "", "## Dataset summary", ""]
    for key, value in data["dataset_summary"].items():
        lines.append(f"- {key}: {value}")
    primary_metric = (
        result.ranked_results()[0].forward.primary_metric if result.extractor_results else "score"
    )
    lines.extend(
        [
            "",
            "## Ranking",
            "",
            f"| rank | extractor | primary score | {primary_metric} | mrr | map |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, item in enumerate(result.ranked_results(), start=1):
        metrics = item.forward.metrics
        primary = metrics.get(primary_metric, float("nan"))
        mrr = metrics.get("mrr", float("nan"))
        mean_average_precision = metrics.get("map", float("nan"))
        lines.append(
            f"| {rank} | {item.name} | {item.primary_score:.4f} | {primary:.4f} | "
            f"{mrr:.4f} | {mean_average_precision:.4f} |"
        )
        lines.append("")
        lines.append(f"Forward metrics for `{item.name}`: {metrics}")
        if item.reverse:
            lines.append(f"Reverse metrics: {item.reverse.metrics}")
    return "\n".join(lines) + "\n"
