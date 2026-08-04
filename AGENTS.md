# AGENTS.md

This file provides persistent project guidance for Codex and other coding agents working on `vertebrae`.

`vertebrae` is a Python package for evaluating feature extractors and transfer-learning backbones. Its primary workflow evaluates labeled embeddings with the external `overlapindex` package and optional `separatix` complexity diagnostics. Dedicated workflows also cover exact retrieval, zero-shot semantic alignment, structured model outputs, and dense segmentation. Results are exposed as practical Python objects and JSON/Markdown reports for data scientists and machine learning practitioners.

Treat the repository as the source of truth. The project now includes single-label, multi-label, regression, named target-view, hierarchical label-view, structured-unit, dense segmentation, retrieval, and zero-shot workflows; custom embedding metrics; optional compression; sparse embedding support; memory-aware streaming; a CLI; local/Ray/Dask execution backends; local/S3/GCS artifact stores; and many optional extractor families.

## Alpha Compatibility Policy

`vertebrae` is in an unreleased alpha state. There are no external legacy contracts that must be preserved, and obsolete APIs do not need a slow deprecation cycle. Prefer the clearest coherent design over compatibility shims, aliases, or staged deprecations when changing pre-release behavior. Remove or replace stale interfaces directly, update all in-repository callers, tests, examples, serialized recipes/artifacts where relevant, and documentation in the same change. Still avoid gratuitous churn: preserve a contract when it remains the best design, not merely because it already exists.

## Project Goals

- Benchmark feature extractors and transfer-learning backbones on labeled datasets.
- Support single-extractor diagnostics and multi-extractor comparisons.
- Support single-label classification, multi-label classification, explicit regression targets, hierarchical label views, and dense segmentation token evaluation where implemented.
- Support named target views and structured embedding units such as regions, tokens, frames, keypoints, depth cells, and latent slots.
- Support dedicated retrieval and zero-shot protocols without conflating them with labeled-overlap benchmarking.
- Allow explicit custom embedding metrics while retaining OverlapIndex as the default labeled-embedding metric.
- Support multi-output extraction from one backbone when users need to compare layers, pooling strategies, heads, or spatial outputs within the same run.
- Use `overlapindex` as the metric engine and `separatix` as the optional complexity-diagnostic engine.
- Keep extraction, caching, compression, scoring, stability, diagnostics, execution, and reporting as separate stages.
- Keep workflows reproducible through serialized recipes, artifact keys, metadata, and report data.
- Preserve local-first ergonomics while keeping artifact-backed distributed workflows usable.

## Package And Tooling

- Package name: `vertebrae`.
- Use Poetry for dependency management and packaging.
- Use the `src/` layout.
- Current Python target in `pyproject.toml`: `>=3.9,<3.15`.
- Current metric dependencies in `pyproject.toml`: `overlapindex>=0.1.3a3` and `separatix>=0.1.0a4`.
- Public CLI entry point: `vertebrae = vertebrae.cli:main`.
- Core dependencies include NumPy, SciPy, scikit-learn, pandas, joblib, pydantic, psutil, pillow, overlapindex, and separatix.
- Optional extras currently include `hf`, `audio`, `timeseries`, `video`, `torch`, `timm`, `torchvision`, `openclip`, `mlp`, `keras`, `tensorflow`, `tensorflow-hub`, `jax`, `trees`, `graph`, `onnx`, `ray`, `dask`, `distributed`, `s3`, `gcs`, and `cloud`.
- Prefer lazy imports and clear install hints for optional dependencies.

Current package structure:

```text
src/vertebrae/
  __init__.py
  adapters.py
  benchmark.py
  cli.py
  config.py
  evaluator.py
  results.py
  retrieval.py
  segmentation.py
  structured.py
  zero_shot.py
  cache/
  compression/
  datasets/
  execution/
  extractors/
  reports/
  scoring/
  utils/
tests/
docs/
examples/
```

## Hard Metric Rule

`vertebrae` must depend on `overlapindex` and must not reimplement OverlapIndex or ContinuousOverlapIndex. This rule governs overlap-based labeled-embedding scoring; the dedicated retrieval and zero-shot scorers implement different protocols and are not substitutes for OverlapIndex.

All direct calls to `overlapindex.OverlapIndex` or `overlapindex.ContinuousOverlapIndex` must go through the internal scoring adapter in `src/vertebrae/scoring/overlap.py`, currently `OverlapIndexScorer`. No other package module, test helper aside, should instantiate them directly.

Classification and multi-label scoring use MiniBatchKMeans-backed `OverlapIndex` internally:

```python
OverlapIndex(
    model_type="MiniBatchKMeans",
    kmeans_k=resolved_k,
    kmeans_kwargs=resolved_kmeans_kwargs,
    offline_chunk_size=offline_chunk_size,
)
```

Regression scoring uses `ContinuousOverlapIndex` internally through `ContinuousOverlapScoringConfig`.

Do not expose a public API option that lets users choose an OverlapIndex backend. Do not expose or document these as user-facing options:

- `model_type`
- `rho`
- `r_hat`
- `match_tracking`
- Fuzzy ART
- Hypersphere ART
- ARTMAP
- KMeans backend selection

Users may configure MiniBatchKMeans-relevant and implemented overlap options, such as:

- `k`
- `min_k`
- `max_k`
- `min_samples_per_cluster`
- `kmeans_kwargs`
- `offline_chunk_size`
- `normalize_embeddings`
- `max_dense_bytes`
- `exclude_classes`

## Current Public Workflow

Primary Python API:

- `BenchmarkDataset` creates validated classification, multi-label, regression, embedding, multimodal, graph, relational embedding, and grouped datasets.
- `EmbeddingUnitDataset` and structured adapters align model-emitted units with unit-level annotations before ordinary labeled-embedding scoring.
- `SegmentationDataset` plus spatial extractors materialize dense segmentation token datasets.
- `Evaluator` is the single-extractor convenience wrapper.
- `Benchmark` runs one or more extractors and returns a `BenchmarkResult`.
- `BenchmarkResult` can rank results, convert rankings to a pandas DataFrame, and write JSON or Markdown reports.
- `RetrievalDataset` and `RetrievalBenchmark` evaluate exact query-gallery retrieval and return `RetrievalBenchmarkResult`.
- `ZeroShotDataset` and `ZeroShotBenchmark` evaluate frozen sample embeddings against frozen prompt prototypes and return `ZeroShotBenchmarkResult`.

One extractor may contribute one or many embedding outputs. Multi-output runs should behave like a structured expansion of a single extractor recipe, not like ad hoc duplication of unrelated extractor classes.

The benchmark flow is:

1. Validate dataset and target metadata.
2. Prepare label views or segmentation token materialization when configured.
3. Prepare memory-aware subsampling when configured or necessary.
4. Fit and run extractor, either in one pass or streaming batches.
5. Expand extractor outputs when an extractor is configured to emit multiple embeddings.
6. Cache raw embedding artifacts when enabled.
7. Apply zero, one, or many compression configs.
8. Score embeddings through the configured `EmbeddingMetric`; the default `OverlapMetric` delegates to `OverlapIndexScorer`.
9. Run stability analysis when enabled.
10. Run gated Separatix diagnostics, including its probe diagnostics, when enabled and the overlap threshold passes.
11. Aggregate warnings, weakest-class diagnostics, recommendations, runtime metadata, and reproducibility metadata.
12. Render reports from serialized result data.

Avoid hidden global state. Each stage should have explicit inputs and outputs.

## Dataset Guidance

`BenchmarkDataset` lives in `src/vertebrae/datasets/base.py` and currently supports:

- `from_arrays(...)`
- `from_dataframe(...)`
- `from_embeddings(...)`
- `from_image_paths(...)`
- `from_audio_paths(...)`
- `from_audio_arrays(...)`
- `from_video_paths(...)`
- `from_video_arrays(...)`
- `from_time_series(...)`
- `from_multimodal(...)`
- `from_graphs(...)`
- `from_segmentation_embeddings(...)`
- `from_embedding_units(...)`
- `from_node_embeddings(...)`
- `from_entity_embeddings(...)`
- `from_edge_embeddings(...)`
- `from_pair_embeddings(...)`
- `from_triplet_embeddings(...)`

Validation currently enforces:

- `X` and `y` lengths match.
- at least one sample is present.
- classification and multi-label targets have at least two classes/labels with at least two samples per class or active label.
- explicit regression targets have at least three samples and at least one non-constant target.
- target metadata is normalized into `target_type`, `label_names`, or `target_names` as appropriate.

Supported target modes are:

- single-label categorical classification, inferred by default from one-dimensional labels;
- multi-label classification, using a 2D indicator matrix or a 1D sequence of label sets;
- explicit regression, only when `target_type="regression"` is passed.

Datasets also provide:

- `with_label_hierarchy(...)`, `label_view(...)`, and `active_label_view()` for hierarchical label projections.
- `with_target_views(...)`, `target_view(...)`, `target_view_names()`, and `active_target_view()` for named aligned classification, multi-label, or regression targets.
- `with_unit_annotations(...)` for attaching structured unit-level annotations.
- `with_groups(...)` and `groups()` for aligned independence groups used by Separatix and segmentation diagnostics.
- `iter_batches(...)` for streaming-safe extractors and sharded execution.
- `stratified_subsample_indices(...)` for classification/multi-label subsampling and random regression subsampling.
- `subset(...)` while preserving original sample indices, hierarchy paths, and groups in metadata.
- `summary()` and `fingerprint()` for reports and artifact keys.

Keep labels in their original semantic form where possible. Preserve modality, target metadata, label views, groups, and provenance because reports and artifact keys depend on them.

Relational embedding constructors remain supervised labeled-embedding diagnostics. They preserve metadata such as `relational_unit`, endpoint ids, graph ids, pair ids, triplet ids, composition method, and optional `edge_index`. Do not silently treat these constructors as retrieval datasets. Retrieval is implemented separately through `RetrievalDataset`, `RetrievalScorer`, and `RetrievalBenchmark`; it supports exact query-gallery ranking metrics including recall, precision, mAP, MRR, and NDCG at configured cutoffs. It is not a recommender-training or graph-link-prediction framework.

## Structured Output Guidance

Structured output support lives in `src/vertebrae/adapters.py`, `src/vertebrae/datasets/embeddings.py`, `src/vertebrae/extractors/structured.py`, and `src/vertebrae/structured.py`.

- Use `EmbeddingUnitDataset`, `UnitAnnotation`, and `TargetView` for labeled model-emitted units and aligned alternate targets.
- Use `StructuredOutputSpec` / `StructuredEmbeddingOutput` and `CallableStructuredExtractor` / `PrecomputedStructuredExtractor` for outputs whose row geometry must be declared and aligned.
- Existing adapters cover regions/detection layouts, sequences, keypoints, depth cells, and latent slots. Keep adapter behavior explicit and serializable.
- `StructuredUnitAligner` plus row-selection policies align emitted rows to annotations; never assume arbitrary token/frame/region ordering.
- Materialization should produce ordinary `BenchmarkDataset` rows with preserved parent-sample ids, unit ids, groups, annotations, target views, and provenance.
- Keep structured artifact materialization deterministic and independently addressable in local and distributed workflows.

## Segmentation And Spatial Guidance

Dense segmentation support lives in `src/vertebrae/datasets/segmentation.py`, `src/vertebrae/extractors/spatial.py`, and `src/vertebrae/segmentation.py`.

- Use `SegmentationDataset` for image samples paired with semantic, instance, or panoptic-style raster annotations.
- Use `SpatialLayout`, `SpatialOutputSpec`, and `SpatialEmbeddingOutput` to declare spatial feature geometry explicitly.
- `CallableSpatialExtractor` and `PrecomputedSpatialExtractor` cover custom and precomputed spatial outputs; Torch, Keras, and Hugging Face vision/video paths may also expose spatial outputs.
- `SegmentationConfig` controls coverage thresholds, ambiguity filtering, background behavior, thing/stuff inclusion, per-class and per-instance caps, and deterministic sampling.
- Materialization flattens retained spatial cells into a grouped `BenchmarkDataset` whose groups identify source images.
- Keep segmentation workflows deterministic and preserve token provenance, layout metadata, class/background handling, and image groups.

Segmentation evaluation does not currently support regression targets.

## Extractor Guidance

All extractors should satisfy the protocol in `src/vertebrae/extractors/base.py`:

```python
fit(X, y=None)
transform(X)
fit_transform(X, y=None)
recipe()
```

Extractor outputs must be numeric dense matrices or scipy sparse matrices. Spatial extractors return declared per-image spatial tensors that are materialized into numeric token embeddings before scoring. Recipes must be serializable enough for cache keys and reports. Do not couple scoring or reporting to live model objects.

Extractors may be single-output or multi-output. A multi-output extractor should surface explicit, serializable output specifications such as hidden layer, pooling choice, head, or spatial layout, and each emitted embedding must remain independently cacheable, compressible, scoreable, and reportable.

Implemented extractor families include:

- `PrecomputedExtractor`
- `MultiOutputExtractor`
- `SklearnExtractor`
- `CallableExtractor`
- `TorchExtractor`
- `KerasExtractor`
- `ONNXExtractor`
- `SentenceTransformerExtractor`
- `HFTextExtractor`
- `HFAudioExtractor`
- `HFTimeSeriesExtractor`
- `HFVisionExtractor`
- `HFVideoExtractor`
- `HFMultimodalExtractor`
- `TimmVisionExtractor`
- `TorchvisionVisionExtractor`
- `OpenCLIPExtractor`
- `SigLIPExtractor`
- `TFHubExtractor`
- `JAXFlaxExtractor`
- `TreeLeafEmbeddingExtractor`
- `GraphModelExtractor`
- `HostedEmbeddingExtractor`
- `CallableRetrievalExtractor`
- `CallableSpatialExtractor`
- `PrecomputedSpatialExtractor`
- `CallableStructuredExtractor`
- `PrecomputedStructuredExtractor`

Optional extractors must lazy-load their heavy dependencies and raise actionable `ImportError` messages when extras are missing. Tests use fake modules heavily; keep optional-dependency behavior testable without network access, GPUs, or real model downloads.

Streaming-safe extractors should set `streaming_safe = True` only when independent batches can be transformed without access to the full dataset. For local Torch, Keras, ONNX, and Hugging Face wrappers, keep user-supplied adapter functions explicit (`collate_fn`, `output_fn`, `input_fn`, processor hooks, pooling/output specs, etc.) rather than guessing arbitrary model conventions.

For timm, torchvision, OpenCLIP/SigLIP, TensorFlow Hub, JAX/Flax, tree-leaf, graph, and hosted embedding wrappers, preserve the same adapter-first posture: users should explicitly choose preprocessing, branch/output selection, batching, request behavior, and serialization metadata. Hosted embedding API wrappers must make retry, batching, provider/model identity, and cache policy visible in recipes; do not hide network calls behind default tests.

Graph extractors should be clear about output level (`graph`, `node`, or `edge`) in recipe metadata. Graph-level datasets use `from_graphs(...)`; node/edge transfer diagnostics should materialize labeled rows through `from_node_embeddings(...)` or `from_edge_embeddings(...)` rather than adding graph-task metrics to the overlap benchmark.

Prefer extending an existing extractor to accept multiple output specs over duplicating extractor classes when the underlying model invocation is shared. Keep the single-output path ergonomic. Extractors that fundamentally yield one embedding per sample may remain single-output. Apply the alpha compatibility policy rather than adding compatibility machinery solely for unreleased APIs.

## Retrieval And Zero-Shot Guidance

Retrieval support lives in `src/vertebrae/datasets/retrieval.py`, `src/vertebrae/extractors/retrieval.py`, `src/vertebrae/scoring/retrieval.py`, `src/vertebrae/retrieval.py`, and `src/vertebrae/execution/`. Zero-shot support follows the corresponding dataset, scoring, benchmark, and execution modules.

- Retrieval is exact, training-free query-gallery evaluation. `RetrievalConfig` controls cosine/dot/squared-L2 similarity, cutoffs, the primary NDCG metric, batching, bidirectionality, pairwise limits, and worst-query reporting.
- Preserve query ids, gallery ids, relevance judgments, modality/branch identity, and exclusions. Enforce memory and pairwise-comparison guards.
- `RetrievalCapableExtractor` or explicit branch-aware adapters should encode query and gallery inputs. Do not guess multimodal branch conventions.
- Zero-shot compares frozen sample embeddings with frozen text prompt prototypes. `ZeroShotClassSpec` defines class prompts and `ZeroShotConfig` controls similarity, top-k cutoffs, the primary classification metric, memory bounds, and worst-sample reporting.
- Do not add learned heads, fitted calibration, prompt search, or training to the zero-shot protocol without making it a distinct explicit workflow.
- Retrieval and zero-shot artifacts have their own keys, shard/merge/compression/scoring jobs, and result builders. Preserve protocol metadata and deterministic ordering rather than forcing them through labeled-overlap artifacts.

## Scoring And Diagnostics Guidance

The general metric protocol lives in `src/vertebrae/scoring/metrics.py`.

- `EmbeddingMetric` is the explicit scoring protocol and returns `MetricResult`.
- `OverlapMetric` is the default adapter around `OverlapIndexScorer`.
- `CallableMetric` supports user metrics, including import-path reconstruction for serialized/distributed workflows.
- `LabelRetrievalMetric` is a labeled-neighbor diagnostic for embedding rows; do not confuse it with the query-gallery retrieval benchmark.
- Custom metrics must declare stable recipes and must not be presented as OverlapIndex results. Separatix and overlap-specific stability behavior should run only where semantically applicable.

`OverlapScoringConfig` currently contains:

```python
OverlapScoringConfig(
    k="auto",
    min_k=10,
    max_k=50,
    min_samples_per_cluster=5,
    kmeans_kwargs=None,
    offline_chunk_size=10_000,
    normalize_embeddings=True,
    max_dense_bytes=2_000_000_000,
    exclude_classes=None,
)
```

`ContinuousOverlapScoringConfig` supports explicit regression targets with fields including `k`, `kmeans_kwargs`, `offline_chunk_size`, `normalize_embeddings`, `max_dense_bytes`, `target_cover`, `n_target_cells`, `target_distance`, `target_scaling`, `n_projections`, `n_null_permutations`, `aggregation`, and `clip`.

Scoring rules:

- Resolve classification `k` per class with `resolve_kmeans_k(...)`.
- If requested or automatic `k` is reduced because a class is too small, surface a warning in the result and reports.
- L2-normalize embeddings by default unless the user disables it.
- Sparse embeddings are allowed upstream and in artifact stores, but are densified only inside scoring/diagnostic adapters when required, guarded by memory limits.
- Preserve diagnostics when available: macro score, per-class or per-target scores, pairwise scores, sparse adjacency, class counts, resolved `k` per class, excluded classes, warnings, and metadata.
- `exclude_classes` is for classification aggregation/reporting, commonly for background classes; do not use it for regression.

`SeparatixConfig` controls optional complexity diagnostics:

- Separatix is enabled by default but gated by overlap thresholds.
- Classification/multi-label diagnostics use `overlap_threshold`; regression diagnostics use `regression_overlap_threshold`.
- Separatix receives normalized embeddings consistently with the overlap config, optional groups, target mode, budget, sample caps, dense memory limits, and optional MLP probe settings.
- Separatix recommendations are complementary diagnostics; they must not replace or mutate overlap scores or rankings.

## Stability And Separatix Probes

`StabilityConfig` supports:

- `mode="prototype"`: fixed embeddings and labels, repeated MiniBatchKMeans seeds.
- `mode="subsample"`: repeated subsamples without replacement.
- `mode="none"` or `enabled=False` to skip stability.

Default wording should be "stability interval," not "confidence interval." Do not imply a formal population confidence interval unless a future statistical bootstrap mode is explicitly implemented.

Probe summaries come from Separatix rather than a vertebrae `ProbeConfig`. Preserve Separatix's target-appropriate primary metric, evaluation/grouping metadata, sampling status, and linear/nonlinear comparison. Optional MLP probes are configured through `SeparatixConfig`. Probe results are secondary diagnostics; keep recommendations modest and tied to the evaluated dataset and protocol.

## Compression Guidance

Compression lives in `src/vertebrae/compression/` and is configured through `EmbeddingCompressionConfig`.

Supported methods:

- `none`
- `pca`
- `incremental_pca`
- `truncated_svd`
- `gaussian_random_projection`
- `sparse_random_projection`
- `prefix_truncate`
- `quantize`

Paired query/gallery and sample/prototype compression helpers live in `src/vertebrae/compression/paired.py`. Fit shared learned transforms on the declared reference side and apply the same transform consistently to both spaces. Keep separate artifacts and provenance for each side.

Compression runs after raw embedding generation and before scoring, stability, and Separatix diagnostics. Raw embedding artifacts should remain reusable; compressed variants use derived artifact keys based on the compression recipe.

Important constraints:

- PCA and incremental PCA require dense input.
- `truncated_svd` is the preferred sparse text reduction path.
- `prefix_truncate` should warn unless `assume_matryoshka=True`.
- Quantization currently supports `float16`, `int8`, and `uint8`; binary or packed-bit quantization is intentionally not part of the package.
- Multiple extractor outputs and multiple compression configs may each expand one extractor into multiple `ExtractorResult` variants.

## Cache And Artifact Stores

Artifact-store abstractions live in `src/vertebrae/cache/`.

Implemented stores:

- `LocalArtifactStore`
- `S3ArtifactStore`
- `GCSArtifactStore`

Use `create_artifact_store(...)` for local paths, `file://`, `s3://`, and `gs://` URIs. S3 and GCS dependencies are optional and must remain lazy.

Artifact stores persist:

- dense arrays as `.npy`
- sparse arrays as `.npz`
- labels and groups as JSON-compatible artifacts
- metadata and manifests as JSON

For multi-output extractors and segmentation materialization, artifact workflows should preserve deterministic per-output boundaries. Each output needs its own metadata and embedding artifact identity so distributed scoring, recompression, diagnostics, and resume workflows can address outputs independently.

Cache keys should be based on dataset fingerprints, extractor recipes, compression recipes, scoring seeds where relevant, target metadata, label views, and package metadata where practical. Avoid cache keys that depend on non-serializable live objects.

## Execution And CLI Guidance

Execution code lives in `src/vertebrae/execution/`.

Implemented backends:

- `LocalBackend`
- `RayBackend`
- `DaskBackend`

Implemented execution primitives include:

- deterministic `ShardSpec`
- `EmbeddingShardJob`
- `EmbeddingMergeJob`
- `CompressionJob`
- `ScoringJob`
- `SeparatixJob`
- embedding shard planning/materialization/merge
- label and group artifact materialization
- segmentation artifact materialization
- persisted embedding scoring
- persisted Separatix diagnostics
- repeated scoring and score collection
- benchmark result construction from artifacts
- structured artifact materialization
- retrieval shard planning/materialization/merge, paired compression, scoring, relevance artifacts, and score collection
- zero-shot shard planning/materialization/merge, protocol artifacts, paired compression, scoring, and result construction

Distributed materialization should support multi-output extractors by writing one artifact set per declared output rather than collapsing outputs into a single opaque blob. Keep manifests explicit about which extractor output, target type, label view, groups, and segmentation metadata each artifact represents.

The CLI in `src/vertebrae/cli.py` supports artifact-backed workflows including:

- `plan`
- `embed-shard`
- `merge-embeddings`
- `write-labels`
- `write-groups`
- `materialize-segmentation`
- `score`
- `diagnose-complexity`
- `compress`
- `score-repeats`
- `collect-scores`
- `benchmark-from-artifacts`
- `slurm-array`
- `slurm-score-array`
- `run-embedding-shards`
- `plan-retrieval`, `embed-retrieval-shard`, `merge-retrieval-embeddings`, `write-retrieval-relevance`, `score-retrieval`, and `compress-retrieval`
- `plan-zero-shot`, `embed-zero-shot-shard`, `merge-zero-shot-embeddings`, `write-zero-shot-protocol`, `score-zero-shot`, `compress-zero-shot`, and `zero-shot-from-artifacts`
- `materialize-structured`

Distributed work should remain artifact-backed. Workers should consume serialized dataset/extractor/config objects or persisted artifacts and emit explicit manifests. Keep row order deterministic and prevent duplicate shard writes.

When extending CLI or execution code, preserve parity between local and distributed multi-output, target-view, label-view, grouped, regression, structured, segmentation, retrieval, and zero-shot behavior where the protocol supports the stage. A user should be able to materialize, merge, compress, score, diagnose, and collect results for each output without special one-off handling outside its normal artifact workflow.

## Memory Guidance

`MemoryConfig` and `src/vertebrae/utils/memory.py` provide memory budgeting and admission checks.

Current behavior includes:

- deriving a budget from `psutil` when no explicit byte limit is supplied,
- probing streaming-safe extractors to infer embedding shape and dtype,
- estimating resident embedding bytes and dense scoring bytes,
- optionally stratified classification/multi-label subsampling or random regression subsampling when a full run would exceed memory,
- preserving memory estimates and subsampling metadata in results.

Do not remove memory guards around sparse-to-dense conversion, continuous scoring, Separatix diagnostics, segmentation token materialization, or streaming materialization. When adding a new workflow that may allocate a dense matrix, route it through existing validation and memory helpers.

## Reporting Guidance

Reports should be useful to practitioners, not just metric dumps.

Current reports include:

- dataset summary, target type, label view, grouping, and segmentation summaries when present,
- extractor summary,
- ranking table for multi-extractor, multi-output, label-view, segmentation, and compression-variant runs,
- global and per-class OverlapIndex scores or continuous overlap scores for regression,
- weakest class/target diagnostics when available,
- stability summary,
- probe summary,
- Separatix recommendation, confidence, skipped reason, and report details when available,
- compression metadata,
- warnings,
- recommendations,
- reproducibility metadata.

Reports must render from `BenchmarkResult` / `ExtractorResult` data, not from live extractor/model objects.

Result names and report text should make multi-output and label-view variants legible. Prefer stable names that preserve the parent extractor identity while distinguishing the output and active view, such as layer, head, pooling, spatial-output, or hierarchy-level suffixes.

Recommendations should be simple and transparent. Phrase conclusions as representation diagnostics for the given dataset, target type, and protocol, not universal model quality.

## Documentation And Examples

Docs live in `docs/` and examples live in `examples/`.

When changing behavior, update the relevant docs page:

- dataset, target, label hierarchy, grouping, or multimodal behavior: `docs/datasets.md`
- extractor behavior: `docs/feature_extractors.md`
- structured units and adapters: README, examples, and the nearest dataset/extractor documentation until a dedicated page exists
- scoring, continuous overlap, stability, probes, or Separatix: `docs/scoring.md`
- compression: `docs/compression.md`
- segmentation behavior: `docs/segmentation.md`
- execution, artifact stores, CLI, SLURM, Ray, or Dask: `docs/distributed_readiness.md`
- report fields: `docs/results_and_reports.md`
- retrieval and zero-shot behavior: README/examples and their implementation tests until dedicated docs pages exist
- runnable workflows: `docs/examples.md`, `examples/README.md`, or example scripts

README examples currently cover precomputed embeddings, multi-label and regression targets, compression, scikit-learn pipelines, local Torch/Keras, Hugging Face audio/time-series/vision/video/text/multimodal extractors, sentence-transformers, multi-extractor comparisons, reports, Separatix, label views, dense segmentation tokens, optional extractor families, and CLI/distributed workflows. Keep README claims aligned with implemented tests.

Examples should include realistic multi-output workflows, such as comparing intermediate and final Hugging Face hidden states from the same model, and should stay network-free or clearly optional unless they are explicitly framed as external-model examples.

## Testing Expectations

Add or update tests for every behavioral change.

The current test suite covers:

- dataset validation, batching, subsetting, modalities, groups, label views, multi-label targets, regression targets, graph inputs, relational embedding constructors, and segmentation embedding constructors,
- auto-k behavior, hidden backend selection, and MiniBatchKMeans-only scorer behavior,
- ContinuousOverlapIndex regression scoring through the adapter,
- Separatix configuration, gating, groups, multi-label/regression target modes, reports, and optional MLP probe wiring,
- sparse embedding preservation and scoring-boundary densification,
- precomputed, callable, sklearn, Torch, Keras, ONNX, sentence-transformers, Hugging Face text/audio/time-series/vision/video/multimodal, timm, torchvision, OpenCLIP/SigLIP, TensorFlow Hub, JAX/Flax, tree-leaf, graph, hosted API, and spatial extractors,
- single-extractor, multi-output, label-view, segmentation, and multi-extractor benchmark workflows,
- named target views, structured unit alignment/materialization, custom metrics, retrieval, and zero-shot workflows,
- compression methods and multi-variant results,
- JSON and Markdown reports,
- stability repeat counts and output shape,
- memory planning and auto-subsampling,
- local artifact stores, S3/GCS lazy dependency behavior,
- local/Ray/Dask execution factories and distributed primitives,
- distributed multi-output artifact materialization, merge, scoring, Separatix diagnostics, and result collection,
- CLI planning, sharding, merging, labels/groups, segmentation materialization, compression, scoring, complexity diagnostics, SLURM script generation, and local backend runs.

Use small synthetic datasets and fake optional modules in default tests. Do not require network access, GPUs, real Hugging Face downloads, cloud credentials, or large files.

## Quality Checks

Before considering a code task complete, run relevant checks:

```bash
poetry run pytest
poetry run ruff check .
```

If type checking is relevant or changed files affect typed boundaries, also run:

```bash
poetry run mypy src
```

If a check cannot be run, state why and what remains unverified.

## Coding Style

- Prefer clear, typed Python.
- Keep public APIs small and consistent with existing dataclass-based config/result objects.
- Validate inputs early and fail with actionable error messages.
- Avoid broad exception swallowing.
- Keep optional-dependency imports lazy and errors user-friendly.
- Preserve sparse matrices until a component explicitly requires dense input.
- Keep comments focused on non-obvious design decisions.
- Do not introduce heavyweight dependencies without a strong reason and optional-extra strategy.
- Avoid unrelated refactors while implementing a specific change.

## Task Execution Guidance For Agents

When implementing new functionality:

1. Start with the simplest local, testable path.
2. Preserve the hard metric rule.
3. Keep artifact boundaries explicit.
4. Add tests beside the behavior.
5. Update docs and examples when user-facing behavior changes.
6. Keep optional model/download workflows lazy and network-free in default tests.
7. Preserve deterministic row order for embeddings, shards, reports, segmentation tokens, label views, and groups.
8. Prefer existing validation, serialization, memory, cache, and execution helpers over ad hoc logic.

Do not add ART/ARTMAP backend support unless the project owner explicitly changes the metric contract.
