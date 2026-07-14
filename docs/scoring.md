# scoring

All overlap scoring in `vertebrae` goes through one internal adapter:
`OverlapIndexScorer`.

`vertebrae` can also run a default Separatix complexity diagnostic through the
internal `SeparatixScorer` adapter. Separatix does not affect extractor ranking;
it adds classifier-complexity guidance on top of the overlap result when enabled.
When a dataset declares groups, vertebrae forwards them to Separatix so supervised
evaluation and structural evidence respect those independence units. A grouped
diagnostic that lacks sufficient cross-group class support is recorded as skipped;
vertebrae never retries it with a row-level split.

## Fixed metric backend

`vertebrae` depends on the external `overlapindex` package and does not reimplement
OverlapIndex. Currently, the backend is fixed internally to MiniBatchKMeans:

```python
OverlapIndex(
    model_type="MiniBatchKMeans",
    kmeans_k=resolved_k,
    kmeans_kwargs=resolved_kmeans_kwargs,
    offline_chunk_size=offline_chunk_size,
)
```

This backend choice is not exposed as a public user option.

## OverlapScoringConfig

Use `OverlapScoringConfig` to tune the supported MiniBatchKMeans-facing settings
for classification and multi-label datasets:

```python
from vertebrae import OverlapScoringConfig

config = OverlapScoringConfig(
    k="auto",
    min_k=10,
    max_k=50,
    min_samples_per_cluster=5,
    kmeans_kwargs={"batch_size": 256},
    offline_chunk_size=10_000,
    normalize_embeddings=True,
    exclude_classes=None,
)
```

Key fields:

- `k`: an integer, a class-to-`k` mapping, or `"auto"`.
- `min_k` and `max_k`: bounds used during automatic per-class resolution.
- `min_samples_per_cluster`: guardrail that prevents too many prototypes for a
  small class.
- `kmeans_kwargs`: extra keyword arguments forwarded to MiniBatchKMeans.
- `offline_chunk_size`: chunk size passed to `OverlapIndex.fit_offline(...)`.
- `normalize_embeddings`: enables L2 row normalization before scoring.
- `max_dense_bytes`: caps sparse-to-dense conversion size at the scoring boundary.
- `exclude_classes`: class id or ids retained during fitting and detailed
  diagnostics but omitted from global macro and weighted aggregation.

Do not reconstruct the global score by averaging per-class scores: reporting-only
excluded classes remain present there. `OverlapScoreResult` records both
`macro_score` and `weighted_score`, plus the effective aggregation classes.

## ContinuousOverlapScoringConfig

Explicit regression datasets use `ContinuousOverlapIndex` internally through the
same `OverlapIndexScorer` adapter. Configure that path with
`ContinuousOverlapScoringConfig`:

```python
from vertebrae import ContinuousOverlapScoringConfig

config = ContinuousOverlapScoringConfig(
    k=8,
    target_cover="auto",
    target_distance="auto",
    n_null_permutations=20,
    aggregation="support_weighted",
)
```

Regression scoring keeps the MiniBatchKMeans backend fixed internally and reports
`OverlapScoreResult.score` as the primary continuous overlap score. Values near
`0.5` are null-like, values above `0.5` indicate useful structure, and values
below `0.5` indicate harmful overlap.

## Automatic k resolution

The default `k="auto"` mode resolves a separate `k` for each class using class size,
`min_k`, `max_k`, and `min_samples_per_cluster`. For multi-label targets, class
size means per-label occurrence count. If a class or label is too small to support
the requested or inferred `k`, the scorer reduces it and records a warning.

That warning is carried into the structured result and final reports so users can see
when a weakly represented class forced a smaller prototype count.

## Scoring inputs

The scorer accepts numeric embedding matrices with single-label, multi-label, or
explicit regression targets.

- Dense inputs are scored directly.
- Sparse inputs are validated, then densified only at the OverlapIndex boundary.
- Single-label targets are passed to OverlapIndex as a one-dimensional label array.
- Multi-label targets are normalized to a dense 0/1 indicator matrix before calling
  OverlapIndex.
- Regression targets are passed to ContinuousOverlapIndex as finite numeric 1D or
  2D targets.
- When `normalize_embeddings=True`, embeddings are L2-normalized row-wise before
  scoring.

This keeps extractor and caching layers free to preserve sparse artifacts longer,
while still meeting the MiniBatchKMeans backend requirements.

## Returned diagnostics

`OverlapIndexScorer.score(...)` returns an `OverlapScoreResult` with:

- `score`
- `macro_score`
- `per_class_scores`
- `pairwise_scores`
- `sparse_adjacency`
- `class_counts`
- `k_per_class`
- `warnings`
- `metadata`

The `metadata` payload records backend details such as normalization, chunk size,
seed, KMeans kwargs, and whether the original input was sparse. Regression results
also preserve continuous null-calibration metadata plus prototype-level summaries,
support, adjacency, and loss diagnostics.

## Stability analysis

Repeated scoring is handled by `run_stability_analysis(...)`, which uses
`OverlapIndexScorer` under the hood. The default stability mode is prototype
stability:

- embeddings and labels stay fixed,
- MiniBatchKMeans seeds change across repeats,
- summaries report mean, standard deviation, min, max, and percentile interval.

Dense and scipy sparse embeddings are both preserved across repeats and subsample
slices. Sparse matrices are densified only inside the overlap scoring adapter, where
`OverlapScoringConfig.max_dense_bytes` remains the admission limit.

Subsample stability is also available when requested through `StabilityConfig`.

```python
from vertebrae import OverlapScoringConfig, StabilityConfig

scoring = OverlapScoringConfig()
stability = StabilityConfig(mode="prototype", repeats=20, interval_level=0.95)
```

`vertebrae` reports these as stability summaries and stability intervals. They are
not formal confidence intervals unless a different statistical protocol is added
explicitly in a future release.

## Separatix diagnostics

Use `SeparatixConfig` to control the optional complexity diagnostic stage:

```python
from vertebrae import SeparatixConfig

config = SeparatixConfig(
    enabled=True,
    overlap_threshold=0.80,
    random_state=42,
)
```

Current behavior:

- Separatix runs on the same evaluated embedding variant that overlap scores.
- It runs after compression and after the main overlap score is available.
- By default it only runs when the overlap gate passes.
- Classification and multi-label datasets use `overlap_threshold`.
- Regression datasets use `regression_overlap_threshold`.
- Multi-label targets are passed to Separatix as dense 0/1 indicator matrices with
  `target_mode="multilabel"`.
- Regression targets are passed with `target_mode="regression"`.
- Optional Separatix MLP probes can be enabled with `mlp_probes=True`.
- The full Separatix report is preserved in JSON outputs, while Markdown reports
  show a compact recommendation, confidence, decision path, key scores, probe
  evidence, evaluation context, and skips.

Separatix follows the same normalization convention as overlap scoring when
`normalize_embeddings=True`. Sparse inputs remain sparse at the vertebrae boundary,
and Separatix uses its own densification policy internally.

Separatix is the only downstream-complexity and probe-style diagnostic path in
vertebrae. Ranking remains based on overlap scores; Separatix probe fields are
reported only when the Separatix diagnostic runs and includes them.

`SeparatixResult.probe_summary` normalizes the existing probe evidence without
fitting another model. It records the best probe, a target-appropriate primary
metric when Separatix declares one, the complete reported metric map, comparable
linear/nonlinear evidence, evaluation and sampling context, grouping counts, and
probe-specific skip reasons. Single-label fallback summaries use Separatix's
balanced-accuracy baseline contract. Multi-label and regression summaries never
invent an accuracy value or select an undeclared primary metric.

These fields are descriptive diagnostics. They do not participate in ranking,
aggregate validity, or vertebrae's benchmark recommendations. Recommendation
confidence and probe-comparison confidence are reported separately because they
describe different evidence.

## Custom embedding metrics

`Benchmark` can score the same full embedding batch with one or more custom metrics.
Every metric must return one finite aggregate `score`; that score is the only value
eligible for ranking. Metrics may also return JSON-safe diagnostics, warnings, and
metadata.

## Retrieval and matching

`RetrievalBenchmark` evaluates frozen query embeddings against a declared gallery and
explicit relevance grades. It is a separate, exact ranking protocol: it does not fit a
head, build an ANN index, mine negatives, or replace OverlapIndex in ordinary labeled
benchmarks. Use it for semantic search, image-text matching, entity lookup, and other
candidate-ranking use cases.

```python
from vertebrae import RetrievalBenchmark, RetrievalConfig, RetrievalDataset, DatasetIdentity
from vertebrae.extractors import PrecomputedExtractor

dataset = RetrievalDataset.from_embeddings(
    query_embeddings,
    gallery_embeddings,
    relevance=[("query-1", "document-9", 2.0)],
    query_ids=["query-1"],
    gallery_ids=["document-9"],
    identity=DatasetIdentity.declared("example-dataset", "1"),
)
result = RetrievalBenchmark(
    dataset,
    [PrecomputedExtractor("candidate")],
    retrieval_config=RetrievalConfig(primary_metric="ndcg@10"),
).run()
```

Relevance may be a dense query-by-gallery grade matrix or sparse
`(query_id, gallery_id, grade)` records. Grades above zero are relevant for binary
metrics; NDCG uses the original grades. Results include NDCG, precision, recall, hit
rate, MRR, mAP, and positive-versus-negative similarity diagnostics. Cosine similarity
is the default, with dot product and squared L2 available explicitly.

For a normal single-label `Benchmark`, `LabelRetrievalMetric` provides opt-in
leave-one-out same-label retrieval after embedding compression. It is not inferred for
multi-label or regression targets because those relevance semantics must be declared.

```python
from vertebrae import Benchmark, CallableMetric, OverlapMetric, OverlapScoringConfig

def label_aware_metric(embeddings, labels, *, target_metadata=None, groups=None, seed=None):
    return {
        "score": 0.87,
        "diagnostics": {"criterion": "my benchmark rule"},
    }

benchmark = Benchmark(
    dataset,
    [extractor],
    metrics=[
        OverlapMetric(config=OverlapScoringConfig(k="auto")),
        CallableMetric("my_metric", label_aware_metric),
    ],
    primary_metric="my_metric",
)
```

OverlapIndex always runs and is available through `ExtractorResult.overlap` and
`ExtractorResult.metrics["overlap"]`. Stability and Separatix continue to use this
built-in overlap metric.
For distributed or CLI scoring, use an importable callable path such as
`my_project.metrics:label_aware_metric` with `vertebrae score --metric ...`.

## Practical guidance

- Keep `normalize_embeddings=True` unless your embedding space already encodes a
  deliberate scale that you want to preserve.
- Watch `k_per_class` and warnings when classes are imbalanced.
- Use prototype stability first; it isolates clustering sensitivity without mixing
  in sampling noise.
- Treat overlap scores as representation diagnostics for a specific dataset and
  protocol, not universal model-quality claims.
