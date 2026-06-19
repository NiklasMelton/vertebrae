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

Use `OverlapScoringConfig` to tune the supported MiniBatchKMeans-facing settings:

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

## Automatic k resolution

The default `k="auto"` mode resolves a separate `k` for each class using class size,
`min_k`, `max_k`, and `min_samples_per_cluster`. For multi-label targets, class
size means per-label occurrence count. If a class or label is too small to support
the requested or inferred `k`, the scorer reduces it and records a warning.

That warning is carried into the structured result and final reports so users can see
when a weakly represented class forced a smaller prototype count.

## Scoring inputs

The scorer accepts numeric embedding matrices with single-label or multi-label
classification targets.

- Dense inputs are scored directly.
- Sparse inputs are validated, then densified only at the OverlapIndex boundary.
- Single-label targets are passed to OverlapIndex as a one-dimensional label array.
- Multi-label targets are normalized to a dense 0/1 indicator matrix before calling
  OverlapIndex.
- When `normalize_embeddings=True`, embeddings are L2-normalized row-wise before
  scoring.

This keeps extractor and caching layers free to preserve sparse artifacts longer,
while still meeting the MiniBatchKMeans backend requirements.

## Returned diagnostics

`OverlapIndexScorer.score(...)` returns an `OverlapScoreResult` with:

- `macro_score`
- `per_class_scores`
- `pairwise_scores`
- `sparse_adjacency`
- `class_counts`
- `k_per_class`
- `warnings`
- `metadata`

The `metadata` payload records backend details such as normalization, chunk size,
seed, KMeans kwargs, and whether the original input was sparse.

## Stability analysis

Repeated scoring is handled by `run_stability_analysis(...)`, which uses
`OverlapIndexScorer` under the hood. The default stability mode is prototype
stability:

- embeddings and labels stay fixed,
- MiniBatchKMeans seeds change across repeats,
- summaries report mean, standard deviation, min, max, and percentile interval.

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
- By default it only runs when `overlap.macro_score >= 0.80`.
- Multi-label targets are passed to Separatix as dense 0/1 indicator matrices with
  `target_mode="multilabel"`.
- The full Separatix report is preserved in JSON outputs, while Markdown reports
  show a compact recommendation, confidence, decision path, key scores, and skips.

Separatix follows the same normalization convention as overlap scoring when
`normalize_embeddings=True`. Sparse inputs remain sparse at the vertebrae boundary,
and Separatix uses its own densification policy internally.

Native vertebrae probes are still available through `ProbeConfig`, but they are
now opt-in quick checks rather than part of the default report path. They are
currently single-label only; when enabled on a multi-label dataset they are skipped
with a warning while overlap scoring and Separatix continue to run.

## Practical guidance

- Keep `normalize_embeddings=True` unless your embedding space already encodes a
  deliberate scale that you want to preserve.
- Watch `k_per_class` and warnings when classes are imbalanced.
- Use prototype stability first; it isolates clustering sensitivity without mixing
  in sampling noise.
- Treat overlap scores as representation diagnostics for a specific dataset and
  protocol, not universal model-quality claims.
