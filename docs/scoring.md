# scoring

All overlap scoring in `vertebrae` goes through one internal adapter:
`OverlapIndexScorer`. This keeps the package aligned with its v1 metric contract and
ensures every benchmark run uses the same backend path.

## Fixed metric backend

`vertebrae` depends on the external `overlapindex` package and does not reimplement
OverlapIndex. In v1, the backend is fixed internally to MiniBatchKMeans:

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

## Automatic k resolution

The default `k="auto"` mode resolves a separate `k` for each class using class size,
`min_k`, `max_k`, and `min_samples_per_cluster`. If a class is too small to support
the requested or inferred `k`, the scorer reduces it and records a warning.

That warning is carried into the structured result and final reports so users can see
when a weakly represented class forced a smaller prototype count.

## Scoring inputs

The scorer accepts numeric embedding matrices and one-dimensional class labels.

- Dense inputs are scored directly.
- Sparse inputs are validated, then densified only at the OverlapIndex boundary.
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

## Practical guidance

- Keep `normalize_embeddings=True` unless your embedding space already encodes a
  deliberate scale that you want to preserve.
- Watch `k_per_class` and warnings when classes are imbalanced.
- Use prototype stability first; it isolates clustering sensitivity without mixing
  in sampling noise.
- Treat overlap scores as representation diagnostics for a specific dataset and
  protocol, not universal model-quality claims.
