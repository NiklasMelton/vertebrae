# distributed readiness

Distributed execution starts with local, artifact-backed sharding. The package keeps
the same boundaries needed for future HPC, Ray, Dask, and multi-GPU backends:

- Dataset validation is separate from feature extraction.
- Extractors produce dense embedding artifacts.
- Streaming-safe extractors can embed deterministic sample batches and materialize
  embeddings without keeping the full raw dataset in memory.
- Scoring consumes embeddings and labels through `OverlapIndexScorer`.
- Reports render from serialized result data.
- `ExecutionBackend` supports `submit`, `gather`, `status`, and `map`, with only
  the local backend implemented.

Distributed backends can shard embedding generation across workers, then submit scoring
jobs over saved embedding and label artifacts. New extractors should keep deterministic
row order, avoid hidden global state, and include all model/preprocessing settings in
`recipe()`.

The concrete local distributed flow is:

```python
from vertebrae import BenchmarkDataset, LocalBackend
from vertebrae.cache.local_store import LocalArtifactStore
from vertebrae.execution import materialize_and_merge_embeddings

store = LocalArtifactStore(".vertebrae_cache")
manifest = materialize_and_merge_embeddings(
    dataset=dataset,
    extractor=extractor,
    store=store,
    execution=LocalBackend(n_jobs=4),
    total_shards=4,
    batch_size=128,
)
```

For HPC schedulers, each array task can run one `EmbeddingShardJob` using
`materialize_embedding_shard(...)`; the final collection task runs
`merge_embedding_shards(...)`. Scoring jobs consume the merged embedding artifact and
the label artifact through `score_embedding_artifact(...)`. Ray and Dask adapters
should submit the same job objects and use the same manifests.

The same flow is available from the CLI. First serialize a dataset and extractor with
`pickle`, then plan, run shards, and merge:

```bash
vertebrae plan \
  --dataset-pickle dataset.pkl \
  --extractor-pickle extractor.pkl \
  --cache-dir .vertebrae_cache \
  --total-shards 8 \
  --batch-size 128 \
  --output-json plan.json

vertebrae embed-shard \
  --dataset-pickle dataset.pkl \
  --extractor-pickle extractor.pkl \
  --cache-dir .vertebrae_cache \
  --total-shards 8 \
  --shard-index 0 \
  --batch-size 128

vertebrae merge-embeddings \
  --cache-dir .vertebrae_cache \
  --plan-json plan.json

vertebrae write-labels \
  --dataset-pickle dataset.pkl \
  --cache-dir .vertebrae_cache

vertebrae score \
  --cache-dir .vertebrae_cache \
  --plan-json plan.json

vertebrae score-repeats \
  --cache-dir .vertebrae_cache \
  --plan-json plan.json \
  --repeats 20 \
  --output-json score_repeats.json

vertebrae collect-scores \
  --cache-dir .vertebrae_cache \
  --score-plan-json score_repeats.json \
  --output-key "$(jq -r .output_key plan.json)/scores/stability"

vertebrae benchmark-from-artifacts \
  --cache-dir .vertebrae_cache \
  --score-key "$(jq -r .score_key plan.json)" \
  --stability-key "$(jq -r .output_key plan.json)/scores/stability" \
  --json-output result.json \
  --markdown-output report.md
```

For SLURM, generate an array script:

```bash
vertebrae slurm-array \
  --dataset-pickle dataset.pkl \
  --extractor-pickle extractor.pkl \
  --cache-dir .vertebrae_cache \
  --total-shards 8 \
  --batch-size 128 \
  --script-output vertebrae_embed.sbatch \
  --job-name vertebrae-embed \
  --time 04:00:00 \
  --mem 16G \
  --cpus-per-task 4
```

Submit the script with `sbatch vertebrae_embed.sbatch`, then run the merge, label,
score, and optional score-array commands shown in the generated script after the
array completes.

`ShardSpec` partitions samples by index modulo the total shard count. This gives
future distributed workers disjoint sample sets and prevents duplicate embedding work
when the same model is deployed across multiple nodes. Local `Benchmark.run()` still
requires a complete embedding artifact before scoring.

Memory admission is handled separately from scheduling. `MemoryConfig` uses `psutil`
to derive an available-memory budget unless an explicit byte limit is supplied. For
streaming-safe extractors, the benchmark probes a small first batch, estimates the
final embedding artifact and dense scoring input, and reuses that probe as the first
materialized batch when no subsampling is needed. If the full plan would exceed the
budget, the local runner records a warning and switches to the largest fitting
class-stratified subsample when possible. This keeps single-GPU sequential embedding
and CPU-distributed analysis workflows from overcommitting memory.

The metric backend remains fixed: all scoring goes through MiniBatchKMeans-backed
`overlapindex.OverlapIndex` via the internal `OverlapIndexScorer` adapter.
