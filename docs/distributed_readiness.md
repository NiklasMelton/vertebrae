# distributed readiness

Distributed execution starts with local, artifact-backed sharding. The package keeps
the same boundaries needed for future HPC, Ray, Dask, and multi-GPU backends:

- Dataset validation is separate from feature extraction.
- Extractors produce dense or sparse embedding artifacts, with one artifact per
  named output when the extractor is multi-output.
- Streaming-safe extractors can embed deterministic sample batches and materialize
  embeddings without keeping the full raw dataset in memory.
- Scoring consumes embeddings and labels through `OverlapIndexScorer`.
- Reports render from serialized result data.
- `ExecutionBackend` supports `submit`, `gather`, `status`, and `map`, with local,
  Ray, and Dask implementations available.

Distributed artifact materialization now supports multi-output extractors by
writing one embedding artifact per named output. Each output remains a normal 2D
embedding artifact for downstream scoring and compression.

Retrieval uses paired endpoint artifacts. `plan-retrieval` produces JSON-native,
deterministic query and gallery shard plans; `embed-retrieval-shard` materializes one
endpoint shard; and `merge-retrieval-embeddings --plan-json ... --side ...` produces
each complete endpoint. Persist the declared relevance with `write-retrieval-relevance`,
apply gallery-fitted paired compression with `compress-retrieval`, and run
`score-retrieval`. Endpoint keys and manifests preserve side, branch, dataset, and
extractor-recipe identities; compression and scoring reject incompatible pairs.

Distributed endpoint workers transform frozen or already-fitted extractor pickles;
they do not fit independently on their shards. This keeps query and gallery embeddings
in one representation space. Local `RetrievalBenchmark` continues to fit compatible
standard extractors once on the gallery before transforming both endpoints.

The same artifact flow works for aligned multi-modal datasets because dataset
pickles preserve structured fields and multi-output extractors still materialize
ordinary per-output embedding artifacts. Missing modalities are not supported in
v1; workers should receive complete aligned samples.

Distributed backends can shard embedding generation across workers, then submit
compression, scoring, and optional diagnostic jobs over saved embedding, label, and
group artifacts. New extractors should keep deterministic row order, avoid hidden
global state, and include all model/preprocessing settings in `recipe()`.

Label artifacts support single-label, multi-label, and explicit regression targets.
Multi-label artifacts store one JSON list of labels per sample and preserve
`target_type`, `label_names`, per-label `class_counts`, labelset counts, mean label
cardinality, and label density in the manifest. Regression artifacts preserve
`target_type`, `target_names`, target statistics, and constant-target diagnostics.
When a dataset materializes an active named target view, label manifests also
preserve `target_view` metadata so downstream scoring and benchmark reconstruction
can distinguish views without changing the embedding artifact. CLI commands still
use the same `--labels-key` arguments for every target type.

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
the label artifact through `score_embedding_artifact(...)`. Ray and Dask backends
submit the same job objects and use the same manifests.

For Ray or Dask clusters, `cache_dir` may point either to a shared filesystem path or
to a cloud object-store URI such as `s3://bucket/prefix` or `gs://bucket/prefix`.
Workers must be able to authenticate to the selected object store.

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

vertebrae write-groups \
  --dataset-pickle dataset.pkl \
  --cache-dir .vertebrae_cache

vertebrae compress \
  --cache-dir .vertebrae_cache \
  --embedding-key "$(jq -r .output_key plan.json)"

vertebrae score \
  --cache-dir .vertebrae_cache \
  --plan-json plan.json

vertebrae score-repeats \
  --cache-dir .vertebrae_cache \
  --plan-json plan.json \
  --repeats 20 \
  --backend local \
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

For segmentation datasets, `vertebrae materialize-segmentation` aligns saved
spatial embedding outputs to raster annotations and writes token-level embedding,
label, and group artifacts. Those artifacts then follow the same `score`,
`diagnose-complexity`, `score-repeats`, and `benchmark-from-artifacts` stages as
ordinary embedding workflows.

Structured unit materialization follows the same artifact philosophy. Native
structured extractors can materialize per-output unit embeddings, labels,
groups, and provenance through `materialize_structured_artifacts(...)`, with one
artifact boundary per named structured output.

The same path is available from the CLI:

```bash
vertebrae materialize-structured \
  --dataset-pickle dataset.pkl \
  --extractor-pickle extractor.pkl \
  --cache-dir .vertebrae_cache \
  --aligner 'tokens=drop_special_rows:{"leading":1,"trailing":1}' \
  --batch-size 16 \
  --output-json structured_bundle.json

vertebrae score \
  --cache-dir .vertebrae_cache \
  --plan-json structured_bundle.json \
  --embedding-key "$(jq -r '.outputs[0].output_key' structured_bundle.json)"

vertebrae diagnose-complexity \
  --cache-dir .vertebrae_cache \
  --plan-json structured_bundle.json \
  --embedding-key "$(jq -r '.outputs[0].output_key' structured_bundle.json)"
```

If the structured or segmentation bundle contains exactly one output, `score`,
`score-repeats`, and `diagnose-complexity` can infer the aligned embedding,
labels, groups, and default score artifact directly from the bundle JSON. If the
bundle contains multiple outputs, pass `--embedding-key` for the selected output
and the CLI will resolve that output's aligned label/group artifacts
automatically.

For structured outputs, `materialize-structured` can attach one standard
aligner helper per output with repeatable `--aligner` flags in the form
`output_name=helper_name:{...json params...}`. Supported helpers mirror the
Python API helpers: `drop_special_rows`, `keep_row_indices`, and
`select_frame_rows`.

To execute shards or score repeats through Ray or Dask instead of the local backend:

```bash
vertebrae run-embedding-shards \
  --dataset-pickle dataset.pkl \
  --extractor-pickle extractor.pkl \
  --cache-dir /shared/vertebrae_cache \
  --total-shards 8 \
  --backend ray \
  --ray-address auto

vertebrae score-repeats \
  --cache-dir /shared/vertebrae_cache \
  --plan-json plan.json \
  --repeats 20 \
  --backend dask \
  --dask-address tcp://scheduler:8786
```

To use cloud artifact stores instead of a shared filesystem:

```bash
vertebrae run-embedding-shards \
  --dataset-pickle dataset.pkl \
  --extractor-pickle extractor.pkl \
  --cache-dir s3://team-bucket/vertebrae/run-001 \
  --s3-region us-east-1 \
  --backend ray \
  --ray-address auto \
  --total-shards 8

vertebrae score-repeats \
  --cache-dir gs://team-bucket/vertebrae/run-001 \
  --gcs-project my-gcp-project \
  --plan-json plan.json \
  --repeats 20 \
  --backend dask \
  --dask-address tcp://scheduler:8786
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
class-stratified subsample when possible. Explicit regression datasets use a
deterministic random subsample instead. This keeps single-GPU sequential embedding
and CPU-distributed analysis workflows from overcommitting memory.

The metric backend remains fixed: all scoring goes through MiniBatchKMeans-backed
`overlapindex.OverlapIndex` or `overlapindex.ContinuousOverlapIndex` via the
internal `OverlapIndexScorer` adapter.

Artifact scoring always includes the built-in overlap metric and can evaluate
additional importable callables in the same job. Repeat `--metric` for each callable
and select one aggregate result for score collection with `--primary-metric`:

```bash
vertebrae score \
  --cache-dir /shared/vertebrae_cache \
  --plan-json plan.json \
  --metric my_project.metrics:calibration_score \
  --metric my_project.metrics:domain_margin \
  --primary-metric calibration_score
```

Each scoring artifact is a `metric_evaluation` payload containing every metric
result and recipe. `collect-scores --metric-name ...` can select a non-primary
metric when building an interval summary.

Artifact-backed workflows can also attach a Separatix diagnostic artifact after
overlap scoring. Use the CLI `diagnose-complexity` command with an embedding key,
labels key, and score key. The score artifact provides the overlap score used to
gate Separatix execution, and `benchmark-from-artifacts --separatix-key ...` can
fold that diagnostic back into the final benchmark-style JSON or Markdown report.
