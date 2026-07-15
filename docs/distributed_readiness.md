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

Artifact keys form a lossless relative namespace shared by local, S3, and GCS stores.
Keys must be non-empty forward-slash paths without absolute, empty, `.`, `..`,
backslash, NUL, or control-character components. Stores reject invalid keys rather
than rewriting them; benign names such as `a..b` and `a__b` are distinct. Named
embedding, structured, and segmentation outputs use
`output-v1-<bounded-slug>--<full-sha256>`. The 40-character slug is derived from an
NFKD-normalized lowercase ASCII display form, but identity comes from the full digest
of the exact, unnormalized UTF-8 output name retained in the manifest.

This key layout intentionally does not read pre-layout named-output, structured, or
segmentation caches. The package is pre-release, so no legacy fallback or migration is
provided; rerunning materialization creates canonical artifacts. Some human-readable
namespace prefixes for single-output, retrieval, zero-shot, label, group, scoring, and
compression artifacts remain familiar, but their authoritative component fingerprints
use identity schema v2 and therefore intentionally produce new keys.

Retrieval uses paired endpoint artifacts. `plan-retrieval` produces JSON-native,
deterministic query and gallery shard plans; `embed-retrieval-shard` materializes one
endpoint shard; and `merge-retrieval-embeddings --plan-json ... --side ...` produces
each complete endpoint. Persist the declared relevance with `write-retrieval-relevance`,
apply gallery-fitted paired compression with `compress-retrieval`, and run
`score-retrieval`. Endpoint keys and manifests preserve side, branch, dataset, and
extractor-recipe identities; compression and scoring reject incompatible pairs.

Zero-shot uses the same endpoint-artifact discipline with a sample endpoint, a
text-prompt endpoint, and a serialized prompt protocol. `plan-zero-shot`,
`embed-zero-shot-shard`, `merge-zero-shot-embeddings`, `write-zero-shot-protocol`,
`compress-zero-shot`, and `score-zero-shot` preserve the fixed prompts and source
dataset identity. Compression is fit on samples and applied to prompts. Planned shard
counts are capped independently for samples and prompts; pass the plan entry to
`embed-zero-shot-shard --plan-json ...` so workers use its endpoint-specific branch,
key, and shard count. Sample manifests deliberately omit prompt-protocol identity so
they can be reused with a revised prompt protocol, while prompt artifacts remain
protocol-specific. `zero-shot-from-artifacts` reconstructs JSON/Markdown reports from
one or more persisted score artifacts. Compressed endpoint pairs carry a shared fitted
compression identity and score artifacts are configuration-specific, so mixed pairs or
different evaluation settings are rejected before ranking.

Profiling config pickles can be passed to planning and extraction/materialization commands
with `--resource-profiling-config-pickle`; plans serialize the config and plan-driven workers
restore it. Each shard manifest stores its complete worker-local profile. Merge validates
model footprint agreement while allowing worker paths and device identities to differ, then
produces a `DistributedResourceProfile` with worker-first latency distribution, summed
compute seconds, aggregate compute throughput, maximum worker RSS/device peaks and their
shard keys, shard persisted bytes, and the merged array's logical and persisted bytes.
Aggregate compute throughput is not cluster wall-clock throughput.

Compression updates the evaluated logical/persisted footprint while retaining the raw
footprint and shared inference/model evidence. Labeled, zero-shot, and retrieval artifact
result builders reconstruct typed profiles; use `retrieval-from-artifacts` or
`zero-shot-from-artifacts` to carry endpoint evidence into ordinary reports.

Protocol artifacts use a versioned semantic-label catalog. Workers score stable label
keys, so non-string labels neither require user class imports nor collapse into
identically rendered strings. Composite label artifacts persist those keys plus the
typed catalog for local, S3, and GCS stores; readers mark decoded keys so subsequent
scoring cannot encode them a second time. Protocol contents are fingerprinted and
revalidated before scoring, and the versioned label encoding is required.

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

For an ordinary labeled benchmark, pass a backend explicitly to make `Benchmark.run()`
orchestrate the complete artifact-backed pipeline. Omitting `execution` retains the
direct in-process path.

```python
from vertebrae import Benchmark, ExecutionConfig, LocalBackend

result = Benchmark(
    dataset,
    [extractor],
    execution=LocalBackend(n_jobs=4),
    execution_config=ExecutionConfig(total_shards=4),
).run()
```

`ExecutionConfig.dispatch_stages` selects which of `embedding`, `compression`,
`scoring`, and `diagnostics` use the backend; omitted stages execute the same artifact
jobs on the driver. Extractors are fitted once on the complete selected dataset before
their fitted state is sent to transform shards. Structured and segmentation workflows
dispatch one materialization job per extractor because unit and spatial alignment are
global, then return ordinary benchmark results.

Dispatched runs always require an artifact store. When `CacheConfig.enabled=False`,
vertebrae uses a unique `runs/<run-id>` namespace and removes it after constructing the
result. Set `retain_intermediate_artifacts=True` to retain run-scoped data and embedding
shards for inspection. Backend or serialization failures raise `BenchmarkExecutionError`;
they never silently rerun locally.

The lower-level equivalent remains available when callers need direct control of every
job and manifest:

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

For Ray or Dask clusters, `cache_dir` must point to a filesystem visible to every worker
or to a cloud object-store URI such as `s3://bucket/prefix` or `gs://bucket/prefix`.
Workers must be able to authenticate to the selected object store.

Dense and sparse arrays use array manifest v2. Data files are immutable and named by
their SHA-256 digest; the manifest records that digest, exact byte size, shape, dtype,
storage format, sparse format, and `nnz`. Readers validate every field before returning
an array.

Execution artifacts that must keep payload and metadata coherent use composite
artifact manifest v2. One manifest atomically selects either an array plus its metadata
or labels plus their metadata, with every JSON component also immutable, digest-named,
size-checked, and SHA-256-checked. A reader never combines an array or labels from one
writer with metadata from another. Standalone `put_json`/`get_json` and
`put_labels`/`get_labels` remain supported low-level store APIs; the composite rule
applies to execution artifacts committed through the artifact methods.

Local publications use same-directory temporary files, `fsync`, and atomic replacement.
S3 and GCS upload every immutable component first and publish the selecting manifest
last, so the manifest is the commit point and concurrent writers have coherent
last-manifest-wins behavior. Cloud stores conservatively retain superseded immutable
digests because safe reclamation requires provider version/generation preconditions;
ordinary check-then-delete cleanup can race a concurrent writer. Use an external,
version-aware lifecycle or garbage-collection policy when reclaiming those objects.
Local stores reclaim unreferenced digests while holding the per-artifact exclusive
publication lock. Array and composite manifest v1 artifacts are rejected as stale and
are not migrated.

The same flow is available from the CLI. Pickle inputs and fitted-extractor bundles are
trusted-input-only: never load files from an untrusted source. First serialize the
dataset and unfitted extractor, fit one trusted bundle on the complete dataset, plan
the work from that bundle, run transform-only shards, and merge:

```bash
vertebrae fit-extractor \
  --dataset-pickle dataset.pkl \
  --extractor-pickle extractor.pkl \
  --output-pickle fitted-extractor.pkl

vertebrae plan \
  --dataset-pickle dataset.pkl \
  --extractor-pickle fitted-extractor.pkl \
  --cache-dir .vertebrae_cache \
  --total-shards 8 \
  --batch-size 128 \
  --output-json plan.json

vertebrae embed-shard \
  --dataset-pickle dataset.pkl \
  --extractor-pickle fitted-extractor.pkl \
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

Driver-managed execution performs the same single fit before dispatch. Generated
SLURM workflows include the fit pre-step and quote all shell arguments; every array
task consumes the resulting fitted bundle rather than fitting its own shard. Ordinary
and retrieval shard counts are capped to their available rows, and a non-streaming
extractor is planned as one full transform rather than being split into independent
batches. Job name, partition, time, memory, CPU, shard, and batch values are validated;
newlines, control characters, invalid ranges, and unsafe scheduler-field punctuation
are rejected before a shell file is written.

When a plan declares `groups_key`, both `score` and `score-repeats` load and
validate that artifact and pass the aligned groups to group-aware custom metrics.
An explicit `--groups-key` overrides the plan value. Group manifests must match
the embedding and label row count and dataset identity; the scoring artifact and
collected repeat summary retain the validated group provenance.

For segmentation datasets, `vertebrae materialize-segmentation` aligns saved
spatial embedding outputs to raster annotations and writes token-level embedding,
label, and group artifacts. Those artifacts then follow the same `score`,
`diagnose-complexity`, `score-repeats`, and `benchmark-from-artifacts` stages as
ordinary embedding workflows.

Structured unit materialization follows the same artifact philosophy. Native
structured extractors can materialize per-output unit embeddings, labels,
groups, and provenance through `materialize_structured_artifacts(...)`, with one
artifact boundary per named structured output.

Structured artifacts preserve resolved single-label, multi-label, or
regression schema metadata. Multi-label manifests retain ordered `label_names`;
regression manifests retain ordered `target_names` and target counts. Sparse
structured matrices are written as `.npz` artifacts and report their sparse
format and nonzero count in the output manifest.

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

For SLURM, generate the fit/plan/array workflow files:

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

The command always writes `vertebrae_embed.sbatch`,
`vertebrae_embed.submit.sh`, and `vertebrae_embed.plan.json`. When the input is an
unfitted extractor it also writes `vertebrae_embed.fit.sbatch`; the submit wrapper
submits that fit job first and gives the array an `afterok` dependency. Start the
workflow with:

```bash
bash vertebrae_embed.submit.sh
```

Do not submit `vertebrae_embed.sbatch` directly when a fit script was generated: the
array requires the fitted bundle and plan produced by the dependency job. After the
array completes, run the merge, label, score, and optional score-array commands shown
in the generated array script.

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
target-preserving subsample: class/label-aware for categorical targets and
non-constant-target-aware for regression. Result metadata records the original and
final row counts plus separate manual, automatic, and cumulative subsample rates, so
an automatic second stage never overwrites the earlier user-requested stage. This keeps
single-GPU sequential embedding and CPU-distributed analysis workflows from
overcommitting memory.

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
result and recipe. Group-aware callables can accept a `groups` keyword argument;
callables that omit it remain valid. `collect-scores --metric-name ...` can select
a non-primary metric when building an interval summary, and rejects repeats that
mix different group protocols.

Artifact-backed workflows can also attach a Separatix diagnostic artifact after
overlap scoring. Use the CLI `diagnose-complexity` command with an embedding key,
labels key, and score key. The score artifact provides the overlap score used to
gate Separatix execution, and `benchmark-from-artifacts --separatix-key ...` can
fold that diagnostic back into the final benchmark-style JSON or Markdown report.
