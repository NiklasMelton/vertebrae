# results and reports

Benchmark runs return structured result objects first, then render reports from those
serialized results. That separation keeps reporting reproducible and avoids coupling
report generation to live model objects.

## Result objects

Single- and multi-extractor workflows aggregate into `BenchmarkResult`, which
contains:

- `dataset_summary`
- `extractor_results`
- `recommendations`
- `metadata`

Each extractor contributes an `ExtractorResult` with:

- extractor identity and type,
- `OverlapScoreResult`,
- a named collection of normalized metric results,
- the selected `primary_metric_name` and aggregate `primary_score`,
- optional stability summary,
- optional Separatix complexity diagnostic,
- embedding metadata,
- runtime timing metadata,
- warnings,
- recommendation label,
- weakest-class diagnostics when available.
- an optional measured `resource_profile` when local resource profiling is enabled.

## Resource profiles and quality cohorts

`ResourceProfilingConfig(enabled=True)` adds a serialized profile containing observed
first-call/warm latency, throughput, host and supported device peaks, model/checkpoint
footprint, measurement context, and logical raw/evaluated embedding bytes.
`BenchmarkResult.quality_cohort()` returns candidates within the configured absolute
primary-score tolerance of the best result while respecting metric direction. It does
not reorder `ranked_results()` or create a composite score. Results whose primary
metric metadata declares `aggregate_valid=False` are excluded from the quality cohort,
resource comparison, and top-candidate recommendations.

Markdown reports keep the main ranking quality-focused and render a separate resource
table for the quality cohort. DataFrames expose the same measurements as resource
columns. Compare latency only when batch size, input workload, device, precision, and
synchronization context are compatible.

Retrieval and zero-shot results keep endpoint profiles separate (`query`/`gallery`
and `samples`/`prompts`). Their DataFrames and Markdown reports show endpoint-prefixed
measurements for the quality cohort, including batch, cache, synchronization, modality,
branch, and measurement-scope context.

Embedding footprints distinguish logical resident bytes from persisted array-object
bytes. Persisted size is the actual `.npy`/`.npz`, S3 object, or GCS blob size and excludes
metadata, labels, relevance, provenance, and reports. Raw and evaluated footprints remain
separate through compression. Missing or failed object stats produce an unavailable status
and warning rather than failing the benchmark.

Artifact-backed results reconstruct typed local or distributed profiles. Distributed
profiles report worker-first latency, maximum worker memory, and aggregate compute
throughput (materialized samples divided by summed worker compute seconds). This is not
cluster wall-clock throughput.

First-call latency is the first observed extractor call after fitting and can include
lazy model loading or compilation. Warm statistics use subsequent real calls; a
single-call run therefore has no warm distribution. Host RSS peaks are measured and
remain distinct from `MemoryConfig` admission estimates. Cache hits do not invoke the
extractor and report inference as `not_measured_cache_hit`.

Embedding metadata exposes `cache_eligible` and `cache_status`. The normal statuses are
`hit`, `miss`, `disabled`, and `bypassed_unsafe_identity`; the last means evaluation
continued but the callable/model identity could not safely authorize reuse. Compression
and other derived artifacts retain the source eligibility and status rather than
silently converting a disabled or unsafe source into a reusable cache.

Device profiles distinguish allocator baseline, absolute peak, and peak increase.
CPU allocator memory is marked not applicable because process RSS is the relevant
portable measurement. Model footprint reports parameter and checkpoint availability
independently, so an ONNX file can have a measured checkpoint footprint while graph
parameter inspection remains unavailable. `partial` means only part of the model or
declared deployment bundle could be measured.

Torch measurements use the extractor's resolved device and synchronize CUDA/MPS when
supported. TensorFlow memory counters are used only for an unambiguous or explicitly
hinted GPU; they never change model placement. Model weight dtypes describe stored
weights and must not be interpreted as execution or autocast precision.

`measurement_scope="profile_window"` is present only when Vertebrae synchronized the
device, successfully reset allocator peak counters, observed at least one extractor
call, and read the same device afterward. Cache hits, failed resets, device placement
changes, JAX accelerators, and multi-device Torch models retain null peaks with an
explicit status and reason. Synchronization status is accumulated across the run: one
failed boundary makes latency host-observed.

`parameter_count` and `parameter_bytes` include frozen parameters. Optional
`trainable_parameter_count` and `trainable_parameter_bytes` describe the trainable
subset, while `in_memory_bytes` covers all reported persistent model state. For Keras
and generic TensorFlow variable containers, `buffer_bytes` remains unavailable when
the framework cannot reliably distinguish frozen parameters from non-parameter state.

For multi-label datasets, `target_type` is `multi_label`, `class_counts` means
per-label occurrence counts, and result metadata preserves `label_names` plus
labelset summary fields. For explicit regression datasets, `target_type` is
`regression`, the primary ranking field is `overlap.score`, and summaries preserve
`target_names`, target statistics, and constant-target diagnostics.

Classification and hierarchy diagnostics also preserve a semantic `label_catalog`.
It maps stable typed keys back to original values and report displays, so values such
as integer `1` and string `"1"` remain distinct. When ordinary display text collides,
the rendered label includes its type.

Multi-output extractors contribute one `ExtractorResult` per named output. Result
names use the form `parent_name:output_name`, and embedding metadata preserves
`parent_extractor_name` and `output_name`.

For multi-modal workflows, dataset summaries preserve the aligned field and
per-field modality metadata, and embedding metadata can also preserve
per-output source details such as `image`, `text`, or `fused`.

Hierarchy-level benchmarks contribute one `ExtractorResult` per evaluated label
view. Those results preserve `label_view` metadata and qualify extractor names with
suffixes such as `extractor[level=family]`.

Named target-view benchmarks contribute one `ExtractorResult` per evaluated target
view. Those results preserve `target_view` metadata and qualify extractor names
with suffixes such as `extractor[target=coarse]`.

When named extractor outputs are mapped to hierarchy levels or target views,
result names preserve those dimensions, such as `extractor:layer_6[level=family]`
or `extractor:final[target=role]`. Embedding metadata keeps the original
`output_name`, and result metadata keeps the active `label_view` or `target_view`.

## Ranking and tabular views

`BenchmarkResult.ranked_results()` sorts extractors by the selected primary metric
score, respecting metrics that declare `higher_is_better=False`. It omits results whose
primary aggregate is marked invalid. If every aggregate is invalid, rankings and
quality-cohort/resource tables are unavailable rather than populated with a fallback
candidate.
`BenchmarkResult.to_dataframe()` includes `primary_metric` and `primary_score` plus
overlap columns whenever OverlapIndex was enabled.

```python
result = benchmark.run()

print(result.to_dataframe())
best = result.ranked_results()[0]
print(best.name, best.primary_metric_name, best.primary_score)
```

The Markdown ranking table uses the same metric-aware summary fields. Its current
core columns are `primary_metric`, `primary_score`, `overlap_score`,
`overlap_macro`, `overlap_weighted`, `stability_interval`, `weakest_class`,
`best_probe`, `probe_metric`, and `probe_score`, followed by embedding,
compression, recommendation, and Separatix fields. Target and label views are
included explicitly so expanded results remain distinguishable.

Probe fields are target-aware. For example, a single-label diagnostic may select
balanced accuracy, a multi-label diagnostic may select macro/micro F1 or sample
Jaccard, and a regression diagnostic may select R². Consumers should inspect
`probe_metric` before interpreting `probe_score`; there is no universal
`probe_accuracy` field.

## JSON and Markdown output

Reports can be written directly from the result object:

```python
result.save_json("result.json")
result.save_markdown("report.md")
```

The JSON report is the most complete machine-readable artifact. The Markdown report
is aimed at practical review and sharing.

JSON persistence is strict and deterministic. Metadata may contain ordinary JSON
values, dataclasses, paths, enums, NumPy scalars or arrays, scipy sparse matrices,
tuples, and sets. Sets receive a stable order and non-string mapping keys receive a
collision-safe encoding. Unsupported live objects, recursive containers, and
non-finite floats raise a path-aware serialization error instead of being silently
converted with `str(...)`.

Markdown table cells are escaped through one shared renderer. Pipes, backslashes, and
line breaks in extractor names, labels, warnings, or metadata cannot create extra rows
or columns.

`ZeroShotBenchmarkResult` is a separate result type for fixed prompt-prototype
evaluation. Its ranking uses the configured zero-shot metric (Top-1 accuracy by
default), while the report retains OverlapIndex as contextual sample-embedding
evidence. The values are intentionally not combined into one universal score.
Its serialized protocol preserves the complete ordered prompt declaration, and a
compressed variant name retains the requested dimension even when compression is
skipped and the reported output dimension is unchanged.

`RetrievalBenchmarkResult` likewise preserves its complete protocol. Artifact-backed
entries contain `forward`, optional `reverse`, and an averaged `primary_score`, matching
local bidirectional evaluation. Comparative reconstruction requires identical
retrieval configuration and protocol fingerprints.

At a high level, reports include:

- dataset summary,
- multi-modal dataset field and modality metadata when available,
- extractor summary,
- target-view metadata when named target views are benchmarked,
- label-view metadata when hierarchy-derived views are benchmarked,
- overlap configuration,
- global macro and weighted overlap scores plus reporting-only class exclusions,
- target type and multi-label or regression summary fields when applicable,
- ranked comparison table for multi-extractor runs,
- global and per-class scores,
- per-output branch or fused source metadata when available,
- Separatix recommendation and confidence when available,
- weakest class,
- stability summary,
- warnings,
- recommendations,
- reproducibility metadata.

The same report structure covers the major problem classes, with target-specific
details:

| problem class | summary metadata | score detail |
| --- | --- | --- |
| single-label classification | class names and counts | per-class and pairwise overlap |
| multi-label classification | label names, cardinality, and density | per-label overlap |
| regression | target names and target statistics | per-target continuous overlap |
| hierarchy or named target views | active and available views | one result variant per evaluated view |
| dense segmentation | source-image groups and token provenance | per-class token overlap |

These values are diagnostics for the evaluated representation and protocol. They
are not substitutes for task-native metrics such as IoU, RMSE, retrieval recall,
or ranking quality.

Segmentation reports also include source-image counts, candidate and retained
tokens, ignored-token reasons, background counts, and spatial layout metadata.

Structured-output reports and tabular summaries also surface `task_family`,
`alignment_mode`, and `alignment_recipe` when raw token, frame, region,
keypoint, depth, or latent-unit outputs were materialized before scoring. The
same fields are preserved in structured artifact manifests so artifact-backed
workflows can inspect alignment choices without reopening Python objects.

Relational embedding datasets report their `relational_unit` metadata, such as
`node`, `edge`, `entity`, `pair`, or `triplet`, plus composition metadata when rows
were derived from endpoint embeddings. These reports still describe supervised
embedding efficacy through overlap or continuous overlap scores; they are not
retrieval, recommender, or ranking benchmark reports.

Separatix is the default classifier-complexity diagnostic when the overlap gate
passes, including multi-label datasets. Probe-style report columns are derived
from Separatix baseline probe metrics when present. Tabular results expose
`probe_metric` and `probe_score` only when the diagnostic declares a suitable
primary metric, plus the best probe, complete metric map, linear/nonlinear
comparison, evaluation mode, sampling, grouping context, and skip reason. There
is intentionally no universal `probe_accuracy` column because it is misleading
for regression and incomplete for multi-label diagnostics.

JSON outputs preserve both the normalized `probe_summary` and the complete raw
Separatix report. Markdown ranking tables show the compact best-probe metric while
per-extractor details show the full descriptive context. When Separatix MLP probes
are enabled, their trigger, status, reason, and comparison payload remain separate
from the ordinary baseline probe summary.

`probe_summary` is required in the current alpha result contract. Serialized
artifacts created before this field was introduced are not backward compatible and
are not reconstructed from the raw Separatix report. The raw report remains
available for detailed evidence and reproducibility, not as a legacy result schema.

## What recommendations mean

Recommendation labels are lightweight practitioner guidance, not absolute verdicts.
They summarize the observed overlap behavior for the evaluated dataset and protocol.

Use them as a triage aid:

- shortlist strong frozen representations,
- flag weak classes for inspection,
- compare multiple candidate extractors under the same benchmark setup.

Separatix recommendations are complementary. They describe the apparent classifier
complexity of the labeled embedding space and do not replace vertebrae's overlap-based
ranking or existing benchmark recommendation label.

## Reproducibility mindset

Because report generation depends on serialized result data rather than live Python
objects, you can archive JSON outputs and regenerate downstream summaries later
without needing the original extractor instance in memory.

That design also fits the package's local-first distributed roadmap: embedding jobs,
scoring jobs, and report rendering can remain separate stages with explicit artifacts
between them.
