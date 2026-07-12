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

For multi-label datasets, `target_type` is `multi_label`, `class_counts` means
per-label occurrence counts, and result metadata preserves `label_names` plus
labelset summary fields. For explicit regression datasets, `target_type` is
`regression`, the primary ranking field is `overlap.score`, and summaries preserve
`target_names`, target statistics, and constant-target diagnostics.

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
score, respecting metrics that declare `higher_is_better=False`.
`BenchmarkResult.to_dataframe()` includes `primary_metric` and `primary_score` plus
overlap columns whenever OverlapIndex was enabled.

```python
result = benchmark.run()

print(result.to_dataframe())
best = result.ranked_results()[0]
print(best.name, best.primary_metric_name, best.primary_score)
```

## JSON and Markdown output

Reports can be written directly from the result object:

```python
result.save_json("result.json")
result.save_markdown("report.md")
```

The JSON report is the most complete machine-readable artifact. The Markdown report
is aimed at practical review and sharing.

`ZeroShotBenchmarkResult` is a separate result type for fixed prompt-prototype
evaluation. Its ranking uses the configured zero-shot metric (Top-1 accuracy by
default), while the report retains OverlapIndex as contextual sample-embedding
evidence. The values are intentionally not combined into one universal score.
Its serialized protocol preserves the complete ordered prompt declaration, and a
compressed variant name retains the requested dimension even when compression is
skipped and the reported output dimension is unchanged.

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
from Separatix baseline probe metrics when present. When Separatix MLP probes are
enabled, JSON outputs preserve the full trigger and comparison payload and
Markdown reports summarize the MLP status.

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
