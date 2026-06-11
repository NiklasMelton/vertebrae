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
- optional stability summary,
- optional native probe summary,
- optional Separatix complexity diagnostic,
- embedding metadata,
- runtime timing metadata,
- warnings,
- recommendation label,
- weakest-class diagnostics when available.

Multi-output extractors contribute one `ExtractorResult` per named output. Result
names use the form `parent_name:output_name`, and embedding metadata preserves
`parent_extractor_name` and `output_name`.

## Ranking and tabular views

`BenchmarkResult.ranked_results()` sorts extractors by descending macro overlap
score. `BenchmarkResult.to_dataframe()` produces a compact comparison table with the
most important fields for side-by-side inspection.

```python
result = benchmark.run()

print(result.to_dataframe())
best = result.ranked_results()[0]
print(best.name, best.overlap.macro_score)
```

## JSON and Markdown output

Reports can be written directly from the result object:

```python
result.save_json("result.json")
result.save_markdown("report.md")
```

The JSON report is the most complete machine-readable artifact. The Markdown report
is aimed at practical review and sharing.

At a high level, reports include:

- dataset summary,
- extractor summary,
- overlap configuration,
- ranked comparison table for multi-extractor runs,
- global and per-class scores,
- Separatix recommendation and confidence when available,
- weakest class,
- stability summary,
- warnings,
- recommendations,
- reproducibility metadata.

Separatix is the default classifier-complexity diagnostic when the overlap gate
passes. Native vertebrae probes remain available as an explicit quick-check
option through `ProbeConfig(enabled=True)`.

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
