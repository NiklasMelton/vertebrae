# scoring

All [OverlapIndex](https://github.com/NiklasMelton/OverlapIndex) scoring in
`vertebrae` goes through one internal adapter: `OverlapIndexScorer`.

`vertebrae` can also run a default
[Separatix](https://github.com/NiklasMelton/Separatix) complexity diagnostic
through the internal `SeparatixScorer` adapter. Separatix does not affect
extractor ranking; it adds classifier-complexity guidance on top of the overlap
result when enabled.
The adapter targets Separatix `0.1.1` or newer. That release uses one canonical
recommendation vocabulary for classification, multi-label, and regression
diagnostics while retaining target-specific explanatory text.
When a dataset declares groups, vertebrae forwards them to Separatix so supervised
evaluation and structural evidence respect those independence units. A grouped
diagnostic that lacks sufficient cross-group class support is recorded as skipped;
vertebrae never retries it with a row-level split.

## Fixed metric backend

`vertebrae` depends on the external
[OverlapIndex](https://github.com/NiklasMelton/OverlapIndex) package and does not
reimplement it. Currently, the backend is fixed internally to MiniBatchKMeans:

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

Classification and multi-label OverlapIndex scores are bounded to `[0, 1]`:
`1.0` indicates perfect class separation and `0.0` indicates perfect class
overlap.

Key fields:

- `k`: an integer, a class-to-`k` mapping, or `"auto"`.
- `min_k` and `max_k`: bounds used during automatic per-class resolution.
- `min_samples_per_cluster`: guardrail that prevents too many prototypes for a
  small class.
- `kmeans_kwargs`: extra keyword arguments forwarded to MiniBatchKMeans.
- `offline_chunk_size`: chunk size passed to `OverlapIndex.fit_offline(...)`.
- `normalize_embeddings`: enables L2 row normalization before scoring.
- `max_dense_bytes`: fallback limit for downstream dense-only diagnostics, including
  Separatix when its own limit is not set. Sparse overlap scoring does not densify
  the full embedding matrix.
- `exclude_classes`: class id or ids retained during fitting and detailed
  diagnostics but omitted from global macro and weighted aggregation.

Typed labels are translated through the dataset's semantic label catalog before the
OverlapIndex adapter is called. `k` mappings and `exclude_classes` may still use the
original typed values; per-class diagnostics are restored to stable keys and readable,
type-disambiguated report labels afterward.

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
`OverlapScoreResult.score` as the primary continuous overlap score. The score is
always bounded to `[0, 1]`: `1.0` indicates no observed harmful continuous-target
overlap, while `0.0` is the permutation-equivalent null endpoint. The unbounded
loss comparison remains available through `OverlapScoreResult.loss_ratio`.

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
- SciPy sparse matrices and sparse arrays are validated and normalized to CSR while
  retaining sparse storage through OverlapIndex.
- Single-label targets are encoded as one-dimensional semantic keys before being
  passed to OverlapIndex, then translated back through the label catalog for reports.
- Multi-label targets, including sparse binary indicators supplied by the user, are
  normalized to a CSR 0/1 indicator before calling OverlapIndex.
- Regression targets are passed to ContinuousOverlapIndex as finite numeric 1D or
  2D targets.
- When `normalize_embeddings=True`, embeddings are L2-normalized row-wise before
  scoring.

This keeps extractor and caching layers free to preserve sparse artifacts longer,
while still meeting the MiniBatchKMeans backend requirements.

The internal `OverlapIndexScorer.score_cross_fitted(...)` path is available for
single-label cross-fitted diagnostics. It uses shuffled stratified folds, fits a fresh
MiniBatchKMeans-backed OverlapIndex on each training fold, scores only held-out rows
against those fixed prototypes through upstream `score_fixed(X, y)`, and returns the
mean fold macro score with fold diagnostics in metadata. It is not a replacement for
ordinary benchmark scoring and is not currently exposed as a public `Benchmark`
configuration. Multi-label and regression targets are rejected because their fold
aggregation semantics require separate protocols.

## Food-101 nonlinear-backbone experiment (Q1 confirmatory protocol)

`examples/food101_nonlinear_backbone_bridge.py` is the independent, frozen
Food-101 protocol used for the linear-accessibility story in the README. It has a
narrow, protocol-bound support flag; its universal `claim_supported` field is always
`false`, and no claim about every backbone or representation is permitted. This page
is the detailed protocol source: it records the frozen design, estimands, gates,
artifact schema, and reproducibility checks rather than duplicating numerical results.

### Frozen Food-101 design

The driver reads the official Food-101 metadata, sorts class names, and selects the
alphabetical first 40 classes. Five deterministic replicates use 80 official-train
images per class for selectors and 52 official-train images per class for head
development. The official test split remains untouched except for a 52-image-per-class
reference cohort. Replicate cohort maps are persisted; roles are disjoint and nested
selector budgets are `64, 68, 72, 80` images per class.

The 52 official-test reference IDs are fixed across all replicates. Split-local
donor/mode/nuisance banks are regenerated from each replicate seed, while the same
seed pattern pairs bank permutations and class-balanced modes across the ten
backbones and both selector methods. The prespecified per-class `{-1,+1}` mode is
part of the semi-synthetic intervention; reference test labels never fit a
selector/probe/head or choose a model. The compact analyzed extraction union is
exactly 28,480 rows: 26,400 official-train cohort rows plus 2,080 fixed official-test
reference rows (rather than the full first-40-class 40,000-row split).

Q1 fixes quality to `q=1` and evaluates exactly `baseline`, `nonlinearity_full`, and
`nuisance_full` arms. Donor, mode, and nuisance banks are split-local and copied in
each backbone's native dimension, so a development or reference row cannot draw a
donor from another split. The panel is the same ten final models used by the prior
backbone study. Each model is extracted once and released before the next model is
loaded.

The selector method is five-fold cross-fitted OverlapIndex through
`OverlapIndexScorer` with fixed `k=10` and the adapter's minimum five training
examples per prototype. The baseline is a fixed five-fold out-of-fold L2 linear
probe (`LogisticRegression(C=1)`). Four fixed heads are fit on development rows and
evaluated once on the reference rows: quadratic degree two is primary; linear,
cosine-distance kNN (`k=15`), and RBF are secondary. No family, standardization, or
hyperparameter search is allowed to alter the endpoint.

Each selector call is timed around scoring only. The paired runtime diagnostic sums
the 120 calls in each replicate (ten backbones by three arms by four budgets) for each
method. Shared feature extraction, reference-head fitting/evaluation, bootstrap work,
and process-launch overhead are excluded, so this measures selector compute rather
than complete experiment wall-clock time.

For each backbone, replicate, arm, and budget, the primary ranking statistic is the
quadratic-head Spearman log-budget AUC. The direct nonlinear effect is the
full-nonlinearity OverlapIndex-minus-OOF-probe AUC difference. The interaction is
the nonlinear-minus-baseline change in that OverlapIndex-minus-probe difference. A
full-nuisance direct contrast and its nuisance-minus-baseline interaction are
reported as diagnostics and do not replace the primary test. Kendall, regret,
exact-best, and within-tolerance summaries are secondary.

Uncertainty is a paired hierarchical bootstrap: resampling keeps each backbone's
replicate cells, arm, method, head, and nested budget grid together before drawing
the declared backbone/replicate hierarchy. The narrow confirmatory flag can be true
only for a canonical complete run when both primary 95% lower bounds are positive,
at least four of the five replicate summaries are positive, the quadratic-regret
gate passes, the complete factorial grid has no missing, duplicate, or nonfinite
cells, and at least 90% of hierarchical bootstrap draws are finite. Failed,
interrupted, partial, and exploratory artifacts never pass these gates. The nuisance
interval, complete-grid status, and finite-draw fraction are retained in the result
for auditability.

### Food-101 execution and artifacts

Install the vision dependencies and run from the repository root:

```bash
poetry install -E backbone-selection
poetry run python examples/food101_nonlinear_backbone_bridge.py
```

The tracked `examples/assets/food101_overlap_vs_linear_probe_story_summary.json`
renders the README figure without requiring the full result artifact:

```bash
poetry run python examples/plot_food101_overlap_vs_linear_probe_story.py
```

Pass a completed result JSON with `--results` when reproducing from a local run.

The CLI exposes `--jobs`, `--device`, `--resume`, `--data-dir`, `--cache-dir`,
`--output-dir`, `--models`, `--embedding-batch-size`, `--budgets`, `--seed`,
`--no-download`, `--replicates`, and `--bootstrap-resamples`; `--replicates` must
remain 5 for the frozen protocol. Worker count, device, cache/output paths, and
resume are operational controls; dataset path, model subset, batch size, budgets,
seed, and bootstrap count remain scientific configuration fields. Fresh-process
`multiprocessing` spawn workers score CPU blocks with
`maxtasksperchild=1`. Each completed block is written atomically to
`<stem>.checkpoints/<20hex>.json`, progress is printed as blocks complete, and merged
rows are sorted by block key. Resume accepts only complete checkpoints whose
deterministic key/path and configuration hash match. Embedding-cache reuse verifies
sample IDs, labels, scientific identity, array SHA-256, dtype, and shape; per-model
cache manifests persist extractor recipes, but recipes are not compared or persisted
in aggregate protocol metadata. Runtime metadata records worker count and resolved
device without changing scientific/cache identity. CUDA, Apple MPS, then CPU
are selected by `--device auto`; MPS is used for sequential extraction only, while
OverlapIndex, probes, fixed heads, bootstrap, and serialization remain CPU work.

Artifacts use the exact stem
`food101_nonlinear_backbone_bridge_food101_k10_<12hex>`, where `<12hex>` is the
scientific configuration hash. The output directory contains
`<stem>_planned_protocol.json`, `<stem>.json`, `<stem>.cohorts.json`,
`<stem>_reference.csv`, `<stem>_selector.csv`, and `<stem>_ranking.csv`;
`<stem>.cohorts.json` records replicate role indices, extracted sample IDs, train/test
counts, and the configuration hash. Model caches are `food101_<model>_final.npy` with
adjacent JSON manifests, and checkpoints are
`<stem>.checkpoints/<20hex>.json`. The aggregate JSON stores `auc_rows` and
compact quadratic `test_prediction_rows`, plus matching `schema_version` and
`protocol_version`, `artifact_status`, configuration/hash and runtime metadata,
three-arm factorial rows, direct/interaction and nuisance bootstrap summaries,
complete-grid and finite-draw/regret gates, top-level
`food101_nonlinearity_supported`, and
`claim_supported=false`. Custom hashes cannot overwrite canonical artifacts. No
ranking, accuracy, support flag, or other result should be cited until a conformant
run has completed and its JSON/CSV artifacts have been inspected.

## Food-101 selector runtime scaling (post-hoc computational benchmark)

`examples/food101_selector_runtime_scaling.py` is intentionally separate from the
Q1 accuracy protocol above. It measures selector compute only and always writes
`claim_supported=false`; it is not evidence for the nonlinear-backbone accuracy
claim. The default grid is `64, 128, 256, 512, 640` nested samples per class over
the same ten frozen backbones and five timing repeats. Each repeat reuses one
deterministic class-balanced nested subset for both methods: five-fold
cross-fitted OverlapIndex (`k=10`) and a five-fold out-of-fold L2 probe.

The driver runs serially with one scoring worker on purpose. Concurrent workers
would fold scheduling, CPU contention, and process-launch effects into a per-call
measurement. The timed interval excludes embedding-cache reads, subset
materialization, imports, warmup, feature extraction, downstream-head evaluation,
and plotting. The paired speedup is probe elapsed time divided by OverlapIndex
elapsed time; the figure annotates the paired-cell median speedup with its
interquartile range. This is descriptive spread across paired backbone/repeat
cells, not a population confidence interval.

With a completed Food-101 experiment cohort and its local cache manifests, the
driver discovers the unique cohort, validates sample/label hashes, and excludes
the fixed official-test rows automatically:

```bash
poetry run python examples/food101_selector_runtime_scaling.py --output-dir examples/output
poetry run python examples/plot_food101_selector_runtime_scaling.py
```

For an exploratory panel, pass repeated `--embedding-manifest` paths and a
`--labels-manifest`, then pass the completed configuration-hashed JSON to the
plotter with `--results`. The driver writes planned/completed/failed JSON,
per-backbone checkpoints, and raw/paired CSV rows, printing progress outside the
timed intervals; `--resume` reuses validated checkpoints. Runtime artifacts are
computational diagnostics with explicit boundaries; do not merge them into the
confirmatory accuracy table or infer total experiment wall-clock time.

In the completed canonical run, the paired median probe-over-OverlapIndex speedup
rose from `0.70×` at 2,560 embeddings to `2.00×` at 25,600 embeddings. OverlapIndex
was faster in all 50 paired cells at both 20,480 and 25,600 embeddings. The compact
tracked summary at
`examples/assets/food101_selector_runtime_scaling_summary.json` records the plotted
aggregates and source-artifact checksum; these remain descriptive, hardware-specific
runtime results rather than representation-quality evidence.

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
slices. Sparse matrices remain CSR through every repeated OverlapIndex call.

Subsample stability is also available when requested through `StabilityConfig`.
It is target-aware by definition:

- single-label targets are sampled per class;
- multi-label targets are sampled to preserve every active label;
- regression targets use target-preserving samples containing at least one
  non-constant target.

Categorical repeats retain at least two occurrences of every original class or
active label, while regression repeats retain at least three rows. The requested
fraction is applied with `floor(count * subsample_fraction)`. If those minimums
cannot be met, validation fails before any repeat is scored and recommends either
a larger fraction or prototype mode; the subset is never silently expanded.

```python
from vertebrae import OverlapScoringConfig, StabilityConfig

scoring = OverlapScoringConfig()
stability = StabilityConfig(mode="prototype", repeats=20, interval_level=0.95)
subsample_stability = StabilityConfig(
    mode="subsample",
    repeats=20,
    subsample_fraction=0.8,
    random_state=42,
)
```

Scoring seeds and sampling seeds use independent deterministic streams. Subsample
results record `sampling_seeds`, `effective_sample_counts`, and
`effective_subsample_fractions` so the sampling plan can be reproduced and audited.
`StabilityConfig.stratified` has been removed: categorical subsampling cannot opt
out of target preservation. Serialized configurations containing that field are
intentionally unsupported.

`vertebrae` reports these as stability summaries and stability intervals. They are
not formal confidence intervals unless a different statistical protocol is added
explicitly in a future release.

## Separatix diagnostics

Use `SeparatixConfig` to control the optional
[Separatix](https://github.com/NiklasMelton/Separatix) complexity diagnostic stage:

```python
from vertebrae import SeparatixConfig

config = SeparatixConfig(
    enabled=True,
    overlap_threshold=0.80,
    random_state=42,
    densify_policy="warn_and_sample",
    mlp_probes=True,
    mlp_trigger_skill_threshold=0.75,
    mlp_min_improvement=0.02,
)
```

Current behavior:

- Separatix runs on the same evaluated embedding variant that overlap scores.
- It runs after compression and after the main overlap score is available.
- By default it only runs when the overlap gate passes.
- Classification and multi-label datasets use `overlap_threshold`.
- Regression datasets use `regression_overlap_threshold`.
- Multi-label targets are passed to Separatix as CSR 0/1 indicator matrices with
  `target_mode="multilabel"`.
- Regression targets are passed with `target_mode="regression"`.
- Optional Separatix MLP probes can be enabled with `mlp_probes=True`.
- `mlp_trigger_skill_threshold` controls only whether MLP computation is attempted;
  `mlp_min_improvement` controls whether paired MLP evidence can override the
  simpler-family guidance. These are intentionally separate so an MLP can be
  selected when the simpler probes are useful but not already near-perfect.
- Linear/nonlinear probe comparisons use the declared metric direction: accuracy/F1
  families favor larger values, while MAE/RMSE favor smaller values. Reported
  improvement and favored-family fields follow that same direction.
- The machine-readable `recommendation` is one of eight shared labels:
  `linear_likely_sufficient`, `smooth_nonlinear_recommended`,
  `kernel_or_local_recommended`, `high_capacity_or_partitioning_recommended`,
  `feedforward_mlp_recommended`, `feature_or_target_bottleneck_likely`,
  `insufficient_data_or_unreliable_geometry`, or `inconclusive`.
  These labels are deliberately target-agnostic. Plain-text recommendation
  headlines and suggested model names still use `target_type`, so a regression
  result does not lose its regression-specific interpretation.
- The full Separatix report is preserved in JSON outputs, while Markdown reports
  show a compact recommendation, confidence, decision path, key scores, probe
  evidence, evaluation context, and skips.

Separatix follows the same normalization convention as overlap scoring when
`normalize_embeddings=True`. Sparse inputs remain sparse at the vertebrae boundary.
`densify_policy` controls unavoidable dense-only diagnostics:

- `"warn_and_sample"` (default) samples within the effective whole-MiB budget derived
  from `max_dense_bytes` and records warnings;
- `"skip"` records the unavailable diagnostic without densifying;
- `"fail"` raises when a required dense operation cannot fit.

`SeparatixResult.preprocessing`, `densification_events`, `skipped_diagnostics`, and
`warnings` expose the sparse and memory audit directly; the complete upstream report
remains available through `SeparatixResult.report`.

Separatix is the only downstream-complexity and probe-style diagnostic path in
vertebrae. Ranking remains based on overlap scores; Separatix probe fields are
reported only when the Separatix diagnostic runs and includes them.

`SeparatixResult.probe_summary` normalizes the existing probe evidence without
fitting another model. It records the best probe, a target-appropriate primary
metric when Separatix declares one, the complete reported metric map, comparable
linear/nonlinear evidence, evaluation and sampling context, grouping counts, and
probe-specific skip reasons. Its evaluation context also records whether the
comparison is estimator-aligned, the cross-validation plan and cohort size, and
the effective train-size summary (the number of rows available to each fitted
fold). This makes it possible to distinguish a genuinely different family
recommendation from a comparison made with a different estimator or training
budget. Single-label fallback summaries use Separatix's balanced-accuracy
baseline contract. Multi-label and regression summaries never invent an accuracy
value or select an undeclared primary metric.

Separatix `0.1.1` also exposes an uncertainty-aware family frontier. The
`SeparatixResult.family_guidance` mapping contains the guidance status, target
type, reason when unavailable, `minimum_recommended_family`,
`plausible_families`, and the `decision_method`. It separately records the
selected family/probe, the selected recipe id, whether an MLP override is active,
and the paired-comparison status and method. The minimum family is the simplest
family supported by the evidence; the plausible set retains alternatives that
cannot be ruled out. Consumers should not treat a family outside that set as
equally supported merely because its point estimate is close.

Probe recipes are versioned, JSON-safe Separatix data rather than opaque live
estimators. Use `separatix_result.probe_recipe(name_or_id)` to retrieve a retained
recipe or `separatix_result.selected_probe_recipe()` for the family/probe selected
by the guidance.
Pass a returned recipe to Separatix's `make_probe_estimator(...)` factory when a
downstream audit must reproduce the exact preprocessing, sketch, kernel, or MLP
architecture used by the diagnostic. Missing or unavailable recipes return
`None`; Vertebrae does not guess a replacement estimator.

These fields are descriptive diagnostics. They do not participate in ranking,
aggregate validity, or vertebrae's benchmark recommendations. Recommendation
confidence and probe-comparison confidence are reported separately because they
describe different evidence. `inconclusive` and
`insufficient_data_or_unreliable_geometry` retain Separatix's target-specific
confidence handling; do not infer a high-confidence regression conclusion from
the shared label alone.

## Custom embedding metrics

`Benchmark` can score the same full embedding batch with one or more custom metrics.
Every metric must return one finite aggregate `score`; that score is the only value
eligible for ranking. Metrics may also return JSON-safe diagnostics, warnings, and
metadata. Vertebrae preserves sparse embeddings at this boundary; a custom metric
that densifies them is responsible for enforcing its own memory limit.

Classification labels and independence groups cross the metric boundary as marked
semantic-key strings in both local and artifact-backed runs. This makes custom-metric
behavior identical for typed classes such as integers, booleans, dates, UUIDs, and
Decimals without importing user label classes on workers. Use the `label_catalog` in
`target_metadata` for original typed provenance and display text; do not perform numeric
arithmetic directly on categorical labels. Regression targets remain numeric.

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

Relevance may be a NumPy/scipy query-by-gallery matrix or sparse
`(query_id, gallery_id, grade)` records. Any other iterable, including a nested Python
list, is interpreted as records; use `RetrievalDataset.from_relevance_matrix(...)` for
an explicit nested-list matrix. Sparse matrices are read from their nonzeros without
densification, and every query must retain at least one eligible positive after
exclusions. Grades above zero are relevant for binary metrics; NDCG uses the original
finite nonnegative grades and stable log-domain gain ratios. Results include NDCG,
precision, recall, hit rate, MRR, mAP, and positive-versus-negative similarity
diagnostics. Cosine similarity is the default, with dot product and squared L2 available
explicitly.

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
