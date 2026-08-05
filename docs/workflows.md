# workflows

Vertebrae shares extraction, caching, compression, execution, and reporting
infrastructure across several evaluation protocols, while keeping each protocol's
scoring semantics explicit. This page is the complete capability inventory; the
[project README](https://github.com/NiklasMelton/vertebrae/blob/develop/README.md) is the shorter narrative introduction.

## Labeled embedding benchmarks

The ordinary `Benchmark` and `Evaluator` workflows support:

- precomputed dense or scipy sparse embeddings;
- NumPy arrays and pandas DataFrames;
- single-label classification, multi-label classification including sparse binary
  indicators, and explicitly declared regression targets;
- hierarchy label views and named target views for scoring one embedding against
  different aligned targets;
- graph-node, graph-edge, entity, pair, triplet-derived, and generic labeled-unit
  embedding diagnostics;
- one extractor, comparisons among several extractors, and several named outputs from
  one backbone;
- scikit-learn transformers and pipelines, custom Python callables, local PyTorch,
  Keras, and ONNX models, and the optional extractor families listed in the
  [extractor guide](https://github.com/NiklasMelton/vertebrae/blob/develop/docs/feature_extractors.md);
- the built-in OverlapIndex metric, additional custom full-batch embedding metrics,
  repeated-run stability analysis, and gated Separatix diagnostics.

Classification and multi-label workflows use discrete OverlapIndex. Explicit
regression uses ContinuousOverlapIndex through the same internal scoring adapter.
These are representation diagnostics for the declared dataset and targets, not
substitutes for task-native behavior metrics.

## Structured and spatial evaluation

Dense segmentation workflows materialize spatial feature cells against semantic mask
labels, with deterministic ambiguity filtering, background handling, class/instance
caps, image groups, and token provenance. The score describes retained token geometry;
it is not IoU, mask accuracy, or boundary quality.

Structured-output workflows materialize native regions, tokens, frames, keypoints,
depth cells, or latent slots into aligned labeled rows. Explicit selection rules handle
model rows that do not correspond directly to annotations. Structured targets may be
single-label, multi-label, or regression where supported. These workflows do not
replace mAP, WER/CER, OKS, depth-error, or reconstruction metrics.

Relational constructors for nodes, edges, entities, pairs, and triplets remain
supervised labeled-embedding diagnostics. They are not silently treated as retrieval,
recommendation, or link-prediction protocols.

## Retrieval and zero-shot protocols

`RetrievalBenchmark` performs exact, training-free query--gallery evaluation with
explicit binary or graded relevance. It reports NDCG, recall, precision, mAP, and MRR
at configured cutoffs and can evaluate both directions. Branch identity, query/gallery
IDs, exclusions, and pairwise memory limits remain part of the protocol.

`ZeroShotBenchmark` compares frozen sample embeddings with frozen prototypes built
from declared text prompts. It reports top-k classification metrics and keeps
OverlapIndex on the sample embeddings as separate context. It does not fit heads,
search prompts against test labels, calibrate scores, or combine zero-shot and overlap
into a universal metric.

Both protocols support paired compression that fits one shared transform on the
declared reference side and applies it to both embedding spaces.

## Monitoring, compression, and reports

`RepresentationMonitor` repeatedly evaluates live extractor outputs during training
while the caller retains control of optimization, checkpointing, and cadence. Named
outputs support time-by-layer trajectories, append-only history, and optional reporters.

Compression variants include PCA, incremental PCA, truncated SVD, Gaussian and sparse
random projection, prefix truncation for declared Matryoshka embeddings, and float16,
int8, or uint8 quantization. Raw artifacts remain reusable and every derived variant
retains its compression recipe and provenance.

Results are practical Python objects with ranking helpers and pandas DataFrame views.
JSON preserves complete structured diagnostics; Markdown provides a review-oriented
report. Reports cover dataset and extractor summaries, target and output variants,
overlap details, stability, Separatix, compression, warnings, recommendations,
resources, and reproducibility metadata.

## Artifact-backed execution

Vertebrae supports local caching and deterministic artifact-backed execution through
the CLI and explicit local, Ray, or Dask backends. Artifact stores may use local paths,
`file://`, `s3://`, or `gs://` locations.

The CLI covers:

- ordinary labeled workflows: `plan`, `fit-extractor`, `embed-shard`,
  `merge-embeddings`, `write-labels`, `write-groups`, `compress`, `score`,
  `diagnose-complexity`, `score-repeats`, `collect-scores`, and
  `benchmark-from-artifacts`;
- structured and spatial workflows: `materialize-structured` and
  `materialize-segmentation`;
- retrieval workflows: `plan-retrieval`, `embed-retrieval-shard`,
  `merge-retrieval-embeddings`, `write-retrieval-relevance`, `compress-retrieval`,
  and `score-retrieval`;
- zero-shot workflows: `plan-zero-shot`, `embed-zero-shot-shard`,
  `merge-zero-shot-embeddings`, `write-zero-shot-protocol`, `compress-zero-shot`,
  `score-zero-shot`, and `zero-shot-from-artifacts`;
- orchestration helpers: `slurm-array`, `slurm-score-array`, and
  `run-embedding-shards`.

Workers consume fitted extractors or persisted artifacts and preserve deterministic
row order. Multi-output, target-view, label-view, grouped, regression, structured,
segmentation, retrieval, and zero-shot metadata remain explicit at artifact boundaries.
See [distributed readiness](https://github.com/NiklasMelton/vertebrae/blob/develop/docs/distributed_readiness.md) for commands, manifests,
trusted-input rules, cloud stores, and memory admission.
