# vertebrae examples

These examples are small enough for a laptop or desktop and are meant to show
practical workflows you can adapt for real datasets. The core examples are
network-free; the Hugging Face examples require optional dependencies and a model
available from the local Hugging Face cache or downloaded on first run.
Their introductory remote model names are intentionally unpinned, so those scripts
explicitly disable embedding caching. For a reusable production run, supply a full
immutable model revision or an explicit, maintained `cache_identity`.

Run them from the project root:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/precomputed_embeddings.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/multilabel_precomputed_embeddings.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/relational_embeddings.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_text_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_tabular_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_wine_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/multi_extractor_comparison.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/dispatched_benchmark.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/cache_reuse.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/zero_shot_callable.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/structured_outputs.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/structured_depth.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/structured_latent_slots.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/torch_local_model.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/representation_monitoring.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/fashion_mnist_visual_suite.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/fashion_mnist_corruption_atlas.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/tiny_shakespeare_transformer_visual_suite.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/fashion_mnist_overfitting.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/keras_local_model.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/onnx_extractor.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/hf_multimodal_image_text.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/caltech101_vision_foundation_models.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/food101_nonlinear_backbone_bridge.py
```

Each script writes reports to `examples/output/`.

## Included workflows

- `precomputed_embeddings.py`: evaluate embeddings that were already generated
  elsewhere.
- `multilabel_precomputed_embeddings.py`: evaluate precomputed embeddings against
  a small multi-label classification target and write JSON/Markdown reports.
- `relational_embeddings.py`: evaluate graph node and edge embeddings through the
  same overlap-based transfer-learning diagnostics as other precomputed
  embeddings. This is not a graph-link, retrieval, or ranking benchmark.
- `sklearn_text_pipeline.py`: build embeddings from a pandas text dataframe with a
  scikit-learn pipeline.
- `sklearn_tabular_pipeline.py`: build embeddings from mixed numeric/categorical
  dataframe columns with a `ColumnTransformer`.
- `sklearn_wine_pipeline.py`: compare real scikit-learn scaling/projection
  pipelines on the bundled UCI Wine dataset.
- `multi_extractor_comparison.py`: compare several extractors on the same labeled
  numeric dataset.
- `dispatched_benchmark.py`: run the ordinary benchmark API through a two-worker,
  artifact-backed local execution backend.
- `cache_reuse.py`: show how an explicit callable `cache_identity` enables safe
  embedding reuse across repeated runs. Callables without a provably portable
  identity still run but deliberately bypass the reusable cache.
- `zero_shot_callable.py`: compare a synthetic frozen image/text-aligned adapter
  with an explicit prompt protocol, without downloads or a learned head.
- `structured_outputs.py`: materialize OCR/layout regions, ASR tokens, and pose
  keypoints directly from native structured extractors, then score them as
  representation-efficacy workflows rather than IoU, WER/CER, or OKS metrics. The
  deterministic lookup closures use manually versioned cache identities.
- `structured_depth.py`: materialize sampled depth cells and score them through
  continuous overlap as a structured regression workflow rather than a depth
  error benchmark.
- `structured_latent_slots.py`: materialize labeled latent slots from raw
  per-parent matrices, using a standard aligner helper to drop unmatched rows
  before scoring.
- `torch_local_model.py`: demonstrate `TorchExtractor` with content-digested checkpoint
  provenance, inference-mode/eval defaults, and user-supplied
  `collate_fn` / `output_fn`.
- `representation_monitoring.py`: train a small local Torch network, monitor two
  explicitly named representations after every epoch, persist summary history and
  caller-managed training state, resume them together with `--resume`, and pivot the
  final DataFrame by epoch and hidden layer.
- `fashion_mnist_visual_suite.py`: train a compact two-block PyTorch CNN on a deterministic
  Fashion-MNIST subset with a fixed held-out validation set, monitor hidden
  pooled convolutional representations and a 128-dimensional embedding every two
  optimizer steps, compare compression variants, evaluate nested product-category
  views, save a reusable trained checkpoint, and generate the README's
  representation-monitoring, compression-frontier, and layer-by-hierarchy figures.
  The first run downloads Fashion-MNIST through torchvision; install all required
  optional dependencies with `poetry install -E visuals`.
- `fashion_mnist_corruption_atlas.py`: reuse that trained checkpoint (or train it once
  when absent), then evaluate a fixed official-test probe under blur, Gaussian noise,
  occlusion, contrast reduction, and rotation at four increasing severities. The atlas
  aligns layer-wise OverlapIndex retention with accuracy and cross-entropy, shows which
  classes lose embedding geometry under severe shift, and marks the first corruption
  tier where each classifier-confusion pair materially increases. CSV protocol tables
  make the visual drill-down reproducible.
- `tiny_shakespeare_transformer_visual_suite.py`: checksum and cache the 1.1 MB Tiny
  Shakespeare corpus, then train either the four-block `fast` character GPT or the
  4.82-million-parameter `quality` profile (`--profile quality`) with six blocks, width
  256, 256-character contexts, and dropout. Four hidden representations are evaluated
  against the exact next character on a fixed held-out validation probe. The probe
  deterministically retains every character with at least two eligible occurrences (up
  to 256 rows per class); classes below the configurable macro-support threshold remain
  in per-token diagnostics but are excluded from macro aggregation. A separate naturally
  distributed validation pass reports cross-entropy, perplexity, and top-1 accuracy.
  Final generation, compression, and reports use the best-validation checkpoint. The
  suite also compares PCA and quantization and writes monitoring, compression-frontier,
  and complete per-character heatmap figures. Install `poetry install -E text-visuals`;
  the default `--device auto` smoke-tests CUDA/ROCm, MPS, and XPU before its CPU fallback,
  while `--no-download` requires an already checksum-valid cache.
- `fashion_mnist_overfitting.py`: compare clean-label and noisy-label CNNs that start from
  identical weights and receive identical images, mini-batch order, optimizer settings,
  and schedules. Per-layer OverlapIndex panels evaluate both models on the same clean
  validation probe and the same corrupted-target subset, separating transferable class
  geometry from alignment with deliberately incorrect labels. Loss defines the shaded
  overfitting region independently of OverlapIndex. The first run downloads Fashion-MNIST
  through torchvision and requires the `visuals` extra.
- `colored_fashion_mnist_shortcut.py`: create a controlled colored Fashion-MNIST
  shortcut-learning experiment with paired, identically initialized control and
  treatment CNNs. The control receives colors balanced within every class; only the
  treatment receives class-correlated colors. Every layer is evaluated against the
  named `intended_class` and `nuisance_color` views on a separate balanced audit probe.
  Three paired seeds are aggregated by default. Correlated, balanced, reversed-color,
  grayscale, and exhaustive all-color tests measure accuracy, prediction flips,
  color-following errors, and robust class×color cell summaries. The script also writes
  aggregate monitoring, paired-effect, and documentation-ready PNG/SVG exemplar figures.
  The first run downloads Fashion-MNIST through torchvision and requires the `visuals`
  extra; use `--repeats 1` for a quick smoke run.
- `keras_local_model.py`: demonstrate `KerasExtractor` with content-digested provenance
  for a locally saved Keras model and user-supplied `collate_fn` / `output_fn`.
- `onnx_extractor.py`: demonstrate `ONNXExtractor` against a local ONNX export
  you provide via `VERTABRAE_ONNX_MODEL_PATH`.
- `hf_audio_extractor.py`: Hugging Face audio backbone API example. Requires optional
  dependencies and a local or downloadable model.
- `hf_multimodal_image_text.py`: aligned image-text benchmarking example with
  image, text, and fused Hugging Face outputs. Requires optional dependencies and
  a local or downloadable model.
- `hf_text_extractor.py`: Hugging Face text backbone API example. Requires optional
  dependencies and a local or downloadable model.
- `hf_time_series_extractor.py`: Hugging Face time-series backbone API example.
  Requires optional dependencies and a local or downloadable model.
- `hf_video_extractor.py`: Hugging Face video backbone API example using
  predecoded clips. Requires optional dependencies and a local or downloadable model.
- `hf_vision_mnist.py`: compare MNIST handwritten digit images with generic,
  final-layer and mid-layer Hugging Face vision embeddings from one model
  configuration plus a scikit-learn image pipeline. Requires optional
  dependencies and local or downloadable data/models.
- `caltech101_vision_foundation_models.py`: compare a laptop-sized Caltech-101
  subset with related category pairs using DINOv2, a tiny supervised ViT baseline,
  and an optional gated DINOv3 extractor. Downloads the dataset archive when it
  is not already present locally.
- `oxford_pets_backbone_selection.py`: screen frozen vision backbones and layers on
  Oxford-IIIT Pet without consulting final test accuracy. A disjoint representation
  probe ranks candidates by clean breed OverlapIndex; DINOv2 and DeiT contribute
  quarter-, middle-, late-, and final-block CLS representations to expose a broad
  transfer-quality range without additional image forward passes. A standardized
  linear-head scatter tests the representation-screening claim, and an OI-ranked
  budget curve shows how quickly that ranking finds useful candidates relative to
  random search. The ordinary head-choice audit runs Separatix on clean head-training
  rows and compares its paired MLP-minus-linear evidence with clean validation
  advantage. A simplified target-view heatmap and background shift-effect plot remain
  separate diagnostics. A relational audit then reuses the cached embeddings for
  balanced same-breed verification with same-species hard negatives. It runs Separatix
  on the combined non-test development pairs (head-train plus validation), reconstructs
  every downstream estimator from the exact versioned probe recipe, and validates
  linear, smooth, local/kernel, and MLP heads only on untouched source-disjoint test
  pairs. The plot marks the selected deployment
  family (including an active MLP override) with a thick border and the plausible core
  alternatives with small orange markers; the serialized guidance also retains the
  minimum recommended family when it differs. Its star is a retrospective
  simplest-near-best test oracle, never a selection input. The experiment writes protocol JSON/CSV data, including
  family guidance, recipe ids, paired evidence, effective train size, and held-out
  test-regret summaries, plus six plot families. Install `poetry install -E
  backbone-selection`; the first run downloads the dataset and requested checkpoints.
  Use `--replot-from examples/output/oxford_pets_backbone_selection.json` to regenerate
  the figures without rerunning feature extraction or head training.
- `food101_nonlinear_backbone_bridge.py`: run the independent Food-101 Q1
  confirmatory experiment. The frozen panel is the alphabetical first 40 classes,
  with five paired replicates, disjoint official-train selector/development cohorts
  (80 and 52 images per class), a fixed 52-image-per-class official-test reference
  cohort, and nested selector budgets `(64, 68, 72, 80)`. Split-local donor,
  mode, and nuisance banks are copied for each backbone; Q1 compares baseline,
  full-nonlinearity, and full-nuisance arms. Five-fold cross-fitted OverlapIndex
  (`k=10`) is paired with a fixed five-fold out-of-fold L2 logistic probe. Fixed
  quadratic, linear, cosine `k=15` kNN, and RBF heads are evaluated on the reference
  rows, with quadratic primary. The detailed protocol and artifact schema are in the
  [scoring guide](../docs/scoring.md#food-101-nonlinear-backbone-experiment-q1-confirmatory-protocol).
  Run it with:

  ```bash
  poetry install -E backbone-selection
  poetry run python examples/food101_nonlinear_backbone_bridge.py
  ```

  Regenerate the README figure from the tracked
  `examples/assets/food101_overlap_vs_linear_probe_story_summary.json` (or pass a
  completed result artifact with `--results`):

  ```bash
  poetry run python examples/plot_food101_overlap_vs_linear_probe_story.py
  ```

  The plot writes `../img/visuals/food101-overlap-vs-linear-probe-story.png` and
  `.svg`. Its runtime panel compares summed selector scoring-call time across 120
  paired calls per replicate; shared feature extraction and downstream-head evaluation
  are excluded. Existing `food101_nonlinear_backbone_bridge_*` filenames and artifact
  stems remain technical identifiers; no universal claim is implied by this experiment.
- `zero_shot_transfer_structure.py`: flagship CIFAR-10 OpenCLIP experiment that
  encodes images once, varies only explicit text prompt sets, and contrasts fixed
  global/per-class OverlapIndex with prompt-sensitive zero-shot accuracy/F1. Requires
  `openclip` and `visuals` extras; downloads CIFAR-10 and the checkpoint on first use.
- `sentence_transformer_extractor.py`: sentence-transformers API example. Requires
  optional dependencies and a local or downloadable model.

The library also now exposes first-class adapters for timm, torchvision,
OpenCLIP/SigLIP-style image-text models, TensorFlow Hub, JAX/Flax, tree leaf
embeddings, graph models, and hosted embedding APIs. Those integrations are
covered in unit tests with fake modules today; dedicated runnable example
scripts can be added as the public ergonomics settle.

Most compact examples configure small `k` values and a few stability repeats so they
finish quickly while still exercising the real MiniBatchKMeans-backed OverlapIndex
path. The Food-101 protocol is intentionally a substantial multi-backbone run; use
its documented checkpoints and resume controls when running the full analysis.

Run the Hugging Face vision example after installing optional dependencies:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install -E hf
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/hf_vision_mnist.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/caltech101_vision_foundation_models.py
```

Run the zero-shot versus transfer-structure experiment after installing its OpenCLIP
and visualization extras:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install -E openclip -E visuals
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/zero_shot_transfer_structure.py
```

To include DINOv3 in the Caltech-101 example, accept the model terms on Hugging
Face and set `VERTABRAE_INCLUDE_DINOV3=1`.
Override `VERTABRAE_CALTECH101_CLASSES` to make the slice easier or harder.
