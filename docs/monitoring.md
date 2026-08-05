# representation monitoring

`RepresentationMonitor` evaluates a live extractor repeatedly while another piece
of code owns the training loop. Each explicit call to `evaluate(...)` constructs a
fresh `Benchmark`, records its result against caller-supplied training coordinates,
and optionally reports a compact snapshot.

Monitoring is intentionally observational. It does not schedule evaluations, manage
optimizers, load checkpoints, install layer hooks, stop training, or run work
asynchronously. Retrieval and zero-shot protocols also remain separate because they
do not return `BenchmarkResult`.

## Basic workflow

Use a fixed held-out probe dataset and retain the same extractor object that wraps the
model being trained:

```python
from vertebrae import (
    ConsoleReporter,
    EvaluationHistoryConfig,
    RepresentationMonitor,
    SeparatixConfig,
    StabilityConfig,
)

monitor = RepresentationMonitor(
    probe_dataset,
    [live_extractor],
    history_config=EvaluationHistoryConfig(
        storage="disk",
        path="training-representations.jsonl",
        detail="summary",
    ),
    reporters=[ConsoleReporter()],
    stability_config=StabilityConfig(enabled=False),
    separatix_config=SeparatixConfig(enabled=False),
)

for epoch in range(number_of_epochs):
    train_one_epoch(model, optimizer, training_data)
    monitor.evaluate(
        epoch=epoch,
        global_step=global_step,
        checkpoint=f"checkpoints/epoch-{epoch}.pt",
        metadata={"training_loss": training_loss},
    )

history = monitor.history.to_dataframe()
layer_curve = history.pivot_table(
    index="epoch",
    columns="hidden_layer",
    values="overlap_score",
)
```

Every evaluation needs at least one of `snapshot_id`, `epoch`, `global_step`,
`timestamp`, or `checkpoint`. Epoch and step values must be nonnegative integers.
Timestamps may be timezone-aware `datetime` objects or ISO-8601 strings and are
stored in UTC. Checkpoint values are provenance only; monitoring neither checks nor
loads the path. Metadata must have string keys and finite values supported by
Vertebrae's strict serializer. Supported scientific Python scalars, arrays, paths,
sets, enums, and dataclasses are normalized deterministically to JSON data.

The automatic `recorded_at` field records when Vertebrae accepted the evaluation.
Identity is based only on the explicit identifier fields. Reusing the same identity
emits a warning but still appends a new record with the next zero-based
`evaluation_index`; caller metadata and `recorded_at` do not make an identity unique.

## Fixed probes and interpretation

Use the same held-out samples, targets, target or label views, structured alignment
rules, segmentation sampling configuration, and random seeds throughout a training
run. Otherwise changes in the history may reflect changes in the evaluation protocol
rather than changes in the representation.

With named model outputs, each evaluation contributes one row per layer/output and
per compression or view variant. `parent_extractor`, `output_name`, `hidden_layer`,
and `pooling` preserve the declared representation identity. The result is therefore
suited to time-by-depth inspection, but it remains a diagnostic for the fixed probe
dataset and configured protocol—not a universal claim about model quality.

Local Torch and Keras adapters never discover layers or install hooks. Declare
ordinary outputs explicitly and make the model or `output_fn` return them:

```python
from vertebrae.extractors import TorchExtractor

extractor = TorchExtractor(
    "live_encoder",
    model=model,
    collate_fn=collate,
    output_fn=lambda raw: {
        "block_1": raw["block_1"],
        "embedding": raw["embedding"],
    },
    outputs=[
        {"name": "block_1", "hidden_layer": 1, "pooling": "identity"},
        {"name": "embedding", "hidden_layer": 2, "pooling": "identity"},
    ],
)
```

The same `outputs` mapping is supported by `KerasExtractor`. A batch invokes the
model once and applies `output_fn` once. `Benchmark` and `Evaluator` expand multiple
outputs into independent results. Direct `transform()` is only available with
exactly one declared output; use `transform_many()` for multiple outputs.

Without an explicit `outputs` declaration, Torch and Keras preserve their legacy
contract and require a 2D numeric result. Explicit declarations default to
`flatten=True`. Multiple selector-free declarations require a mapping keyed by the
declared output names; selector-free members of mixed configurations must likewise
be present by name.

## Fresh computation and cost

Monitoring defaults to `CacheConfig(enabled=False)` and always sets
`force_recompute=True`, including when an enabled cache is supplied. An evaluation
never reads an earlier embedding, compression, scoring, stability, or Separatix
artifact. Enabled caches may retain the latest artifacts under their normal keys,
but snapshot identifiers do not alter those keys and are not a historical embedding
store.

Every explicit `evaluate()` runs the complete configured benchmark stack:
extraction, compression variants, metrics, stability, Separatix gating/diagnostics,
resource profiling, and reporting. The caller controls cadence. For frequent checks,
use fewer stability repeats, disable stability, tighten Separatix sample/probe
budgets, or disable Separatix and run a fuller evaluation less often.

## History storage

Memory-backed summary history is the default:

```python
monitor = RepresentationMonitor(probe_dataset, [extractor])
```

Summary records retain tidy operational rows, including identifiers, scores,
aggregate metric columns, compact stability and Separatix fields, warnings, runtimes,
and compact resource fields. They omit per-class and pairwise diagnostics, complete
stability repeats, full Separatix reports, and nested resource profiles.

`detail="full"` additionally stores the complete `BenchmarkResult.to_dict()` payload.
Failed full-detail evaluations also retain traceback text; summary failures retain
only their type and message. A failed evaluation always contributes one row with null
result fields.

Disk history is local, append-only UTF-8 JSONL:

```python
config = EvaluationHistoryConfig(
    storage="disk",
    path="training-representations.jsonl",
    detail="summary",
    resume=True,
)
```

The first line is a schema-version-2 manifest and every later line is one evaluation. Each
line is flushed before `evaluate()` returns. Disk history supports one writer; it
does not provide locking or a remote store. A nonempty existing file is rejected
unless `resume=True`. Resume validates the schema, detail level, dataset identity,
extractor declarations, and resolved benchmark protocol before reading or appending
records. Mismatches raise without changing the file. Resume also rejects malformed
or truncated records, restores duplicate detection, and continues the evaluation
index. Earlier schema-version-1 monitoring files are not compatible with this
unreleased alpha format and must be recreated.

Whether caching is enabled is part of protocol identity because it can affect
persisted-artifact metadata and resource measurements. Cache locations, provider
credentials/options, and live-checkpoint content digests are excluded.

Protocol validation cannot identify the current live weights. To resume training, the
caller must restore the matching model and optimizer state and continue the recorded
epoch/global-step coordinates before constructing the resumed monitor.

For read-only inspection:

```python
from vertebrae import EvaluationHistory

history = EvaluationHistory.load("training-representations.jsonl")
all_rows = history.to_dataframe()
latest_rows = history.latest_dataframe()
```

Memory and disk modes return the same DataFrame schema.

## Failures and reporters

`error_policy="raise"` is the default. Benchmark failures are committed before the
original exception is re-raised. With `error_policy="continue"`, the same failure is
recorded and `evaluate()` returns `None`.

History validation, strict serialization, and disk writes always raise because the
record cannot be guaranteed. Reporters are different: they run only after a record is
committed, and reporter exceptions become warnings without changing evaluation status
or interrupting training.

`ConsoleReporter` prints the context, output/layer identity, primary score, overlap
score, and Separatix recommendation or skip state. Reporting is opt-in; the default is
silent.

See the network-free
[`examples/representation_monitoring.py`](https://github.com/NiklasMelton/vertebrae/blob/develop/examples/representation_monitoring.py)
workflow for a complete local Torch training loop with two named representations,
caller-managed checkpoint restoration via `--resume`, strict JSONL resume, and a
final epoch-by-layer pivot.

For a visual real-data workflow,
[`examples/fashion_mnist_visual_suite.py`](https://github.com/NiklasMelton/vertebrae/blob/develop/examples/fashion_mnist_visual_suite.py)
trains a compact two-block PyTorch CNN on Fashion-MNIST, monitors two pooled
convolutional representations and its 128-dimensional embedding from
initialization through three epochs at optimizer-step cadence, evaluates only against
a stratified held-out validation set, disables Separatix and stability repeats, and
renders the network architecture beside the layer-wise OverlapIndex trajectories.

[`examples/fashion_mnist_corruption_atlas.py`](https://github.com/NiklasMelton/vertebrae/blob/develop/examples/fashion_mnist_corruption_atlas.py)
reuses the suite's trained checkpoint as a model-release audit. A fixed, stratified
official-test probe is evaluated clean and under blur, noise, occlusion, contrast
reduction, and rotation at four increasing severities. Layer-wise OverlapIndex retention
localizes representation damage; accuracy and cross-entropy quantify behavior; and the
class and pairwise panels identify affected labels and the first new confusions. The
protocol uses the same fixed `k=5` for every condition so layer retention is comparable
within this probe.

[`examples/fashion_mnist_overfitting.py`](https://github.com/NiklasMelton/vertebrae/blob/develop/examples/fashion_mnist_overfitting.py)
extends that monitoring pattern into a paired memorization experiment. Clean-label and
noisy-label CNNs start from identical weights and receive the same images, mini-batch
order, optimizer settings, and schedule. Both models are monitored on the same clean
validation probe and the same subset of deliberately corrupted training targets, making
per-layer differences attributable to the training-target treatment. The plot marks the
noisy model's minimum clean validation loss and shades the later region, so its
overfitting label is defined independently of the representation metric being observed.

## Case-study protocols and interpretation

The project README keeps short versions of the following stories. This section retains
their protocol choices, reported measurements, interpretation aids, and reproduction
commands.

### Corruption and deployment-shift atlas

The corruption atlas evaluates blur, Gaussian noise, occlusion, contrast reduction,
and rotation at four severities on the untouched Fashion-MNIST test split. A fixed
`k=5` makes within-layer OI retention, `corrupted OI / clean OI`, comparable across
conditions. Retention can exceed one when a corruption compacts classes, so the atlas
shows accuracy and cross-entropy beside OI rather than treating retention as behavior.

Severe noise retains `0.89` of clean embedding OI with accuracy `0.733`. Severe
contrast is a backbone failure: first- and second-block retention fall to `0.11` and
`0.37`, with accuracy `0.299`. Occlusion drives both convolutional layers to roughly
`0.24` retention and accuracy to `0.448`. Rotation instead preserves first-block OI
and `0.92` of second-block OI while the embedding falls to `0.73` and accuracy to
`0.407`, pointing toward reuse of early features and later-block fine-tuning. Mild
occlusion first increases Sandal/Sneaker interference; mild rotation first increases
Sneaker/Ankle boot interference.

| Observed pattern | Practical interpretation | Likely next experiment |
| --- | --- | --- |
| Accuracy worsens while early-layer OI stays near `1.0` and late-layer OI falls | Low-level features remain, but task-specific geometry is fragile | Retrain the head or later blocks; test calibration and targeted augmentation |
| OI falls first in early layers | The shift affects preprocessing or backbone extraction | Change normalization/augmentation or fine-tune the backbone |
| OI declines before the headline metric moves | Representation margins may be eroding early | Expand the probe and use OI as an early warning |
| Accuracy falls while OI remains stable | Structure remains, but the decision rule or confidence may be failing | Refit the head, inspect calibration, and audit thresholds |
| Damage concentrates in a class or pair | Aggregate quality hides slice-specific risk | Add targeted data and class- or pair-specific gates |

Apply the pattern by freezing one checkpoint and representative probe, applying
deployment-relevant shifts to the same rows, extracting the same named layers, and
turning observed failure boundaries into regression tests. Comparisons are probe- and
layer-specific; absolute OI values from unrelated probes are not interchangeable.

### Paired overfitting monitor

The overfitting example replaces 40% of a stratified 1,000-row training subset with a
deterministic different label. Clean and noisy CNNs share initial weights, images,
mini-batch order, optimizer settings, and a 20-epoch schedule. A disjoint clean
2,000-row probe measures transferable geometry; deliberately relabeled rows measure
organization around incorrect targets. Clean validation cross-entropy, not OI, defines
the overfitting region.

From epochs 10 to 20, the noisy model's final-embedding OI on the corruption probe rises
from `0.039` to `0.313`, while the clean control remains near zero. Clean-validation
embedding OI rises from `0.669` to `0.718` for the control but stalls near `0.63` for
the noisy treatment. Early features remain similar; the second block diverges, and the
final embedding develops geometry around arbitrary targets.

| Monitoring pattern | Practical interpretation | Possible response |
| --- | --- | --- |
| Held-out loss worsens while clean-probe OI stays near the reference | The head or confidence may be overfitting | Early-stop, calibrate, or regularize/retrain the head |
| Late-layer clean-probe OI falls behind the reference | Task-specific geometry is degrading | Increase regularization or augmentation, freeze earlier layers, or restore a checkpoint |
| OI rises on a noise probe while clean-probe OI stalls | The representation is organizing around noisy targets | Audit labels, deduplicate, reweight the slice, or use noise-robust training |
| Early layers stay stable but the embedding diverges | Overfitting is localized near the head | Reuse the backbone and retrain later layers |

OI is not an overfitting detector by itself. It localizes representation changes after
a held-out metric or another external signal establishes overfitting.

### Shortcut learning with named target views

[`examples/colored_fashion_mnist_shortcut.py`](https://github.com/NiklasMelton/vertebrae/blob/develop/examples/colored_fashion_mnist_shortcut.py)
asks whether a CNN learns garment shape or a correlated color. Two compact CNNs receive
the same 2,000 garments, class targets, initialization, mini-batch order, optimizer,
and 12-epoch schedule. The control balances color within every class; the shortcut
treatment uses each class's canonical color with 85% probability.

A held-out 2,000-garment probe is rendered grayscale, canonical, independently colored,
and with a reversed mapping. The representation probe balances every class×color cell
and registers `intended_class` and `nuisance_color` views. Five paired seeds provide
the reported mean and standard deviation; OI stability repeats are disabled so the
uncertainty describes model-training variation.

| Model | Correlated ID | Balanced colors | Reversed colors | Color removed |
| --- | ---: | ---: | ---: | ---: |
| Independent-color control | 82.63% ± 1.87 | 82.76% ± 1.49 | 82.81% ± 1.56 | 82.73% ± 1.68 |
| Shortcut treatment | **91.42% ± 0.70** | 68.95% ± 2.43 | 47.27% ± 4.02 | 78.25% ± 1.77 |

The treatment gains 8.79 points in-distribution but trails by 13.81 points under
balanced colors and 35.54 under reversal. At epoch 12, named views localize the effect:

| Layer | Intended: control | Intended: shortcut | Color: control | Color: shortcut |
| --- | ---: | ---: | ---: | ---: |
| Conv block 1 | 0.307 ± 0.012 | 0.263 ± 0.015 | 0.031 ± 0.012 | 0.076 ± 0.014 |
| Conv block 2 | 0.477 ± 0.033 | 0.266 ± 0.030 | 0.009 ± 0.003 | 0.194 ± 0.056 |
| Final embedding | 0.675 ± 0.016 | 0.557 ± 0.022 | 0.000 ± 0.000 | 0.020 ± 0.010 |

The control organizes later layers around garment class while suppressing color. The
treatment retains more color structure, especially in block 2, and weaker garment
geometry at every layer. Low final-embedding color OI is not a contradiction: a
classifier can exploit a weak or class-conditional color direction without arranging
all colors into globally separated clusters.

| Behavioral diagnostic | Control | Shortcut | Difference |
| --- | ---: | ---: | ---: |
| Accuracy across all colors | 82.70% ± 1.64 | 68.28% ± 2.33 | −14.42 points |
| Correct under every color | 81.23% ± 1.76 | 37.03% ± 3.10 | −44.20 points |
| Prediction changes under recoloring | 3.42% ± 0.74 | 62.17% ± 3.22 | +58.75 points |
| Errors following the displayed color's class | 11.26% ± 0.15 | 28.49% ± 1.87 | +17.22 points |
| 10th-percentile class×color accuracy | 65.98% ± 5.95 | 6.75% ± 6.14 | −59.23 points |
| Mean of five weakest cells | 40.96% ± 16.24 | 0.16% ± 0.21 | −40.80 points |

To transfer the audit, name intended and nuisance targets, decorrelate them on a fixed
probe, compare with a controlled reference, monitor identical layers, intervene on the
nuisance independently, and report paired effects across seeds. Named views do not
remove confounding by themselves, and representation separability is not causal proof.

```bash
poetry install -E visuals
poetry run python examples/colored_fashion_mnist_shortcut.py --repeats 5
```

### Hierarchical Fashion-MNIST and Tiny Shakespeare

The Fashion-MNIST suite also evaluates the same representations against nested
department, garment-group, and exact-class views before and after training.

[`examples/tiny_shakespeare_transformer_visual_suite.py`](https://github.com/NiklasMelton/vertebrae/blob/develop/examples/tiny_shakespeare_transformer_visual_suite.py)
uses the first 90% of the checksum-pinned 1.1 MB corpus for training, the next 5% for
a class-balanced probe and naturally distributed validation, and leaves the final 5%
untouched. With the default 64-character context, the fixed probe contains 11,429 rows
across 59 characters: 51 contribute to macro OI, eight remain diagnostic-only below
the 50-occurrence threshold, and one singleton is omitted.

The fast profile observes token-plus-position, blocks 1/2, and normalized block 4; the
quality profile observes token-plus-position, blocks 2/4, and normalized block 6. In
the documented 5,000-step CPU run, validation cross-entropy falls from `4.1555` to
`1.6097` (perplexity `63.78` to `5.00`), top-1 accuracy rises from `0.8%` to `51.6%`,
and final-block macro OI rises from `0.173` to `0.351`. Token-plus-position changes
little, localizing learned geometry to the transformer blocks.

| Profile | Architecture | Context / batch | Default steps | Monitored states |
| --- | --- | ---: | ---: | --- |
| `fast` | 4 blocks, 4 heads, width 128, MLP 512, no dropout | 64 / 12 | 30,000 | token+position, blocks 1/2, final block 4 |
| `quality` | 6 blocks, 8 heads, width 256, MLP 1024, dropout 0.1 | 256 / 32 | 10,000 | token+position, blocks 2/4, final block 6 |

The quality profile has about 4.82 million parameters and samples 8,192 training
characters per step. Final outputs restore the lowest-validation-loss checkpoint.
`--device auto` smoke-tests CUDA/ROCm, Apple MPS, and Intel XPU before CPU; an explicit
backend fails instead of silently moving the run.

```bash
poetry install -E text-visuals
poetry run python examples/tiny_shakespeare_transformer_visual_suite.py --profile quality
```

Use `--profile fast --steps 5000` to reproduce the checked figures and `--no-download`
after the checksum-valid corpus has been cached. The script writes PNG/SVG figures, CSV
history, benchmark and compression JSON, and deterministic generated text.

The upstream [OverlapIndex](https://github.com/NiklasMelton/OverlapIndex) and
[Separatix](https://github.com/NiklasMelton/Separatix) repositories also have
their own visual examples that users may find informative.
