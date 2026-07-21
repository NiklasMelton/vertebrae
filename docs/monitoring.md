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
[`examples/representation_monitoring.py`](../examples/representation_monitoring.py)
workflow for a complete local Torch training loop with two named representations,
caller-managed checkpoint restoration via `--resume`, strict JSONL resume, and a
final epoch-by-layer pivot.

For a visual real-data workflow,
[`examples/fashion_mnist_visual_suite.py`](../examples/fashion_mnist_visual_suite.py)
trains a compact two-block PyTorch CNN on Fashion-MNIST, monitors two pooled
convolutional representations and its 128-dimensional embedding from
initialization through three epochs at optimizer-step cadence, evaluates only against
a stratified held-out validation set, disables Separatix and stability repeats, and
renders the network architecture beside the layer-wise OverlapIndex trajectories.
