# zero-shot semantic alignment

`ZeroShotBenchmark` evaluates a frozen text-aligned backbone by comparing each
sample embedding with fixed text prompt prototypes. It is training-free: no classifier,
prompt search, calibration, threshold fitting, or model head is learned.

This is a capability-specific protocol, not a universal representation-quality score.
It answers whether the model's shared embedding space can address this dataset's
declared classes with the supplied language. Every result also reports OverlapIndex on
the same sample embeddings as separate contextual evidence; the scores are never combined.

## Fixed prompts and class prototypes

Prompts are explicit protocol input. Use fully rendered strings when classes need
domain-specific language, or explicitly provide templates:

```python
from vertebrae import BenchmarkDataset, ZeroShotBenchmark, ZeroShotDataset, DatasetIdentity

dataset = BenchmarkDataset.from_arrays(images, labels, modality="image", identity=DatasetIdentity.declared("example-dataset", "1"))
protocol = ZeroShotDataset.from_templates(
    dataset,
    templates=("a photo of {label}", "a close-up of {label}"),
)
result = ZeroShotBenchmark(
    protocol,
    [clip_extractor],
    sample_branch="image_branch",
    text_branch="text_branch",
).run()
```

The source dataset must have single-label targets. Every observed class requires one
or more non-empty prompts. Prompt order is preserved in the protocol fingerprint;
duplicate prompts are rejected. Template strings are parsed with `string.Formatter`
and may reference only the `label` field; positional, conversion, format-spec, and
unknown fields are rejected. Template IDs must be nonempty and the protocol must have
exact class-by-template coverage before scoring. For a prompt ensemble, vertebrae L2-normalizes each
prompt embedding, averages all prompts for a class with equal weight, then normalizes
the class prototype. It never chooses the best prompt after observing labels.

The protocol fingerprint hashes the complete ordered prompt declaration. Changing any
prompt, template ID, label order, or protocol metadata produces a distinct prompt
artifact and cache identity, including changes beyond a fixed list length.

Class labels retain their semantic values in the in-memory protocol. Strings and
numbers work directly, as do UUIDs, enums, dates/times, `Decimal`, `Fraction`, and
dataclass values. Arbitrary custom objects must be converted to a stable value first;
vertebrae rejects them while constructing the protocol instead of failing later during
artifact planning.

Artifact protocols use the versioned `vertebrae.semantic-label/v1` encoding. Ordinary
string labels remain readable, while non-string labels receive collision-resistant keys
and an ordered catalog containing their exact typed identity and display value. This
keeps values such as integer `1` and string `"1"`, or a UUID and its textual form,
distinct through distributed scoring and report reconstruction. User-defined enum and
dataclass classes do not need to be importable on scoring workers.

The complete protocol recipe is strict JSON and includes sample-label order, typed
sample IDs, prompts, template IDs, metadata, and the label catalog. Artifact scoring
recomputes its fingerprint and checks duplicated manifest fields against the recipe;
corrupt or stale protocol payloads are rejected before embeddings are scored. Missing
or unknown label-encoding schemas are rejected.

`sample_branch` and `text_branch` are explicit because only extractors that declare
independently encodable, shared-space branches can support zero-shot evaluation.
OpenCLIP, SigLIP, compatible Hugging Face multimodal extractors, and custom
`CallableRetrievalExtractor` adapters are appropriate. Ordinary pooled backbones are
not implicitly treated as zero-shot-capable.

When candidates use different endpoint names, declare the pair beside each extractor
instead of forcing a global convention:

```python
from vertebrae import ZeroShotCandidate

result = ZeroShotBenchmark(
    protocol,
    [
        ZeroShotCandidate(openclip, "image_branch", "text_branch"),
        ZeroShotCandidate(custom_adapter, "query", "gallery"),
    ],
).run()
```

The global-branch form remains supported. OpenCLIP's ordinary single-output
default remains image-only, but its native `text_branch` is always independently
available to retrieval and zero-shot evaluation.

## Metrics and interpretation

The default rank is Top-1 `accuracy`; `macro_f1` and `balanced_accuracy` are
first-class configurable primary metrics. Reports always include all three, valid
Top-K accuracy values, per-class precision/recall/F1, a confusion matrix, prompt
coherence, correct-class margins, and deterministic tie warnings.

`ZeroShotConfig.similarity="squared_l2"` ranks prototypes by negative squared
Euclidean distance, so larger scores correspond to closer prototypes just as larger
cosine and dot-product scores correspond to better matches.

Cosine scoring requires nonzero sample rows and nonzero prompt/prototype rows. Sample
zero vectors remain valid for dot-product and squared-L2 scoring; prompt groups must
always be nonempty and produce a nonzero prototype.

| zero-shot | overlap | interpretation |
| --- | --- | --- |
| strong | strong | Text-addressable and well-structured frozen representation. |
| strong | weak | Native prompt alignment works, but do not infer broad transfer quality. |
| weak | strong | Promising transfer structure with weak text/prompt alignment. |
| weak | weak | Weak fit for this declared dataset and fixed protocol. |

Poor zero-shot performance can reflect taxonomy wording, domain terminology, prompt
coverage, or a weak text branch. It does not prove that sample embeddings are poor
features for a separately trained downstream model.

## Flagship experiment: zero-shot versus transfer structure

[`examples/zero_shot_transfer_structure.py`](../examples/zero_shot_transfer_structure.py)
turns the two-axis interpretation into a reproducible real-image experiment. It uses
a deterministic balanced CIFAR-10 test slice and OpenCLIP, encodes the image endpoint
once, and scores OverlapIndex once. It then scores three fully declared text protocols
against that same in-memory image embedding matrix: bare class names, a photo template,
and a CIFAR-10-context template. No prompt selection is performed after labels are
scored.

```bash
poetry install -E openclip -E visuals
poetry run python examples/zero_shot_transfer_structure.py
```

The example writes `zero_shot_transfer_structure.png` and an auditable CSV of every
plotted point. The first panel states the fixed macro sample OverlapIndex and compares
zero-shot accuracy with horizontal prompt-set bars, including the full accuracy range
caused by wording alone. The second panel sorts classes by per-class OverlapIndex and
connects each class's zero-shot F1 values; its two-line row label reports the unchanged
OverlapIndex while an in-bounds annotation shows the prompt-driven F1 range. This
avoids implying that the experiment expects a correlation between overlap and F1 while
keeping both metrics visible. It makes the practical next step legible:

- high overlap with prompt-sensitive, weak zero-shot scores suggests improving class
  descriptions or training a supervised head;
- high overlap with consistently weak prompts suggests that the image features may
  still transfer well even though this text branch is a poor fit;
- low overlap across prompt sets is evidence to investigate another backbone or data
  representation before spending effort on prompt wording.

The CIFAR-10 subset size, dataset cache directory, OpenCLIP model/checkpoint, and
device are configurable through the environment variables documented at the top of
the script. It downloads CIFAR-10 and the checkpoint on first use.

## Memory-bounded exact scoring

`ZeroShotConfig.sample_batch_size` controls how many samples are compared with the
class prototypes at once; its default is `128`. Scoring remains exact and preserves
declared-class-order tie breaking, but it never needs to allocate the complete
sample-by-class score matrix.

`ZeroShotConfig.max_dense_bytes` limits the estimated float64 scorer working set,
including dense sample and prompt endpoints, prototypes, result accumulators, and one
configured score block. If that block does not fit, lower `sample_batch_size`, compress
the embeddings, or raise the explicit limit. Dense and sparse inputs use the same
admission check before conversion or scoring.

`ZeroShotBenchmark(memory_config=...)` separately bounds local sample and prompt
embedding materialization. Streaming batches spill to temporary local staging when
allowed; otherwise the run fails as soon as the next batch would exceed the resident
budget instead of accumulating the complete endpoint first.

## Compression and artifact workflows

Compression variants are supported. Vertebrae fits the label-free compressor on sample
embeddings and applies the same fitted transform to prompt embeddings before building
prototypes. Sample embedding cache keys use the source dataset identity, so changing
only prompts can reuse sample embeddings when the source dataset and extractor have a
stable cache identity and raw caching is enabled. An unsafe or explicitly disabled raw
cache also makes every paired compressed variant ineligible for reuse.

For `CallableRetrievalExtractor`, top-level importable callables are recorded in its
recipe and cache key. Lambdas or nested callables must supply `cache_identity` to use
the local cache or canonical artifact plans; otherwise local caching is skipped with a
warning rather than guessing an unsafe function identity. Functions defined in a
script's `__main__` module are likewise nonportable and need `cache_identity`.

Artifact-backed workflows mirror retrieval: use `plan-zero-shot`,
`embed-zero-shot-shard`, `merge-zero-shot-embeddings`, and
`write-zero-shot-protocol`; optionally apply `compress-zero-shot`; then call
`score-zero-shot` with the merged sample key, prompt key, and protocol key.
Each endpoint's planned shard count is capped at its row count, so execute
`embed-zero-shot-shard --plan-json PLAN --side ... --shard-index ...` from the plan
rather than assuming samples and prompts have the same count. Prompt text, ordered
labels, and template IDs are preserved in JSON and artifact metadata, while compact
Markdown shows prompt counts only.

Use `zero-shot-from-artifacts --score-key SCORE ... --json-output REPORT.json
--markdown-output REPORT.md` to reconstruct a ranked report from one or more score
artifacts sharing the same protocol. The report retains zero-shot alignment and
OverlapIndex as separate metrics.

Sample and prompt endpoints must be either both raw or produced by the same paired
compression artifact. Vertebrae records the fitted compression-pair identity and
rejects mixed endpoints, even when their dimensions match. Score artifact keys include
the zero-shot and overlap configurations, so different similarity or ranking settings
do not overwrite one another; reports combine only matching evaluation settings.

When compression is skipped because a requested dimension is not smaller than the
embedding dimension, result names retain that requested dimension so variants remain
distinct while metadata reports the actual output dimension.
