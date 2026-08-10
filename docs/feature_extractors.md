# feature extractors

## Choosing an extractor

`vertebrae` supports these extractor families:

- `PrecomputedExtractor`: uses embeddings supplied by the user.
- `SklearnExtractor`: fits or applies scikit-learn transformers and pipelines.
- `CallableExtractor`: wraps custom Python feature functions.
- `TorchExtractor`: wraps a locally loaded `torch.nn.Module` with user-supplied batch and output adapters.
- `TimmVisionExtractor`: lazy-loads timm vision backbones with explicit preprocessing and output selection.
- `TorchvisionVisionExtractor`: lazy-loads torchvision vision backbones with explicit weights/preprocessing handling.
- `ONNXExtractor`: wraps a local ONNX Runtime session with user-supplied input and output adapters.
- `SentenceTransformerExtractor`: lazy-loads sentence-transformers models.
- `HFTextExtractor`: lazy-loads Hugging Face text backbones with explicit pooling.
- `HFAudioExtractor`: lazy-loads Hugging Face audio backbones with explicit pooling.
- `HFMultimodalExtractor`: lazy-loads Hugging Face multi-modal backbones with
  explicit branch and fused output selection.
- `HFTimeSeriesExtractor`: lazy-loads Hugging Face time-series backbones with explicit pooling.
- `HFVideoExtractor`: lazy-loads Hugging Face video backbones with explicit pooling.
- `HFVisionExtractor`: lazy-loads Hugging Face vision backbones when optional
  dependencies are installed.
- `OpenCLIPExtractor`: lazy-loads OpenCLIP-style image/text backbones with explicit branch outputs.
- `SigLIPExtractor`: provides an ergonomic SigLIP-style image/text wrapper on top of the Hugging Face multi-modal path.
- `TFHubExtractor`: wraps a TensorFlow Hub module with explicit input and output adapters.
- `JAXFlaxExtractor`: wraps a JAX/Flax apply function or model object with explicit adapter hooks.
- `TreeLeafEmbeddingExtractor`: turns fitted XGBoost, LightGBM, or CatBoost ensembles into dense or sparse leaf embeddings.
- `GraphModelExtractor`: wraps graph-level PyG or DGL models with explicit batching and output adapters.
- `HostedEmbeddingExtractor`: wraps hosted embedding APIs behind explicit batch, retry, and cache-policy settings.

Dense segmentation workflows use `SpatialOutputSpec`, `SpatialLayout`, and
`SpatialEmbeddingOutput`. `CallableSpatialExtractor` and
`PrecomputedSpatialExtractor` cover explicit adapters; Torch and Keras support
`spatial_output_fn`, while `HFVisionExtractor` supports explicit
`spatial_outputs`. Spatial geometry must be declared rather than inferred from
ambiguous model outputs.

Structured unit workflows use `StructuredOutputSpec` and
`StructuredEmbeddingOutput`. `CallableStructuredExtractor` and
`PrecomputedStructuredExtractor` cover explicit adapters, while
`TorchExtractor`, `KerasExtractor`, `HFTextExtractor`, `HFAudioExtractor`,
`HFVisionExtractor`, `HFVideoExtractor`, `HFTimeSeriesExtractor`,
`HFMultimodalExtractor`, `ONNXExtractor`, `TimmVisionExtractor`,
`TorchvisionVisionExtractor`, `TFHubExtractor`, and `JAXFlaxExtractor` can
expose native `transform_structured(...)` outputs alongside ordinary pooled
embeddings.
Structured outputs are intended for raw token, frame, region, keypoint, or
other per-parent unit matrices that should be materialized and scored as unit
embeddings without first precomputing a flat embedding dataset by hand.
Each parent matrix may be a dense NumPy array or scipy sparse matrix. Sparse
matrices are passed to explicit aligners without densification and remain
sparse when rows are combined. Within one named output, feature dimension,
dtype, and dense-versus-sparse representation must remain stable across
parents and batches.

Today the extractor families with native structured-output coverage are:

- Explicit structured adapters: `CallableStructuredExtractor`,
  `PrecomputedStructuredExtractor`
- Local model wrappers: `TorchExtractor`, `KerasExtractor`, `ONNXExtractor`
- Hugging Face families: `HFTextExtractor`, `HFAudioExtractor`,
  `HFVisionExtractor`, `HFVideoExtractor`, `HFTimeSeriesExtractor`,
  `HFMultimodalExtractor`
- Vision backbone adapters: `TimmVisionExtractor`, `TorchvisionVisionExtractor`
- Other adapter-first wrappers: `TFHubExtractor`, `JAXFlaxExtractor`

Those structured outputs can now support token, frame, region, keypoint, depth,
and latent-slot materialization workflows as long as the model path can expose
one explicit 2D unit matrix per parent sample. `vertebrae` still treats them as
embedding-efficacy diagnostics, not task-native detection, OCR, ASR, pose,
depth, or generative evaluation engines.

## Common extractor protocol

Every extractor implements:

```python
fit(X, y=None)
transform(X)
fit_transform(X, y=None)
recipe()
```

## Multi-output and structured outputs

Some extractors can also emit multiple named embedding matrices from one model
pass. `Benchmark` and `Evaluator` score each named output as a separate result.
Exact output names remain visible in results and recipes. For persisted artifacts,
vertebrae derives a readable, collision-resistant segment from each name instead of
using the name as a path: `output-v1-<slug>--<sha256>`. The slug is a lowercase ASCII
description capped at 40 characters, while the full SHA-256 digest is computed from
the exact, unnormalized UTF-8 name. Names such as `a/b` and `a_b` therefore remain
independent even when their readable slugs match.

Native multi-output support is available for:

- `TorchExtractor`
- `KerasExtractor`
- `HFTextExtractor`
- `HFAudioExtractor`
- `HFMultimodalExtractor`
- `HFTimeSeriesExtractor`
- `HFVideoExtractor`
- `HFVisionExtractor`
- `MultiOutputExtractor`
- `TimmVisionExtractor`
- `TorchvisionVisionExtractor`
- `OpenCLIPExtractor`
- `TFHubExtractor`
- `JAXFlaxExtractor`

### Named Hugging Face outputs

For Hugging Face backbones, pass explicit output specs:

```python
extractor = HFVisionExtractor(
    name="mnist_vit",
    # This introductory remote name is intentionally unpinned; cache reuse is bypassed.
    model_id="farleyknight-org-username/vit-base-mnist",
    outputs=[
        {"name": "final_cls", "pooling": "cls"},
        {"name": "mid_cls", "pooling": "cls", "hidden_layer": 6},
    ],
)
```

Use `processor_kwargs` only for `AutoImageProcessor.from_pretrained(...)` options such
as `use_fast`. Use `preprocess_kwargs` for arguments applied to each processor call,
such as per-run resizing or padding overrides. Keeping these mappings separate avoids
silently forwarding construction-only options into image preprocessing.

### Named local-model outputs

Local Torch and Keras models use the same explicit ordinary `outputs` mapping. These
adapters do not discover internal layers or install hooks; the model or `output_fn`
must expose every declared representation:

```python
extractor = TorchExtractor(
    name="local_encoder",
    model=model,
    collate_fn=collate_fn,
    output_fn=lambda raw: {
        "middle": raw["middle"],
        "final": raw["final"],
    },
    outputs=[
        {
            "name": "middle",
            "selector": "middle",
            "hidden_layer": 2,
            "pooling": "mean",
        },
        {
            "name": "final",
            "selector": "final",
            "hidden_layer": 4,
            "pooling": "cls",
        },
    ],
)
```

The same constructor shape applies to `KerasExtractor`. Each batch invokes the model
once, applies `output_fn` once, and materializes all named outputs. Selectors are
dotted paths and may address sequence positions with numeric components. `flatten`
defaults to `True` for explicit declarations. `hidden_layer`, `pooling`, `flatten`,
and `metadata` are explicit provenance and become part of the extractor recipe and
cache identity.

When `outputs` is omitted, the legacy single-output path remains strict: the model or
`output_fn` must return a 2D numeric matrix and higher-rank arrays are rejected rather
than flattened. Declare an output explicitly to opt into flattening. When multiple
outputs omit selectors, the model or `output_fn` must return a mapping whose keys
exactly match the declared names. In a mixed configuration, every selector-free
output must still be present by name in the mapping.

`transform()` remains the ergonomic single-output path. With multiple declarations it
raises with guidance to use `Benchmark`, `Evaluator`, or `transform_many()`.

### Multimodal outputs

For paired image-text models, `HFMultimodalExtractor` works with aligned
structured dataset inputs and explicit named branch or fused outputs:

```python
from vertebrae import BenchmarkDataset, DatasetIdentity
from vertebrae.extractors import HFMultimodalExtractor

dataset = BenchmarkDataset.from_multimodal(
    inputs={"image": images, "caption": captions},
    labels=labels,
    modalities={"image": "image", "caption": "text"},
    identity=DatasetIdentity.declared("example-dataset", "1"),
)

extractor = HFMultimodalExtractor(
    name="clip_like",
    # This introductory remote name is intentionally unpinned; cache reuse is bypassed.
    model_id="openai/clip-vit-base-patch32",
    input_modalities={"image": "image", "caption": "text"},
    outputs=[
        {"name": "image_branch", "source": "image", "model_output": "image_embeds"},
        {"name": "text_branch", "source": "text", "model_output": "text_embeds"},
        {"name": "fused", "source": "fused", "model_output": "pooler_output"},
    ],
)
```

Extractor recipes are serialized into result metadata and cache keys. Scoring consumes
numeric embeddings and labels, not live model objects. Embeddings may be dense NumPy
arrays, scipy sparse matrices, or scipy sparse arrays. Sparse embeddings are
normalized to CSR, stored as `.npz` artifacts, and remain sparse through
MiniBatchKMeans-backed overlap scoring and when passed into Separatix. Separatix may
perform bounded internal densification for diagnostics that inherently require it.

Cache identity schema v2 hashes the complete typed extractor recipe. Callable and
live-model extractors accept an explicit `cache_identity`. Without one, cache reuse is
allowed only when Vertebrae can prove the identity is stable: import-resolvable
callables include their implementation digest, model identifiers that are actual local
paths include a content digest, and remote model names require a 40--64 hexadecimal
immutable revision. A separately declared checkpoint path is still digested as
provenance, but cannot prove the state of an already-loaded live model. Closures, fitted
in-memory models, opaque referenced globals, and unpinned remote models still run, but
bypass reusable caches and record `cache_status="bypassed_unsafe_identity"`. Importable
callable identities include referenced modules, helper callables, and exact global
configuration values. Optional extractor recipes also record installed backend
distribution versions, so an inference-runtime upgrade invalidates reuse. This policy
also applies to every compressed or otherwise derived artifact.

The complete extras mapping, including Keras, TensorFlow, visualization, distributed,
and cloud-store bundles, lives in the
[installation guide](https://github.com/NiklasMelton/vertebrae/blob/develop/docs/installation.md).
Common optional model extras are:

```bash
poetry install -E torch
poetry install -E hf
poetry install -E timm
poetry install -E torchvision
poetry install -E openclip
poetry install -E audio
poetry install -E timeseries
poetry install -E video
poetry install -E onnx
poetry install -E tensorflow-hub
poetry install -E jax
poetry install -E trees
poetry install -E graph
```

## Local and scikit-learn adapters

`SklearnExtractor` wraps fitted or fit-capable scikit-learn transformers and pipelines.
`CallableExtractor` wraps a user function when an existing adapter is not appropriate.
Both retain their serializable recipe and cache identity alongside generated embeddings.

`TorchExtractor` is intended for users who already have a trained local PyTorch model
loaded in memory. They provide a `collate_fn` that converts raw inputs into model
inputs, and an `output_fn` when the model output needs to be projected to an
embedding matrix. By default extraction temporarily switches the module to evaluation
mode, runs under `torch.inference_mode()`, and restores the module's prior training
state afterward. Set `inference_mode=False` only when an adapter genuinely requires
autograd or training-mode behavior.

`KerasExtractor` follows the same adapter-first shape. The caller controls batch
coercion and output selection rather than relying on model-specific conventions:

```python
import numpy as np

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.extractors import KerasExtractor


def collate_fn(batch):
    return np.asarray(batch, dtype=np.float32)


dataset = BenchmarkDataset.from_arrays(
    features,
    labels,
    modality="tabular",
    identity=DatasetIdentity.declared("example-dataset", "1"),
)
extractor = KerasExtractor(
    name="local_keras",
    model=model,
    collate_fn=collate_fn,
    call_method="call",
    checkpoint_paths=["/path/to/model.keras"],
    cache_identity="local-keras-model-v1",
    recipe_data={"checkpoint": "/path/to/model.keras"},
)

result = Evaluator(dataset=dataset, extractor=extractor).run()
```

`checkpoint_paths` contributes content-digested provenance and profiling evidence; a
path copied only into `recipe_data` is descriptive. Because the adapter receives an
already-loaded live object, the path cannot prove which weights are in memory, so
reusable caching still requires a maintained `cache_identity`. Notebook-local, nested,
or otherwise nonportable adapter functions need an explicit identity for the same
reason.

`ONNXExtractor` is intended for exported inference graphs. Users supply an optional
`input_fn` and `output_fn` when model inputs or outputs need reshaping, tokenization,
or selection from multi-input/multi-output sessions.

## Hugging Face adapters

Hugging Face support is explicit by modality. All remote examples below use unpinned
introductory model names or placeholders, so they disable reusable caching. Production
runs should use a full immutable model revision or a maintained `cache_identity`.

| adapter | dataset input | common output controls |
| --- | --- | --- |
| `HFTextExtractor` | strings | pooling, hidden layer, named outputs, structured tokens |
| `HFVisionExtractor` | PIL/NumPy images or paths | image mode, pooling, hidden layer, spatial/structured outputs |
| `HFAudioExtractor` | waveforms, paths, or sampling-rate dictionaries | pooling, feature masks, structured frames |
| `HFMultimodalExtractor` | aligned named modality fields | branch/fused source, model output, selector, pooling |
| `HFTimeSeriesExtractor` | dense series and optional observed/time features | pooling, hidden layer, structured cells |
| `HFVideoExtractor` | paths or predecoded clips | frame windows, pooling, hidden layer, structured frames |

The vision and multimodal examples above demonstrate named outputs. The remaining
modality-specific constructor shapes are:

```python
from vertebrae import BenchmarkDataset, CacheConfig, DatasetIdentity, Evaluator
from vertebrae.extractors import HFAudioExtractor, HFTextExtractor

text_dataset = BenchmarkDataset.from_arrays(
    texts,
    labels,
    modality="text",
    identity=DatasetIdentity.declared("text-example", "1"),
)
text_extractor = HFTextExtractor(
    name="bert_text",
    model_id="bert-base-uncased",
    pooling="mean",
)

audio_dataset = BenchmarkDataset.from_audio_arrays(
    audio=waveforms,
    labels=labels,
    sampling_rate=16_000,
    identity=DatasetIdentity.declared("audio-example", "1"),
)
audio_extractor = HFAudioExtractor(
    name="wav2vec2_base",
    model_id="facebook/wav2vec2-base",
    pooling="mean",
)

text_result = Evaluator(
    dataset=text_dataset,
    extractor=text_extractor,
    cache_config=CacheConfig(enabled=False),
).run()
audio_result = Evaluator(
    dataset=audio_dataset,
    extractor=audio_extractor,
    cache_config=CacheConfig(enabled=False),
).run()
```

```python
from vertebrae import BenchmarkDataset, DatasetIdentity
from vertebrae.extractors import HFTimeSeriesExtractor, HFVideoExtractor

time_series_dataset = BenchmarkDataset.from_time_series(
    series=series,
    labels=labels,
    identity=DatasetIdentity.declared("series-example", "1"),
)
time_series_extractor = HFTimeSeriesExtractor(
    name="patchtst",
    model_id="some-local-or-hf-timeseries-model",
    pooling="mean",
)

video_dataset = BenchmarkDataset.from_video_arrays(
    frames=clips,
    labels=labels,
    frame_rate=24.0,
    identity=DatasetIdentity.declared("video-example", "1"),
)
video_extractor = HFVideoExtractor(
    name="videomae_base",
    model_id="MCG-NJU/videomae-base",
    pooling="mean",
    num_frames=16,
)
```

Text extractors validate that inputs are sequences of strings. Vision extractors accept
PIL images, NumPy image arrays, or image paths. Audio extractors accept waveform
arrays, audio paths, or structured dictionaries containing `array` / `path` and
`sampling_rate`. Padded Hugging Face audio pooling derives a frame-level feature mask
through the model helper; models without that helper require an explicit
`feature_mask_fn` when input and output time axes differ. Video extractors accept video
paths, predecoded frame arrays with
shape `(time, height, width, channels)`, or structured dictionaries containing
`frames` / `path`; configured time windows are also applied to predecoded frames using
their validated frame rate. Time-series extractors accept dense arrays with shape `(n, time)`
or `(n, time, channels)`, plus optional structured fields such as
`observed_mask` and `time_features`.

## Other optional extractor families

The corrected optional-wrapper details are intentionally explicit:

- Hugging Face text `last_token` pooling selects the highest position marked attended,
  so left and right padding behave equivalently. Structured token output removes only
  positions marked by the tokenizer's `special_tokens_mask`; it does not assume a fixed
  number of leading or trailing special tokens.
- Hugging Face multimodal ordinary outputs honor each declared dotted `selector` after
  resolving `model_output`, then apply the output's hidden-layer and pooling choices.
- Shared vision coercion accepts `alpha_mode="drop"`, `"black_background"`, or
  `"white_background"`. Compositing is applied consistently before RGB/grayscale mode
  conversion. Hugging Face vision treats `hidden_layer=0` as an actual layer selection,
  distinct from `None` (the model's final/default output).
- `TorchvisionVisionExtractor(weights=None)` converts HWC images to stacked CHW floating
  tensors; integer images are scaled to `[0, 1]`. Supply `preprocess_fn` for any other
  normalization or layout.
- `TreeLeafEmbeddingExtractor` flattens every non-sample leaf axis before dense or
  one-hot encoding. `TFHubExtractor` honors `batch_size` for ordinary and structured
  outputs rather than invoking the Hub module on the full input at once.
- `HostedEmbeddingExtractor` validates that every response batch has exactly the
  requested row count and that feature width remains constant across batches.

Public batch sizes, retry counts/backoff values, names, and output specifications are
validated at construction. Booleans are not accepted as integers, counts must be in
range, numeric backoff values must be finite, and output names must be nonblank and
unique.

Graph extractors operate at graph level by default: each sample corresponds to one
graph object and each output row corresponds to one graph embedding. Use
`BenchmarkDataset.from_graphs(...)` for graph-level workflows.

For transfer-learning diagnostics over node or edge embeddings, keep the same
embedding-efficacy contract: materialize one embedding row per labeled node or
edge and evaluate it with `BenchmarkDataset.from_node_embeddings(...)` or
`BenchmarkDataset.from_edge_embeddings(...)`. `GraphModelExtractor` accepts
`output_level="graph"`, `"node"`, or `"edge"` as recipe metadata for wrappers whose
model/output adapter already returns that level, but it does not add graph task
metrics or ranking protocols.

Hosted API extractors are streaming-safe, but reusable benchmark artifacts require
both `cache_embeddings=True` and an explicit stable `cache_identity`. Either omission
keeps the network call visible to the current evaluation without opting its responses
into persistent reuse.

Text-aligned extractors that expose `encode_retrieval(...)` can also participate in
the explicit `ZeroShotBenchmark` protocol. Zero-shot evaluation requires separately
declared sample and text branches in one shared embedding space; it does not infer
zero-shot capability from an arbitrary pooled extractor output.

`HFMultimodalExtractor` accepts dict inputs keyed by declared field names. For
common image-text models it maps image fields to processor `images` and text
fields to processor `text` by default. Use `input_map` or `input_fn` for custom
processor shapes, and `output_fn` when model outputs need explicit projection
before named output validation.

## Streaming and memory planning

For structured outputs on adapter-style extractors, reuse the same forward path
and expose per-parent 2D unit matrices with explicit specs:

```python
extractor = ONNXExtractor(
    name="layout_tokens",
    model_path="layout.onnx",
    input_fn=prepare_inputs,
    output_fn=lambda outputs: {"regions": outputs[0]},
    structured_outputs=[
        {"name": "regions", "unit_type": "region"},
    ],
)
```

Streaming-safe extractors, including Hugging Face backbones and precomputed embeddings,
can be embedded batch-by-batch through `EmbeddingConfig(batch_size=...)`. This is
intended for large raw data where only the embedding artifact should persist.

`streaming_safe=False` is a hard contract: Vertebrae performs one transform on the
complete selected input. For streaming-safe extraction, every batch must return the
same exact unique output names, per-output row count, feature width, dtype,
dense-versus-sparse form, sparse format, recipe, metadata, and complete parent
coverage. Structured and spatial extractors must emit every declared output on every
batch, with no extras or duplicates. Assembly uses explicit indexed writes and rejects
shape mismatches instead of relying on NumPy broadcasting.

When output shape is not known ahead of time, streaming-safe extractors are probed on a
small first batch. The inferred embedding dimension and dtype are used with
`MemoryConfig` to estimate whether the full embedding artifact and its actual dense or
sparse scoring representation fit in memory before the full job runs. Result metadata
also retains the hypothetical dense footprint for capacity planning.

## Resource profiling adapters

`ResourceProfilingConfig(enabled=True)` observes the actual calls made by local
`Benchmark` and `Evaluator` runs. Portable profiling covers call latency, throughput,
process RSS, and logical embedding bytes. Optional extractor-owned
`ResourceProfileAdapter` hooks add device synchronization, allocator peaks, model
parameter bytes, and explicitly declared checkpoint artifacts.

`RetrievalBenchmark` and `ZeroShotBenchmark` use the same `EmbeddingConfig` and
`ResourceProfilingConfig` contracts. Frozen branch encoders are called in deterministic
batches (128 by default), preserve endpoint row order, and combine sparse batches without
densifying. Standard retrieval extractors use whole-endpoint calls unless they declare
streaming safety. Each endpoint has its own profiler, so latency, memory, cache status, and
storage evidence are never blended.

Native adapters cover local Torch/Keras/ONNX models plus the owned Torch, TensorFlow,
and JAX families: Hugging Face text, vision, audio, time-series, video, and multimodal
models; sentence-transformers; timm; torchvision; OpenCLIP/SigLIP; graph models;
TensorFlow Hub; and JAX/Flax. Loaded model placement takes precedence over profiling
hints, and adapters do not load a model merely to inspect it.

Torch, Keras, and high-level model wrappers accept `checkpoint_paths=` for explicit
files. ONNX always counts `model_path` and accepts `external_data_paths=`. A directory
is counted only when a custom adapter returns
`DeploymentArtifact(path, recursive=True)`; this prevents accidental measurement of
an unrelated model cache. Vertebrae never infers checkpoint locations from
`recipe_data`, model names, Hub handles, or external caches.

Checkpoint declarations contribute content digests to the recipe, but do not by
themselves make an already-loaded Torch, Keras, graph, or JAX model cache-safe because
Vertebrae cannot verify that the live state matches the file. Those adapters require an
explicit maintained `cache_identity`. Profiling-device hints remain profiling evidence
rather than extraction semantics and do not change reusable embedding cache keys.

Custom adapters implement the typed `ResourceProfileAdapter` contract. Subclass
`BaseResourceProfileAdapter` and override only supported hooks, returning typed
payloads such as `ResourceAdapterMetadata`, `DeviceMemoryMeasurement`,
`ModelFootprintMeasurement`, and `DeploymentArtifact`. Hook failures are retained as
profile warnings and do not abort quality scoring.

```python
from vertebrae import (
    BaseResourceProfileAdapter,
    DeploymentArtifact,
    ModelFootprintMeasurement,
)


class MyModelResources(BaseResourceProfileAdapter):
    def model_footprint(self):
        return ModelFootprintMeasurement(
            status="measured",
            parameter_count=1_000_000,
            parameter_bytes=4_000_000,
        )

    def deployment_artifacts(self):
        return (DeploymentArtifact("weights/model.bin"),)
```

Keras and TensorFlow Hub accept `profiling_device=` when allocator measurement would
otherwise be ambiguous; this hint does not move the model. Keras 3 selects hooks for
its active TensorFlow, Torch, or JAX backend. TensorFlow and single-device CUDA Torch
runs expose resettable allocator peaks. JAX supplies native execution barriers and
device/parameter metadata, but its benchmark-scoped peak is unavailable because no
portable resettable allocator window exists. Stored weight dtypes are reported
separately from execution/autocast precision.

Torch models spanning multiple devices synchronize every active accelerator for
latency correctness, but allocator memory is marked unavailable rather than collapsing
per-device peaks into a misleading total. Model parameter totals include frozen
parameters; trainable parameter counts and bytes are reported separately.

For unsupported backends, portable measurements remain available and framework-only
fields are marked unavailable. Multi-output calls share one inference profile because
the underlying forward pass is shared, while every output retains its own embedding
storage footprint.
