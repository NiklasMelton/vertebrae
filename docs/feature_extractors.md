# feature extractors

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

Every extractor implements:

```python
fit(X, y=None)
transform(X)
fit_transform(X, y=None)
recipe()
```

Some extractors can also emit multiple named embedding matrices from one model
pass. `Benchmark` and `Evaluator` score each named output as a separate result.

Native multi-output support is available for:

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

For Hugging Face backbones, pass explicit output specs:

```python
extractor = HFVisionExtractor(
    name="mnist_vit",
    model_id="farleyknight-org-username/vit-base-mnist",
    outputs=[
        {"name": "final_cls", "pooling": "cls"},
        {"name": "mid_cls", "pooling": "cls", "hidden_layer": 6},
    ],
)
```

For paired image-text models, `HFMultimodalExtractor` works with aligned
structured dataset inputs and explicit named branch or fused outputs:

```python
from vertebrae import BenchmarkDataset
from vertebrae.extractors import HFMultimodalExtractor

dataset = BenchmarkDataset.from_multimodal(
    inputs={"image": images, "caption": captions},
    labels=labels,
    modalities={"image": "image", "caption": "text"},
)

extractor = HFMultimodalExtractor(
    name="clip_like",
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
arrays or scipy sparse matrices. Sparse embeddings are stored as `.npz` artifacts and
converted to dense arrays only at the MiniBatchKMeans-backed OverlapIndex scoring
boundary, with `OverlapScoringConfig.max_dense_bytes` guarding memory use.

Optional model extractors require:

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

`TorchExtractor` is intended for users who already have a trained local PyTorch model
loaded in memory. They provide a `collate_fn` that converts raw inputs into model
inputs, and an `output_fn` when the model output needs to be projected to an
embedding matrix.

`ONNXExtractor` is intended for exported inference graphs. Users supply an optional
`input_fn` and `output_fn` when model inputs or outputs need reshaping, tokenization,
or selection from multi-input/multi-output sessions.

Text extractors validate that inputs are sequences of strings. Vision extractors accept
PIL images, NumPy image arrays, or image paths. Audio extractors accept waveform
arrays, audio paths, or structured dictionaries containing `array` / `path` and
`sampling_rate`. Video extractors accept video paths, predecoded frame arrays with
shape `(time, height, width, channels)`, or structured dictionaries containing
`frames` / `path`. Time-series extractors accept dense arrays with shape `(n, time)`
or `(n, time, channels)`, plus optional structured fields such as
`observed_mask` and `time_features`.

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

Hosted API
extractors are streaming-safe, but benchmark artifact reuse is only enabled when
the extractor sets `cache_embeddings=True`.

`HFMultimodalExtractor` accepts dict inputs keyed by declared field names. For
common image-text models it maps image fields to processor `images` and text
fields to processor `text` by default. Use `input_map` or `input_fn` for custom
processor shapes, and `output_fn` when model outputs need explicit projection
before named output validation.

Streaming-safe extractors, including Hugging Face backbones and precomputed embeddings,
can be embedded batch-by-batch through `EmbeddingConfig(batch_size=...)`. This is
intended for large raw data where only the embedding artifact should persist.

When output shape is not known ahead of time, streaming-safe extractors are probed on a
small first batch. The inferred embedding dimension and dtype are used with
`MemoryConfig` to estimate whether the full embedding artifact and dense scoring input
fit in memory before the full job runs.
