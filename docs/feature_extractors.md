# feature extractors

`vertebrae` supports these extractor families:

- `PrecomputedExtractor`: uses embeddings supplied by the user.
- `SklearnExtractor`: fits or applies scikit-learn transformers and pipelines.
- `CallableExtractor`: wraps custom Python feature functions.
- `TorchExtractor`: wraps a locally loaded `torch.nn.Module` with user-supplied batch and output adapters.
- `ONNXExtractor`: wraps a local ONNX Runtime session with user-supplied input and output adapters.
- `SentenceTransformerExtractor`: lazy-loads sentence-transformers models.
- `HFTextExtractor`: lazy-loads Hugging Face text backbones with explicit pooling.
- `HFAudioExtractor`: lazy-loads Hugging Face audio backbones with explicit pooling.
- `HFTimeSeriesExtractor`: lazy-loads Hugging Face time-series backbones with explicit pooling.
- `HFVideoExtractor`: lazy-loads Hugging Face video backbones with explicit pooling.
- `HFVisionExtractor`: lazy-loads Hugging Face vision backbones when optional
  dependencies are installed.

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
- `HFTimeSeriesExtractor`
- `HFVideoExtractor`
- `HFVisionExtractor`
- `MultiOutputExtractor`

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

Extractor recipes are serialized into result metadata and cache keys. Scoring consumes
numeric embeddings and labels, not live model objects. Embeddings may be dense NumPy
arrays or scipy sparse matrices. Sparse embeddings are stored as `.npz` artifacts and
converted to dense arrays only at the MiniBatchKMeans-backed OverlapIndex scoring
boundary, with `OverlapScoringConfig.max_dense_bytes` guarding memory use.

Optional model extractors require:

```bash
poetry install -E torch
poetry install -E hf
poetry install -E audio
poetry install -E timeseries
poetry install -E video
poetry install -E onnx
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

Streaming-safe extractors, including Hugging Face backbones and precomputed embeddings,
can be embedded batch-by-batch through `EmbeddingConfig(batch_size=...)`. This is
intended for large raw data where only the embedding artifact should persist.

When output shape is not known ahead of time, streaming-safe extractors are probed on a
small first batch. The inferred embedding dimension and dtype are used with
`MemoryConfig` to estimate whether the full embedding artifact and dense scoring input
fit in memory before the full job runs.
