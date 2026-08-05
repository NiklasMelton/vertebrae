# installation

Install the core package from PyPI:

```bash
pip install vertebrae
```

For local development, install the Poetry development group:

```bash
poetry install --with dev
```

Optional dependencies are split by model family, modality, execution backend, and
artifact store so a core installation stays lightweight. The commands below preserve
the complete install mapping used throughout the examples and extractor guide.

## Hugging Face and modality extras

The general Hugging Face extra also installs sentence-transformers support:

```bash
pip install "vertebrae[hf]"
```

Modality-specific Hugging Face dependencies can be installed independently:

```bash
pip install "vertebrae[audio]"
pip install "vertebrae[timeseries]"
pip install "vertebrae[video]"
```

See [feature extractors](https://github.com/NiklasMelton/vertebrae/blob/develop/docs/feature_extractors.md#hugging-face-adapters) for the text,
vision, audio, multimodal, time-series, and video adapter contracts.

## Local model and optional extractor families

Local PyTorch and vision-family wrappers:

```bash
pip install "vertebrae[torch]"
pip install "vertebrae[timm]"
pip install "vertebrae[torchvision]"
pip install "vertebrae[openclip]"
```

Local Keras, TensorFlow, and TensorFlow Hub wrappers:

```bash
pip install "vertebrae[keras]"
pip install "vertebrae[tensorflow]"
pip install "vertebrae[tensorflow-hub]"
```

ONNX Runtime support:

```bash
pip install "vertebrae[onnx]"
```

JAX/Flax, tree-ensemble, and graph-model support:

```bash
pip install "vertebrae[jax]"
pip install "vertebrae[trees]"
pip install "vertebrae[graph]"
```

All heavyweight integrations are imported lazily. Installing an extra makes the
corresponding adapter available; it does not download model weights or call a hosted
service.

## Visual example suites

The Fashion-MNIST visual suite, including the monitoring, compression, corruption,
overfitting, and shortcut-learning examples, uses:

```bash
pip install "vertebrae[visuals]"
```

The Tiny Shakespeare transformer visual suite uses:

```bash
pip install "vertebrae[text-visuals]"
```

The example scripts document which datasets or checkpoints are downloaded on first
use. Introductory remote model names are intentionally unpinned and disable reusable
embedding caching unless the caller supplies an immutable revision or maintained
`cache_identity`.

## Distributed execution

Install one backend or the combined distributed bundle:

```bash
pip install "vertebrae[ray]"
pip install "vertebrae[dask]"
pip install "vertebrae[distributed]"
```

Ray and Dask execute the same artifact-backed jobs as the local backend. Cluster
workers must share a filesystem or use a supported cloud artifact store.

## Cloud artifact stores

Install an individual provider or both providers:

```bash
pip install "vertebrae[s3]"
pip install "vertebrae[gcs]"
pip install "vertebrae[cloud]"
```

Provider credentials are never bundled with vertebrae. Workers must be able to
authenticate independently for the selected `s3://` or `gs://` location. See
[distributed readiness](https://github.com/NiklasMelton/vertebrae/blob/develop/docs/distributed_readiness.md) for backend and artifact-store
configuration.
