# embedding compression

`vertebrae` can apply an optional embedding-compression step after feature
extraction and before OverlapIndex scoring, stability analysis, Separatix
diagnostics.
This is useful when you want to:

- compare raw embeddings against compressed variants,
- test whether a lower-dimensional representation preserves class separation,
- evaluate sparse text embeddings with `TruncatedSVD`,
- study matryoshka-style prefix truncation,
- measure the effect of lossy precision reduction such as `float16` or `int8`.

Compression runs on the embedding artifact, not inside the extractor itself. The
raw embedding cache stays reusable, and each compression recipe gets its own
derived artifact key and report metadata.

## Configuration

Use `EmbeddingCompressionConfig` with `Evaluator` or `Benchmark`:

```python
from vertebrae import BenchmarkDataset, EmbeddingCompressionConfig, Evaluator
from vertebrae.config import CacheConfig, StabilityConfig
from vertebrae.extractors import PrecomputedExtractor

dataset = BenchmarkDataset.from_embeddings(embeddings=Z, labels=y)

compression = EmbeddingCompressionConfig(
    enabled=True,
    method="pca",
    n_components=128,
)

result = Evaluator(
    dataset=dataset,
    extractor=PrecomputedExtractor(name="baseline"),
    compression_config=compression,
    stability_config=StabilityConfig(enabled=False),
    cache_config=CacheConfig(cache_dir=".vertebrae_cache"),
).run()
```

To compare multiple compression variants in one run, pass
`compression_configs=[...]` to `Benchmark`.

## Supported methods

### `none`

Disables compression and preserves the current workflow.

### `pca`

Dense PCA for low- to medium-dimensional dense embeddings.

- Requires dense input.
- Supports `n_components` or `preserve_variance`.
- Supports `whiten=True`.

### `incremental_pca`

Dense PCA variant for larger dense embeddings.

- Requires dense input.
- Requires `n_components`.
- Supports `whiten=True`.

### `truncated_svd`

Recommended for sparse text embeddings such as TF-IDF features.

- Accepts sparse input.
- Requires `n_components`.

### `gaussian_random_projection`

Fast dense or sparse projection for coarse dimensionality reduction.

- Accepts sparse input.
- Requires `n_components`.

### `sparse_random_projection`

Sparse-friendly random projection for very high-dimensional sparse matrices.

- Accepts sparse input.
- Requires `n_components`.

### `prefix_truncate`

Keeps the first `n_components` dimensions without fitting a model.

- Accepts dense and sparse input.
- Requires `n_components`.
- Use `assume_matryoshka=True` when the embedding source is intentionally
  dimension-ordered, such as matryoshka-trained or shortened embeddings.

When `assume_matryoshka=False`, `vertebrae` records a warning so reports make it
clear this is being treated as a prefix-based diagnostic rather than a claimed
model-native shortening path.

### `quantize`

Applies a lossy precision-reduction step and returns numeric embeddings that can
still be scored by OverlapIndex.

Supported precisions:

- `float16`: direct cast for dense or sparse embeddings.
- `int8`: dense scalar quantize/dequantize round trip using symmetric per-dimension scaling.
- `uint8`: dense scalar quantize/dequantize round trip using affine per-dimension min/max scaling.

Binary and packed-bit quantization are intentionally not included here because
they are more appropriate for retrieval or ANN index evaluation than for
MiniBatchKMeans-backed OverlapIndex scoring.

## Metadata and reports

Each extractor result includes `compression_metadata` describing the applied
compression. Depending on the method, this may include:

- `method`
- `precision`
- `original_dim`
- `compressed_dim`
- `dtype`
- `compression_ratio`
- `explained_variance_total`
- `assume_matryoshka`
- quantization calibration metadata
- warnings

Markdown reports and `result.to_dataframe()` include compression columns so raw
and compressed variants can be compared side by side.

## CLI

The artifact-based CLI also supports compression:

```bash
vertebrae compress \
  --cache-dir .vertebrae_cache \
  --embedding-key embeddings/... \
  --method prefix_truncate \
  --n-components 256 \
  --assume-matryoshka
```

Example quantization run:

```bash
vertebrae compress \
  --cache-dir .vertebrae_cache \
  --embedding-key embeddings/... \
  --method quantize \
  --precision int8
```

The resulting compressed artifact can be passed to `vertebrae score` just like a
raw embedding artifact.

## Choosing a method

- Use `truncated_svd` for sparse text features.
- Use `pca` for dense embeddings when you want an interpretable learned reduction.
- Use `prefix_truncate` for matryoshka-style or dimension-shortened embeddings.
- Use `quantize` when you want to see how precision reduction affects the
  diagnostic, not when you need a retrieval index benchmark.
