# vertebrae examples

These examples are small enough for a laptop or desktop and are meant to show
practical workflows you can adapt for real datasets. The core examples are
network-free; the Hugging Face examples require optional dependencies and a model
available from the local Hugging Face cache or downloaded on first run.

Run them from the project root:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/precomputed_embeddings.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/multilabel_precomputed_embeddings.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_text_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_tabular_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_wine_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/multi_extractor_comparison.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/cache_reuse.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/torch_local_model.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/keras_local_model.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/onnx_extractor.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/hf_multimodal_image_text.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/caltech101_vision_foundation_models.py
```

Each script writes reports to `examples/output/`.

## Included workflows

- `precomputed_embeddings.py`: evaluate embeddings that were already generated
  elsewhere.
- `multilabel_precomputed_embeddings.py`: evaluate precomputed embeddings against
  a small multi-label classification target.
- `sklearn_text_pipeline.py`: build embeddings from a pandas text dataframe with a
  scikit-learn pipeline.
- `sklearn_tabular_pipeline.py`: build embeddings from mixed numeric/categorical
  dataframe columns with a `ColumnTransformer`.
- `sklearn_wine_pipeline.py`: compare real scikit-learn scaling/projection
  pipelines on the bundled UCI Wine dataset.
- `multi_extractor_comparison.py`: compare several extractors on the same labeled
  numeric dataset.
- `cache_reuse.py`: show how embedding caching avoids recomputing extractor output
  on repeated runs.
- `torch_local_model.py`: demonstrate `TorchExtractor` with a locally loaded
  PyTorch checkpoint and user-supplied `collate_fn` / `output_fn`.
- `keras_local_model.py`: demonstrate `KerasExtractor` with a locally saved
  Keras model and user-supplied `collate_fn` / `output_fn`.
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
- `sentence_transformer_extractor.py`: sentence-transformers API example. Requires
  optional dependencies and a local or downloadable model.

The library also now exposes first-class adapters for timm, torchvision,
OpenCLIP/SigLIP-style image-text models, TensorFlow Hub, JAX/Flax, tree leaf
embeddings, graph models, and hosted embedding APIs. Those integrations are
covered in unit tests with fake modules today; dedicated runnable example
scripts can be added as the public ergonomics settle.

The examples configure small `k` values and a few stability repeats so they finish
quickly while still exercising the real MiniBatchKMeans-backed OverlapIndex path.

Run the Hugging Face vision example after installing optional dependencies:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install -E hf
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/hf_vision_mnist.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/caltech101_vision_foundation_models.py
```

To include DINOv3 in the Caltech-101 example, accept the model terms on Hugging
Face and set `VERTABRAE_INCLUDE_DINOV3=1`.
Override `VERTABRAE_CALTECH101_CLASSES` to make the slice easier or harder.
