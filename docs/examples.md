# examples

Runnable examples live in `examples/`:

- `precomputed_embeddings.py`: score synthetic precomputed embeddings.
- `relational_embeddings.py`: evaluate graph node and edge embeddings as
  supervised transfer-learning diagnostics.
- `sklearn_text_pipeline.py`: evaluate a TF-IDF + SVD text pipeline.
- `sklearn_tabular_pipeline.py`: evaluate mixed numeric/categorical tabular features.
- `sklearn_wine_pipeline.py`: compare real scaling/projection pipelines on the
  bundled UCI Wine dataset.
- `multi_extractor_comparison.py`: compare multiple local extractors.
- `representation_monitoring.py`: train a local Torch model, evaluate two named
  representations after each epoch, persist summary JSONL plus caller-managed
  training state, resume them together with `--resume`, and pivot overlap by epoch
  and layer.
- `cache_reuse.py`: demonstrate safe embedding cache reuse with an explicit callable
  `cache_identity`.
- `zero_shot_callable.py`: compare a synthetic frozen image/text-aligned adapter with
  an explicit prompt protocol, without downloads or a learned head.
- `structured_outputs.py`: materialize OCR/layout regions, ASR tokens, and pose
  keypoints directly from native structured extractors, then score them as
  representation-efficacy workflows rather than IoU, WER/CER, or OKS metrics.
- `structured_depth.py`: evaluate sampled depth cells as regression-style
  structured unit embeddings rather than RMSE or other depth-estimation metrics.
- `structured_latent_slots.py`: evaluate latent-slot embeddings with an explicit
  aligner that drops unmatched leading and trailing rows before scoring.
- `hf_audio_extractor.py`: demonstrate the Hugging Face audio API.
- `hf_multimodal_image_text.py`: demonstrate aligned image-text benchmarking with
  branch and fused Hugging Face outputs.
- `hf_text_extractor.py`: demonstrate the Hugging Face text API.
- `hf_time_series_extractor.py`: demonstrate the Hugging Face time-series API.
- `hf_video_extractor.py`: demonstrate the Hugging Face video API with predecoded clips.
- `hf_vision_mnist.py`: compare MNIST handwritten digit images with generic,
  final-layer, and mid-layer Hugging Face vision embeddings from one model
  configuration plus a scikit-learn image pipeline.
- `caltech101_vision_foundation_models.py`: compare a laptop-sized Caltech-101
  subset with related category pairs using DINOv2, a tiny supervised ViT baseline,
  and optional gated DINOv3 embeddings. It reuses a local Caltech-101 download
  when available, otherwise it downloads the dataset archive.
- `sentence_transformer_extractor.py`: demonstrate the sentence-transformers API.

Additional first-class extractor families now supported by the library include
timm, torchvision, OpenCLIP/SigLIP-style image-text models, TensorFlow Hub,
JAX/Flax adapters, tree leaf embeddings, graph models, and hosted embedding
APIs. The test suite covers these with fake modules and synthetic inputs; the
repository does not yet ship dedicated runnable example scripts for each one.

Run local examples from the repository root:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_text_pipeline.py
```

The Hugging Face examples require optional dependencies and a model available from
a local cache or from Hugging Face. Their remote names are intentionally unpinned and
the scripts explicitly disable embedding caching; pin an immutable revision or maintain
an explicit `cache_identity` before opting a real deployment into reuse.

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install -E hf
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/hf_vision_mnist.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/caltech101_vision_foundation_models.py
```

Set `VERTABRAE_INCLUDE_DINOV3=1` for the Caltech-101 example after accepting the
DINOv3 model terms on Hugging Face.
Override `VERTABRAE_CALTECH101_CLASSES` to make the slice easier or harder.
