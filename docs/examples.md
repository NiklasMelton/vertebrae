# examples

Runnable examples live in `examples/`:

- `precomputed_embeddings.py`: score synthetic precomputed embeddings.
- `sklearn_text_pipeline.py`: evaluate a TF-IDF + SVD text pipeline.
- `sklearn_tabular_pipeline.py`: evaluate mixed numeric/categorical tabular features.
- `sklearn_wine_pipeline.py`: compare real scaling/projection pipelines on the
  bundled UCI Wine dataset.
- `multi_extractor_comparison.py`: compare multiple local extractors.
- `cache_reuse.py`: demonstrate embedding cache reuse.
- `hf_text_extractor.py`: demonstrate the Hugging Face text API.
- `hf_vision_kmnist.py`: compare KMNIST handwritten character images with small
  Hugging Face vision backbones.
- `sentence_transformer_extractor.py`: demonstrate the sentence-transformers API.

Run local examples from the repository root:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_text_pipeline.py
```

The Hugging Face examples require optional dependencies and a model available from
a local cache or from Hugging Face:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install -E hf
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/hf_vision_kmnist.py
```
