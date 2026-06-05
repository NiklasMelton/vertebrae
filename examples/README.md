# vertebrae examples

These examples are small enough for a laptop or desktop and are meant to show
practical workflows you can adapt for real datasets. The core examples are
network-free; the Hugging Face examples require optional dependencies and a model
available from the local Hugging Face cache or downloaded on first run.

Run them from the project root:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/precomputed_embeddings.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_text_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_tabular_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_wine_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/multi_extractor_comparison.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/cache_reuse.py
```

Each script writes reports to `examples/output/`.

## Included workflows

- `precomputed_embeddings.py`: evaluate embeddings that were already generated
  elsewhere.
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
- `hf_text_extractor.py`: Hugging Face text backbone API example. Requires optional
  dependencies and a local or downloadable model.
- `hf_vision_digits.py`: compare real scikit-learn handwritten digit images with
  small real Hugging Face vision backbones. Requires optional dependencies and
  local or downloadable models.
- `sentence_transformer_extractor.py`: sentence-transformers API example. Requires
  optional dependencies and a local or downloadable model.

The examples configure small `k` values and a few stability repeats so they finish
quickly while still exercising the real MiniBatchKMeans-backed OverlapIndex path.

Run the Hugging Face vision example after installing optional dependencies:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install -E hf
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/hf_vision_digits.py
```
