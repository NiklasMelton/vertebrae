# vertebrae examples

These examples are small, local, and network-free. They are meant to show practical
workflows you can adapt for real datasets.

Run them from the project root:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/precomputed_embeddings.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/sklearn_text_pipeline.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/multi_extractor_comparison.py
POETRY_VIRTUALENVS_IN_PROJECT=true poetry run python examples/cache_reuse.py
```

Each script writes reports to `examples/output/`.

## Included workflows

- `precomputed_embeddings.py`: evaluate embeddings that were already generated
  elsewhere.
- `sklearn_text_pipeline.py`: build embeddings from a pandas text dataframe with a
  scikit-learn pipeline.
- `multi_extractor_comparison.py`: compare several extractors on the same labeled
  numeric dataset.
- `cache_reuse.py`: show how embedding caching avoids recomputing extractor output
  on repeated runs.

The examples configure small `k` values and a few stability repeats so they finish
quickly while still exercising the real MiniBatchKMeans-backed OverlapIndex path.
