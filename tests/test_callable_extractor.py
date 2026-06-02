import numpy as np
import pytest

from vertebrae.extractors import CallableExtractor


def test_callable_extractor_validates_numeric_2d_output():
    extractor = CallableExtractor(
        "stats",
        lambda X: np.column_stack([np.mean(X, axis=1), np.std(X, axis=1)]),
        modality="tabular",
        recipe_data={"features": ["mean", "std"]},
    )

    output = extractor.fit_transform(np.arange(12, dtype=float).reshape(4, 3))

    assert output.shape == (4, 2)
    assert extractor.recipe()["recipe_data"] == {"features": ["mean", "std"]}


def test_callable_extractor_rejects_1d_output():
    extractor = CallableExtractor("bad", lambda X: np.asarray([1, 2, 3]))

    with pytest.raises(ValueError, match="2D"):
        extractor.transform([[1], [2], [3]])
