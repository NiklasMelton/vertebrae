import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vertebrae.extractors import SklearnExtractor


def test_sklearn_extractor_pipeline_fit_transform():
    X = np.arange(60, dtype=float).reshape(20, 3)
    pipeline = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=2))])
    extractor = SklearnExtractor("pca", pipeline)

    embeddings = extractor.fit_transform(X)

    assert embeddings.shape == (20, 2)
    assert extractor.already_fitted is True
    assert extractor.recipe()["pipeline_class"].endswith(".Pipeline")
