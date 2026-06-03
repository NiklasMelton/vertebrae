import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

from vertebrae.extractors import SklearnExtractor


def test_sklearn_extractor_pipeline_fit_transform():
    X = np.arange(60, dtype=float).reshape(20, 3)
    pipeline = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=2))])
    extractor = SklearnExtractor("pca", pipeline)

    embeddings = extractor.fit_transform(X)

    assert embeddings.shape == (20, 2)
    assert extractor.already_fitted is True
    assert extractor.recipe()["pipeline_class"].endswith(".Pipeline")


def test_sklearn_text_pipeline_tfidf_svd_normalizer():
    texts = [
        "billing invoice refund",
        "billing payment receipt",
        "server api timeout",
        "api server latency",
        "dashboard chart filter",
        "product dashboard export",
    ]
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("svd", TruncatedSVD(n_components=3, random_state=42)),
            ("norm", Normalizer()),
        ]
    )
    extractor = SklearnExtractor("tfidf_svd", pipeline)

    embeddings = extractor.fit_transform(texts)

    assert embeddings.shape == (6, 3)


def test_sklearn_tabular_column_transformer_pipeline():
    df = pd.DataFrame(
        {
            "age": [21, 25, 40, 44, 52, 58],
            "income": [42, 48, 85, 88, 97, 102],
            "state": ["CA", "CA", "NY", "NY", "TX", "TX"],
        }
    )
    pipeline = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), ["age", "income"]),
                        ("state", OneHotEncoder(sparse_output=False), ["state"]),
                    ]
                ),
            ),
            ("pca", PCA(n_components=2, random_state=3)),
        ]
    )
    extractor = SklearnExtractor("tabular", pipeline)

    embeddings = extractor.fit_transform(df)

    assert embeddings.shape == (6, 2)


def test_sklearn_sparse_output_too_large_to_densify():
    extractor = SklearnExtractor(
        "tfidf_sparse",
        TfidfVectorizer(),
        max_dense_bytes=1,
    )

    with pytest.raises(ValueError, match="Sparse extractor output"):
        extractor.fit_transform(["alpha beta gamma", "alpha beta", "delta epsilon"])


def test_sklearn_allow_sparse_errors_clearly():
    extractor = SklearnExtractor(
        "tfidf_sparse",
        TfidfVectorizer(),
        allow_sparse=True,
    )

    embeddings = extractor.fit_transform(["alpha beta gamma", "alpha beta", "delta epsilon"])

    assert embeddings.shape == (3, 5)
    assert hasattr(embeddings, "tocsr")


class AlreadyFittedTransformer:
    def __init__(self):
        self.fit_called = False

    def fit(self, X, y=None):
        self.fit_called = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, :2]


def test_sklearn_already_fitted_calls_only_transform():
    transformer = AlreadyFittedTransformer()
    extractor = SklearnExtractor("fitted", transformer, already_fitted=True)

    embeddings = extractor.fit_transform(np.arange(12).reshape(4, 3))

    assert embeddings.shape == (4, 2)
    assert transformer.fit_called is False


def test_sklearn_rejects_1d_output():
    class BadTransformer:
        def fit_transform(self, X, y=None):
            return np.asarray([1, 2, 3])

    extractor = SklearnExtractor("bad", BadTransformer())

    with pytest.raises(ValueError, match="2D"):
        extractor.fit_transform([[1], [2], [3]])
