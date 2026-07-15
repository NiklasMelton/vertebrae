"""Leaf-index embedding extractor for tree ensembles."""

from typing import Any, Dict, Optional

import numpy as np

from vertebrae.extractors._identity import (
    cache_identity_fields,
    validate_cache_identity,
    validate_extractor_name,
)
from vertebrae.extractors._utils import (
    optional_dependency_versions,
    snapshot_mapping,
    validate_bool,
    validate_choice,
)
from vertebrae.utils.validation import ensure_numeric_matrix


class TreeLeafEmbeddingExtractor:
    """Extract dense or sparse leaf embeddings from fitted tree ensembles."""

    def __init__(
        self,
        name: str,
        model: Any,
        backend: str = "auto",
        encoding: str = "dense",
        sparse_output: bool = True,
        recipe_data: Optional[Dict[str, Any]] = None,
        cache_identity: Optional[str] = None,
    ) -> None:
        backend = validate_choice(backend, "backend", {"auto", "xgboost", "lightgbm", "catboost"})
        encoding = validate_choice(encoding, "encoding", {"dense", "one_hot"})
        self.name = validate_extractor_name(name)
        self.model = model
        self.backend = backend
        self.encoding = encoding
        self.sparse_output = validate_bool(sparse_output, "sparse_output")
        self.recipe_data = snapshot_mapping(recipe_data, "recipe_data")
        self.modality = "tabular"
        self.extractor_type = "tree_leaf_embeddings"
        self.streaming_safe = encoding == "dense"
        self._column_maps: Optional[list[dict[int, int]]] = None
        self.cache_identity = validate_cache_identity(cache_identity)

    def fit(self, X: Any, y: Any = None) -> "TreeLeafEmbeddingExtractor":
        if self.encoding == "one_hot":
            leaves = self._leaf_matrix(X)
            self._column_maps = _build_column_maps(leaves)
        return self

    def transform(self, X: Any) -> Any:
        leaves = self._leaf_matrix(X)
        if self.encoding == "dense":
            return ensure_numeric_matrix(
                leaves.astype(np.float32, copy=False),
                f"TreeLeafEmbeddingExtractor '{self.name}' output",
                allow_sparse=False,
            )
        if self._column_maps is None:
            self._column_maps = _build_column_maps(leaves)
        matrix = _one_hot_leaf_matrix(
            leaves,
            column_maps=self._column_maps,
            sparse_output=self.sparse_output,
        )
        return ensure_numeric_matrix(
            matrix,
            f"TreeLeafEmbeddingExtractor '{self.name}' output",
            allow_sparse=True,
        )

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X, y).transform(X)

    def recipe(self) -> Dict[str, Any]:
        backend = self._resolved_backend()
        recipe = {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "backend": backend,
            "encoding": self.encoding,
            "sparse_output": self.sparse_output,
            "model_class": self.model.__class__.__module__ + "." + self.model.__class__.__name__,
            "recipe_data": self.recipe_data,
            "dependency_versions": optional_dependency_versions(backend),
            "streaming_safe": self.streaming_safe,
        }
        recipe.update(cache_identity_fields(explicit=self.cache_identity, state_required=True))
        return recipe

    def _leaf_matrix(self, X: Any) -> np.ndarray:
        values = np.asarray(self._predict_leaf_indices(X))
        if values.ndim == 0:
            raise ValueError("Tree leaf indices must contain one row per sample.")
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)
        return values.astype(np.int64, copy=False)

    def _predict_leaf_indices(self, X: Any) -> Any:
        backend = self._resolved_backend()
        if backend == "xgboost":
            if hasattr(self.model, "apply"):
                return self.model.apply(X)
            return self.model.predict(X, pred_leaf=True)
        if backend == "lightgbm":
            return self.model.predict(X, pred_leaf=True)
        if backend == "catboost":
            return self.model.calc_leaf_indexes(X)
        raise ValueError(f"Unsupported tree backend '{backend}'.")

    def _resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        module_name = self.model.__class__.__module__.lower()
        if "xgboost" in module_name:
            return "xgboost"
        if "lightgbm" in module_name:
            return "lightgbm"
        if "catboost" in module_name:
            return "catboost"
        raise ValueError(
            "Could not infer tree backend. Pass backend='xgboost', 'lightgbm', or 'catboost'."
        )


def _build_column_maps(values: np.ndarray) -> list[dict[int, int]]:
    column_maps: list[dict[int, int]] = []
    offset = 0
    for column in range(values.shape[1]):
        unique = sorted({int(item) for item in values[:, column].tolist()})
        mapping = {leaf: offset + index for index, leaf in enumerate(unique)}
        column_maps.append(mapping)
        offset += len(mapping)
    return column_maps


def _one_hot_leaf_matrix(
    values: np.ndarray,
    column_maps: list[dict[int, int]],
    sparse_output: bool,
) -> Any:
    from scipy import sparse

    rows = []
    cols = []
    for row_index in range(values.shape[0]):
        for column_index, mapping in enumerate(column_maps):
            value = int(values[row_index, column_index])
            if value not in mapping:
                continue
            rows.append(row_index)
            cols.append(mapping[value])
    data = np.ones(len(rows), dtype=np.float32)
    width = sum(len(mapping) for mapping in column_maps)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(values.shape[0], width))
    return matrix if sparse_output else matrix.toarray()
