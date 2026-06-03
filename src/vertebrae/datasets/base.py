"""Dataset abstraction for benchmark inputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np

from vertebrae.cache.fingerprint import fingerprint_array_like
from vertebrae.execution.jobs import SampleBatch, ShardSpec
from vertebrae.utils.labels import class_counts
from vertebrae.utils.validation import is_sparse_matrix


@dataclass
class BenchmarkDataset:
    """A labeled dataset prepared for feature extraction or scoring.

    Attributes:
        X: Input samples, tabular frame, image paths, or embedding matrix.
        y: One-dimensional label array.
        modality: Dataset modality such as `"text"`, `"tabular"`, or `"embeddings"`.
        input_col: Source dataframe input column or columns.
        label_col: Source dataframe label column.
        metadata: User and construction metadata.
    """

    X: Any
    y: np.ndarray
    modality: str
    input_col: Optional[Union[str, list[str]]] = None
    label_col: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        X: Any,
        y: Any,
        modality: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create a dataset from array-like inputs and labels.

        Args:
            X: Input samples or feature matrix.
            y: Class labels.
            modality: Dataset modality.
            metadata: Optional metadata to preserve.

        Returns:
            Validated benchmark dataset.
        """

        dataset = cls(X=X, y=np.asarray(y), modality=modality, metadata=metadata or {})
        dataset.validate()
        return dataset

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        input_col: Union[str, list[str]],
        label_col: str,
        modality: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create a dataset from a pandas DataFrame.

        Args:
            df: DataFrame containing inputs and labels.
            input_col: Input column name or list of tabular feature columns.
            label_col: Label column name.
            modality: Dataset modality.
            metadata: Optional metadata to preserve.

        Returns:
            Validated benchmark dataset.

        Raises:
            ValueError: If requested columns are missing.
        """

        input_cols = [input_col] if isinstance(input_col, str) else list(input_col)
        missing = [column for column in input_cols if column not in df.columns]
        if missing:
            raise ValueError(f"input_col contains columns not present in the dataframe: {missing}.")
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' is not present in the dataframe.")
        merged_metadata = {
            "source": "dataframe",
            "columns": list(df.columns),
            "input_columns": input_cols,
        }
        merged_metadata.update(metadata or {})
        X = df[input_col].to_numpy() if isinstance(input_col, str) else df[input_cols].copy()
        dataset = cls(
            X=X,
            y=df[label_col].to_numpy(),
            modality=modality,
            input_col=input_col,
            label_col=label_col,
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_image_paths(
        cls,
        paths: Any,
        labels: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create an image dataset from filesystem paths.

        Args:
            paths: Image file paths.
            labels: Class labels.
            metadata: Optional metadata to preserve.

        Returns:
            Validated image dataset.
        """

        merged_metadata = {"source": "image_paths"}
        merged_metadata.update(metadata or {})
        dataset = cls(
            X=np.asarray(paths, dtype=object),
            y=np.asarray(labels),
            modality="image",
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_embeddings(
        cls,
        embeddings: Any,
        labels: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create a dataset from precomputed dense or sparse embeddings.

        Args:
            embeddings: Dense array or scipy sparse embedding matrix.
            labels: Class labels.
            metadata: Optional metadata to preserve.

        Returns:
            Validated embedding dataset.
        """

        merged_metadata = {"precomputed_embeddings": True}
        merged_metadata.update(metadata or {})
        dataset = cls(
            X=embeddings if is_sparse_matrix(embeddings) else np.asarray(embeddings),
            y=np.asarray(labels),
            modality="embeddings",
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    def validate(self) -> None:
        """Validate dataset shape and labels.

        Raises:
            ValueError: If sample counts mismatch, labels are missing, too few
                classes are present, or a class has fewer than two samples.
        """

        if self.y.ndim != 1:
            raise ValueError("Labels must be one-dimensional.")
        n_samples = _num_samples(self.X)
        if n_samples != len(self.y):
            raise ValueError(
                f"X and y must have the same length; got {n_samples} and {len(self.y)}."
            )
        if len(self.y) == 0:
            raise ValueError("Dataset must contain at least one sample.")
        if _has_missing_labels(self.y):
            raise ValueError("Labels must be non-missing.")
        counts = self.class_counts()
        if len(counts) < 2:
            raise ValueError("Dataset must contain at least two classes.")
        small = {label: count for label, count in counts.items() if count < 2}
        if small:
            raise ValueError(f"Each class must contain at least 2 samples; found {small}.")

    def class_counts(self) -> Dict[Any, int]:
        """Count samples per class.

        Returns:
            Mapping from original label values to sample counts.
        """

        return class_counts(np.asarray(self.y))

    def iter_batches(
        self,
        batch_size: int,
        shard: Optional[ShardSpec] = None,
    ) -> Iterator[SampleBatch]:
        """Yield deterministic sample batches for embedding.

        Args:
            batch_size: Maximum samples per batch.
            shard: Optional non-overlapping shard assignment.

        Yields:
            Sample batches with original dataset indices and sliced inputs.

        Raises:
            ValueError: If `batch_size` is less than one.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        shard = shard or ShardSpec()
        indices = shard.indices(len(self.y))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield SampleBatch(indices=batch_indices, X=_take_samples(self.X, batch_indices))

    def summary(self) -> Dict[str, Any]:
        """Summarize the dataset for result metadata and reports.

        Returns:
            JSON-compatible dataset summary.
        """

        return {
            "n_samples": int(len(self.y)),
            "n_classes": int(len(self.class_counts())),
            "class_counts": self.class_counts(),
            "modality": self.modality,
            "input_col": self.input_col,
            "label_col": self.label_col,
            "metadata": self.metadata,
        }

    def fingerprint(self) -> str:
        """Compute a conservative dataset fingerprint for caching.

        Returns:
            Stable hash string derived from inputs, labels, modality, and metadata.
        """

        return fingerprint_array_like(
            {
                "X": self.X,
                "y": self.y,
                "modality": self.modality,
                "input_col": self.input_col,
                "label_col": self.label_col,
                "metadata": self.metadata,
            }
        )


def _has_missing_labels(y: np.ndarray) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(y).any())
    except ImportError:
        if y.dtype.kind in {"f", "c"}:
            return bool(np.isnan(y).any())
        return any(label is None for label in y)


def _num_samples(X: Any) -> int:
    if is_sparse_matrix(X):
        return int(X.shape[0])
    return len(X)


def _take_samples(X: Any, indices: np.ndarray) -> Any:
    if is_sparse_matrix(X):
        return X[indices]
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    if isinstance(X, np.ndarray):
        return X[indices]
    if isinstance(X, tuple):
        return tuple(X[int(index)] for index in indices)
    if isinstance(X, list):
        return [X[int(index)] for index in indices]
    try:
        return X[indices]
    except Exception:
        return [X[int(index)] for index in indices]
