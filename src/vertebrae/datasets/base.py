"""Dataset abstraction for benchmark inputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from vertebrae.cache.fingerprint import fingerprint_array_like
from vertebrae.utils.labels import class_counts


@dataclass
class BenchmarkDataset:
    """A labeled dataset prepared for feature extraction or scoring."""

    X: Any
    y: np.ndarray
    modality: str
    input_col: Optional[str] = None
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
        dataset = cls(X=X, y=np.asarray(y), modality=modality, metadata=metadata or {})
        dataset.validate()
        return dataset

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        input_col: str,
        label_col: str,
        modality: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        if input_col not in df.columns:
            raise ValueError(f"input_col '{input_col}' is not present in the dataframe.")
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' is not present in the dataframe.")
        merged_metadata = {"source": "dataframe", "columns": list(df.columns)}
        merged_metadata.update(metadata or {})
        dataset = cls(
            X=df[input_col].to_numpy(),
            y=df[label_col].to_numpy(),
            modality=modality,
            input_col=input_col,
            label_col=label_col,
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
        merged_metadata = {"precomputed_embeddings": True}
        merged_metadata.update(metadata or {})
        dataset = cls(
            X=np.asarray(embeddings),
            y=np.asarray(labels),
            modality="embeddings",
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    def validate(self) -> None:
        if self.y.ndim != 1:
            raise ValueError("Labels must be one-dimensional.")
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have the same length; got {len(self.X)} and {len(self.y)}."
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
        return class_counts(np.asarray(self.y))

    def summary(self) -> Dict[str, Any]:
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
