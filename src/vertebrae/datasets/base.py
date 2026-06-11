"""Dataset abstraction for benchmark inputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional, Union, cast

import numpy as np

from vertebrae.cache.fingerprint import fingerprint_array_like
from vertebrae.execution.jobs import SampleBatch, ShardSpec
from vertebrae.utils.labels import (
    class_counts,
    default_label_view_metadata,
    hierarchy_depth,
    label_view_from_paths,
    normalize_label_paths,
    normalize_level_names,
)
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
    def from_audio_paths(
        cls,
        paths: Any,
        labels: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create an audio dataset from filesystem paths.

        Args:
            paths: Audio file paths.
            labels: Class labels.
            metadata: Optional metadata to preserve.

        Returns:
            Validated audio dataset.
        """

        merged_metadata = {"source": "audio_paths"}
        merged_metadata.update(metadata or {})
        dataset = cls(
            X={"path": np.asarray(paths, dtype=object)},
            y=np.asarray(labels),
            modality="audio",
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_audio_arrays(
        cls,
        audio: Any,
        labels: Any,
        sampling_rate: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create an audio dataset from waveform arrays.

        Args:
            audio: Sequence of 1D or 2D waveform arrays.
            labels: Class labels.
            sampling_rate: Shared sampling rate for every sample.
            metadata: Optional metadata to preserve.

        Returns:
            Validated audio dataset.
        """

        label_array = np.asarray(labels)
        merged_metadata = {
            "source": "audio_arrays",
            "sampling_rate": int(sampling_rate),
        }
        merged_metadata.update(metadata or {})
        dataset = cls(
            X={
                "array": _coerce_object_sequence(audio),
                "sampling_rate": np.full(len(label_array), int(sampling_rate), dtype=int),
            },
            y=label_array,
            modality="audio",
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_video_paths(
        cls,
        paths: Any,
        labels: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create a video dataset from filesystem paths.

        Args:
            paths: Video file paths.
            labels: Class labels.
            metadata: Optional metadata to preserve.

        Returns:
            Validated video dataset.
        """

        merged_metadata = {"source": "video_paths"}
        merged_metadata.update(metadata or {})
        dataset = cls(
            X={"path": np.asarray(paths, dtype=object)},
            y=np.asarray(labels),
            modality="video",
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_video_arrays(
        cls,
        frames: Any,
        labels: Any,
        frame_rate: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create a video dataset from predecoded frame arrays.

        Args:
            frames: Sequence of per-sample clip arrays, typically `(time, height, width, channels)`.
            labels: Class labels.
            frame_rate: Optional shared frame rate or sequence aligned to `frames`.
            metadata: Optional metadata to preserve.

        Returns:
            Validated video dataset.
        """

        label_array = np.asarray(labels)
        merged_metadata: Dict[str, Any] = {"source": "video_arrays"}
        payload: Dict[str, Any] = {"frames": _coerce_object_sequence(frames)}
        if frame_rate is not None:
            if np.isscalar(frame_rate):
                resolved_rate = float(cast(Any, frame_rate))
                payload["frame_rate"] = np.full(len(label_array), resolved_rate, dtype=float)
                merged_metadata["frame_rate"] = resolved_rate
            else:
                payload["frame_rate"] = np.asarray(frame_rate, dtype=float)
        merged_metadata.update(metadata or {})
        dataset = cls(
            X=payload,
            y=label_array,
            modality="video",
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_time_series(
        cls,
        series: Any,
        labels: Any,
        observed_mask: Any = None,
        time_features: Any = None,
        timestamps: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create a time-series dataset from aligned sequence inputs.

        Args:
            series: Sequence array with shape `(n_samples, time)` or
                `(n_samples, time, channels)`.
            labels: Class labels.
            observed_mask: Optional boolean or numeric mask aligned to `series`.
            time_features: Optional numeric time features aligned to `series`.
            timestamps: Optional timestamp annotations preserved for reporting.
            metadata: Optional metadata to preserve.

        Returns:
            Validated time-series dataset.
        """

        merged_metadata = {"source": "time_series"}
        merged_metadata.update(metadata or {})
        payload: Dict[str, Any] = {"series": np.asarray(series)}
        if observed_mask is not None:
            payload["observed_mask"] = np.asarray(observed_mask)
        if time_features is not None:
            payload["time_features"] = np.asarray(time_features)
        if timestamps is not None:
            payload["timestamps"] = np.asarray(timestamps)
        dataset = cls(
            X=payload,
            y=np.asarray(labels),
            modality="time_series",
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_multimodal(
        cls,
        inputs: Dict[str, Any],
        labels: Any,
        modalities: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create a dataset from aligned multi-modal sample fields.

        Args:
            inputs: Mapping from field name to aligned per-sample values.
            labels: Class labels.
            modalities: Mapping from field name to modality string.
            metadata: Optional metadata to preserve.

        Returns:
            Validated multi-modal dataset.
        """

        if not isinstance(inputs, dict):
            raise ValueError("inputs must be a dict mapping field names to aligned sample data.")
        if not isinstance(modalities, dict):
            raise ValueError("modalities must be a dict mapping field names to modality names.")
        if not inputs:
            raise ValueError("inputs must not be empty.")
        input_keys = list(inputs.keys())
        modality_keys = list(modalities.keys())
        if set(input_keys) != set(modality_keys):
            raise ValueError(
                "inputs and modalities must contain the same field names; "
                f"got {sorted(input_keys)} and {sorted(modality_keys)}."
            )

        label_array = np.asarray(labels)
        normalized_inputs: Dict[str, Any] = {}
        for field_name in input_keys:
            normalized_inputs[field_name] = _normalize_multimodal_field(
                inputs[field_name],
                n_samples=len(label_array),
                field_name=field_name,
            )
            if _field_has_missing_top_level_values(normalized_inputs[field_name]):
                raise ValueError(
                    "Multi-modal inputs must include a non-missing value for every sample; "
                    f"field '{field_name}' contains missing values."
                )

        merged_metadata: Dict[str, Any] = {
            "source": "multimodal",
            "input_fields": input_keys,
            "modalities": {key: str(modalities[key]) for key in input_keys},
        }
        merged_metadata.update(metadata or {})
        dataset = cls(
            X=normalized_inputs,
            y=label_array,
            modality="multimodal",
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

    def with_label_hierarchy(
        self,
        label_paths: Any,
        level_names: Optional[Iterable[Any]] = None,
    ) -> "BenchmarkDataset":
        """Return a dataset annotated with hierarchical label paths."""

        normalized_paths = normalize_label_paths(label_paths, n_samples=len(self.y))
        max_depth = hierarchy_depth(normalized_paths)
        normalized_level_names = normalize_level_names(level_names, max_depth=max_depth)
        hierarchy_metadata = {
            "paths": [list(path) for path in normalized_paths],
            "max_depth": int(max_depth),
            "level_names": (
                list(normalized_level_names) if normalized_level_names is not None else None
            ),
        }
        metadata = dict(self.metadata)
        metadata["label_hierarchy"] = hierarchy_metadata
        dataset = BenchmarkDataset(
            X=self.X,
            y=np.asarray(self.y),
            modality=self.modality,
            input_col=self.input_col,
            label_col=self.label_col,
            metadata=metadata,
        )
        dataset.validate()
        return dataset

    def label_view(self, level: Any, name: Optional[str] = None) -> "BenchmarkDataset":
        """Project hierarchical labels to a single requested level."""

        hierarchy = self.metadata.get("label_hierarchy")
        if hierarchy is None:
            raise ValueError("label_view(...) requires label hierarchy metadata on the dataset.")
        labels, view_metadata = label_view_from_paths(
            hierarchy["paths"],
            level=level,
            level_names=hierarchy.get("level_names"),
        )
        if name is not None:
            view_metadata["name"] = str(name)
            view_metadata["key"] = f"hierarchy:{view_metadata['level']}:{view_metadata['name']}"
        metadata = dict(self.metadata)
        metadata["label_view"] = view_metadata
        dataset = BenchmarkDataset(
            X=self.X,
            y=labels,
            modality=self.modality,
            input_col=self.input_col,
            label_col=self.label_col,
            metadata=metadata,
        )
        dataset.validate()
        return dataset

    def active_label_view(self) -> Dict[str, Any]:
        """Return metadata describing the active dataset label view."""

        view = self.metadata.get("label_view")
        if view is None:
            return default_label_view_metadata()
        return dict(view)

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

    def stratified_subsample_indices(
        self,
        rate: float,
        random_state: int = 42,
        min_samples_per_class: int = 2,
    ) -> np.ndarray:
        """Select class-stratified sample indices without replacement.

        Args:
            rate: Fraction of each class to keep. Must be in `(0, 1]`.
            random_state: Random seed for reproducible selection.
            min_samples_per_class: Minimum retained samples per class when possible.

        Returns:
            Sorted original sample indices for the stratified subset.

        Raises:
            ValueError: If `rate` is outside `(0, 1]`.
        """

        if not 0.0 < rate <= 1.0:
            raise ValueError("subsample rate must be in (0, 1].")
        if rate >= 1.0:
            return np.arange(len(self.y), dtype=int)
        rng = np.random.default_rng(random_state)
        selected = []
        for label in np.unique(self.y):
            class_indices = np.flatnonzero(self.y == label)
            target = int(np.floor(len(class_indices) * rate))
            if len(class_indices) >= min_samples_per_class:
                target = max(min_samples_per_class, target)
            target = max(1, min(len(class_indices), target))
            selected.extend(rng.choice(class_indices, size=target, replace=False).tolist())
        return np.asarray(sorted(selected), dtype=int)

    def subset(self, indices: Any, metadata: Optional[Dict[str, Any]] = None) -> "BenchmarkDataset":
        """Create a dataset subset by original sample indices.

        Args:
            indices: Sample indices to retain.
            metadata: Additional metadata to merge into the subset.

        Returns:
            Validated dataset containing only the selected samples.
        """

        index_array = np.asarray(indices, dtype=int)
        parent_indices = self.metadata.get("sample_indices")
        if parent_indices is None:
            sample_indices = index_array.tolist()
        else:
            sample_indices = np.asarray(parent_indices, dtype=int)[index_array].tolist()
        merged_metadata = dict(self.metadata)
        merged_metadata.update(
            {
                "subset": True,
                "parent_n_samples": int(len(self.y)),
                "sample_indices": sample_indices,
            }
        )
        hierarchy = merged_metadata.get("label_hierarchy")
        if hierarchy is not None:
            paths = hierarchy.get("paths")
            if paths is not None:
                hierarchy = dict(hierarchy)
                hierarchy["paths"] = np.asarray(paths, dtype=object)[index_array].tolist()
                merged_metadata["label_hierarchy"] = hierarchy
        merged_metadata.update(metadata or {})
        dataset = BenchmarkDataset(
            X=_take_samples(self.X, index_array),
            y=self.y[index_array],
            modality=self.modality,
            input_col=self.input_col,
            label_col=self.label_col,
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

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
            "label_view": self.active_label_view(),
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
    if isinstance(X, dict):
        lengths = {key: _num_samples(value) for key, value in X.items()}
        unique_lengths = set(lengths.values())
        if not unique_lengths:
            return 0
        if len(unique_lengths) != 1:
            raise ValueError(f"Structured dataset fields must align in length; found {lengths}.")
        return unique_lengths.pop()
    if is_sparse_matrix(X):
        return int(X.shape[0])
    return len(X)


def _take_samples(X: Any, indices: np.ndarray) -> Any:
    if isinstance(X, dict):
        return {key: _take_samples(value, indices) for key, value in X.items()}
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


def _coerce_object_sequence(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == object:
        return value
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError("Expected a sequence of per-sample values.") from exc
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def _normalize_multimodal_field(value: Any, n_samples: int, field_name: str) -> Any:
    if is_sparse_matrix(value):
        normalized = value
    elif hasattr(value, "iloc"):
        normalized = value
    elif isinstance(value, np.ndarray):
        normalized = value
    else:
        normalized = _coerce_object_sequence(value)
    actual_n_samples = _num_samples(normalized)
    if actual_n_samples != n_samples:
        raise ValueError(
            f"Multi-modal field '{field_name}' must have {n_samples} samples; "
            f"got {actual_n_samples}."
        )
    return normalized


def _field_has_missing_top_level_values(value: Any) -> bool:
    if is_sparse_matrix(value):
        return False
    if hasattr(value, "isna"):
        try:
            return bool(value.isna().any())
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        if value.ndim > 1 and value.dtype != object:
            return False
        items = value.tolist() if value.ndim == 1 else list(value)
        return any(_is_missing_scalar(item) for item in items)
    if isinstance(value, (list, tuple)):
        return any(_is_missing_scalar(item) for item in value)
    return False


def _is_missing_scalar(value: Any) -> bool:
    try:
        import pandas as pd

        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except ImportError:
        pass
    if value is None:
        return True
    if isinstance(value, float):
        return bool(np.isnan(value))
    return False
