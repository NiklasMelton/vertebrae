"""Dataset abstraction for benchmark inputs."""

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Dict, Iterable, Iterator, Optional, Union, cast

import numpy as np

from vertebrae.cache.fingerprint import canonical_json_exact
from vertebrae.datasets.identity import DatasetIdentity
from vertebrae.execution.jobs import SampleBatch, ShardSpec
from vertebrae.utils.labels import (
    MULTI_LABEL_TARGET,
    REGRESSION_TARGET,
    class_counts,
    coerce_label_input,
    default_label_view_metadata,
    default_target_view_metadata,
    hierarchy_depth,
    label_view_from_paths,
    labels_from_jsonable,
    labels_to_jsonable,
    normalize_label_paths,
    normalize_level_names,
    normalize_targets,
    regression_subsample_indices,
    stratified_label_indices,
    target_summary,
)
from vertebrae.utils.semantic_labels import label_display, semantic_label_key
from vertebrae.utils.validation import (
    ensure_numeric_matrix,
    is_sparse_matrix,
    validate_row_indices,
)

_ROW_ALIGNED_METADATA_KEY = "_row_aligned_metadata_keys"
_TARGET_METADATA_KEYS = frozenset({"target_type", "label_names", "target_names"})
_STRUCTURAL_METADATA_KEYS = frozenset(
    {
        _ROW_ALIGNED_METADATA_KEY,
        "columns",
        "composition",
        "edge_index",
        "embedding_source",
        "entity_ids",
        "entity_type",
        "frame_rate",
        "group_name",
        "groups",
        "input_columns",
        "input_fields",
        "label_catalog",
        "label_hierarchy",
        "label_view",
        "modalities",
        "modality_detail",
        "node_ids",
        "pair_ids",
        "parent_row_positions",
        "parent_ids",
        "parent_n_samples",
        "precomputed_embeddings",
        "relational_embeddings",
        "relational_unit",
        "sample_indices",
        "sampling_rate",
        "segmentation_embeddings",
        "source",
        "subset",
        "target_view",
        "target_views",
        "triplet_ids",
        "unit_annotation_task_family",
        "unit_annotation_unit_type",
        "unit_annotations",
        "unit_coordinates",
        "unit_embeddings",
        "unit_ids",
        "unit_positions",
        "unit_provenance",
        "unit_spans",
        "unit_type",
    }
    | _TARGET_METADATA_KEYS
)


@dataclass
class TargetView:
    """Declarative target view aligned to an existing dataset sample axis."""

    name: str
    targets: Any
    target_type: str = "auto"
    label_names: Optional[Iterable[Any]] = None
    target_names: Optional[Iterable[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnitAnnotation:
    """Declarative per-parent unit targets and provenance."""

    labels: Any
    unit_ids: Any = None
    positions: Any = None
    spans: Any = None
    coordinates: Any = None
    provenance: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    label_names: Optional[Iterable[Any]] = None
    target_type: str = "auto"
    target_names: Optional[Iterable[str]] = None


@dataclass
class BenchmarkDataset:
    """A labeled dataset prepared for feature extraction or scoring.

    Attributes:
        X: Input samples, tabular frame, image paths, or embedding matrix.
        y: Single-label, multi-label, or explicit regression targets.
        modality: Dataset modality such as `"text"`, `"tabular"`, or `"embeddings"`.
        input_col: Source dataframe input column or columns.
        label_col: Source dataframe label column.
        metadata: User and construction metadata.
    """

    X: Any
    y: np.ndarray
    modality: str
    identity: DatasetIdentity
    input_col: Optional[Union[str, list[str]]] = None
    label_col: Optional[Union[str, list[str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _identity_key_cache: Optional[str] = field(default=None, init=False, repr=False)

    @classmethod
    def from_arrays(
        cls,
        X: Any,
        y: Any,
        modality: str,
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
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

        dataset = cls(
            X=X,
            y=coerce_label_input(y),
            modality=modality,
            identity=identity,
            metadata=_metadata_with_target_metadata(
                _merge_user_metadata({}, metadata, reserved=_TARGET_METADATA_KEYS),
                label_names=label_names,
                target_type=target_type,
                target_names=target_names,
            ),
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        input_col: Union[str, list[str]],
        label_col: Union[str, list[str]],
        modality: str,
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
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
        label_cols = [label_col] if isinstance(label_col, str) else list(label_col)
        missing_labels = [column for column in label_cols if column not in df.columns]
        if missing_labels:
            raise ValueError(
                f"label_col contains columns not present in the dataframe: {missing_labels}."
            )
        resolved_label_names = label_names
        if (
            resolved_label_names is None
            and not isinstance(label_col, str)
            and target_type != REGRESSION_TARGET
        ):
            resolved_label_names = label_cols
        resolved_target_names = target_names
        if (
            resolved_target_names is None
            and not isinstance(label_col, str)
            and target_type == REGRESSION_TARGET
        ):
            resolved_target_names = [str(column) for column in label_cols]
        merged_metadata = {
            "source": "dataframe",
            "columns": list(df.columns),
            "input_columns": input_cols,
        }
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=resolved_label_names,
            target_type=target_type,
            target_names=resolved_target_names,
        )
        X = df[input_col].to_numpy() if isinstance(input_col, str) else df[input_cols].copy()
        labels = (
            df[label_col].to_numpy() if isinstance(label_col, str) else df[label_cols].to_numpy()
        )
        dataset = cls(
            X=X,
            y=labels,
            modality=modality,
            identity=identity,
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
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
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
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        dataset = cls(
            X=np.asarray(paths, dtype=object),
            y=coerce_label_input(labels),
            modality="image",
            identity=identity,
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_audio_paths(
        cls,
        paths: Any,
        labels: Any,
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
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
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        dataset = cls(
            X={"path": np.asarray(paths, dtype=object)},
            y=coerce_label_input(labels),
            modality="audio",
            identity=identity,
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
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
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

        resolved_sampling_rate = _strict_positive_integer(sampling_rate, "sampling_rate")
        label_array = coerce_label_input(labels)
        waveforms = [
            _validated_numeric_sample(
                waveform,
                "audio waveform",
                allowed_ranks=(1, 2),
            )
            for waveform in list(audio)
        ]
        merged_metadata = {
            "source": "audio_arrays",
            "sampling_rate": resolved_sampling_rate,
        }
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        dataset = cls(
            X={
                "array": _coerce_object_sequence(waveforms),
                "sampling_rate": np.full(len(label_array), resolved_sampling_rate, dtype=int),
            },
            y=label_array,
            modality="audio",
            identity=identity,
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_video_paths(
        cls,
        paths: Any,
        labels: Any,
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
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
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        dataset = cls(
            X={"path": np.asarray(paths, dtype=object)},
            y=coerce_label_input(labels),
            modality="video",
            identity=identity,
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_video_arrays(
        cls,
        frames: Any,
        labels: Any,
        *,
        identity: DatasetIdentity,
        frame_rate: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
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

        label_array = coerce_label_input(labels)
        clips = [
            _validated_numeric_sample(
                clip,
                "video frame array",
                allowed_ranks=(4,),
            )
            for clip in list(frames)
        ]
        merged_metadata: Dict[str, Any] = {"source": "video_arrays"}
        payload: Dict[str, Any] = {"frames": _coerce_object_sequence(clips)}
        if frame_rate is not None:
            rate_array = np.asarray(frame_rate)
            if rate_array.ndim == 0:
                resolved_rate = _strict_positive_real(rate_array.item(), "frame_rate")
                payload["frame_rate"] = np.full(len(label_array), resolved_rate, dtype=float)
                merged_metadata["frame_rate"] = resolved_rate
            else:
                payload["frame_rate"] = _validated_frame_rates(
                    rate_array,
                    len(label_array),
                )
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        dataset = cls(
            X=payload,
            y=label_array,
            modality="video",
            identity=identity,
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_time_series(
        cls,
        series: Any,
        labels: Any,
        *,
        identity: DatasetIdentity,
        observed_mask: Any = None,
        time_features: Any = None,
        timestamps: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
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

        series_array = _validated_numeric_sample(
            series,
            "time-series values",
            allowed_ranks=(2, 3),
        )
        merged_metadata = {"source": "time_series"}
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        payload: Dict[str, Any] = {"series": series_array}
        if observed_mask is not None:
            mask = _validated_numeric_sample(
                observed_mask,
                "observed_mask",
                allowed_ranks=(series_array.ndim,),
                allow_bool=True,
            )
            if mask.shape != series_array.shape:
                raise ValueError("observed_mask must have the same shape as series.")
            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError("observed_mask must contain only binary 0/1 values.")
            payload["observed_mask"] = mask
        if time_features is not None:
            features = _validated_numeric_sample(
                time_features,
                "time_features",
                allowed_ranks=(2, 3),
            )
            if features.shape[:2] != series_array.shape[:2]:
                raise ValueError(
                    "time_features must align with the sample and time axes of series."
                )
            payload["time_features"] = features
        if timestamps is not None:
            timestamp_values = np.asarray(timestamps)
            if timestamp_values.ndim not in (2, 3) or any(
                size < 1 for size in timestamp_values.shape
            ):
                raise ValueError("timestamps must be a non-empty 2D or 3D array aligned to series.")
            if timestamp_values.shape[:2] != series_array.shape[:2]:
                raise ValueError("timestamps must align with the sample and time axes of series.")
            _validate_timestamp_values(timestamp_values)
            payload["timestamps"] = timestamp_values
        dataset = cls(
            X=payload,
            y=coerce_label_input(labels),
            modality="time_series",
            identity=identity,
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
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
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

        label_array = coerce_label_input(labels)
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
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        dataset = cls(
            X=normalized_inputs,
            y=label_array,
            modality="multimodal",
            identity=identity,
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_graphs(
        cls,
        graphs: Any,
        labels: Any,
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> "BenchmarkDataset":
        """Create a graph dataset from aligned graph objects."""

        merged_metadata = {"source": "graphs"}
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        dataset = cls(
            X=np.asarray(graphs, dtype=object),
            y=coerce_label_input(labels),
            modality="graph",
            identity=identity,
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_embeddings(
        cls,
        embeddings: Any,
        labels: Any,
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> "BenchmarkDataset":
        """Create a dataset from precomputed dense or sparse embeddings.

        Args:
            embeddings: Dense array or scipy sparse embedding matrix.
            labels: Class labels.
            metadata: Optional metadata to preserve.

        Returns:
            Validated embedding dataset.
        """

        matrix = ensure_numeric_matrix(embeddings, "embeddings", allow_sparse=True)
        merged_metadata = _merge_user_metadata(
            {"precomputed_embeddings": True},
            metadata,
        )
        return cls._from_prepared_embeddings(
            matrix,
            labels,
            identity=identity,
            metadata=merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    @classmethod
    def _from_prepared_embeddings(
        cls,
        embeddings: Any,
        labels: Any,
        *,
        identity: DatasetIdentity,
        metadata: Dict[str, Any],
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> "BenchmarkDataset":
        """Build an embedding dataset from constructor-owned structural metadata."""

        merged_metadata = _metadata_with_target_metadata(
            metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        dataset = cls(
            X=embeddings,
            y=coerce_label_input(labels),
            modality="embeddings",
            identity=identity,
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_segmentation_embeddings(
        cls,
        embeddings: Any,
        labels: Any,
        image_ids: Any,
        *,
        identity: DatasetIdentity,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create a grouped token dataset from precomputed segmentation features."""

        matrix = ensure_numeric_matrix(embeddings, "segmentation embeddings", allow_sparse=True)
        merged_metadata = _merge_user_metadata(
            {
                "precomputed_embeddings": True,
                "segmentation_embeddings": True,
            },
            metadata,
        )
        return cls._from_prepared_embeddings(
            matrix,
            labels,
            identity=identity,
            metadata=merged_metadata,
        ).with_groups(image_ids, name="image_id")

    @classmethod
    def from_embedding_units(
        cls,
        embeddings: Any,
        labels: Any,
        unit_ids: Any,
        *,
        identity: DatasetIdentity,
        parent_ids: Any = None,
        unit_type: str = "unit",
        positions: Any = None,
        spans: Any = None,
        coordinates: Any = None,
        provenance: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
        target_views: Optional[Iterable[TargetView]] = None,
    ) -> "EmbeddingUnitDataset":
        """Create a generic grouped unit dataset from precomputed embeddings."""

        return EmbeddingUnitDataset.from_units(
            embeddings=embeddings,
            labels=labels,
            identity=identity,
            unit_ids=unit_ids,
            parent_ids=parent_ids,
            unit_type=unit_type,
            positions=positions,
            spans=spans,
            coordinates=coordinates,
            provenance=provenance,
            metadata=metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
            target_views=target_views,
        )

    @classmethod
    def from_node_embeddings(
        cls,
        embeddings: Any,
        labels: Any,
        *,
        identity: DatasetIdentity,
        node_ids: Any = None,
        edge_index: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> "BenchmarkDataset":
        """Create an embedding dataset for labeled graph nodes.

        This is an embedding-efficacy view of a graph: every row is one node
        embedding and labels or regression targets remain aligned row-wise.
        """

        matrix = ensure_numeric_matrix(embeddings, "node embeddings", allow_sparse=True)
        n_nodes = int(matrix.shape[0])
        resolved_node_ids = _aligned_ids(node_ids, n_nodes, name="node_ids")
        merged_metadata = {
            "precomputed_embeddings": True,
            "relational_embeddings": True,
            "relational_unit": "node",
            "modality_detail": "graph_node_embeddings",
            "node_ids": resolved_node_ids,
        }
        if edge_index is not None:
            merged_metadata["edge_index"] = _normalize_edge_like_index(
                edge_index,
                n_rows=None,
                name="edge_index",
            ).tolist()
        merged_metadata = _register_row_aligned_metadata(merged_metadata, "node_ids")
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        return cls._from_prepared_embeddings(
            matrix,
            labels,
            identity=identity,
            metadata=merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    @classmethod
    def from_entity_embeddings(
        cls,
        embeddings: Any,
        labels: Any,
        *,
        identity: DatasetIdentity,
        entity_ids: Any = None,
        entity_type: str = "entity",
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> "BenchmarkDataset":
        """Create an embedding dataset for labeled entities.

        Use this for user, item, query, document, or other entity embeddings
        when the evaluation target is attached to the entity row itself.
        """

        matrix = ensure_numeric_matrix(embeddings, "entity embeddings", allow_sparse=True)
        n_entities = int(matrix.shape[0])
        resolved_entity_ids = _aligned_ids(entity_ids, n_entities, name="entity_ids")
        normalized_entity_type = str(entity_type)
        merged_metadata = {
            "precomputed_embeddings": True,
            "relational_embeddings": True,
            "relational_unit": "entity",
            "entity_type": normalized_entity_type,
            "modality_detail": f"{normalized_entity_type}_embeddings",
            "entity_ids": resolved_entity_ids,
        }
        merged_metadata = _register_row_aligned_metadata(merged_metadata, "entity_ids")
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        return cls._from_prepared_embeddings(
            matrix,
            labels,
            identity=identity,
            metadata=merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    @classmethod
    def from_edge_embeddings(
        cls,
        *,
        identity: DatasetIdentity,
        edge_embeddings: Any = None,
        labels: Any = None,
        edge_index: Any = None,
        node_embeddings: Any = None,
        node_ids: Any = None,
        composition: str = "hadamard",
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> "BenchmarkDataset":
        """Create an embedding dataset for labeled graph edges.

        Pass either precomputed `edge_embeddings`, or pass `node_embeddings`
        plus `edge_index` to compose one embedding row per edge.
        """

        if labels is None:
            raise ValueError("labels must be provided for edge embedding evaluation.")
        normalized_edge_index = _normalize_edge_like_index(
            edge_index,
            n_rows=None,
            name="edge_index",
        )
        if edge_embeddings is None:
            if node_embeddings is None:
                raise ValueError("node_embeddings are required when edge_embeddings is not given.")
            matrix = _compose_index_pairs(
                node_embeddings,
                normalized_edge_index,
                ids=node_ids,
                composition=composition,
                owner="edge embeddings",
            )
            source = "composed_node_embeddings"
        else:
            matrix = ensure_numeric_matrix(edge_embeddings, "edge embeddings", allow_sparse=True)
            source = "precomputed_edge_embeddings"
            if matrix.shape[0] != normalized_edge_index.shape[0]:
                raise ValueError(
                    "edge_embeddings and edge_index must have the same number of rows; "
                    f"got {matrix.shape[0]} and {normalized_edge_index.shape[0]}."
                )
        merged_metadata = {
            "precomputed_embeddings": True,
            "relational_embeddings": True,
            "relational_unit": "edge",
            "modality_detail": "graph_edge_embeddings",
            "edge_index": normalized_edge_index.tolist(),
            "composition": composition if edge_embeddings is None else None,
            "embedding_source": source,
        }
        if node_ids is not None:
            merged_metadata["node_ids"] = _aligned_ids(
                node_ids,
                _num_samples(node_embeddings) if node_embeddings is not None else len(node_ids),
                name="node_ids",
            )
        merged_metadata = _register_row_aligned_metadata(merged_metadata, "edge_index")
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        return cls._from_prepared_embeddings(
            matrix,
            labels,
            identity=identity,
            metadata=merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    @classmethod
    def from_pair_embeddings(
        cls,
        *,
        identity: DatasetIdentity,
        pair_embeddings: Any = None,
        labels: Any = None,
        pairs: Any = None,
        entity_embeddings: Any = None,
        entity_ids: Any = None,
        composition: str = "abs_diff",
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> "BenchmarkDataset":
        """Create an embedding dataset for labeled entity pairs."""

        if labels is None:
            raise ValueError("labels must be provided for pair embedding evaluation.")
        normalized_pairs = _normalize_edge_like_index(pairs, n_rows=None, name="pairs")
        if pair_embeddings is None:
            if entity_embeddings is None:
                raise ValueError(
                    "entity_embeddings are required when pair_embeddings is not given."
                )
            matrix = _compose_index_pairs(
                entity_embeddings,
                normalized_pairs,
                ids=entity_ids,
                composition=composition,
                owner="pair embeddings",
            )
            source = "composed_entity_embeddings"
        else:
            matrix = ensure_numeric_matrix(pair_embeddings, "pair embeddings", allow_sparse=True)
            source = "precomputed_pair_embeddings"
            if matrix.shape[0] != normalized_pairs.shape[0]:
                raise ValueError(
                    "pair_embeddings and pairs must have the same number of rows; "
                    f"got {matrix.shape[0]} and {normalized_pairs.shape[0]}."
                )
        merged_metadata = {
            "precomputed_embeddings": True,
            "relational_embeddings": True,
            "relational_unit": "pair",
            "modality_detail": "pair_embeddings",
            "pair_ids": normalized_pairs.tolist(),
            "composition": composition if pair_embeddings is None else None,
            "embedding_source": source,
        }
        if entity_ids is not None:
            merged_metadata["entity_ids"] = _aligned_ids(
                entity_ids,
                _num_samples(entity_embeddings)
                if entity_embeddings is not None
                else len(entity_ids),
                name="entity_ids",
            )
        merged_metadata = _register_row_aligned_metadata(merged_metadata, "pair_ids")
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        return cls._from_prepared_embeddings(
            matrix,
            labels,
            identity=identity,
            metadata=merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    @classmethod
    def from_triplet_embeddings(
        cls,
        *,
        identity: DatasetIdentity,
        triplet_embeddings: Any = None,
        labels: Any = None,
        triplets: Any = None,
        entity_embeddings: Any = None,
        entity_ids: Any = None,
        composition: str = "abs_diff",
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> "BenchmarkDataset":
        """Create an embedding dataset for supervised triplet-derived rows."""

        if labels is None:
            raise ValueError("labels must be provided for triplet embedding evaluation.")
        normalized_triplets = _normalize_triplets(triplets)
        if triplet_embeddings is None:
            if entity_embeddings is None:
                raise ValueError(
                    "entity_embeddings are required when triplet_embeddings is not given."
                )
            matrix = _compose_triplets(
                entity_embeddings,
                normalized_triplets,
                ids=entity_ids,
                composition=composition,
            )
            source = "composed_entity_embeddings"
        else:
            matrix = ensure_numeric_matrix(
                triplet_embeddings,
                "triplet embeddings",
                allow_sparse=True,
            )
            source = "precomputed_triplet_embeddings"
            if matrix.shape[0] != normalized_triplets.shape[0]:
                raise ValueError(
                    "triplet_embeddings and triplets must have the same number of rows; "
                    f"got {matrix.shape[0]} and {normalized_triplets.shape[0]}."
                )
        merged_metadata = {
            "precomputed_embeddings": True,
            "relational_embeddings": True,
            "relational_unit": "triplet",
            "modality_detail": "triplet_embeddings",
            "triplet_ids": normalized_triplets.tolist(),
            "composition": composition if triplet_embeddings is None else None,
            "embedding_source": source,
        }
        if entity_ids is not None:
            merged_metadata["entity_ids"] = _aligned_ids(
                entity_ids,
                _num_samples(entity_embeddings)
                if entity_embeddings is not None
                else len(entity_ids),
                name="entity_ids",
            )
        merged_metadata = _register_row_aligned_metadata(merged_metadata, "triplet_ids")
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        return cls._from_prepared_embeddings(
            matrix,
            labels,
            identity=identity,
            metadata=merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    def validate(self) -> None:
        """Validate dataset shape and labels.

        Raises:
            ValueError: If sample counts mismatch, labels are missing, too few
                classes are present, or a class has fewer than two samples.
        """

        if not isinstance(self.identity, DatasetIdentity):
            raise TypeError("identity must be a DatasetIdentity.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        self.metadata = dict(self.metadata)
        label_names = self.metadata.get("label_names")
        requested_target_type = self.metadata.get("target_type", "auto")
        target_names = self.metadata.get("target_names")
        normalized_labels, label_metadata = normalize_targets(
            self.y,
            label_names=label_names,
            target_type=requested_target_type,
            target_names=target_names,
        )
        self.y = normalized_labels
        n_samples = _num_samples(self.X)
        if n_samples != len(self.y):
            raise ValueError(
                f"X and y must have the same length; got {n_samples} and {len(self.y)}."
            )
        if len(self.y) == 0:
            raise ValueError("Dataset must contain at least one sample.")
        self.metadata["target_type"] = label_metadata["target_type"]
        if label_metadata["target_type"] != REGRESSION_TARGET:
            declared_catalog = self.metadata.get("label_catalog")
            declared_keys = (
                {str(item.get("key")) for item in declared_catalog if isinstance(item, dict)}
                if isinstance(declared_catalog, list)
                else set()
            )
            observed_keys = set(label_metadata.get("class_counts", {}))
            if not declared_catalog or not observed_keys.issubset(declared_keys):
                self.metadata["label_catalog"] = list(label_metadata.get("label_catalog", []))
        else:
            self.metadata.pop("label_catalog", None)
        if label_metadata["target_type"] == "multi_label":
            self.metadata["label_names"] = list(label_metadata["label_names"])
            self.metadata.pop("target_names", None)
        elif label_metadata["target_type"] == REGRESSION_TARGET:
            self.metadata["target_names"] = list(label_metadata["target_names"])
            self.metadata.pop("label_names", None)
        else:
            self.metadata.pop("label_names", None)
            self.metadata.pop("target_names", None)
        _validate_row_aligned_metadata(self.metadata, n_samples=n_samples)
        if label_metadata["target_type"] == REGRESSION_TARGET:
            if len(self.y) < 3:
                raise ValueError("Regression datasets must contain at least 3 samples.")
            if not label_metadata["nonconstant_targets"]:
                raise ValueError(
                    "Regression datasets must contain at least one non-constant target."
                )
            return
        counts = label_metadata["class_counts"]
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

        counts = class_counts(
            self.y,
            label_names=self.metadata.get("label_names"),
            target_type=self.metadata.get("target_type", "auto"),
            target_names=self.metadata.get("target_names"),
        )
        label_view = self.active_label_view()
        catalog = label_view.get("label_catalog", [])
        if label_view.get("kind") == "hierarchy" and catalog:
            return {label_display(key, catalog): count for key, count in counts.items()}
        return counts

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
        dataset = type(self)(
            X=self.X,
            y=coerce_label_input(self.y),
            modality=self.modality,
            identity=self._derive_identity("with_label_hierarchy", hierarchy_metadata),
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
        metadata["label_catalog"] = list(view_metadata.get("label_catalog", []))
        dataset = type(self)(
            X=self.X,
            y=labels,
            modality=self.modality,
            identity=self._derive_identity("label_view", view_metadata),
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

    def with_target_views(
        self,
        target_views: Iterable[TargetView],
    ) -> "BenchmarkDataset":
        """Return a dataset annotated with aligned named target views."""

        resolved = _normalize_target_views(
            target_views=target_views,
            n_samples=len(self.y),
            X=self.X,
            modality=self.modality,
            input_col=self.input_col,
            label_col=self.label_col,
            base_metadata=self.metadata,
            dataset_type=type(self),
            identity=self.identity,
        )
        metadata = dict(self.metadata)
        metadata["target_views"] = resolved
        dataset = type(self)(
            X=self.X,
            y=coerce_label_input(self.y),
            modality=self.modality,
            identity=self._derive_identity("with_target_views", resolved),
            input_col=self.input_col,
            label_col=self.label_col,
            metadata=metadata,
        )
        dataset.validate()
        return dataset

    def target_view(self, name: str) -> "BenchmarkDataset":
        """Materialize one named target view as an ordinary benchmark dataset."""

        views = self.metadata.get("target_views")
        if views is None:
            raise ValueError("target_view(...) requires target view metadata on the dataset.")
        try:
            view = views[str(name)]
        except KeyError as exc:
            available = sorted(views)
            raise ValueError(
                f"Unknown target view {name!r}. Available target views: {available}."
            ) from exc
        labels = labels_from_jsonable(
            view["targets"],
            label_names=view.get("label_names"),
            target_type=view.get("target_type", "auto"),
            target_names=view.get("target_names"),
        )
        metadata = dict(self.metadata)
        metadata.pop("label_names", None)
        metadata.pop("target_names", None)
        metadata["target_type"] = view.get("target_type", "auto")
        metadata["target_view"] = {
            "kind": "named_target",
            "name": str(name),
            "key": f"target:{name}",
            "target_type": view.get("target_type", "auto"),
            "metadata": dict(view.get("metadata", {})),
        }
        metadata = _metadata_with_target_metadata(
            metadata,
            label_names=view.get("label_names"),
            target_type=view.get("target_type", "auto"),
            target_names=view.get("target_names"),
        )
        dataset = type(self)(
            X=self.X,
            y=labels,
            modality=self.modality,
            identity=self._derive_identity("target_view", {"name": str(name), "view": view}),
            input_col=self.input_col,
            label_col=self.label_col,
            metadata=metadata,
        )
        dataset.validate()
        return dataset

    def target_view_names(self) -> list[str]:
        """Return the registered target view names in insertion order."""

        views = self.metadata.get("target_views", {})
        return [str(name) for name in views]

    def active_target_view(self) -> Dict[str, Any]:
        """Return metadata describing the active dataset target view."""

        view = self.metadata.get("target_view")
        if view is None:
            return default_target_view_metadata()
        return dict(view)

    def with_unit_annotations(
        self,
        annotations: Iterable[UnitAnnotation],
        unit_type: str = "unit",
        task_family: Optional[str] = None,
    ) -> "BenchmarkDataset":
        """Return a dataset with aligned per-parent structured unit annotations."""

        resolved = _normalize_unit_annotations(
            annotations=annotations,
            n_samples=len(self.y),
            unit_type=unit_type,
        )
        metadata = dict(self.metadata)
        metadata["unit_annotations"] = resolved
        metadata["unit_annotation_unit_type"] = str(unit_type)
        if task_family is not None:
            metadata["unit_annotation_task_family"] = str(task_family)
        metadata = _register_row_aligned_metadata(metadata, "unit_annotations")
        dataset = type(self)(
            X=self.X,
            y=coerce_label_input(self.y),
            modality=self.modality,
            identity=self._derive_identity(
                "with_unit_annotations",
                {"annotations": resolved, "unit_type": unit_type, "task_family": task_family},
            ),
            input_col=self.input_col,
            label_col=self.label_col,
            metadata=metadata,
        )
        dataset.validate()
        return dataset

    def unit_annotations(self) -> Optional[list[Dict[str, Any]]]:
        """Return aligned structured unit annotations when configured."""

        annotations = self.metadata.get("unit_annotations")
        if annotations is None:
            return None
        return [dict(annotation) for annotation in annotations]

    def with_groups(self, groups: Any, name: str = "group") -> "BenchmarkDataset":
        """Return a dataset with aligned independence-group identifiers."""

        group_array = np.asarray(groups, dtype=object)
        if group_array.ndim != 1:
            raise ValueError("groups must be one-dimensional.")
        if len(group_array) != len(self.y):
            raise ValueError(
                f"groups and samples must have the same length; got {len(group_array)} "
                f"and {len(self.y)}."
            )
        for value in group_array:
            normalized = value.item() if hasattr(value, "item") else value
            if _is_missing_identifier(normalized):
                raise ValueError("groups values must be non-missing.")
            try:
                hash(normalized)
                canonical_json_exact(normalized)
            except TypeError as exc:
                raise ValueError(
                    "groups values must be hashable and have deterministic exact identities."
                ) from exc
        metadata = dict(self.metadata)
        metadata["groups"] = group_array.tolist()
        metadata["group_name"] = str(name)
        metadata = _register_row_aligned_metadata(metadata, "groups")
        dataset = type(self)(
            X=self.X,
            y=coerce_label_input(self.y),
            modality=self.modality,
            identity=self._derive_identity(
                "with_groups", {"groups": group_array, "name": str(name)}
            ),
            input_col=self.input_col,
            label_col=self.label_col,
            metadata=metadata,
        )
        dataset.validate()
        return dataset

    def groups(self) -> Optional[np.ndarray]:
        """Return aligned independence groups when configured."""

        values = self.metadata.get("groups")
        return None if values is None else np.asarray(values, dtype=object)

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

        batch_size = _strict_positive_integer(batch_size, "batch_size")
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
        """Select target-aware sample indices without replacement.

        Args:
            rate: Requested fraction of samples to keep. Must be in `(0, 1]`.
            random_state: Random seed for reproducible selection.
            min_samples_per_class: Minimum retained samples per class when possible.

        Returns:
            Sorted original sample indices for the stratified subset.

        Raises:
            ValueError: If `rate` is outside `(0, 1]`.
        """

        if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, Real):
            raise TypeError("subsample rate must be a finite real number.")
        resolved_rate = float(rate)
        if not np.isfinite(resolved_rate) or not 0.0 < resolved_rate <= 1.0:
            raise ValueError("subsample rate must be in (0, 1].")
        if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, Integral):
            raise TypeError("random_state must be an integer.")
        if isinstance(min_samples_per_class, (bool, np.bool_)) or not isinstance(
            min_samples_per_class, Integral
        ):
            raise TypeError("min_samples_per_class must be an integer.")
        if int(min_samples_per_class) < 1:
            raise ValueError("min_samples_per_class must be >= 1.")
        resolved_random_state = int(random_state)
        resolved_min_samples = int(min_samples_per_class)
        if resolved_rate >= 1.0:
            return np.arange(len(self.y), dtype=int)
        if self.metadata.get("target_type") == REGRESSION_TARGET:
            n_take = min(len(self.y), max(3, int(np.floor(len(self.y) * resolved_rate))))
            return regression_subsample_indices(
                self.y,
                n_take=n_take,
                random_state=resolved_random_state,
            )
        return stratified_label_indices(
            self.y,
            rate=resolved_rate,
            random_state=resolved_random_state,
            min_samples_per_class=resolved_min_samples,
            label_names=self.metadata.get("label_names"),
            target_type=self.metadata.get("target_type", "auto"),
            target_names=self.metadata.get("target_names"),
        )

    def subset(self, indices: Any, metadata: Optional[Dict[str, Any]] = None) -> "BenchmarkDataset":
        """Create a dataset subset by original sample indices.

        Args:
            indices: Sample indices to retain.
            metadata: Additional metadata to merge into the subset.

        Returns:
            Validated dataset containing only the selected samples.
        """

        _reject_reserved_metadata(metadata, _STRUCTURAL_METADATA_KEYS, owner="subset metadata")
        index_array = validate_row_indices(indices, len(self.y), name="subset indices")
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
            }
        )
        hierarchy = merged_metadata.get("label_hierarchy")
        if hierarchy is not None:
            paths = hierarchy.get("paths")
            if paths is not None:
                hierarchy = dict(hierarchy)
                hierarchy["paths"] = np.asarray(paths, dtype=object)[index_array].tolist()
                merged_metadata["label_hierarchy"] = hierarchy
        for key in merged_metadata.get(_ROW_ALIGNED_METADATA_KEY, []):
            values = merged_metadata.get(key)
            if values is not None:
                merged_metadata[key] = _take_aligned_metadata(values, index_array)
        merged_metadata["sample_indices"] = sample_indices
        merged_metadata["parent_row_positions"] = index_array.tolist()
        merged_metadata = _register_row_aligned_metadata(
            merged_metadata,
            "sample_indices",
            "parent_row_positions",
        )
        target_views = merged_metadata.get("target_views")
        if target_views is not None:
            subset_views = {}
            for name, view in target_views.items():
                labels = labels_from_jsonable(
                    view["targets"],
                    label_names=view.get("label_names"),
                    target_type=view.get("target_type", "auto"),
                    target_names=view.get("target_names"),
                )
                subset_labels = labels[index_array]
                subset_views[name] = _target_view_entry(
                    name=str(name),
                    targets=subset_labels,
                    target_type=view.get("target_type", "auto"),
                    label_names=view.get("label_names"),
                    target_names=view.get("target_names"),
                    metadata=view.get("metadata"),
                    X=_take_samples(self.X, index_array),
                    modality=self.modality,
                    input_col=self.input_col,
                    label_col=self.label_col,
                    base_metadata=merged_metadata,
                    dataset_type=type(self),
                    identity=self.identity,
                )
            merged_metadata["target_views"] = subset_views
        merged_metadata.update(metadata or {})
        dataset = type(self)(
            X=_take_samples(self.X, index_array),
            y=self.y[index_array],
            modality=self.modality,
            identity=self._derive_identity(
                "subset", {"indices": index_array, "metadata": metadata or {}}
            ),
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

        labels = target_summary(
            self.y,
            label_names=self.metadata.get("label_names"),
            target_type=self.metadata.get("target_type", "auto"),
            target_names=self.metadata.get("target_names"),
        )
        if self.metadata.get("label_catalog") is not None:
            labels["label_catalog"] = list(self.metadata["label_catalog"])
        report_metadata = dict(self.metadata)
        report_metadata.pop(_ROW_ALIGNED_METADATA_KEY, None)
        groups = report_metadata.pop("groups", None)
        target_views = report_metadata.pop("target_views", None)
        unit_ids = report_metadata.pop("unit_ids", None)
        parent_ids = report_metadata.pop("parent_ids", None)
        unit_positions = report_metadata.pop("unit_positions", None)
        unit_spans = report_metadata.pop("unit_spans", None)
        unit_coordinates = report_metadata.pop("unit_coordinates", None)
        unit_provenance = report_metadata.pop("unit_provenance", None)
        unit_annotations = report_metadata.pop("unit_annotations", None)
        summary = {
            "n_samples": int(len(self.y)),
            "n_classes": labels["n_classes"],
            "class_counts": labels["class_counts"],
            "target_type": labels["target_type"],
            "modality": self.modality,
            "input_col": self.input_col,
            "label_col": self.label_col,
            "label_view": self.active_label_view(),
            "target_view": self.active_target_view(),
            "identity": self.identity.descriptor(self.identity_key()),
            "metadata": report_metadata,
        }
        if target_views is not None:
            summary["available_target_views"] = [
                {
                    "name": str(name),
                    "target_type": view.get("target_type", "auto"),
                    "metadata": dict(view.get("metadata", {})),
                }
                for name, view in target_views.items()
            ]
        if self.metadata.get("unit_embeddings"):
            summary["units"] = {
                "provided": True,
                "unit_type": self.metadata.get("unit_type", "unit"),
                "n_units": int(len(self.y)),
                "has_parent_ids": parent_ids is not None,
                "has_positions": unit_positions is not None,
                "has_spans": unit_spans is not None,
                "has_coordinates": unit_coordinates is not None,
                "has_provenance": unit_provenance is not None,
                "n_distinct_units": (
                    int(len({semantic_label_key(value) for value in unit_ids}))
                    if unit_ids is not None
                    else int(len(self.y))
                ),
            }
            if parent_ids is not None:
                summary["units"]["n_parents"] = int(
                    len({semantic_label_key(value) for value in parent_ids})
                )
        if unit_annotations is not None:
            summary["structured_units"] = {
                "provided": True,
                "unit_type": self.metadata.get("unit_annotation_unit_type", "unit"),
                "task_family": self.metadata.get("unit_annotation_task_family"),
                "n_parents": int(len(unit_annotations)),
                "n_units": int(
                    sum(len(annotation.get("labels", [])) for annotation in unit_annotations)
                ),
            }
        if groups is not None:
            summary["grouping"] = {
                "provided": True,
                "name": self.metadata.get("group_name", "group"),
                "n_groups": int(len({semantic_label_key(value) for value in groups})),
            }
        for key in (
            "label_names",
            "label_catalog",
            "labelset_counts",
            "mean_label_cardinality",
            "label_density",
            "n_targets",
            "target_names",
            "target_means",
            "target_variances",
            "constant_targets",
            "nonconstant_targets",
        ):
            if key in labels:
                summary[key] = labels[key]
        return summary

    def identity_key(self) -> str:
        """Resolve and memoize the dataset's explicit identity key.

        Treat the dataset and its identity-bearing values as immutable after this
        method is called.
        """

        if self._identity_key_cache is None:
            self._identity_key_cache = self.identity.resolve(self._identity_payload())
        return self._identity_key_cache

    def _identity_payload(self) -> Dict[str, Any]:
        return {
            "X": self.X,
            "y": self.y,
            "modality": self.modality,
            "input_col": self.input_col,
            "label_col": self.label_col,
            "metadata": self.metadata,
        }

    def _derive_identity(self, operation: str, recipe: Any) -> DatasetIdentity:
        return DatasetIdentity.derived(self.identity_key(), operation, recipe)


class EmbeddingUnitDataset(BenchmarkDataset):
    """Generic embedding dataset for structured units such as boxes or tokens."""

    @classmethod
    def from_units(
        cls,
        embeddings: Any,
        labels: Any,
        unit_ids: Any,
        *,
        identity: DatasetIdentity,
        parent_ids: Any = None,
        unit_type: str = "unit",
        positions: Any = None,
        spans: Any = None,
        coordinates: Any = None,
        provenance: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
        target_views: Optional[Iterable[TargetView]] = None,
    ) -> "EmbeddingUnitDataset":
        """Create a structured unit dataset from aligned embedding rows."""

        matrix = ensure_numeric_matrix(embeddings, "unit embeddings", allow_sparse=True)
        n_units = int(matrix.shape[0])
        resolved_unit_ids = _validated_unit_ids(unit_ids, n_units=n_units)
        resolved_parent_ids = _optional_aligned_values(
            parent_ids,
            n_units=n_units,
            name="parent_ids",
        )
        merged_metadata = {
            "precomputed_embeddings": True,
            "unit_embeddings": True,
            "unit_type": str(unit_type),
            "unit_ids": resolved_unit_ids,
        }
        if resolved_parent_ids is not None:
            merged_metadata["parent_ids"] = resolved_parent_ids
        for key, values in (
            ("unit_positions", positions),
            ("unit_spans", spans),
            ("unit_coordinates", coordinates),
            ("unit_provenance", provenance),
        ):
            resolved = _optional_aligned_values(values, n_units=n_units, name=key)
            if resolved is not None:
                merged_metadata[key] = resolved
        row_keys = ["unit_ids"]
        row_keys.extend(
            key
            for key in (
                "parent_ids",
                "unit_positions",
                "unit_spans",
                "unit_coordinates",
                "unit_provenance",
            )
            if key in merged_metadata
        )
        merged_metadata = _register_row_aligned_metadata(merged_metadata, *row_keys)
        merged_metadata = _merge_user_metadata(merged_metadata, metadata)
        dataset = cast(
            EmbeddingUnitDataset,
            cls._from_prepared_embeddings(
                matrix,
                labels,
                identity=identity,
                metadata=merged_metadata,
                label_names=label_names,
                target_type=target_type,
                target_names=target_names,
            ),
        )
        if resolved_parent_ids is not None:
            dataset = cast(
                EmbeddingUnitDataset,
                dataset.with_groups(resolved_parent_ids, name="parent_id"),
            )
        if target_views is not None:
            dataset = cast(EmbeddingUnitDataset, dataset.with_target_views(target_views))
        return dataset


def _normalize_unit_annotations(
    annotations: Iterable[UnitAnnotation],
    n_samples: int,
    unit_type: str,
) -> list[Dict[str, Any]]:
    resolved = list(annotations)
    if len(resolved) != n_samples:
        raise ValueError(
            "unit annotations must align to dataset samples; "
            f"got {len(resolved)} annotations for {n_samples} samples."
        )
    entries = []
    expected_schema: Optional[Dict[str, Any]] = None
    for index, annotation in enumerate(resolved):
        if not isinstance(annotation, UnitAnnotation):
            raise ValueError("unit annotations must contain UnitAnnotation entries.")
        labels, target_metadata = normalize_targets(
            annotation.labels,
            label_names=annotation.label_names,
            target_type=annotation.target_type,
            target_names=annotation.target_names,
        )
        resolved_target_type = str(target_metadata["target_type"])
        resolved_label_names = (
            list(target_metadata["label_names"])
            if resolved_target_type == MULTI_LABEL_TARGET
            else None
        )
        resolved_target_names = (
            list(target_metadata["target_names"])
            if resolved_target_type == REGRESSION_TARGET
            else None
        )
        if resolved_target_type == REGRESSION_TARGET and target_metadata["n_targets"] == 1:
            labels = np.asarray(labels, dtype=float).reshape(-1)
        unit_count = int(len(labels))
        if unit_count < 1:
            raise ValueError(
                f"{unit_type} annotations for sample {index} must contain at least one unit."
            )
        schema = {
            "target_type": resolved_target_type,
            "label_names": resolved_label_names,
            "target_names": resolved_target_names,
        }
        if expected_schema is None:
            expected_schema = schema
        else:
            _validate_unit_annotation_schema(
                expected=expected_schema,
                actual=schema,
                sample_index=index,
                unit_type=unit_type,
            )
        entry = {
            "labels": labels_to_jsonable(
                labels,
                label_names=resolved_label_names,
                target_type=resolved_target_type,
                target_names=resolved_target_names,
            ),
            "label_names": resolved_label_names,
            "target_type": resolved_target_type,
            "target_names": resolved_target_names,
            "metadata": dict(annotation.metadata),
        }
        resolved_unit_ids = _validated_local_unit_ids(
            annotation.unit_ids,
            n_units=unit_count,
            sample_index=index,
            unit_type=unit_type,
        )
        if resolved_unit_ids is not None:
            entry["unit_ids"] = resolved_unit_ids
        for key, values in (
            ("positions", annotation.positions),
            ("spans", annotation.spans),
            ("coordinates", annotation.coordinates),
            ("provenance", annotation.provenance),
        ):
            resolved_values = _optional_aligned_values(values, n_units=unit_count, name=key)
            if resolved_values is not None:
                entry[key] = resolved_values
        entries.append(entry)
    return entries


def _validate_unit_annotation_schema(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    sample_index: int,
    unit_type: str,
) -> None:
    if actual["target_type"] != expected["target_type"]:
        raise ValueError(
            f"{unit_type} annotation target_type for sample {sample_index} is "
            f"{actual['target_type']!r}; expected {expected['target_type']!r} to match sample 0."
        )
    if actual["target_type"] == MULTI_LABEL_TARGET and (
        actual["label_names"] != expected["label_names"]
    ):
        raise ValueError(
            f"{unit_type} annotation label_names for sample {sample_index} do not match "
            "sample 0. Provide the same ordered label_names for every multi-label parent."
        )
    if actual["target_type"] == REGRESSION_TARGET and (
        actual["target_names"] != expected["target_names"]
    ):
        raise ValueError(
            f"{unit_type} annotation target_names for sample {sample_index} do not match "
            "sample 0; regression target count and ordered names must be identical."
        )


def _validated_local_unit_ids(
    values: Any,
    n_units: int,
    sample_index: int,
    unit_type: str,
) -> Optional[list[Any]]:
    unit_ids = _optional_aligned_values(values, n_units=n_units, name="unit_ids")
    if unit_ids is None:
        return None
    try:
        canonical_ids = _exact_identifier_keys(unit_ids, name="unit_ids")
    except ValueError as exc:
        raise ValueError(
            f"{unit_type} unit_ids for sample {sample_index} must contain hashable values "
            "with deterministic exact identities."
        ) from exc
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError(
            f"{unit_type} unit_ids for sample {sample_index} must be unique within the parent."
        )
    return unit_ids


def _metadata_with_target_metadata(
    metadata: Optional[Dict[str, Any]],
    label_names: Optional[Iterable[Any]],
    target_type: str,
    target_names: Optional[Iterable[str]],
) -> Dict[str, Any]:
    merged = dict(metadata or {})
    if label_names is not None:
        merged["label_names"] = list(label_names)
    if target_names is not None:
        merged["target_names"] = list(target_names)
    merged["target_type"] = target_type
    return merged


def _merge_user_metadata(
    base: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
    *,
    reserved: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Merge descriptive metadata without permitting structural overrides."""

    if metadata is None:
        return dict(base)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary when provided.")
    owned = set(base) | set(reserved or ()) | set(_STRUCTURAL_METADATA_KEYS)
    _reject_reserved_metadata(metadata, owned, owner="metadata")
    merged = dict(base)
    merged.update(metadata)
    return merged


def _reject_reserved_metadata(
    metadata: Optional[Dict[str, Any]],
    reserved: Iterable[str],
    *,
    owner: str,
) -> None:
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise TypeError(f"{owner} must be a dictionary when provided.")
    conflicts = sorted(set(metadata) & set(reserved))
    if conflicts:
        raise ValueError(f"{owner} cannot override constructor-owned structural keys: {conflicts}.")


def _register_row_aligned_metadata(metadata: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    merged = dict(metadata)
    registered = list(merged.get(_ROW_ALIGNED_METADATA_KEY, []))
    for key in keys:
        if key not in registered:
            registered.append(key)
    merged[_ROW_ALIGNED_METADATA_KEY] = registered
    return merged


def _validate_row_aligned_metadata(metadata: Dict[str, Any], n_samples: int) -> None:
    registered = metadata.get(_ROW_ALIGNED_METADATA_KEY, [])
    if isinstance(registered, (str, bytes)) or not isinstance(registered, (list, tuple)):
        raise TypeError(f"{_ROW_ALIGNED_METADATA_KEY} must be a sequence of metadata keys.")
    if any(not isinstance(key, str) or not key for key in registered):
        raise TypeError(f"{_ROW_ALIGNED_METADATA_KEY} entries must be non-empty strings.")
    if len(set(registered)) != len(registered):
        raise ValueError(f"{_ROW_ALIGNED_METADATA_KEY} must not contain duplicate keys.")
    for key in registered:
        if key not in metadata:
            raise ValueError(f"Registered row-aligned metadata key {key!r} is missing.")
        try:
            length = len(metadata[key])
        except TypeError as exc:
            raise ValueError(f"Row-aligned metadata {key!r} must be a sized sequence.") from exc
        if length != n_samples:
            raise ValueError(
                f"Row-aligned metadata {key!r} must have length {n_samples}; got {length}."
            )


def _take_aligned_metadata(values: Any, indices: np.ndarray) -> list[Any]:
    array = np.asarray(values, dtype=object)
    return array[indices].tolist()


def _normalize_target_views(
    target_views: Iterable[TargetView],
    n_samples: int,
    X: Any,
    modality: str,
    input_col: Optional[Union[str, list[str]]],
    label_col: Optional[Union[str, list[str]]],
    base_metadata: Dict[str, Any],
    dataset_type: type[BenchmarkDataset],
    identity: DatasetIdentity,
) -> Dict[str, Dict[str, Any]]:
    resolved: Dict[str, Dict[str, Any]] = {}
    for view in target_views:
        if not isinstance(view, TargetView):
            raise ValueError("target_views must contain TargetView entries.")
        if not isinstance(view.name, str) or not view.name.strip():
            raise ValueError("TargetView.name must be a non-empty string.")
        normalized_name = view.name.strip()
        if normalized_name in resolved:
            raise ValueError(f"Duplicate target view name {normalized_name!r}.")
        resolved[normalized_name] = _target_view_entry(
            name=normalized_name,
            targets=view.targets,
            target_type=view.target_type,
            label_names=view.label_names,
            target_names=view.target_names,
            metadata=view.metadata,
            X=X,
            modality=modality,
            input_col=input_col,
            label_col=label_col,
            base_metadata=base_metadata,
            dataset_type=dataset_type,
            identity=identity,
        )
        if (
            len(
                labels_from_jsonable(
                    resolved[normalized_name]["targets"],
                    label_names=resolved[normalized_name].get("label_names"),
                    target_type=resolved[normalized_name].get("target_type", "auto"),
                    target_names=resolved[normalized_name].get("target_names"),
                )
            )
            != n_samples
        ):
            raise ValueError(f"Target view {normalized_name!r} must have length {n_samples}.")
    if not resolved:
        raise ValueError("target_views must not be empty.")
    return resolved


def _target_view_entry(
    name: str,
    targets: Any,
    target_type: str,
    label_names: Optional[Iterable[Any]],
    target_names: Optional[Iterable[str]],
    metadata: Optional[Dict[str, Any]],
    X: Any,
    modality: str,
    input_col: Optional[Union[str, list[str]]],
    label_col: Optional[Union[str, list[str]]],
    base_metadata: Dict[str, Any],
    dataset_type: type[BenchmarkDataset],
    identity: DatasetIdentity,
) -> Dict[str, Any]:
    view_metadata = _metadata_with_target_metadata(
        {
            key: value
            for key, value in base_metadata.items()
            if key
            not in {
                "label_names",
                "target_names",
                "target_type",
                "target_views",
                "target_view",
            }
        },
        label_names=label_names,
        target_type=target_type,
        target_names=target_names,
    )
    candidate = dataset_type(
        X=X,
        y=coerce_label_input(targets),
        modality=modality,
        identity=identity,
        input_col=input_col,
        label_col=label_col,
        metadata=view_metadata,
    )
    candidate.validate()
    summary = target_summary(
        candidate.y,
        label_names=candidate.metadata.get("label_names"),
        target_type=candidate.metadata.get("target_type", "auto"),
        target_names=candidate.metadata.get("target_names"),
    )
    return {
        "name": str(name),
        "targets": labels_to_jsonable(
            candidate.y,
            label_names=candidate.metadata.get("label_names"),
            target_type=candidate.metadata.get("target_type", "auto"),
            target_names=candidate.metadata.get("target_names"),
        ),
        "target_type": candidate.metadata.get("target_type", "auto"),
        "label_names": candidate.metadata.get("label_names"),
        "target_names": candidate.metadata.get("target_names"),
        "metadata": dict(metadata or {}),
        "summary": summary,
    }


def _validated_unit_ids(values: Any, n_units: int) -> list[Any]:
    unit_ids = _aligned_ids(values, n_units, name="unit_ids")
    return unit_ids


def _optional_aligned_values(values: Any, n_units: int, name: str) -> Optional[list[Any]]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 0:
        raise ValueError(f"{name} must align to the unit rows.")
    if len(arr) != n_units:
        raise ValueError(f"{name} must have length {n_units}; got {len(arr)}.")
    return arr.tolist()


def _aligned_ids(values: Any, expected: int, name: str) -> list[Any]:
    if values is None:
        return list(range(expected))
    ids = np.asarray(values, dtype=object)
    if ids.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(ids) != expected:
        raise ValueError(f"{name} must have length {expected}; got {len(ids)}.")
    normalized = ids.tolist()
    canonical = _exact_identifier_keys(normalized, name=name)
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"{name} must be unique under exact typed identity.")
    return normalized


def _normalize_edge_like_index(value: Any, n_rows: Optional[int], name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} must be provided.")
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with two columns.")
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"{name} must contain source and target columns.")
    if n_rows is not None and arr.shape[0] != n_rows:
        raise ValueError(f"{name} must have {n_rows} rows; got {arr.shape[0]}.")
    return arr


def _normalize_triplets(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("triplets must be provided.")
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 2:
        raise ValueError("triplets must be a 2D array with three columns.")
    if arr.shape[1] != 3 and arr.shape[0] == 3:
        arr = arr.T
    if arr.shape[1] != 3:
        raise ValueError("triplets must contain anchor, positive, and negative columns.")
    return arr


def _compose_index_pairs(
    embeddings: Any,
    pairs: np.ndarray,
    ids: Any,
    composition: str,
    owner: str,
) -> Any:
    matrix = ensure_numeric_matrix(embeddings, owner, allow_sparse=True)
    left = _positions_for_ids(pairs[:, 0], ids=ids, n_rows=matrix.shape[0], name="pair ids")
    right = _positions_for_ids(pairs[:, 1], ids=ids, n_rows=matrix.shape[0], name="pair ids")
    left_embeddings = matrix[left]
    right_embeddings = matrix[right]
    return _compose_embedding_rows(left_embeddings, right_embeddings, composition, owner)


def _compose_triplets(
    embeddings: Any,
    triplets: np.ndarray,
    ids: Any,
    composition: str,
) -> Any:
    anchor_positive = triplets[:, [0, 1]]
    anchor_negative = triplets[:, [0, 2]]
    positive_rows = _compose_index_pairs(
        embeddings,
        anchor_positive,
        ids=ids,
        composition=composition,
        owner="triplet positive-pair embeddings",
    )
    negative_rows = _compose_index_pairs(
        embeddings,
        anchor_negative,
        ids=ids,
        composition=composition,
        owner="triplet negative-pair embeddings",
    )
    return _concat_matrices(positive_rows, negative_rows)


def _positions_for_ids(values: Any, ids: Any, n_rows: int, name: str) -> np.ndarray:
    values_array = np.asarray(values, dtype=object)
    if ids is None:
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in values_array.tolist()
        ):
            raise TypeError(
                f"{name} must contain exact integer row positions when ids are not provided."
            )
        positions = np.asarray([int(value) for value in values_array.tolist()], dtype=int)
        if np.any(positions < 0) or np.any(positions >= n_rows):
            raise ValueError(f"{name} contain row positions outside [0, {n_rows}).")
        return positions
    id_values = _aligned_ids(ids, n_rows, name="ids")
    lookup = {
        canonical: index
        for index, canonical in enumerate(_exact_identifier_keys(id_values, name="ids"))
    }
    requested = _exact_identifier_keys(values_array.tolist(), name=name)
    missing = [
        value
        for value, canonical in zip(values_array.tolist(), requested)
        if canonical not in lookup
    ]
    if missing:
        raise ValueError(f"{name} contain unknown ids: {missing[:5]}.")
    return np.asarray([lookup[canonical] for canonical in requested], dtype=int)


def _exact_identifier_keys(values: Iterable[Any], name: str) -> list[str]:
    keys = []
    for value in values:
        normalized = value.item() if hasattr(value, "item") else value
        if _is_missing_identifier(normalized):
            raise ValueError(f"{name} entries must be non-missing.")
        try:
            hash(normalized)
            keys.append(canonical_json_exact(normalized))
        except TypeError as exc:
            raise ValueError(
                f"{name} entries must be hashable and have deterministic exact identities."
            ) from exc
    return keys


def _is_missing_identifier(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return bool(np.isnan(value))
    return False


def _compose_embedding_rows(left: Any, right: Any, composition: str, owner: str) -> Any:
    if left.shape != right.shape:
        raise ValueError(f"{owner} pair components must have matching shapes.")
    if composition == "concat":
        return _concat_matrices(left, right)
    if composition == "hadamard":
        return left.multiply(right) if is_sparse_matrix(left) else left * right
    if composition == "average":
        return (left + right) * 0.5
    if composition == "abs_diff":
        diff = left - right
        if is_sparse_matrix(diff):
            diff = diff.copy()
            diff.data = np.abs(diff.data)
            return diff
        return np.abs(diff)
    raise ValueError("composition must be one of: 'concat', 'hadamard', 'abs_diff', 'average'.")


def _concat_matrices(left: Any, right: Any) -> Any:
    if is_sparse_matrix(left) or is_sparse_matrix(right):
        from scipy import sparse

        return sparse.hstack([left, right], format="csr")
    return np.concatenate([np.asarray(left), np.asarray(right)], axis=1)


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


def _strict_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an exact positive integer.")
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be > 0.")
    return resolved


def _strict_positive_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real number.")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.")
    return resolved


def _validated_numeric_sample(
    value: Any,
    name: str,
    *,
    allowed_ranks: tuple[int, ...],
    allow_bool: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in allowed_ranks:
        expected = " or ".join(str(rank) for rank in allowed_ranks)
        raise ValueError(f"{name} must have rank {expected}; got shape {array.shape}.")
    if array.size == 0 or any(size < 1 for size in array.shape):
        raise ValueError(f"{name} must be non-empty on every axis.")
    boolean = np.issubdtype(array.dtype, np.bool_)
    if (not np.issubdtype(array.dtype, np.number) and not boolean) or (boolean and not allow_bool):
        raise TypeError(f"{name} must contain numeric values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_timestamp_values(values: np.ndarray) -> None:
    """Reject missing and non-finite timestamp cells without coercing their type."""

    if values.dtype.kind in {"M", "m"}:
        if bool(np.any(np.isnat(values))):
            raise ValueError("timestamps must not contain missing or NaT values.")
        return
    if np.issubdtype(values.dtype, np.number):
        if not bool(np.all(np.isfinite(values))):
            raise ValueError("timestamps must contain only finite values.")
        return
    for value in values.reshape(-1).tolist():
        if _is_missing_scalar(value):
            raise ValueError("timestamps must not contain missing or NaT values.")
        if isinstance(value, np.datetime64) or isinstance(value, np.timedelta64):
            if bool(np.isnat(value)):
                raise ValueError("timestamps must not contain missing or NaT values.")
        elif isinstance(value, (Real, complex, np.number)) and not bool(
            np.isfinite(cast(Any, value))
        ):
            raise ValueError("timestamps must contain only finite values.")
        elif hasattr(value, "is_finite") and callable(value.is_finite):
            if not bool(value.is_finite()):
                raise ValueError("timestamps must contain only finite values.")


def _validated_frame_rates(value: np.ndarray, n_samples: int) -> np.ndarray:
    if value.ndim != 1 or len(value) != n_samples:
        raise ValueError(f"frame_rate must be scalar or have length {n_samples}.")
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
        raise TypeError("frame_rate values must be finite positive real numbers.")
    rates = value.astype(float, copy=False)
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("frame_rate values must be finite and > 0.")
    return rates


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
