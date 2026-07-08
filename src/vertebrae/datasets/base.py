"""Dataset abstraction for benchmark inputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional, Union, cast

import numpy as np

from vertebrae.cache.fingerprint import fingerprint_array_like
from vertebrae.execution.jobs import SampleBatch, ShardSpec
from vertebrae.utils.labels import (
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
    stratified_label_indices,
    target_summary,
)
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


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
    input_col: Optional[Union[str, list[str]]] = None
    label_col: Optional[Union[str, list[str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        X: Any,
        y: Any,
        modality: str,
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
            metadata=_metadata_with_target_metadata(
                metadata,
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
        merged_metadata.update(metadata or {})
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
        merged_metadata.update(metadata or {})
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
        merged_metadata.update(metadata or {})
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

        label_array = coerce_label_input(labels)
        merged_metadata = {
            "source": "audio_arrays",
            "sampling_rate": int(sampling_rate),
        }
        merged_metadata.update(metadata or {})
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
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
        merged_metadata.update(metadata or {})
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

        merged_metadata = {"source": "time_series"}
        merged_metadata.update(metadata or {})
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        payload: Dict[str, Any] = {"series": np.asarray(series)}
        if observed_mask is not None:
            payload["observed_mask"] = np.asarray(observed_mask)
        if time_features is not None:
            payload["time_features"] = np.asarray(time_features)
        if timestamps is not None:
            payload["timestamps"] = np.asarray(timestamps)
        dataset = cls(
            X=payload,
            y=coerce_label_input(labels),
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
        merged_metadata.update(metadata or {})
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
            metadata=merged_metadata,
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_graphs(
        cls,
        graphs: Any,
        labels: Any,
        metadata: Optional[Dict[str, Any]] = None,
        label_names: Optional[Iterable[Any]] = None,
        target_type: str = "auto",
        target_names: Optional[Iterable[str]] = None,
    ) -> "BenchmarkDataset":
        """Create a graph dataset from aligned graph objects."""

        merged_metadata = {"source": "graphs"}
        merged_metadata.update(metadata or {})
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

        merged_metadata = {"precomputed_embeddings": True}
        merged_metadata.update(metadata or {})
        merged_metadata = _metadata_with_target_metadata(
            merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )
        dataset = cls(
            X=embeddings if is_sparse_matrix(embeddings) else np.asarray(embeddings),
            y=coerce_label_input(labels),
            modality="embeddings",
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkDataset":
        """Create a grouped token dataset from precomputed segmentation features."""

        merged_metadata = {
            "precomputed_embeddings": True,
            "segmentation_embeddings": True,
            **(metadata or {}),
        }
        return cls.from_embeddings(
            embeddings,
            labels,
            metadata=merged_metadata,
        ).with_groups(image_ids, name="image_id")

    @classmethod
    def from_embedding_units(
        cls,
        embeddings: Any,
        labels: Any,
        unit_ids: Any,
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
        merged_metadata.update(metadata or {})
        return cls.from_embeddings(
            matrix,
            labels,
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
        merged_metadata.update(metadata or {})
        return cls.from_embeddings(
            matrix,
            labels,
            metadata=merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    @classmethod
    def from_edge_embeddings(
        cls,
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
        merged_metadata.update(metadata or {})
        return cls.from_embeddings(
            matrix,
            labels,
            metadata=merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    @classmethod
    def from_pair_embeddings(
        cls,
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
        merged_metadata.update(metadata or {})
        return cls.from_embeddings(
            matrix,
            labels,
            metadata=merged_metadata,
            label_names=label_names,
            target_type=target_type,
            target_names=target_names,
        )

    @classmethod
    def from_triplet_embeddings(
        cls,
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
        merged_metadata.update(metadata or {})
        return cls.from_embeddings(
            matrix,
            labels,
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
        if label_metadata["target_type"] == "multi_label":
            self.metadata["label_names"] = list(label_metadata["label_names"])
            self.metadata.pop("target_names", None)
        elif label_metadata["target_type"] == REGRESSION_TARGET:
            self.metadata["target_names"] = list(label_metadata["target_names"])
            self.metadata.pop("label_names", None)
        else:
            self.metadata.pop("label_names", None)
            self.metadata.pop("target_names", None)
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

        return class_counts(
            self.y,
            label_names=self.metadata.get("label_names"),
            target_type=self.metadata.get("target_type", "auto"),
            target_names=self.metadata.get("target_names"),
        )

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
        dataset = type(self)(
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
        )
        metadata = dict(self.metadata)
        metadata["target_views"] = resolved
        dataset = type(self)(
            X=self.X,
            y=coerce_label_input(self.y),
            modality=self.modality,
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

    def with_groups(self, groups: Any, name: str = "group") -> "BenchmarkDataset":
        """Return a dataset with aligned independence-group identifiers."""

        group_array = np.asarray(groups)
        if group_array.ndim != 1:
            raise ValueError("groups must be one-dimensional.")
        if len(group_array) != len(self.y):
            raise ValueError(
                f"groups and samples must have the same length; got {len(group_array)} "
                f"and {len(self.y)}."
            )
        for value in group_array:
            try:
                hash(value.item() if hasattr(value, "item") else value)
            except TypeError as exc:
                raise ValueError("groups values must be hashable.") from exc
        metadata = dict(self.metadata)
        metadata["groups"] = group_array.tolist()
        metadata["group_name"] = str(name)
        dataset = type(self)(
            X=self.X,
            y=coerce_label_input(self.y),
            modality=self.modality,
            input_col=self.input_col,
            label_col=self.label_col,
            metadata=metadata,
        )
        dataset.validate()
        return dataset

    def groups(self) -> Optional[np.ndarray]:
        """Return aligned independence groups when configured."""

        values = self.metadata.get("groups")
        return None if values is None else np.asarray(values)

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
        if self.metadata.get("target_type") == REGRESSION_TARGET:
            return _random_subsample_indices(
                n_samples=len(self.y),
                rate=rate,
                random_state=random_state,
            )
        return stratified_label_indices(
            self.y,
            rate=rate,
            random_state=random_state,
            min_samples_per_class=min_samples_per_class,
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
        groups = merged_metadata.get("groups")
        if groups is not None:
            merged_metadata["groups"] = np.asarray(groups, dtype=object)[index_array].tolist()
        relational_unit = str(merged_metadata.get("relational_unit", ""))
        relational_metadata_keys: Dict[str, tuple[str, ...]] = {
            "node": ("node_ids",),
            "entity": ("entity_ids",),
            "edge": ("edge_index",),
            "pair": ("pair_ids",),
            "triplet": ("triplet_ids",),
        }
        aligned_metadata_keys = relational_metadata_keys.get(relational_unit, ())
        for key in aligned_metadata_keys:
            values = merged_metadata.get(key)
            if values is not None:
                merged_metadata[key] = np.asarray(values, dtype=object)[index_array].tolist()
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
                )
            merged_metadata["target_views"] = subset_views
        merged_metadata.update(metadata or {})
        dataset = type(self)(
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

        labels = target_summary(
            self.y,
            label_names=self.metadata.get("label_names"),
            target_type=self.metadata.get("target_type", "auto"),
            target_names=self.metadata.get("target_names"),
        )
        report_metadata = dict(self.metadata)
        groups = report_metadata.pop("groups", None)
        target_views = report_metadata.pop("target_views", None)
        unit_ids = report_metadata.pop("unit_ids", None)
        parent_ids = report_metadata.pop("parent_ids", None)
        unit_positions = report_metadata.pop("unit_positions", None)
        unit_spans = report_metadata.pop("unit_spans", None)
        unit_coordinates = report_metadata.pop("unit_coordinates", None)
        unit_provenance = report_metadata.pop("unit_provenance", None)
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
                    int(len(set(unit_ids))) if unit_ids is not None else int(len(self.y))
                ),
            }
            if parent_ids is not None:
                summary["units"]["n_parents"] = int(len(set(parent_ids)))
        if groups is not None:
            summary["grouping"] = {
                "provided": True,
                "name": self.metadata.get("group_name", "group"),
                "n_groups": int(len(set(groups))),
            }
        for key in (
            "label_names",
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


class EmbeddingUnitDataset(BenchmarkDataset):
    """Generic embedding dataset for structured units such as boxes or tokens."""

    @classmethod
    def from_units(
        cls,
        embeddings: Any,
        labels: Any,
        unit_ids: Any,
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
        merged_metadata.update(metadata or {})
        dataset = cast(
            EmbeddingUnitDataset,
            cls.from_embeddings(
                matrix,
                labels,
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


def _normalize_target_views(
    target_views: Iterable[TargetView],
    n_samples: int,
    X: Any,
    modality: str,
    input_col: Optional[Union[str, list[str]]],
    label_col: Optional[Union[str, list[str]]],
    base_metadata: Dict[str, Any],
    dataset_type: type[BenchmarkDataset],
) -> Dict[str, Dict[str, Any]]:
    resolved: Dict[str, Dict[str, Any]] = {}
    for view in target_views:
        if not isinstance(view, TargetView):
            raise ValueError("target_views must contain TargetView entries.")
        if not view.name:
            raise ValueError("TargetView.name must be a non-empty string.")
        if view.name in resolved:
            raise ValueError(f"Duplicate target view name {view.name!r}.")
        resolved[view.name] = _target_view_entry(
            name=view.name,
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
        )
        if len(labels_from_jsonable(
            resolved[view.name]["targets"],
            label_names=resolved[view.name].get("label_names"),
            target_type=resolved[view.name].get("target_type", "auto"),
            target_names=resolved[view.name].get("target_names"),
        )) != n_samples:
            raise ValueError(
                f"Target view {view.name!r} must have length {n_samples}."
            )
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
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unit_ids must be unique.")
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
    return ids.tolist()


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
        try:
            positions = values_array.astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must contain integer row positions when ids are not provided."
            ) from exc
        if np.any(positions < 0) or np.any(positions >= n_rows):
            raise ValueError(f"{name} contain row positions outside [0, {n_rows}).")
        return positions
    id_values = _aligned_ids(ids, n_rows, name="ids")
    lookup = {value: index for index, value in enumerate(id_values)}
    missing = [value for value in values_array if value not in lookup]
    if missing:
        raise ValueError(f"{name} contain unknown ids: {missing[:5]}.")
    return np.asarray([lookup[value] for value in values_array], dtype=int)


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


def _random_subsample_indices(
    n_samples: int,
    rate: float,
    random_state: int,
) -> np.ndarray:
    n_take = max(2, int(np.floor(n_samples * rate)))
    n_take = min(n_samples, n_take)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(np.arange(n_samples), size=n_take, replace=False))


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
