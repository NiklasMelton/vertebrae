"""Image-aligned semantic, instance, and panoptic segmentation datasets."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from vertebrae.cache.fingerprint import canonical_json_exact
from vertebrae.datasets.identity import DatasetIdentity
from vertebrae.execution.jobs import SampleBatch, ShardSpec
from vertebrae.utils.semantic_labels import semantic_label_key
from vertebrae.utils.validation import validate_row_indices


@dataclass(frozen=True)
class SegmentationAnnotation:
    """Canonical raster annotation for one image."""

    semantic: Any
    instance: Optional[Any] = None
    ignore_labels: tuple[Any, ...] = ()
    class_metadata: Dict[Any, Dict[str, Any]] = field(default_factory=dict)

    def normalized(self) -> "SegmentationAnnotation":
        semantic = np.asarray(self.semantic)
        if semantic.ndim != 2:
            raise ValueError("Segmentation semantic masks must be two-dimensional.")
        if semantic.size == 0 or any(size < 1 for size in semantic.shape):
            raise ValueError("Segmentation semantic masks must contain at least one pixel.")
        instance = None if self.instance is None else np.asarray(self.instance)
        if instance is not None and instance.shape != semantic.shape:
            raise ValueError("Instance and semantic masks must have the same shape.")
        semantic_keys = _validate_mask_cells(semantic, "Segmentation semantic masks")
        if instance is not None:
            _validate_mask_cells(instance, "Segmentation instance masks")
        ignore_labels = tuple(self.ignore_labels)
        for value in ignore_labels:
            _stable_scalar_key(value, "Segmentation ignore_labels")
        class_metadata = _normalize_class_metadata(self.class_metadata, semantic_keys)
        return SegmentationAnnotation(
            semantic=semantic,
            instance=instance,
            ignore_labels=ignore_labels,
            class_metadata=class_metadata,
        )


@dataclass
class SegmentationDataset:
    """Image samples paired with raster segmentation annotations."""

    X: Any
    annotations: list[SegmentationAnnotation]
    identity: DatasetIdentity
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality: str = "segmentation"
    _identity_key_cache: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_arrays(
        cls,
        images: Any,
        semantic_masks: Iterable[Any],
        *,
        identity: DatasetIdentity,
        instance_masks: Optional[Iterable[Any]] = None,
        ignore_labels: Iterable[Any] = (),
        class_metadata: Optional[Dict[Any, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SegmentationDataset":
        semantics = list(semantic_masks)
        instances = [None] * len(semantics) if instance_masks is None else list(instance_masks)
        resolved_ignore_labels = tuple(ignore_labels)
        if len(instances) != len(semantics):
            raise ValueError("instance_masks and semantic_masks must have the same length.")
        annotations = [
            SegmentationAnnotation(
                semantic=semantic,
                instance=instance,
                ignore_labels=resolved_ignore_labels,
                class_metadata=dict(class_metadata or {}),
            )
            for semantic, instance in zip(semantics, instances)
        ]
        return cls(
            X=_coerce_samples(images),
            annotations=annotations,
            identity=identity,
            metadata=metadata or {},
        )

    @classmethod
    def from_image_paths(
        cls,
        paths: Any,
        semantic_masks: Iterable[Any],
        *,
        identity: DatasetIdentity,
        **kwargs: Any,
    ) -> "SegmentationDataset":
        metadata = {"source": "segmentation_image_paths", **kwargs.pop("metadata", {})}
        return cls.from_arrays(
            np.asarray(paths, dtype=object),
            semantic_masks,
            identity=identity,
            metadata=metadata,
            **kwargs,
        )

    def validate(self) -> None:
        if not isinstance(self.identity, DatasetIdentity):
            raise TypeError("identity must be a DatasetIdentity.")
        self.annotations = list(self.annotations)
        self.metadata = dict(self.metadata)
        if len(self.X) != len(self.annotations):
            raise ValueError(
                f"images and annotations must have the same length; got {len(self.X)} "
                f"and {len(self.annotations)}."
            )
        if not self.annotations:
            raise ValueError("SegmentationDataset must contain at least one image.")
        if any(
            not isinstance(annotation, SegmentationAnnotation) for annotation in self.annotations
        ):
            raise TypeError("annotations must contain SegmentationAnnotation values.")
        self.annotations = [annotation.normalized() for annotation in self.annotations]
        sample_indices = self.metadata.get("sample_indices")
        if sample_indices is not None:
            normalized_indices = list(sample_indices)
            if len(normalized_indices) != len(self.annotations):
                raise ValueError(
                    "metadata sample_indices must have one entry per image; "
                    f"got {len(normalized_indices)} and {len(self.annotations)}."
                )
            if any(
                isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
                for index in normalized_indices
            ):
                raise TypeError("metadata sample_indices entries must be integers.")
            if any(int(index) < 0 for index in normalized_indices):
                raise ValueError("metadata sample_indices entries must be non-negative.")
            self.metadata["sample_indices"] = [int(index) for index in normalized_indices]

    def iter_batches(
        self,
        batch_size: int,
        shard: Optional[ShardSpec] = None,
    ) -> Iterable[SampleBatch]:
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be an integer.")
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        indices = (shard or ShardSpec()).indices(len(self.annotations))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield SampleBatch(indices=batch_indices, X=_take(self.X, batch_indices))

    def subset(self, indices: Any) -> "SegmentationDataset":
        raw_indices = np.asarray(indices, dtype=object)
        if raw_indices.ndim == 1 and raw_indices.size == 0:
            raise ValueError("SegmentationDataset subsets must contain at least one image.")
        index_array = validate_row_indices(indices, len(self.annotations), name="subset indices")
        parent_indices = self.metadata.get("sample_indices")
        if parent_indices is None:
            sample_indices = index_array.tolist()
        else:
            sample_indices = np.asarray(parent_indices, dtype=int)[index_array].tolist()
        return SegmentationDataset(
            X=_take(self.X, index_array),
            annotations=[self.annotations[int(index)] for index in index_array],
            identity=DatasetIdentity.derived(
                self.identity_key(), "subset", {"indices": index_array}
            ),
            metadata={
                **self.metadata,
                "subset": True,
                "parent_n_images": len(self.annotations),
                "sample_indices": sample_indices,
                "parent_row_positions": index_array.tolist(),
            },
        )

    def summary(self) -> Dict[str, Any]:
        labels: set[str] = set()
        instance_ids: set[str] = set()
        for annotation in self.annotations:
            labels.update(
                semantic_label_key(value) for value in annotation.semantic.reshape(-1).tolist()
            )
            if annotation.instance is not None:
                instance_ids.update(
                    semantic_label_key(value) for value in annotation.instance.reshape(-1).tolist()
                )
        return {
            "n_images": len(self.annotations),
            "n_classes_raw": len(labels),
            "n_instances_raw": len(instance_ids),
            "modality": self.modality,
            "identity": self.identity.descriptor(self.identity_key()),
            "metadata": self.metadata,
        }

    def identity_key(self) -> str:
        if self._identity_key_cache is None:
            self._identity_key_cache = self.identity.resolve(
                {
                    "X": self.X,
                    "annotations": [
                        {
                            "semantic": annotation.semantic,
                            "instance": annotation.instance,
                            "ignore_labels": annotation.ignore_labels,
                            "class_metadata": annotation.class_metadata,
                        }
                        for annotation in self.annotations
                    ],
                    "metadata": self.metadata,
                }
            )
        return self._identity_key_cache


def _coerce_samples(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    items = list(value)
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def _take(value: Any, indices: np.ndarray) -> Any:
    if isinstance(value, np.ndarray):
        return value[indices]
    return [value[int(index)] for index in indices]


def _validate_mask_cells(mask: np.ndarray, name: str) -> set[str]:
    keys: set[str] = set()
    for value in mask.reshape(-1).tolist():
        keys.add(_stable_scalar_key(value, name))
    return keys


def _stable_scalar_key(value: Any, name: str) -> str:
    normalized = value.item() if isinstance(value, np.generic) else value
    if normalized is None or _is_missing_scalar(normalized):
        raise ValueError(f"{name} must contain non-missing scalar values.")
    if isinstance(normalized, (Mapping, list, tuple, set, frozenset, np.ndarray)):
        raise TypeError(f"{name} must contain scalar values.")
    try:
        hash(normalized)
        return canonical_json_exact(normalized)
    except TypeError as exc:
        raise TypeError(
            f"{name} values must be hashable and have stable exact identities."
        ) from exc


def _is_missing_scalar(value: Any) -> bool:
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _normalize_class_metadata(
    value: Any,
    semantic_keys: set[str],
) -> Dict[Any, Dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("Segmentation class_metadata must be a mapping.")
    normalized: Dict[Any, Dict[str, Any]] = {}
    for label, metadata in value.items():
        key = _stable_scalar_key(label, "Segmentation class_metadata labels")
        if key not in semantic_keys:
            raise ValueError(
                f"Segmentation class_metadata label {label!r} is not present in the semantic mask."
            )
        if not isinstance(metadata, Mapping):
            raise TypeError("Segmentation class_metadata values must be mappings.")
        item = dict(metadata)
        try:
            canonical_json_exact(item)
        except TypeError as exc:
            raise ValueError(
                "Segmentation class_metadata values must have stable exact JSON identities."
            ) from exc
        normalized[label] = item
    return normalized
