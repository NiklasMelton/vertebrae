"""Image-aligned semantic, instance, and panoptic segmentation datasets."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np

from vertebrae.datasets.identity import DatasetIdentity
from vertebrae.execution.jobs import SampleBatch, ShardSpec


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
        instance = None if self.instance is None else np.asarray(self.instance)
        if instance is not None and instance.shape != semantic.shape:
            raise ValueError("Instance and semantic masks must have the same shape.")
        return SegmentationAnnotation(
            semantic=semantic,
            instance=instance,
            ignore_labels=tuple(self.ignore_labels),
            class_metadata=dict(self.class_metadata),
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
        if len(instances) != len(semantics):
            raise ValueError("instance_masks and semantic_masks must have the same length.")
        annotations = [
            SegmentationAnnotation(
                semantic=semantic,
                instance=instance,
                ignore_labels=tuple(ignore_labels),
                class_metadata=dict(class_metadata or {}),
            )
            for semantic, instance in zip(semantics, instances)
        ]
        dataset = cls(
            X=_coerce_samples(images),
            annotations=annotations,
            identity=identity,
            metadata=metadata or {},
        )
        dataset.validate()
        return dataset

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
        if len(self.X) != len(self.annotations):
            raise ValueError(
                f"images and annotations must have the same length; got {len(self.X)} "
                f"and {len(self.annotations)}."
            )
        if not self.annotations:
            raise ValueError("SegmentationDataset must contain at least one image.")
        self.annotations = [annotation.normalized() for annotation in self.annotations]

    def iter_batches(
        self,
        batch_size: int,
        shard: Optional[ShardSpec] = None,
    ) -> Iterable[SampleBatch]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        indices = (shard or ShardSpec()).indices(len(self.annotations))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield SampleBatch(indices=batch_indices, X=_take(self.X, batch_indices))

    def subset(self, indices: Any) -> "SegmentationDataset":
        index_array = np.asarray(indices, dtype=int)
        return SegmentationDataset(
            X=_take(self.X, index_array),
            annotations=[self.annotations[int(index)] for index in index_array],
            identity=DatasetIdentity.derived(
                self.identity_key(), "subset", {"indices": index_array}
            ),
            metadata={**self.metadata, "sample_indices": index_array.tolist()},
        )

    def summary(self) -> Dict[str, Any]:
        labels = set()
        instance_ids = set()
        for annotation in self.annotations:
            labels.update(np.unique(annotation.semantic).tolist())
            if annotation.instance is not None:
                instance_ids.update(np.unique(annotation.instance).tolist())
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
