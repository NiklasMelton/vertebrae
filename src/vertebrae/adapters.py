"""Typed structured-unit adapters built on top of UnitAnnotation workflows."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from vertebrae.datasets.base import BenchmarkDataset, UnitAnnotation
from vertebrae.utils.labels import REGRESSION_TARGET, coerce_label_input


@dataclass
class RegionAnnotation:
    """Per-parent detection or document-layout region annotations."""

    labels: Any
    boxes: Any = None
    polygons: Any = None
    unit_ids: Any = None
    provenance: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    label_names: Optional[Iterable[Any]] = None
    target_type: str = "auto"
    target_names: Optional[Iterable[str]] = None
    coordinate_format: str = "xyxy"
    normalized: Optional[bool] = None
    page_id: Optional[Any] = None
    document_id: Optional[Any] = None


@dataclass
class SequenceAnnotation:
    """Per-parent OCR, ASR, or sequence-labeling annotations."""

    labels: Any
    unit_ids: Any = None
    positions: Any = None
    spans: Any = None
    tokens: Any = None
    provenance: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    label_names: Optional[Iterable[Any]] = None
    target_type: str = "auto"
    target_names: Optional[Iterable[str]] = None
    sequence_id: Optional[Any] = None
    utterance_id: Optional[Any] = None


@dataclass
class KeypointAnnotation:
    """Per-parent pose or keypoint annotations."""

    labels: Any
    coordinates: Any
    unit_ids: Any = None
    visibility: Any = None
    provenance: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    label_names: Optional[Iterable[Any]] = None
    target_type: str = "auto"
    target_names: Optional[Iterable[str]] = None
    person_id: Optional[Any] = None
    frame_id: Optional[Any] = None
    coordinate_space: str = "xy"
    skeleton: Any = None


@dataclass
class DepthAnnotation:
    """Per-parent sampled depth annotations for regression-style unit targets."""

    labels: Any
    coordinates: Any
    unit_ids: Any = None
    provenance: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    target_type: str = REGRESSION_TARGET
    target_names: Optional[Iterable[str]] = None
    valid: Any = None
    depth_units: Optional[str] = None
    scaling: Optional[str] = None


@dataclass
class LatentSlotAnnotation:
    """Per-parent latent-slot or generative structured-unit annotations."""

    labels: Any
    slot_ids: Any = None
    provenance: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    label_names: Optional[Iterable[Any]] = None
    target_type: str = "auto"
    target_names: Optional[Iterable[str]] = None
    source_component_ids: Any = None
    ordered: bool = True


class DetectionLayoutAdapter:
    """Normalize detection and layout annotations into structured unit annotations."""

    def __init__(self, unit_type: str = "region", task_family: str = "detection_layout") -> None:
        self.unit_type = unit_type
        self.task_family = task_family

    def attach(
        self,
        dataset: BenchmarkDataset,
        annotations: Iterable[RegionAnnotation],
    ) -> BenchmarkDataset:
        resolved = []
        for index, annotation in enumerate(annotations):
            _require_instance(annotation, RegionAnnotation, "RegionAnnotation", index)
            unit_count = _label_count(annotation.labels)
            boxes = _optional_aligned(annotation.boxes, unit_count, "boxes", index)
            polygons = _optional_aligned(annotation.polygons, unit_count, "polygons", index)
            provenance = _merge_provenance(
                base=annotation.provenance,
                n_units=unit_count,
                extras=[
                    ("box", boxes),
                    ("polygon", polygons),
                    ("page_id", annotation.page_id),
                    ("document_id", annotation.document_id),
                ],
                sample_index=index,
                field_name="provenance",
            )
            metadata = dict(annotation.metadata)
            metadata.update(
                {
                    "task_family": self.task_family,
                    "coordinate_format": annotation.coordinate_format,
                    "normalized": annotation.normalized,
                    "has_polygons": polygons is not None,
                }
            )
            resolved.append(
                UnitAnnotation(
                    labels=annotation.labels,
                    unit_ids=annotation.unit_ids,
                    coordinates=boxes if boxes is not None else polygons,
                    provenance=provenance,
                    metadata=metadata,
                    label_names=annotation.label_names,
                    target_type=annotation.target_type,
                    target_names=annotation.target_names,
                )
            )
        return dataset.with_unit_annotations(
            resolved,
            unit_type=self.unit_type,
            task_family=self.task_family,
        )


class SequenceLabelingAdapter:
    """Normalize OCR, ASR, and token-sequence annotations."""

    def __init__(self, unit_type: str = "token", task_family: str = "sequence") -> None:
        self.unit_type = unit_type
        self.task_family = task_family

    def attach(
        self,
        dataset: BenchmarkDataset,
        annotations: Iterable[SequenceAnnotation],
    ) -> BenchmarkDataset:
        resolved = []
        for index, annotation in enumerate(annotations):
            _require_instance(annotation, SequenceAnnotation, "SequenceAnnotation", index)
            unit_count = _label_count(annotation.labels)
            positions = (
                _optional_aligned(annotation.positions, unit_count, "positions", index)
                if annotation.positions is not None
                else list(range(unit_count))
            )
            spans = _optional_aligned(annotation.spans, unit_count, "spans", index)
            tokens = _optional_aligned(annotation.tokens, unit_count, "tokens", index)
            provenance = _merge_provenance(
                base=annotation.provenance,
                n_units=unit_count,
                extras=[
                    ("token_text", tokens),
                    ("sequence_id", annotation.sequence_id),
                    ("utterance_id", annotation.utterance_id),
                ],
                sample_index=index,
                field_name="provenance",
            )
            metadata = dict(annotation.metadata)
            metadata["task_family"] = self.task_family
            resolved.append(
                UnitAnnotation(
                    labels=annotation.labels,
                    unit_ids=annotation.unit_ids,
                    positions=positions,
                    spans=spans,
                    provenance=provenance,
                    metadata=metadata,
                    label_names=annotation.label_names,
                    target_type=annotation.target_type,
                    target_names=annotation.target_names,
                )
            )
        return dataset.with_unit_annotations(
            resolved,
            unit_type=self.unit_type,
            task_family=self.task_family,
        )


class KeypointAdapter:
    """Normalize keypoint annotations into structured unit annotations."""

    def __init__(self, unit_type: str = "keypoint", task_family: str = "keypoint") -> None:
        self.unit_type = unit_type
        self.task_family = task_family

    def attach(
        self,
        dataset: BenchmarkDataset,
        annotations: Iterable[KeypointAnnotation],
    ) -> BenchmarkDataset:
        resolved = []
        for index, annotation in enumerate(annotations):
            _require_instance(annotation, KeypointAnnotation, "KeypointAnnotation", index)
            unit_count = _label_count(annotation.labels)
            coordinates = _required_aligned(
                annotation.coordinates,
                unit_count,
                "coordinates",
                index,
            )
            visibility = _optional_aligned(annotation.visibility, unit_count, "visibility", index)
            provenance = _merge_provenance(
                base=annotation.provenance,
                n_units=unit_count,
                extras=[
                    ("visibility", visibility),
                    ("person_id", annotation.person_id),
                    ("frame_id", annotation.frame_id),
                ],
                sample_index=index,
                field_name="provenance",
            )
            metadata = dict(annotation.metadata)
            metadata.update(
                {
                    "task_family": self.task_family,
                    "coordinate_space": annotation.coordinate_space,
                    "skeleton": annotation.skeleton,
                }
            )
            resolved.append(
                UnitAnnotation(
                    labels=annotation.labels,
                    unit_ids=annotation.unit_ids,
                    coordinates=coordinates,
                    provenance=provenance,
                    metadata=metadata,
                    label_names=annotation.label_names,
                    target_type=annotation.target_type,
                    target_names=annotation.target_names,
                )
            )
        return dataset.with_unit_annotations(
            resolved,
            unit_type=self.unit_type,
            task_family=self.task_family,
        )


class DepthAdapter:
    """Normalize sampled depth annotations into regression-style structured units."""

    def __init__(self, unit_type: str = "depth_sample", task_family: str = "depth") -> None:
        self.unit_type = unit_type
        self.task_family = task_family

    def attach(
        self,
        dataset: BenchmarkDataset,
        annotations: Iterable[DepthAnnotation],
    ) -> BenchmarkDataset:
        resolved = []
        for index, annotation in enumerate(annotations):
            _require_instance(annotation, DepthAnnotation, "DepthAnnotation", index)
            labels = coerce_label_input(annotation.labels)
            if labels.ndim not in (1, 2):
                raise ValueError(
                    f"DepthAnnotation labels for sample {index} must be one- or two-dimensional."
                )
            unit_count = int(labels.shape[0])
            coordinates = _required_aligned(
                annotation.coordinates,
                unit_count,
                "coordinates",
                index,
            )
            valid = _optional_aligned(annotation.valid, unit_count, "valid", index)
            if valid is not None:
                valid_mask = np.asarray(valid, dtype=bool)
                labels = labels[valid_mask]
                coordinates = [coordinates[i] for i, keep in enumerate(valid_mask.tolist()) if keep]
                unit_ids = _masked_aligned(annotation.unit_ids, valid_mask, "unit_ids", index)
                provenance = _masked_aligned(annotation.provenance, valid_mask, "provenance", index)
            else:
                unit_ids = annotation.unit_ids
                provenance = annotation.provenance
            numeric_labels = np.asarray(labels, dtype=float)
            if not np.all(np.isfinite(numeric_labels)):
                raise ValueError(
                    f"DepthAnnotation labels for sample {index} must be finite for retained units."
                )
            retained_count = int(numeric_labels.shape[0])
            provenance_rows = _merge_provenance(
                base=provenance,
                n_units=retained_count,
                extras=[],
                sample_index=index,
                field_name="provenance",
            )
            metadata = dict(annotation.metadata)
            metadata.update(
                {
                    "task_family": self.task_family,
                    "depth_units": annotation.depth_units,
                    "scaling": annotation.scaling,
                    "sampling": "explicit",
                }
            )
            resolved.append(
                UnitAnnotation(
                    labels=numeric_labels,
                    unit_ids=unit_ids,
                    coordinates=coordinates,
                    provenance=provenance_rows,
                    metadata=metadata,
                    target_type=annotation.target_type,
                    target_names=annotation.target_names,
                )
            )
        return dataset.with_unit_annotations(
            resolved,
            unit_type=self.unit_type,
            task_family=self.task_family,
        )


class LatentSlotAdapter:
    """Normalize latent-slot annotations into structured unit annotations."""

    def __init__(self, unit_type: str = "latent_slot", task_family: str = "latent_slot") -> None:
        self.unit_type = unit_type
        self.task_family = task_family

    def attach(
        self,
        dataset: BenchmarkDataset,
        annotations: Iterable[LatentSlotAnnotation],
    ) -> BenchmarkDataset:
        resolved = []
        for index, annotation in enumerate(annotations):
            _require_instance(annotation, LatentSlotAnnotation, "LatentSlotAnnotation", index)
            unit_count = _label_count(annotation.labels)
            source_component_ids = _optional_aligned(
                annotation.source_component_ids,
                unit_count,
                "source_component_ids",
                index,
            )
            provenance = _merge_provenance(
                base=annotation.provenance,
                n_units=unit_count,
                extras=[("source_component_id", source_component_ids)],
                sample_index=index,
                field_name="provenance",
            )
            metadata = dict(annotation.metadata)
            metadata.update(
                {
                    "task_family": self.task_family,
                    "ordered": annotation.ordered,
                }
            )
            resolved.append(
                UnitAnnotation(
                    labels=annotation.labels,
                    unit_ids=annotation.slot_ids,
                    provenance=provenance,
                    metadata=metadata,
                    label_names=annotation.label_names,
                    target_type=annotation.target_type,
                    target_names=annotation.target_names,
                )
            )
        return dataset.with_unit_annotations(
            resolved,
            unit_type=self.unit_type,
            task_family=self.task_family,
        )


def _label_count(labels: Any) -> int:
    resolved = coerce_label_input(labels)
    if resolved.ndim < 1:
        raise ValueError("structured labels must be at least one-dimensional.")
    return int(resolved.shape[0])


def _optional_aligned(
    values: Any,
    n_units: int,
    field_name: str,
    sample_index: int,
) -> Optional[List[Any]]:
    if values is None:
        return None
    resolved = list(values)
    if len(resolved) != n_units:
        raise ValueError(
            f"{field_name} for sample {sample_index} must align to {n_units} units; "
            f"got {len(resolved)}."
        )
    return resolved


def _required_aligned(values: Any, n_units: int, field_name: str, sample_index: int) -> List[Any]:
    resolved = _optional_aligned(values, n_units, field_name, sample_index)
    if resolved is None:
        raise ValueError(f"{field_name} is required for sample {sample_index}.")
    return resolved


def _masked_aligned(values: Any, mask: np.ndarray, field_name: str, sample_index: int) -> Any:
    resolved = _optional_aligned(values, len(mask), field_name, sample_index)
    if resolved is None:
        return None
    return [value for value, keep in zip(resolved, mask.tolist()) if keep]


def _merge_provenance(
    base: Any,
    n_units: int,
    extras: Sequence[tuple[str, Any]],
    sample_index: int,
    field_name: str,
) -> List[Dict[str, Any]]:
    rows = _optional_aligned(base, n_units, field_name, sample_index)
    provenance = [
        dict(row if row is not None else {})
        for row in (rows if rows is not None else [{} for _ in range(n_units)])
    ]
    for key, value in extras:
        if value is None:
            continue
        if isinstance(value, list) and len(value) == n_units:
            for idx, item in enumerate(value):
                provenance[idx][key] = item
        else:
            for row in provenance:
                row[key] = value
    return provenance


def _require_instance(value: Any, cls: type, name: str, sample_index: int) -> None:
    if not isinstance(value, cls):
        raise ValueError(f"Expected {name} for sample {sample_index}.")
