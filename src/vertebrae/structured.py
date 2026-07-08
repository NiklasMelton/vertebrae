"""Structured unit materialization parallel to spatial segmentation flows."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from vertebrae.datasets.base import BenchmarkDataset, TargetView
from vertebrae.utils.labels import labels_from_jsonable


@dataclass
class StructuredMaterialization:
    """One flattened, grouped structured embedding output."""

    name: str
    dataset: BenchmarkDataset
    provenance: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class StructuredAlignment:
    """One explicit one-to-one mapping from embeddings to annotation rows."""

    annotation_indices: Sequence[int]
    embedding_indices: Sequence[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructuredUnitAligner:
    """Named callable that aligns structured embeddings to annotated units."""

    def __init__(
        self,
        name: str,
        align_fn: Callable[[np.ndarray, Dict[str, Any]], Any],
        recipe_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = str(name)
        self.align_fn = align_fn
        self.recipe_data = dict(recipe_data or {})

    def align(self, embeddings: np.ndarray, annotation: Dict[str, Any]) -> StructuredAlignment:
        raw = self.align_fn(embeddings, annotation)
        return _normalize_alignment(raw)

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "align_fn": _callable_name(self.align_fn),
            "recipe_data": self.recipe_data,
        }


def materialize_structured_outputs(
    dataset: BenchmarkDataset,
    extractor: Any,
    batch_size: int = 16,
    aligners: Optional[Mapping[str, StructuredUnitAligner]] = None,
) -> List[StructuredMaterialization]:
    """Extract, align, and flatten declared structured outputs."""

    dataset.validate()
    annotations = dataset.unit_annotations()
    if not annotations:
        raise ValueError(
            "Structured materialization requires dataset unit annotations. "
            "Use BenchmarkDataset.with_unit_annotations(...)."
        )
    extractor.fit(dataset.X, None)
    resolved_aligners = dict(aligners or {})
    collected: Dict[str, Dict[str, Any]] = {}
    for batch in dataset.iter_batches(batch_size=batch_size):
        outputs = list(extractor.transform_structured(batch.X))
        for output in outputs:
            if len(output.embeddings) != len(batch.indices):
                raise ValueError(
                    f"Structured output {output.name!r} returned {len(output.embeddings)} parents "
                    f"for a batch of {len(batch.indices)}."
                )
            bucket = collected.setdefault(
                output.name,
                {
                    "unit_type": output.unit_type,
                    "recipe": output.recipe,
                    "metadata": output.metadata,
                    "rows": [],
                    "alignment_mode": (
                        "explicit" if output.name in resolved_aligners else "strict"
                    ),
                    "alignment_recipe": (
                        resolved_aligners[output.name].recipe()
                        if output.name in resolved_aligners
                        else None
                    ),
                    "n_annotation_units": 0,
                    "n_embedding_units": 0,
                },
            )
            if bucket["unit_type"] != output.unit_type:
                raise ValueError(
                    f"Structured output {output.name!r} changed unit type between batches."
                )
            for local_index, parent_index in enumerate(batch.indices):
                annotation = annotations[int(parent_index)]
                n_annotation_units = len(
                    labels_from_jsonable(
                        annotation["labels"],
                        label_names=annotation.get("label_names"),
                        target_type=annotation.get("target_type", "auto"),
                        target_names=annotation.get("target_names"),
                    )
                )
                n_embedding_units = int(np.asarray(output.embeddings[local_index]).shape[0])
                parent_rows = _materialize_parent_rows(
                    parent_index=int(parent_index),
                    embeddings=output.embeddings[local_index],
                    annotation=annotation,
                    output_name=output.name,
                    unit_type=output.unit_type,
                    aligner=resolved_aligners.get(output.name),
                )
                bucket["rows"].extend(parent_rows)
                bucket["n_annotation_units"] += n_annotation_units
                bucket["n_embedding_units"] += n_embedding_units

    materializations = []
    for output_name, bucket in collected.items():
        rows = bucket["rows"]
        if not rows:
            raise ValueError(f"Structured output {output_name!r} produced no valid units.")
        target_views = _materialized_target_views(dataset, rows)
        benchmark_dataset = BenchmarkDataset.from_embedding_units(
            embeddings=np.vstack([row["embedding"] for row in rows]),
            labels=np.asarray([row["label"] for row in rows], dtype=object),
            unit_ids=[row["unit_id"] for row in rows],
            parent_ids=[row["parent_id"] for row in rows],
            unit_type=bucket["unit_type"],
            positions=[row["position"] for row in rows],
            spans=[row["span"] for row in rows],
            coordinates=[row["coordinates"] for row in rows],
            provenance=[row["unit_provenance"] for row in rows],
            metadata={
                "structured": {
                    "unit_type": bucket["unit_type"],
                    "n_parents": len({row["parent_id"] for row in rows}),
                    "n_units": len(rows),
                },
                "structured_output": output_name,
                "source_dataset_fingerprint": dataset.fingerprint(),
            },
            label_names=rows[0]["label_names"],
            target_type=rows[0]["target_type"],
            target_names=rows[0]["target_names"],
            target_views=target_views or None,
        )
        materializations.append(
            StructuredMaterialization(
                name=output_name,
                dataset=benchmark_dataset,
                provenance=[_provenance(row) for row in rows],
                metadata={
                    "unit_type": bucket["unit_type"],
                    "n_parents": len({row["parent_id"] for row in rows}),
                    "n_units": len(rows),
                    "n_annotation_units": bucket["n_annotation_units"],
                    "n_embedding_units": bucket["n_embedding_units"],
                    "output_recipe": bucket["recipe"],
                    "output_metadata": bucket["metadata"],
                    "alignment_mode": bucket["alignment_mode"],
                    "alignment_recipe": bucket["alignment_recipe"],
                    "task_family": dataset.metadata.get("unit_annotation_task_family"),
                    "target_type": rows[0]["target_type"],
                },
            )
        )
    return materializations


def _materialize_parent_rows(
    parent_index: int,
    embeddings: Any,
    annotation: Dict[str, Any],
    output_name: str,
    unit_type: str,
    aligner: Optional[StructuredUnitAligner] = None,
) -> List[Dict[str, Any]]:
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError(f"Structured output {output_name!r} for parent {parent_index} must be 2D.")
    labels = labels_from_jsonable(
        annotation["labels"],
        label_names=annotation.get("label_names"),
        target_type=annotation.get("target_type", "auto"),
        target_names=annotation.get("target_names"),
    )
    unit_ids = annotation.get("unit_ids") or [
        f"{parent_index}:{index}" for index in range(len(labels))
    ]
    positions = annotation.get("positions") or [None] * len(labels)
    spans = annotation.get("spans") or [None] * len(labels)
    coordinates = annotation.get("coordinates") or [None] * len(labels)
    provenance = annotation.get("provenance") or [None] * len(labels)
    if aligner is None:
        if int(matrix.shape[0]) != int(len(labels)):
            raise ValueError(
                f"Structured output {output_name!r} returned "
                f"{matrix.shape[0]} {unit_type} rows for "
                f"parent {parent_index}, but annotations contain {len(labels)} labels."
            )
        annotation_indices = np.arange(len(labels), dtype=int)
        embedding_indices = np.arange(matrix.shape[0], dtype=int)
        alignment_metadata = {}
    else:
        alignment = aligner.align(matrix, annotation)
        annotation_indices, embedding_indices = _validated_alignment_indices(
            alignment,
            n_annotations=len(labels),
            n_embeddings=int(matrix.shape[0]),
            output_name=output_name,
            parent_index=parent_index,
        )
        alignment_metadata = dict(alignment.metadata)
    rows = []
    for alignment_index, (annotation_index, embedding_index) in enumerate(
        zip(annotation_indices.tolist(), embedding_indices.tolist())
    ):
        rows.append(
            {
                "embedding": matrix[embedding_index : embedding_index + 1],
                "label": labels[annotation_index],
                "unit_id": unit_ids[annotation_index],
                "parent_id": parent_index,
                "position": positions[annotation_index],
                "span": spans[annotation_index],
                "coordinates": coordinates[annotation_index],
                "unit_provenance": provenance[annotation_index],
                "annotation_metadata": dict(annotation.get("metadata", {})),
                "output_name": output_name,
                "unit_index": alignment_index,
                "annotation_index": annotation_index,
                "embedding_index": embedding_index,
                "alignment_metadata": alignment_metadata,
                "label_names": annotation.get("label_names"),
                "target_type": annotation.get("target_type", "auto"),
                "target_names": annotation.get("target_names"),
                "n_annotation_units": len(labels),
                "n_embedding_units": int(matrix.shape[0]),
            }
        )
    return rows


def _materialized_target_views(
    dataset: BenchmarkDataset,
    rows: List[Dict[str, Any]],
) -> List[TargetView]:
    views = dataset.metadata.get("target_views") or {}
    resolved: List[TargetView] = []
    if not views:
        return resolved
    for name, view in views.items():
        parent_labels = labels_from_jsonable(
            view["targets"],
            label_names=view.get("label_names"),
            target_type=view.get("target_type", "auto"),
            target_names=view.get("target_names"),
        )
        labels = np.asarray([parent_labels[row["parent_id"]] for row in rows], dtype=object)
        resolved.append(
            TargetView(
                name=str(name),
                targets=labels,
                target_type=view.get("target_type", "auto"),
                label_names=view.get("label_names"),
                target_names=view.get("target_names"),
                metadata=dict(view.get("metadata", {})),
            )
        )
    return resolved


def _provenance(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "parent_id": row["parent_id"],
        "unit_id": row["unit_id"],
        "unit_index": row["unit_index"],
        "annotation_index": row["annotation_index"],
        "embedding_index": row["embedding_index"],
        "output_name": row["output_name"],
        "position": row["position"],
        "span": row["span"],
        "coordinates": row["coordinates"],
        "annotation_metadata": row["annotation_metadata"],
        "unit_provenance": row["unit_provenance"],
        "alignment_metadata": row["alignment_metadata"],
    }


def _normalize_alignment(raw: Any) -> StructuredAlignment:
    if isinstance(raw, StructuredAlignment):
        return raw
    if isinstance(raw, Mapping):
        return StructuredAlignment(
            annotation_indices=raw.get("annotation_indices", []),
            embedding_indices=raw.get("embedding_indices", []),
            metadata=dict(raw.get("metadata", {})),
        )
    annotation_indices = []
    embedding_indices = []
    for pair in list(raw):
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(
                "Structured aligners must return "
                "(annotation_index, embedding_index) pairs."
            )
        annotation_indices.append(int(pair[0]))
        embedding_indices.append(int(pair[1]))
    return StructuredAlignment(
        annotation_indices=annotation_indices,
        embedding_indices=embedding_indices,
    )


def _validated_alignment_indices(
    alignment: StructuredAlignment,
    n_annotations: int,
    n_embeddings: int,
    output_name: str,
    parent_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    annotation_indices = np.asarray(alignment.annotation_indices, dtype=int)
    embedding_indices = np.asarray(alignment.embedding_indices, dtype=int)
    if annotation_indices.ndim != 1 or embedding_indices.ndim != 1:
        raise ValueError(
            f"Structured aligner for output {output_name!r} parent {parent_index} must return "
            "one-dimensional annotation and embedding indices."
        )
    if len(annotation_indices) != len(embedding_indices):
        raise ValueError(
            f"Structured aligner for output {output_name!r} parent {parent_index} returned "
            f"{len(annotation_indices)} annotation indices but "
            f"{len(embedding_indices)} embedding indices."
        )
    if len(set(annotation_indices.tolist())) != len(annotation_indices):
        raise ValueError(
            f"Structured aligner for output {output_name!r} parent {parent_index} "
            "returned duplicate annotation indices."
        )
    if len(set(embedding_indices.tolist())) != len(embedding_indices):
        raise ValueError(
            f"Structured aligner for output {output_name!r} parent {parent_index} "
            "returned duplicate embedding indices."
        )
    if len(annotation_indices) and (
        annotation_indices.min() < 0 or annotation_indices.max() >= n_annotations
    ):
        raise ValueError(
            f"Structured aligner for output {output_name!r} parent {parent_index} returned "
            "annotation indices outside the available annotation rows."
        )
    if len(embedding_indices) and (
        embedding_indices.min() < 0 or embedding_indices.max() >= n_embeddings
    ):
        raise ValueError(
            f"Structured aligner for output {output_name!r} parent {parent_index} returned "
            "embedding indices outside the available embedding rows."
        )
    return annotation_indices, embedding_indices


def _callable_name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '<unknown>')}.{getattr(fn, '__qualname__', repr(fn))}"
