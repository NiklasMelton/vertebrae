"""Structured unit materialization parallel to spatial segmentation flows."""

from dataclasses import dataclass
from typing import Any, Dict, List

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


def materialize_structured_outputs(
    dataset: BenchmarkDataset,
    extractor: Any,
    batch_size: int = 16,
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
                },
            )
            if bucket["unit_type"] != output.unit_type:
                raise ValueError(
                    f"Structured output {output.name!r} changed unit type between batches."
                )
            for local_index, parent_index in enumerate(batch.indices):
                bucket["rows"].extend(
                    _materialize_parent_rows(
                        parent_index=int(parent_index),
                        embeddings=output.embeddings[local_index],
                        annotation=annotations[int(parent_index)],
                        output_name=output.name,
                        unit_type=output.unit_type,
                    )
                )

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
                    "output_recipe": bucket["recipe"],
                    "output_metadata": bucket["metadata"],
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
) -> List[Dict[str, Any]]:
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError(
            f"Structured output {output_name!r} for parent {parent_index} must be 2D."
        )
    labels = labels_from_jsonable(
        annotation["labels"],
        label_names=annotation.get("label_names"),
        target_type=annotation.get("target_type", "auto"),
        target_names=annotation.get("target_names"),
    )
    if int(matrix.shape[0]) != int(len(labels)):
        raise ValueError(
            f"Structured output {output_name!r} returned {matrix.shape[0]} {unit_type} rows for "
            f"parent {parent_index}, but annotations contain {len(labels)} labels."
        )
    unit_ids = annotation.get("unit_ids") or [
        f"{parent_index}:{index}" for index in range(len(labels))
    ]
    positions = annotation.get("positions") or [None] * len(labels)
    spans = annotation.get("spans") or [None] * len(labels)
    coordinates = annotation.get("coordinates") or [None] * len(labels)
    provenance = annotation.get("provenance") or [None] * len(labels)
    rows = []
    for unit_index in range(len(labels)):
        rows.append(
            {
                "embedding": matrix[unit_index : unit_index + 1],
                "label": labels[unit_index],
                "unit_id": unit_ids[unit_index],
                "parent_id": parent_index,
                "position": positions[unit_index],
                "span": spans[unit_index],
                "coordinates": coordinates[unit_index],
                "unit_provenance": provenance[unit_index],
                "annotation_metadata": dict(annotation.get("metadata", {})),
                "output_name": output_name,
                "unit_index": unit_index,
                "label_names": annotation.get("label_names"),
                "target_type": annotation.get("target_type", "auto"),
                "target_names": annotation.get("target_names"),
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
        "output_name": row["output_name"],
        "position": row["position"],
        "span": row["span"],
        "coordinates": row["coordinates"],
        "annotation_metadata": row["annotation_metadata"],
        "unit_provenance": row["unit_provenance"],
    }
