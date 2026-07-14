"""Structured unit materialization parallel to spatial segmentation flows."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from vertebrae.datasets.base import BenchmarkDataset, TargetView
from vertebrae.datasets.identity import DatasetIdentity
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


def drop_special_rows(
    leading: int = 0,
    trailing: int = 0,
    *,
    name: str = "drop_special_rows",
) -> StructuredUnitAligner:
    """Return an aligner that drops fixed leading/trailing embedding rows."""

    if int(leading) < 0:
        raise ValueError("drop_special_rows(...) requires leading >= 0.")
    if int(trailing) < 0:
        raise ValueError("drop_special_rows(...) requires trailing >= 0.")
    policy = _DropSpecialRowsPolicy(leading=int(leading), trailing=int(trailing))
    return StructuredUnitAligner(
        name=name,
        align_fn=policy,
        recipe_data={
            "policy": "drop_special_rows",
            "leading": int(leading),
            "trailing": int(trailing),
        },
    )


def keep_row_indices(
    embedding_indices: Sequence[int],
    annotation_indices: Optional[Sequence[int]] = None,
    *,
    name: str = "keep_row_indices",
) -> StructuredUnitAligner:
    """Return an aligner that keeps an explicit subset of embedding rows."""

    resolved_embedding_indices = _coerce_index_array(
        embedding_indices,
        arg_name="embedding_indices",
        helper_name="keep_row_indices",
    )
    if annotation_indices is None:
        resolved_annotation_indices = np.arange(len(resolved_embedding_indices), dtype=int)
    else:
        resolved_annotation_indices = _coerce_index_array(
            annotation_indices,
            arg_name="annotation_indices",
            helper_name="keep_row_indices",
        )
        if len(resolved_annotation_indices) != len(resolved_embedding_indices):
            raise ValueError(
                "keep_row_indices(...) requires annotation_indices and "
                "embedding_indices to have the same length."
            )
    policy = _KeepRowIndicesPolicy(
        embedding_indices=tuple(int(index) for index in resolved_embedding_indices.tolist()),
        annotation_indices=tuple(int(index) for index in resolved_annotation_indices.tolist()),
    )
    return StructuredUnitAligner(
        name=name,
        align_fn=policy,
        recipe_data={
            "policy": "keep_row_indices",
            "embedding_indices": resolved_embedding_indices.tolist(),
            "annotation_indices": resolved_annotation_indices.tolist(),
        },
    )


def select_frame_rows(
    *,
    every_n: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    indices_metadata_key: Optional[str] = None,
    start: int = 0,
    name: str = "select_frame_rows",
) -> StructuredUnitAligner:
    """Return an aligner that maps annotations to selected frame rows."""

    modes = [
        every_n is not None,
        indices is not None,
        indices_metadata_key is not None,
    ]
    if sum(bool(mode) for mode in modes) != 1:
        raise ValueError(
            "select_frame_rows(...) requires exactly one of every_n, indices, "
            "or indices_metadata_key."
        )
    if int(start) < 0:
        raise ValueError("select_frame_rows(...) requires start >= 0.")
    resolved_every_n = None
    resolved_indices = None
    resolved_indices_metadata_key = None
    if every_n is not None:
        resolved_every_n = int(every_n)
        if resolved_every_n <= 0:
            raise ValueError("select_frame_rows(...) requires every_n > 0.")
    if indices is not None:
        resolved_indices = _coerce_index_array(
            indices,
            arg_name="indices",
            helper_name="select_frame_rows",
        )
    if indices_metadata_key is not None:
        resolved_indices_metadata_key = str(indices_metadata_key).strip()
        if not resolved_indices_metadata_key:
            raise ValueError("select_frame_rows(...) requires a non-empty indices_metadata_key.")
    policy = _SelectFrameRowsPolicy(
        every_n=resolved_every_n,
        indices=(
            tuple(int(index) for index in resolved_indices.tolist())
            if resolved_indices is not None
            else None
        ),
        indices_metadata_key=resolved_indices_metadata_key,
        start=int(start),
    )
    recipe_data: Dict[str, Any] = {
        "policy": "select_frame_rows",
        "start": int(start),
    }
    if resolved_every_n is not None:
        recipe_data["every_n"] = resolved_every_n
    if resolved_indices is not None:
        recipe_data["indices"] = resolved_indices.tolist()
    if resolved_indices_metadata_key is not None:
        recipe_data["indices_metadata_key"] = resolved_indices_metadata_key
    return StructuredUnitAligner(
        name=name,
        align_fn=policy,
        recipe_data=recipe_data,
    )


def materialize_structured_outputs(
    dataset: BenchmarkDataset,
    extractor: Any,
    batch_size: int = 16,
    aligners: Optional[Mapping[str, StructuredUnitAligner]] = None,
    resource_profiler: Optional[Any] = None,
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
        batch_x = batch.X

        def call(values: Any = batch_x) -> List[Any]:
            return list(extractor.transform_structured(values))

        outputs = (
            resource_profiler.measure_call(
                call,
                samples=len(batch.indices),
                call_type="transform_structured",
            )
            if resource_profiler is not None
            else call()
        )
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
            identity=DatasetIdentity.derived(
                dataset.identity_key(),
                "structured_materialization",
                {
                    "output_name": output_name,
                    "output_recipe": bucket["recipe"],
                    "unit_type": bucket["unit_type"],
                    "provenance": [_provenance(row) for row in rows],
                },
            ),
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
                "source_dataset_identity_key": dataset.identity_key(),
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
                "Structured aligners must return " "(annotation_index, embedding_index) pairs."
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


@dataclass(frozen=True)
class _DropSpecialRowsPolicy:
    leading: int = 0
    trailing: int = 0

    def __call__(
        self,
        embeddings: np.ndarray,
        annotation: Dict[str, Any],
    ) -> StructuredAlignment:
        n_annotations = _annotation_length(annotation)
        n_embeddings = int(np.asarray(embeddings).shape[0])
        stop = n_embeddings - self.trailing if self.trailing else n_embeddings
        embedding_indices = np.arange(self.leading, stop, dtype=int)
        if len(embedding_indices) != n_annotations:
            raise ValueError(
                "drop_special_rows(...) retained "
                f"{len(embedding_indices)} embedding rows, but the parent annotation has "
                f"{n_annotations} labeled units."
            )
        return StructuredAlignment(
            annotation_indices=np.arange(n_annotations, dtype=int).tolist(),
            embedding_indices=embedding_indices.tolist(),
            metadata={
                "policy": "drop_special_rows",
                "leading": self.leading,
                "trailing": self.trailing,
                "selected_embedding_indices": embedding_indices.tolist(),
            },
        )


@dataclass(frozen=True)
class _KeepRowIndicesPolicy:
    embedding_indices: tuple[int, ...]
    annotation_indices: tuple[int, ...]

    def __call__(
        self,
        embeddings: np.ndarray,
        annotation: Dict[str, Any],
    ) -> StructuredAlignment:
        _ = embeddings
        _ = annotation
        return StructuredAlignment(
            annotation_indices=list(self.annotation_indices),
            embedding_indices=list(self.embedding_indices),
            metadata={
                "policy": "keep_row_indices",
                "selected_embedding_indices": list(self.embedding_indices),
                "selected_annotation_indices": list(self.annotation_indices),
            },
        )


@dataclass(frozen=True)
class _SelectFrameRowsPolicy:
    every_n: Optional[int] = None
    indices: Optional[tuple[int, ...]] = None
    indices_metadata_key: Optional[str] = None
    start: int = 0

    def __call__(
        self,
        embeddings: np.ndarray,
        annotation: Dict[str, Any],
    ) -> StructuredAlignment:
        n_annotations = _annotation_length(annotation)
        embedding_indices = self._resolve_embedding_indices(embeddings, annotation)
        if len(embedding_indices) != n_annotations:
            raise ValueError(
                "select_frame_rows(...) retained "
                f"{len(embedding_indices)} embedding rows, but the parent annotation has "
                f"{n_annotations} labeled units."
            )
        metadata: Dict[str, Any] = {
            "policy": "select_frame_rows",
            "selected_embedding_indices": embedding_indices.tolist(),
            "start": self.start,
        }
        if self.every_n is not None:
            metadata["every_n"] = self.every_n
        if self.indices is not None:
            metadata["indices"] = list(self.indices)
        if self.indices_metadata_key is not None:
            metadata["indices_metadata_key"] = self.indices_metadata_key
        return StructuredAlignment(
            annotation_indices=np.arange(n_annotations, dtype=int).tolist(),
            embedding_indices=embedding_indices.tolist(),
            metadata=metadata,
        )

    def _resolve_embedding_indices(
        self,
        embeddings: np.ndarray,
        annotation: Dict[str, Any],
    ) -> np.ndarray:
        n_embeddings = int(np.asarray(embeddings).shape[0])
        if self.every_n is not None:
            return np.arange(self.start, n_embeddings, self.every_n, dtype=int)
        if self.indices is not None:
            return np.asarray(self.indices, dtype=int)
        metadata = annotation.get("metadata", {})
        if self.indices_metadata_key not in metadata:
            raise ValueError(
                "select_frame_rows(...) could not find sampled frame indices under "
                f"annotation metadata key {self.indices_metadata_key!r}."
            )
        assert self.indices_metadata_key is not None
        return _coerce_index_array(
            metadata[self.indices_metadata_key],
            arg_name=self.indices_metadata_key,
            helper_name="select_frame_rows",
        )


def _annotation_length(annotation: Dict[str, Any]) -> int:
    return len(
        labels_from_jsonable(
            annotation["labels"],
            label_names=annotation.get("label_names"),
            target_type=annotation.get("target_type", "auto"),
            target_names=annotation.get("target_names"),
        )
    )


def _coerce_index_array(
    values: Sequence[int],
    *,
    arg_name: str,
    helper_name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=int)
    if array.ndim != 1:
        raise ValueError(f"{helper_name}(...) requires {arg_name} to be one-dimensional.")
    if len(array) and np.any(array < 0):
        raise ValueError(f"{helper_name}(...) requires {arg_name} to be non-negative.")
    return array
