"""Structured unit materialization parallel to spatial segmentation flows."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from vertebrae.cache.fingerprint import hash_json_exact
from vertebrae.config import MemoryConfig
from vertebrae.datasets.base import BenchmarkDataset, TargetView
from vertebrae.datasets.identity import DatasetIdentity
from vertebrae.extractors._identity import portable_callable_identity, validate_cache_identity
from vertebrae.utils.labels import MULTI_LABEL_TARGET, REGRESSION_TARGET, labels_from_jsonable
from vertebrae.utils.memory import (
    IncrementalMatrixStager,
    IncrementalMetadataStager,
    MatrixAssembly,
    admit_final_metadata,
    estimate_final_row_metadata_bytes,
    estimate_object_resident_bytes,
)
from vertebrae.utils.validation import ensure_numeric_matrix, is_sparse_matrix


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
        align_fn: Callable[[Any, Dict[str, Any]], Any],
        recipe_data: Optional[Dict[str, Any]] = None,
        cache_identity: Optional[str] = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("StructuredUnitAligner.name must be a non-empty string.")
        if not callable(align_fn):
            raise TypeError("StructuredUnitAligner.align_fn must be callable.")
        self.name = name
        self.align_fn = align_fn
        self.recipe_data = dict(recipe_data or {})
        hash_json_exact(self.recipe_data)
        self.cache_identity = validate_cache_identity(cache_identity)
        self._align_fn_identity = portable_callable_identity(align_fn)
        if self.cache_identity is None and self._align_fn_identity is None:
            raise ValueError(
                "StructuredUnitAligner.align_fn must be an importable top-level callable "
                "with a complete portable identity, or cache_identity must be provided for "
                "closures, lambdas, bound methods, and stateful callable objects."
            )

    def align(self, embeddings: Any, annotation: Dict[str, Any]) -> StructuredAlignment:
        raw = self.align_fn(embeddings, annotation)
        return _normalize_alignment(raw)

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "align_fn": _callable_name(self.align_fn),
            "align_fn_identity": self._align_fn_identity,
            "recipe_data": self.recipe_data,
            "cache_identity": self.cache_identity,
            "cache_safe": True,
        }


def drop_special_rows(
    leading: int = 0,
    trailing: int = 0,
    *,
    name: str = "drop_special_rows",
) -> StructuredUnitAligner:
    """Return an aligner that drops fixed leading/trailing embedding rows."""

    resolved_leading = _strict_nonnegative_int(leading, "leading", "drop_special_rows")
    resolved_trailing = _strict_nonnegative_int(trailing, "trailing", "drop_special_rows")
    policy = _DropSpecialRowsPolicy(
        leading=resolved_leading,
        trailing=resolved_trailing,
    )
    return StructuredUnitAligner(
        name=name,
        align_fn=policy,
        recipe_data={
            "policy": "drop_special_rows",
            "leading": resolved_leading,
            "trailing": resolved_trailing,
        },
        cache_identity="vertebrae.drop_special_rows/v1",
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
        cache_identity="vertebrae.keep_row_indices/v1",
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
    resolved_start = _strict_nonnegative_int(start, "start", "select_frame_rows")
    resolved_every_n = None
    resolved_indices = None
    resolved_indices_metadata_key = None
    if every_n is not None:
        resolved_every_n = _strict_nonnegative_int(
            every_n,
            "every_n",
            "select_frame_rows",
        )
        if resolved_every_n == 0:
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
        start=resolved_start,
    )
    recipe_data: Dict[str, Any] = {
        "policy": "select_frame_rows",
        "start": resolved_start,
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
        cache_identity="vertebrae.select_frame_rows/v1",
    )


def materialize_structured_outputs(
    dataset: BenchmarkDataset,
    extractor: Any,
    batch_size: int = 16,
    aligners: Optional[Mapping[str, StructuredUnitAligner]] = None,
    resource_profiler: Optional[Any] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> List[StructuredMaterialization]:
    """Extract, align, and flatten declared structured outputs."""

    return list(
        iter_materialize_structured_outputs(
            dataset,
            extractor,
            batch_size=batch_size,
            aligners=aligners,
            resource_profiler=resource_profiler,
            memory_config=memory_config,
        )
    )


def iter_materialize_structured_outputs(
    dataset: BenchmarkDataset,
    extractor: Any,
    batch_size: int = 16,
    aligners: Optional[Mapping[str, StructuredUnitAligner]] = None,
    resource_profiler: Optional[Any] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> Iterator[StructuredMaterialization]:
    """Yield structured outputs one at a time after shared extraction and alignment."""

    resolved_memory = memory_config or MemoryConfig()
    with (
        IncrementalMatrixStager(
            resolved_memory,
            purpose="Structured candidate embeddings",
        ) as stager,
        IncrementalMetadataStager(
            resolved_memory,
            purpose="Structured candidate metadata",
            matrix_stager=stager,
        ) as metadata_stager,
    ):
        yield from _iter_materialize_structured_outputs(
            dataset,
            extractor,
            batch_size=batch_size,
            aligners=aligners,
            resource_profiler=resource_profiler,
            stager=stager,
            metadata_stager=metadata_stager,
        )


def _iter_materialize_structured_outputs(
    dataset: BenchmarkDataset,
    extractor: Any,
    batch_size: int,
    aligners: Optional[Mapping[str, StructuredUnitAligner]],
    resource_profiler: Optional[Any],
    stager: IncrementalMatrixStager,
    metadata_stager: IncrementalMetadataStager,
) -> Iterator[StructuredMaterialization]:
    """Implement structured extraction within a cleanup-scoped row stager."""

    dataset.validate()
    annotations = dataset.unit_annotations()
    if not annotations:
        raise ValueError(
            "Structured materialization requires dataset unit annotations. "
            "Use BenchmarkDataset.with_unit_annotations(...)."
        )
    source_parent_ids = list(dataset.metadata.get("sample_indices", range(len(annotations))))
    if len(source_parent_ids) != len(annotations):
        raise ValueError("Structured source sample IDs must align with unit annotations.")
    extractor_recipe = dict(extractor.recipe())
    extractor.fit(dataset.X, None)
    resolved_aligners = dict(aligners or {})
    expected_names = _declared_structured_output_names(extractor)
    unknown_aligners = sorted(set(resolved_aligners) - set(expected_names))
    if unknown_aligners:
        raise ValueError(f"Structured aligners contain unknown outputs: {unknown_aligners}.")
    collected: Dict[str, Dict[str, Any]] = {}
    effective_batch_size = (
        batch_size if bool(getattr(extractor, "streaming_safe", False)) else len(annotations)
    )
    for batch in dataset.iter_batches(batch_size=effective_batch_size):
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
        _validate_output_names(outputs, expected_names, workflow="Structured")
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
                    "alignment_mode": (
                        "explicit" if output.name in resolved_aligners else "strict"
                    ),
                    "alignment_recipe": (
                        resolved_aligners[output.name].recipe()
                        if output.name in resolved_aligners
                        else None
                    ),
                    "cache_safe": (
                        resolved_aligners[output.name].recipe().get("cache_safe", False)
                        if output.name in resolved_aligners
                        else True
                    ),
                    "n_annotation_units": 0,
                    "n_embedding_units": 0,
                    "embedding_contract": None,
                    "output_contract": _structured_output_contract(output),
                },
            )
            if _structured_output_contract(output) != bucket["output_contract"]:
                raise ValueError(
                    f"Structured output {output.name!r} changed its declared recipe or "
                    "metadata between batches."
                )
            if bucket["unit_type"] != output.unit_type:
                raise ValueError(
                    f"Structured output {output.name!r} changed unit type between batches."
                )
            for local_index, parent_position in enumerate(batch.indices):
                local_parent_position = int(parent_position)
                parent_id = source_parent_ids[local_parent_position]
                annotation = annotations[local_parent_position]
                matrix = ensure_numeric_matrix(
                    output.embeddings[local_index],
                    f"Structured output {output.name!r} for parent {parent_id!r}",
                    allow_sparse=True,
                )
                _validate_embedding_contract(
                    bucket,
                    matrix,
                    output_name=output.name,
                    parent_index=parent_id,
                )
                n_annotation_units = len(
                    labels_from_jsonable(
                        annotation["labels"],
                        label_names=annotation.get("label_names"),
                        target_type=annotation.get("target_type", "auto"),
                        target_names=annotation.get("target_names"),
                    )
                )
                n_embedding_units = int(matrix.shape[0])
                for row in _materialize_parent_rows(
                    parent_index=parent_id,
                    parent_position=local_parent_position,
                    embeddings=matrix,
                    annotation=annotation,
                    output_name=output.name,
                    unit_type=output.unit_type,
                    aligner=resolved_aligners.get(output.name),
                ):
                    row["embedding_ref"] = stager.append(
                        output.name,
                        row.pop("embedding"),
                    )
                    metadata_stager.append(output.name, row)
                    metadata_stager.add_member(
                        output.name,
                        "parents",
                        "",
                        hash_json_exact(row["parent_id"]),
                    )
                bucket["n_annotation_units"] += n_annotation_units
                bucket["n_embedding_units"] += n_embedding_units

    retained_final_metadata_bytes = 0
    retained_output_matrix_bytes = 0
    for output_name in list(collected):
        bucket = collected.pop(output_name)
        n_rows = metadata_stager.count_rows(output_name)
        if not n_rows:
            raise ValueError(f"Structured output {output_name!r} produced no valid units.")

        def row_factory(
            resolved_output_name: str = output_name,
        ) -> Iterator[Dict[str, Any]]:
            return (row for _, row in metadata_stager.iter_rows(resolved_output_name))

        first_row = next(row_factory())
        n_parents = metadata_stager.member_count(output_name, "parents")
        row_metadata_bytes = estimate_final_row_metadata_bytes(
            row_factory(),
            expected_rows=n_rows,
            expansion_factor=3.0,
            purpose="Structured final row metadata",
        )
        (
            target_view_retained_bytes,
            target_view_transient_bytes,
        ) = _estimate_materialized_target_view_bytes(
            dataset,
            row_factory,
            expected_rows=n_rows,
        )
        retained_metadata_bytes = row_metadata_bytes + target_view_retained_bytes
        final_metadata_bytes = retained_metadata_bytes + target_view_transient_bytes
        retained_final_metadata_bytes = admit_final_metadata(
            stager,
            retained_metadata_bytes,
            purpose="Structured final row metadata",
            retained_bytes=retained_final_metadata_bytes,
            transient_bytes=target_view_transient_bytes,
        )
        assembly = _stack_embedding_rows(
            row_factory(),
            stager,
            output_name=output_name,
            force_disk=stager.memory_config.allow_disk_spill,
        )
        embeddings = assembly.matrix
        if assembly.strategy == "in_memory":
            stager.reserve_metadata(
                assembly.required_bytes,
                purpose="Structured retained output matrices",
            )
            retained_output_matrix_bytes += assembly.required_bytes
        stager.reserve_metadata(
            target_view_transient_bytes,
            purpose="Structured target-view materialization workspace",
        )
        try:
            target_views = _materialized_target_views(dataset, row_factory)
            provenance = [_provenance(row) for row in row_factory()]
            benchmark_dataset = BenchmarkDataset.from_embedding_units(
                embeddings=embeddings,
                labels=_flattened_labels(row_factory(), first_row["target_type"], n_rows),
                unit_ids=[row["unit_id"] for row in row_factory()],
                identity=DatasetIdentity.derived(
                    dataset.identity_key(),
                    "structured_materialization",
                    {
                        "output_name": output_name,
                        "output_recipe": bucket["recipe"],
                        "source_extractor_recipe": extractor_recipe,
                        "alignment_recipe": bucket["alignment_recipe"],
                        "cache_safe": bool(
                            bucket["cache_safe"] and extractor_recipe.get("cache_safe") is not False
                        ),
                        "unit_type": bucket["unit_type"],
                        "provenance": provenance,
                    },
                ),
                parent_ids=[row["parent_id"] for row in row_factory()],
                unit_type=bucket["unit_type"],
                positions=[row["position"] for row in row_factory()],
                spans=[row["span"] for row in row_factory()],
                coordinates=[row["coordinates"] for row in row_factory()],
                provenance=[row["unit_provenance"] for row in row_factory()],
                metadata={
                    "structured": {
                        "unit_type": bucket["unit_type"],
                        "n_parents": n_parents,
                        "n_units": n_rows,
                    },
                    "structured_output": output_name,
                    "source_dataset_identity_key": dataset.identity_key(),
                    "source_extractor_recipe": extractor_recipe,
                },
                label_names=first_row["label_names"],
                target_type=first_row["target_type"],
                target_names=first_row["target_names"],
                target_views=target_views or None,
            )
        finally:
            stager.release_metadata(target_view_transient_bytes)
        memory_metadata = _assembly_metadata(assembly)
        memory_metadata["metadata_staging_strategy"] = metadata_stager.strategy
        memory_metadata["final_metadata_required_bytes"] = final_metadata_bytes
        memory_metadata["final_row_metadata_required_bytes"] = row_metadata_bytes
        memory_metadata["target_view_metadata_required_bytes"] = (
            target_view_retained_bytes + target_view_transient_bytes
        )
        memory_metadata["target_view_retained_metadata_required_bytes"] = target_view_retained_bytes
        memory_metadata["target_view_transient_required_bytes"] = target_view_transient_bytes
        memory_metadata["cumulative_final_metadata_required_bytes"] = retained_final_metadata_bytes
        memory_metadata["cumulative_retained_output_bytes"] = (
            stager.memory_config.model_memory_bytes
            + stager.memory_config.raw_batch_memory_bytes
            + retained_final_metadata_bytes
            + retained_output_matrix_bytes
        )
        memory_metadata["fixed_model_and_batch_bytes"] = (
            stager.memory_config.model_memory_bytes + stager.memory_config.raw_batch_memory_bytes
        )
        memory_metadata["final_metadata_strategy"] = "resident"
        metadata_stager.discard_output(output_name)
        materialization = StructuredMaterialization(
            name=output_name,
            dataset=benchmark_dataset,
            provenance=provenance,
            metadata={
                "unit_type": bucket["unit_type"],
                "n_parents": n_parents,
                "n_units": n_rows,
                "n_annotation_units": bucket["n_annotation_units"],
                "n_embedding_units": bucket["n_embedding_units"],
                "output_recipe": bucket["recipe"],
                "output_metadata": bucket["metadata"],
                "source_extractor_recipe": extractor_recipe,
                "alignment_mode": bucket["alignment_mode"],
                "alignment_recipe": bucket["alignment_recipe"],
                "task_family": dataset.metadata.get("unit_annotation_task_family"),
                "target_type": first_row["target_type"],
                "memory": memory_metadata,
            },
        )
        yield materialization
        del (
            assembly,
            benchmark_dataset,
            bucket,
            embeddings,
            materialization,
            first_row,
            final_metadata_bytes,
            memory_metadata,
            provenance,
            retained_metadata_bytes,
            row_metadata_bytes,
            row_factory,
            target_view_retained_bytes,
            target_view_transient_bytes,
            target_views,
        )


def _materialize_parent_rows(
    parent_index: Any,
    parent_position: int,
    embeddings: Any,
    annotation: Dict[str, Any],
    output_name: str,
    unit_type: str,
    aligner: Optional[StructuredUnitAligner] = None,
) -> Iterator[Dict[str, Any]]:
    matrix = ensure_numeric_matrix(
        embeddings,
        f"Structured output {output_name!r} for parent {parent_index}",
        allow_sparse=True,
    )
    labels = labels_from_jsonable(
        annotation["labels"],
        label_names=annotation.get("label_names"),
        target_type=annotation.get("target_type", "auto"),
        target_names=annotation.get("target_names"),
    )
    local_unit_ids = annotation.get("unit_ids") or list(range(len(labels)))
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
    row_source = matrix.tocsr(copy=False) if is_sparse_matrix(matrix) else matrix
    for alignment_index, (annotation_index, embedding_index) in enumerate(
        zip(annotation_indices.tolist(), embedding_indices.tolist())
    ):
        local_unit_id = local_unit_ids[annotation_index]
        unit_id = _global_unit_id(parent_index, local_unit_id)
        yield {
            "embedding": row_source[embedding_index : embedding_index + 1],
            "label": labels[annotation_index],
            "unit_id": unit_id,
            "local_unit_id": local_unit_id,
            "parent_id": parent_index,
            "parent_position": parent_position,
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


def _materialized_target_views(
    dataset: BenchmarkDataset,
    row_factory: Callable[[], Iterator[Dict[str, Any]]],
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
        labels = np.asarray(
            [parent_labels[row["parent_position"]] for row in row_factory()],
            dtype=object,
        )
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


def _estimate_materialized_target_view_bytes(
    dataset: BenchmarkDataset,
    row_factory: Callable[[], Iterator[Dict[str, Any]]],
    *,
    expected_rows: int,
) -> tuple[int, int]:
    """Estimate target-view allocations before creating retained row arrays.

    Structured materialization duplicates every selected parent target into a
    transient ``TargetView`` and the normalized target-view entry retained by
    the final dataset. Iterate staged rows instead of allocating either shape
    up front, and include the largest per-view decoding workspace in the peak.
    """

    views = dataset.metadata.get("target_views") or {}
    if not views:
        return 0, 0
    # The source mapping has the same normalized structure as the mapping
    # retained by one output. Measuring the complete graph at once preserves
    # shared-object accounting without allocating the destination mapping.
    retained_bytes = estimate_object_resident_bytes(views)
    transient_bytes = estimate_object_resident_bytes([])
    target_view_base_bytes = estimate_object_resident_bytes(
        TargetView(name="", targets=np.empty(0, dtype=object))
    )
    maximum_decode_bytes = 0
    for name, view in views.items():
        serialized_targets = view["targets"]
        if not isinstance(serialized_targets, list):
            raise RuntimeError(f"Structured target view {name!r} has a non-list target payload.")
        maximum_decode_bytes = max(
            maximum_decode_bytes,
            1024
            + len(serialized_targets) * 64
            + 2 * estimate_object_resident_bytes(serialized_targets),
        )
        transient_bytes += (
            target_view_base_bytes
            + estimate_object_resident_bytes(str(name))
            + estimate_object_resident_bytes(view.get("metadata", {}))
        )
        observed_rows = 0
        selected_value_bytes = 0
        for observed_rows, row in enumerate(row_factory(), start=1):
            if observed_rows > expected_rows:
                raise RuntimeError("Structured target-view staging produced extra final rows.")
            parent_position = row["parent_position"]
            try:
                value = serialized_targets[parent_position]
            except (IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"Structured target view {name!r} does not align with parent "
                    f"position {parent_position!r}."
                ) from exc
            selected_value_bytes += estimate_object_resident_bytes(value)
        if observed_rows != expected_rows:
            raise RuntimeError(
                "Structured target-view staging produced "
                f"{observed_rows} rows; expected {expected_rows}."
            )
        transient_bytes += expected_rows * 8 + selected_value_bytes
        extra_rows = max(0, expected_rows - len(serialized_targets))
        if extra_rows:
            average_value_bytes = selected_value_bytes // expected_rows
            retained_bytes += extra_rows * (8 + average_value_bytes)
    return retained_bytes, transient_bytes + maximum_decode_bytes


def _provenance(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "parent_id": row["parent_id"],
        "parent_position": row["parent_position"],
        "unit_id": row["unit_id"],
        "local_unit_id": row["local_unit_id"],
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


def _global_unit_id(parent_index: Any, local_unit_id: Any) -> str:
    digest = hash_json_exact(
        {
            "parent_id": parent_index,
            "local_unit_id": local_unit_id,
        }
    )
    return f"structured-unit-v1-{digest}"


def _validate_embedding_contract(
    bucket: Dict[str, Any],
    matrix: Any,
    output_name: str,
    parent_index: Any,
) -> None:
    contract = {
        "embedding_dim": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "sparse": is_sparse_matrix(matrix),
    }
    expected = bucket["embedding_contract"]
    if expected is None:
        bucket["embedding_contract"] = contract
        return
    if contract != expected:
        raise ValueError(
            f"Structured output {output_name!r} changed embedding format for parent "
            f"{parent_index}; expected dimension {expected['embedding_dim']}, dtype "
            f"{expected['dtype']}, and sparse={expected['sparse']}, but received dimension "
            f"{contract['embedding_dim']}, dtype {contract['dtype']}, and "
            f"sparse={contract['sparse']}."
        )


def _stack_embedding_rows(
    rows: Iterable[Dict[str, Any]],
    stager: IncrementalMatrixStager,
    *,
    output_name: str,
    force_disk: bool = False,
) -> MatrixAssembly:
    return stager.assemble(
        output_name,
        (row["embedding_ref"] for row in rows),
        purpose="Structured materialized embeddings",
        force_disk=force_disk,
    )


def _assembly_metadata(assembly: MatrixAssembly) -> Dict[str, Any]:
    return {
        "strategy": assembly.strategy,
        "staging_strategy": assembly.staging_strategy,
        "required_bytes": assembly.required_bytes,
        "budget_bytes": assembly.budget_bytes,
    }


def _flattened_labels(
    rows: Iterable[Dict[str, Any]],
    target_type: str,
    n_rows: int,
) -> np.ndarray:
    labels = [row["label"] for row in rows]
    if len(labels) != n_rows:  # pragma: no cover - staging invariant
        raise RuntimeError("Structured metadata staging changed row count during assembly.")
    if target_type == REGRESSION_TARGET:
        return np.asarray(labels, dtype=float)
    if target_type == MULTI_LABEL_TARGET:
        result = np.empty(len(labels), dtype=object)
        result[:] = labels
        return result
    return np.asarray(labels, dtype=object)


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
        annotation_indices.append(_strict_index_value(pair[0], "annotation_index"))
        embedding_indices.append(_strict_index_value(pair[1], "embedding_index"))
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
    annotation_indices = _coerce_index_array(
        alignment.annotation_indices,
        arg_name="annotation_indices",
        helper_name="Structured aligner",
    )
    embedding_indices = _coerce_index_array(
        alignment.embedding_indices,
        arg_name="embedding_indices",
        helper_name="Structured aligner",
    )
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
    value_type = type(fn)
    module = getattr(fn, "__module__", value_type.__module__)
    qualname = getattr(fn, "__qualname__", value_type.__qualname__)
    return f"{module}.{qualname}"


@dataclass(frozen=True)
class _DropSpecialRowsPolicy:
    leading: int = 0
    trailing: int = 0

    def __call__(
        self,
        embeddings: Any,
        annotation: Dict[str, Any],
    ) -> StructuredAlignment:
        n_annotations = _annotation_length(annotation)
        n_embeddings = int(embeddings.shape[0])
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
        embeddings: Any,
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
        embeddings: Any,
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
        embeddings: Any,
        annotation: Dict[str, Any],
    ) -> np.ndarray:
        n_embeddings = int(embeddings.shape[0])
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
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{helper_name}(...) requires {arg_name} to be one-dimensional.")
    if raw.size == 0:
        return np.asarray([], dtype=int)
    if np.issubdtype(raw.dtype, np.integer) and not np.issubdtype(raw.dtype, np.bool_):
        array = raw.astype(int, copy=False)
    elif raw.dtype == object and all(
        isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))
        for item in raw.tolist()
    ):
        array = raw.astype(int)
    else:
        raise TypeError(f"{helper_name}(...) requires {arg_name} to contain non-boolean integers.")
    if len(array) and np.any(array < 0):
        raise ValueError(f"{helper_name}(...) requires {arg_name} to be non-negative.")
    return array


def _strict_index_value(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Structured aligner {name} values must be non-boolean integers.")
    return int(value)


def _strict_nonnegative_int(value: Any, name: str, helper_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{helper_name}(...) requires {name} to be a non-boolean integer.")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{helper_name}(...) requires {name} >= 0.")
    return resolved


def _declared_structured_output_names(extractor: Any) -> List[str]:
    method = getattr(extractor, "structured_output_specs", None)
    if not callable(method):
        raise ValueError(
            f"Extractor {getattr(extractor, 'name', '<unknown>')!r} must declare "
            "structured_output_specs()."
        )
    raw_names = [getattr(spec, "name", None) for spec in list(method())]
    if not raw_names or any(not isinstance(name, str) or not name.strip() for name in raw_names):
        raise ValueError("Structured output specs must use non-empty string names.")
    names = [name for name in raw_names if isinstance(name, str)]
    if len(names) != len(set(names)):
        raise ValueError("Structured output spec names must be unique.")
    return names


def _validate_output_names(outputs: List[Any], expected: List[str], *, workflow: str) -> None:
    actual = [getattr(output, "name", None) for output in outputs]
    if any(not isinstance(name, str) or not name for name in actual):
        raise ValueError(f"{workflow} extractor returned an invalid output name.")
    if len(actual) != len(set(actual)):
        raise ValueError(f"{workflow} extractor returned duplicate output names.")
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(
            f"{workflow} extractor returned outputs {actual!r}; expected {expected!r} on every "
            "batch."
        )


def _structured_output_contract(output: Any) -> Dict[str, Any]:
    return {
        "unit_type": output.unit_type,
        "recipe": hash_json_exact(dict(output.recipe)),
        "metadata": hash_json_exact(dict(output.metadata)),
    }
