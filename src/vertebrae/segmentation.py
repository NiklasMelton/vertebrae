"""Spatial feature alignment and token materialization for segmentation."""

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np

from vertebrae.cache.fingerprint import hash_json_exact
from vertebrae.config import MemoryConfig, SegmentationConfig
from vertebrae.datasets.base import BenchmarkDataset
from vertebrae.datasets.identity import DatasetIdentity
from vertebrae.datasets.segmentation import SegmentationAnnotation, SegmentationDataset
from vertebrae.extractors.spatial import SpatialLayout
from vertebrae.utils.labels import semantic_label_key
from vertebrae.utils.memory import (
    IncrementalMatrixStager,
    IncrementalMetadataStager,
    MatrixAssembly,
    admit_final_metadata,
    estimate_final_row_metadata_bytes,
)


@dataclass
class SegmentationMaterialization:
    """One flattened, grouped segmentation embedding output."""

    name: str
    dataset: BenchmarkDataset
    provenance: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def materialize_segmentation_outputs(
    dataset: SegmentationDataset,
    extractor: Any,
    config: Optional[SegmentationConfig] = None,
    batch_size: int = 16,
    resource_profiler: Optional[Any] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> List[SegmentationMaterialization]:
    """Extract, align, sample, and flatten all declared spatial outputs."""

    return list(
        iter_materialize_segmentation_outputs(
            dataset,
            extractor,
            config=config,
            batch_size=batch_size,
            resource_profiler=resource_profiler,
            memory_config=memory_config,
        )
    )


def iter_materialize_segmentation_outputs(
    dataset: SegmentationDataset,
    extractor: Any,
    config: Optional[SegmentationConfig] = None,
    batch_size: int = 16,
    resource_profiler: Optional[Any] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> Iterator[SegmentationMaterialization]:
    """Yield flattened spatial outputs one at a time after shared extraction."""

    resolved = config or SegmentationConfig()
    resolved_memory = memory_config or MemoryConfig()
    with (
        IncrementalMatrixStager(
            resolved_memory,
            purpose="Segmentation candidate embeddings",
        ) as stager,
        IncrementalMetadataStager(
            resolved_memory,
            purpose="Segmentation candidate metadata",
            matrix_stager=stager,
        ) as metadata_stager,
    ):
        yield from _iter_materialize_segmentation_outputs(
            dataset,
            extractor,
            resolved=resolved,
            batch_size=batch_size,
            resource_profiler=resource_profiler,
            stager=stager,
            metadata_stager=metadata_stager,
        )


def _iter_materialize_segmentation_outputs(
    dataset: SegmentationDataset,
    extractor: Any,
    *,
    resolved: SegmentationConfig,
    batch_size: int,
    resource_profiler: Optional[Any],
    stager: IncrementalMatrixStager,
    metadata_stager: IncrementalMetadataStager,
) -> Iterator[SegmentationMaterialization]:
    """Implement spatial extraction within a cleanup-scoped row stager."""

    dataset.validate()
    source_image_indices = dataset.metadata.get("sample_indices")
    if source_image_indices is None:
        source_image_indices = list(range(len(dataset.annotations)))
    source_image_indices = list(source_image_indices)
    if len(source_image_indices) != len(dataset.annotations):
        raise ValueError("Segmentation source sample IDs must align with annotations.")
    extractor_recipe = dict(extractor.recipe())
    extractor.fit(dataset.X, None)
    expected_names = _declared_spatial_output_names(extractor)
    collected: Dict[str, Dict[str, Any]] = {}
    effective_batch_size = (
        batch_size
        if bool(getattr(extractor, "streaming_safe", False))
        else len(dataset.annotations)
    )
    for batch in dataset.iter_batches(batch_size=effective_batch_size):
        batch_x = batch.X

        def call(values: Any = batch_x) -> List[Any]:
            return list(extractor.transform_spatial(values))

        outputs = (
            resource_profiler.measure_call(
                call,
                samples=len(batch.indices),
                call_type="transform_spatial",
            )
            if resource_profiler is not None
            else call()
        )
        _validate_output_names(outputs, expected_names)
        for output in outputs:
            if len(output.embeddings) != len(batch.indices):
                raise ValueError(
                    f"Spatial output {output.name!r} returned {len(output.embeddings)} images "
                    f"for a batch of {len(batch.indices)}."
                )
            bucket = collected.setdefault(
                output.name,
                {
                    "layout": output.layout,
                    "recipe": output.recipe,
                    "metadata": output.metadata,
                    "annotation_transform": output.annotation_transform,
                    "candidate_tokens": 0,
                    "ignored_tokens": {},
                    "output_contract": _spatial_output_contract(output),
                    "embedding_contract": None,
                },
            )
            if _spatial_output_contract(output) != bucket["output_contract"]:
                raise ValueError(
                    f"Spatial output {output.name!r} changed its layout, recipe, metadata, or "
                    "annotation transform between batches."
                )
            for local_index, image_index in enumerate(batch.indices):
                source_image_index = source_image_indices[int(image_index)]
                _validate_spatial_embedding_contract(
                    bucket,
                    output.embeddings[local_index],
                    output_name=output.name,
                    image_id=source_image_index,
                )
                for candidate in _align_image(
                    output.embeddings[local_index],
                    output.layout,
                    dataset.annotations[int(image_index)],
                    image_index=source_image_index,
                    output_name=output.name,
                    config=resolved,
                    annotation_transform=output.annotation_transform,
                ):
                    bucket["candidate_tokens"] += 1
                    reason = candidate["drop_reason"]
                    if reason is not None:
                        ignored = bucket["ignored_tokens"]
                        ignored[reason] = ignored.get(reason, 0) + 1
                        continue
                    candidate["embedding_ref"] = stager.append(
                        output.name,
                        candidate.pop("embedding"),
                    )
                    metadata_stager.append(
                        output.name,
                        candidate,
                        priority_key=f"{_priority(candidate, resolved.random_state):064x}",
                        group_key=hash_json_exact(candidate["image_id"]),
                        row_key=int(candidate["row"]),
                        column_key=int(candidate["column"]),
                    )

    retained_final_metadata_bytes = 0
    retained_output_matrix_bytes = 0
    for output_name in list(collected):
        bucket = collected.pop(output_name)
        n_selected = _sample_candidates(metadata_stager, output_name, resolved)
        if not n_selected:
            raise ValueError(f"Segmentation output {output_name!r} produced no valid tokens.")

        def selected_factory(
            resolved_output_name: str = output_name,
        ) -> Iterator[Dict[str, Any]]:
            return (
                row
                for _, row in metadata_stager.iter_rows(
                    resolved_output_name,
                    order="final",
                    selected_only=True,
                )
            )

        final_metadata_bytes = estimate_final_row_metadata_bytes(
            selected_factory(),
            expected_rows=n_selected,
            expansion_factor=2.5,
            purpose="Segmentation final row metadata",
        )
        retained_final_metadata_bytes = admit_final_metadata(
            stager,
            final_metadata_bytes,
            purpose="Segmentation final row metadata",
            retained_bytes=retained_final_metadata_bytes,
        )
        assembly = stager.assemble(
            output_name,
            (candidate["embedding_ref"] for candidate in selected_factory()),
            purpose="Segmentation materialized embeddings",
            force_disk=stager.memory_config.allow_disk_spill,
        )
        embeddings = assembly.matrix
        if assembly.strategy == "in_memory":
            stager.reserve_metadata(
                assembly.required_bytes,
                purpose="Segmentation retained output matrices",
            )
            retained_output_matrix_bytes += assembly.required_bytes
        labels = _object_array(
            (candidate["label"] for candidate in selected_factory()),
            n_selected,
        )
        groups = _object_array(
            (candidate["image_id"] for candidate in selected_factory()),
            n_selected,
        )
        provenance = [_provenance(candidate) for candidate in selected_factory()]
        token_metadata = _materialization_metadata(
            bucket,
            selected_factory,
            n_selected,
            metadata_stager,
            output_name,
            bucket["layout"],
            resolved,
        )
        benchmark_dataset = BenchmarkDataset.from_embeddings(
            embeddings,
            labels,
            identity=DatasetIdentity.derived(
                dataset.identity_key(),
                "segmentation_materialization",
                {
                    "output_name": output_name,
                    "output_recipe": bucket["recipe"],
                    "source_extractor_recipe": extractor_recipe,
                    "annotation_transform": _callable_name(bucket["annotation_transform"]),
                    "config": resolved,
                    "provenance": provenance,
                },
            ),
            metadata={
                "segmentation": token_metadata,
                "spatial_output": output_name,
                "source_dataset_identity_key": dataset.identity_key(),
                "source_extractor_recipe": extractor_recipe,
            },
        ).with_groups(groups, name="image_id")
        materialization = SegmentationMaterialization(
            name=output_name,
            dataset=benchmark_dataset,
            provenance=provenance,
            metadata={
                **token_metadata,
                "output_recipe": bucket["recipe"],
                "output_metadata": bucket["metadata"],
                "source_extractor_recipe": extractor_recipe,
                "cache_safe": extractor_recipe.get("cache_safe") is not False,
                "memory": {
                    **_assembly_metadata(assembly),
                    "metadata_staging_strategy": metadata_stager.strategy,
                    "final_metadata_required_bytes": final_metadata_bytes,
                    "cumulative_final_metadata_required_bytes": (retained_final_metadata_bytes),
                    "cumulative_retained_output_bytes": (
                        stager.memory_config.model_memory_bytes
                        + stager.memory_config.raw_batch_memory_bytes
                        + retained_final_metadata_bytes
                        + retained_output_matrix_bytes
                    ),
                    "fixed_model_and_batch_bytes": (
                        stager.memory_config.model_memory_bytes
                        + stager.memory_config.raw_batch_memory_bytes
                    ),
                    "final_metadata_strategy": "resident",
                },
            },
        )
        metadata_stager.discard_output(output_name)
        yield materialization
        del (
            assembly,
            benchmark_dataset,
            bucket,
            embeddings,
            final_metadata_bytes,
            groups,
            labels,
            materialization,
            provenance,
            selected_factory,
            token_metadata,
        )


def _assembly_metadata(assembly: MatrixAssembly) -> Dict[str, Any]:
    return {
        "strategy": assembly.strategy,
        "staging_strategy": assembly.staging_strategy,
        "required_bytes": assembly.required_bytes,
        "budget_bytes": assembly.budget_bytes,
    }


def _align_image(
    features: Any,
    layout: SpatialLayout,
    annotation: SegmentationAnnotation,
    image_index: Any,
    output_name: str,
    config: SegmentationConfig,
    annotation_transform: Optional[Any] = None,
) -> Iterator[Dict[str, Any]]:
    grid = _feature_grid(features, layout)
    if annotation_transform is not None:
        transformed = annotation_transform(annotation)
        annotation = (
            transformed
            if isinstance(transformed, SegmentationAnnotation)
            else SegmentationAnnotation(
                semantic=transformed,
                ignore_labels=annotation.ignore_labels,
                class_metadata=annotation.class_metadata,
            )
        ).normalized()
    mask = np.asarray(annotation.semantic)
    height, width = mask.shape
    for row in range(layout.grid_height):
        y0 = int(np.floor(row * height / layout.grid_height))
        y1 = max(y0 + 1, int(np.floor((row + 1) * height / layout.grid_height)))
        for column in range(layout.grid_width):
            x0 = int(np.floor(column * width / layout.grid_width))
            x1 = max(x0 + 1, int(np.floor((column + 1) * width / layout.grid_width)))
            region = mask[y0:y1, x0:x1]
            labels, counts = np.unique(region, return_counts=True)
            ranked = np.argsort(-counts)
            label = labels[ranked[0]]
            best = float(counts[ranked[0]] / region.size)
            second = float(counts[ranked[1]] / region.size) if len(ranked) > 1 else 0.0
            reason = None
            if any(label == ignored for ignored in annotation.ignore_labels):
                reason = "ignored_label"
            elif best < config.coverage_threshold:
                reason = "coverage"
            elif best - second < config.ambiguity_margin:
                reason = "ambiguity"
            class_info = annotation.class_metadata.get(label, {})
            is_background = bool(class_info.get("background", label == config.background_label))
            is_thing = bool(class_info.get("is_thing", annotation.instance is not None))
            if is_background and config.background_mode == "ignore":
                reason = "background"
            if is_thing and not config.include_things:
                reason = "thing_disabled"
            if not is_thing and not is_background and not config.include_stuff:
                reason = "stuff_disabled"
            resolved_label = label
            instance_id = None
            if annotation.instance is not None and is_thing and not is_background:
                instance_region = np.asarray(annotation.instance)[y0:y1, x0:x1]
                matching = instance_region[region == label]
                if matching.size:
                    ids, id_counts = np.unique(matching, return_counts=True)
                    eligible = [
                        index
                        for index, candidate_id in enumerate(ids)
                        if not any(
                            _values_equal(candidate_id, ignored_id)
                            for ignored_id in config.ignore_instance_ids
                        )
                    ]
                    if eligible:
                        best_id_index = max(eligible, key=lambda index: int(id_counts[index]))
                        instance_id = ids[best_id_index]
            yield {
                "embedding": np.asarray(grid[row, column], dtype=float).reshape(1, -1),
                "label": resolved_label,
                "image_id": image_index,
                "row": row,
                "column": column,
                "bounds": [y0, x0, y1, x1],
                "coverage": best,
                "second_coverage": second,
                "instance_id": instance_id,
                "is_background": is_background,
                "is_thing": is_thing,
                "output_name": output_name,
                "drop_reason": reason,
            }


def _feature_grid(features: Any, layout: SpatialLayout) -> np.ndarray:
    array = np.asarray(features)
    if array.ndim == 2:
        tokens = array[layout.special_tokens :]
        expected = layout.grid_height * layout.grid_width
        if tokens.shape[0] != expected:
            raise ValueError(
                f"Spatial token count {tokens.shape[0]} does not match declared grid "
                f"{layout.grid_height}x{layout.grid_width}."
            )
        return tokens.reshape(layout.grid_height, layout.grid_width, -1)
    if array.ndim != 3:
        raise ValueError("Per-image spatial features must be [tokens, dim] or a 3D feature map.")
    if layout.channel_axis not in {-1, 0, 2}:
        raise ValueError("channel_axis must identify the first or last feature-map axis.")
    if layout.channel_axis in {0}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[:2] != (layout.grid_height, layout.grid_width):
        raise ValueError(
            f"Feature map shape {array.shape[:2]} does not match declared grid "
            f"{layout.grid_height}x{layout.grid_width}."
        )
    return array


def _sample_candidates(
    stager: IncrementalMetadataStager,
    output_name: str,
    config: SegmentationConfig,
) -> int:
    selected = 0
    for reference, candidate in stager.iter_rows(output_name, order="priority"):
        label = candidate["label"]
        label_key = semantic_label_key(label)
        instance_key = hash_json_exact(
            {
                "image_id": candidate["image_id"],
                "label": label,
                "instance_id": candidate["instance_id"],
            }
        )
        if candidate["is_background"] and config.max_background_tokens is not None:
            if (
                stager.counter_value(output_name, "class_counts", label_key)
                >= config.max_background_tokens
            ):
                continue
        if config.max_tokens_per_class is not None:
            if (
                stager.counter_value(output_name, "class_counts", label_key)
                >= config.max_tokens_per_class
            ):
                continue
        if candidate["instance_id"] is not None:
            known_instance = stager.has_member(
                output_name,
                "class_instances",
                label_key,
                instance_key,
            )
            if not known_instance and config.max_instances_per_class is not None:
                if (
                    stager.member_count(output_name, "class_instances", label_key)
                    >= config.max_instances_per_class
                ):
                    continue
            if config.max_tokens_per_instance is not None:
                if (
                    stager.counter_value(output_name, "instance_counts", instance_key)
                    >= config.max_tokens_per_instance
                ):
                    continue
            stager.add_member(
                output_name,
                "class_instances",
                label_key,
                instance_key,
            )
            stager.increment_counter(
                output_name,
                "instance_counts",
                instance_key,
            )
            stager.add_member(
                output_name,
                "selected_instances",
                "",
                instance_key,
            )
        stager.mark_selected(reference, output_name)
        stager.increment_counter(output_name, "class_counts", label_key)
        stager.add_member(
            output_name,
            "selected_images",
            "",
            semantic_label_key(candidate["image_id"]),
        )
        if candidate["is_background"]:
            stager.increment_counter(output_name, "summary", "background_tokens")
        selected += 1
    return selected


def _priority(candidate: Dict[str, Any], seed: int) -> int:
    return int(
        hash_json_exact(
            {
                "seed": seed,
                "image_id": candidate["image_id"],
                "output_name": candidate["output_name"],
                "row": candidate["row"],
                "column": candidate["column"],
            }
        ),
        16,
    )


def _materialization_metadata(
    bucket: Dict[str, Any],
    selected_factory: Callable[[], Iterator[Dict[str, Any]]],
    n_selected: int,
    stager: IncrementalMetadataStager,
    output_name: str,
    layout: SpatialLayout,
    config: SegmentationConfig,
) -> Dict[str, Any]:
    return {
        "candidate_tokens": bucket["candidate_tokens"],
        "retained_tokens": n_selected,
        "n_images": stager.member_count(output_name, "selected_images"),
        "n_instances": stager.member_count(output_name, "selected_instances"),
        "background_tokens": stager.counter_value(
            output_name,
            "summary",
            "background_tokens",
        ),
        "background_labels": _unique_semantic_values(
            candidate["label"] for candidate in selected_factory() if candidate["is_background"]
        ),
        "ignored_tokens": dict(bucket["ignored_tokens"]),
        "layout": asdict(layout),
        "config": asdict(config),
    }


def _object_array(values: Iterable[Any], expected: int) -> np.ndarray:
    result = np.empty(expected, dtype=object)
    observed = 0
    for observed, value in enumerate(values, start=1):
        if observed > expected:  # pragma: no cover - staging invariant
            raise RuntimeError("Segmentation metadata staging produced extra selected rows.")
        result[observed - 1] = value
    if observed != expected:  # pragma: no cover - staging invariant
        raise RuntimeError("Segmentation metadata staging lost selected rows.")
    return result


def _provenance(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in candidate.items() if key not in {"embedding", "embedding_ref"}
    }


def _values_equal(left: Any, right: Any) -> bool:
    result = left == right
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)


def _unique_semantic_values(values: Iterable[Any]) -> List[Any]:
    observed: Dict[str, Any] = {}
    for value in values:
        observed.setdefault(semantic_label_key(value), value)
    return list(observed.values())


def _declared_spatial_output_names(extractor: Any) -> List[str]:
    method = getattr(extractor, "spatial_output_specs", None)
    if not callable(method):
        raise ValueError(
            f"Extractor {getattr(extractor, 'name', '<unknown>')!r} must declare "
            "spatial_output_specs()."
        )
    raw_names = [getattr(spec, "name", None) for spec in list(method())]
    if not raw_names or any(not isinstance(name, str) or not name.strip() for name in raw_names):
        raise ValueError("Spatial output specs must use non-empty string names.")
    names = [name for name in raw_names if isinstance(name, str)]
    if len(names) != len(set(names)):
        raise ValueError("Spatial output spec names must be unique.")
    return names


def _validate_output_names(outputs: List[Any], expected: List[str]) -> None:
    actual = [getattr(output, "name", None) for output in outputs]
    if any(not isinstance(name, str) or not name for name in actual):
        raise ValueError("Spatial extractor returned an invalid output name.")
    if len(actual) != len(set(actual)):
        raise ValueError("Spatial extractor returned duplicate output names.")
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(
            f"Spatial extractor returned outputs {actual!r}; expected {expected!r} on every "
            "batch."
        )


def _spatial_output_contract(output: Any) -> Dict[str, Any]:
    return {
        "layout": asdict(output.layout),
        "recipe": hash_json_exact(dict(output.recipe)),
        "metadata": hash_json_exact(dict(output.metadata)),
        "annotation_transform": _callable_name(output.annotation_transform),
    }


def _validate_spatial_embedding_contract(
    bucket: Dict[str, Any],
    features: Any,
    *,
    output_name: str,
    image_id: Any,
) -> None:
    grid = _feature_grid(features, bucket["layout"])
    contract = {"embedding_dim": int(grid.shape[-1]), "dtype": str(grid.dtype)}
    expected = bucket["embedding_contract"]
    if expected is None:
        bucket["embedding_contract"] = contract
    elif contract != expected:
        raise ValueError(
            f"Spatial output {output_name!r} changed embedding contract for image "
            f"{image_id!r}; expected {expected}, received {contract}."
        )


def _callable_name(function: Any) -> Optional[str]:
    if function is None:
        return None
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise ValueError("Annotation transforms must expose a stable module and qualified name.")
    return f"{module}:{qualname}"
