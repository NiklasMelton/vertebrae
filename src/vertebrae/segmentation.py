"""Spatial feature alignment and token materialization for segmentation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from vertebrae.config import SegmentationConfig
from vertebrae.datasets.base import BenchmarkDataset
from vertebrae.datasets.identity import DatasetIdentity
from vertebrae.datasets.segmentation import SegmentationAnnotation, SegmentationDataset
from vertebrae.extractors.spatial import SpatialLayout


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
) -> List[SegmentationMaterialization]:
    """Extract, align, sample, and flatten all declared spatial outputs."""

    resolved = config or SegmentationConfig()
    dataset.validate()
    source_image_indices = dataset.metadata.get("sample_indices")
    if source_image_indices is None:
        source_image_indices = list(range(len(dataset.annotations)))
    extractor.fit(dataset.X, None)
    collected: Dict[str, Dict[str, Any]] = {}
    for batch in dataset.iter_batches(batch_size=batch_size):
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
                    "candidates": [],
                },
            )
            if bucket["layout"] != output.layout:
                raise ValueError(f"Spatial output {output.name!r} changed layout between batches.")
            for local_index, image_index in enumerate(batch.indices):
                source_image_index = int(source_image_indices[int(image_index)])
                bucket["candidates"].extend(
                    _align_image(
                        output.embeddings[local_index],
                        output.layout,
                        dataset.annotations[int(image_index)],
                        image_index=source_image_index,
                        output_name=output.name,
                        config=resolved,
                        annotation_transform=output.annotation_transform,
                    )
                )

    materializations = []
    for output_name, bucket in collected.items():
        selected = _sample_candidates(bucket["candidates"], resolved)
        if not selected:
            raise ValueError(f"Segmentation output {output_name!r} produced no valid tokens.")
        embeddings = np.vstack([candidate["embedding"] for candidate in selected])
        labels = np.asarray([candidate["label"] for candidate in selected], dtype=object)
        groups = np.asarray([candidate["image_id"] for candidate in selected])
        token_metadata = _materialization_metadata(
            bucket["candidates"], selected, bucket["layout"], resolved
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
                    "config": resolved,
                    "provenance": [_provenance(candidate) for candidate in selected],
                },
            ),
            metadata={
                "segmentation": token_metadata,
                "spatial_output": output_name,
                "source_dataset_identity_key": dataset.identity_key(),
            },
        ).with_groups(groups, name="image_id")
        materializations.append(
            SegmentationMaterialization(
                name=output_name,
                dataset=benchmark_dataset,
                provenance=[_provenance(candidate) for candidate in selected],
                metadata={
                    **token_metadata,
                    "output_recipe": bucket["recipe"],
                    "output_metadata": bucket["metadata"],
                },
            )
        )
    return materializations


def _align_image(
    features: Any,
    layout: SpatialLayout,
    annotation: SegmentationAnnotation,
    image_index: int,
    output_name: str,
    config: SegmentationConfig,
    annotation_transform: Optional[Any] = None,
) -> List[Dict[str, Any]]:
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
    candidates = []
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
            candidates.append(
                {
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
            )
    return candidates


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
    candidates: Iterable[Dict[str, Any]],
    config: SegmentationConfig,
) -> List[Dict[str, Any]]:
    valid = [candidate for candidate in candidates if candidate["drop_reason"] is None]
    valid.sort(key=lambda candidate: _priority(candidate, config.random_state))
    selected = []
    class_counts: Dict[Any, int] = {}
    instance_counts: Dict[Any, int] = {}
    class_instances: Dict[Any, set] = {}
    for candidate in valid:
        label = candidate["label"]
        instance_key = (candidate["image_id"], label, candidate["instance_id"])
        if candidate["is_background"] and config.max_background_tokens is not None:
            if class_counts.get(label, 0) >= config.max_background_tokens:
                continue
        if config.max_tokens_per_class is not None:
            if class_counts.get(label, 0) >= config.max_tokens_per_class:
                continue
        if candidate["instance_id"] is not None:
            instances = class_instances.setdefault(label, set())
            if instance_key not in instances and config.max_instances_per_class is not None:
                if len(instances) >= config.max_instances_per_class:
                    continue
            if config.max_tokens_per_instance is not None:
                if instance_counts.get(instance_key, 0) >= config.max_tokens_per_instance:
                    continue
            instances.add(instance_key)
            instance_counts[instance_key] = instance_counts.get(instance_key, 0) + 1
        selected.append(candidate)
        class_counts[label] = class_counts.get(label, 0) + 1
    return sorted(
        selected,
        key=lambda candidate: (
            candidate["image_id"],
            candidate["row"],
            candidate["column"],
        ),
    )


def _priority(candidate: Dict[str, Any], seed: int) -> int:
    import hashlib

    payload = (
        f"{seed}|{candidate['image_id']}|{candidate['output_name']}|"
        f"{candidate['row']}|{candidate['column']}"
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def _materialization_metadata(
    candidates: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    layout: SpatialLayout,
    config: SegmentationConfig,
) -> Dict[str, Any]:
    ignored: Dict[str, int] = {}
    for candidate in candidates:
        if candidate["drop_reason"]:
            reason = candidate["drop_reason"]
            ignored[reason] = ignored.get(reason, 0) + 1
    return {
        "candidate_tokens": len(candidates),
        "retained_tokens": len(selected),
        "n_images": len({candidate["image_id"] for candidate in selected}),
        "n_instances": len(
            {
                (candidate["image_id"], candidate["label"], candidate["instance_id"])
                for candidate in selected
                if candidate["instance_id"] is not None
            }
        ),
        "background_tokens": sum(candidate["is_background"] for candidate in selected),
        "background_labels": list(
            {candidate["label"] for candidate in selected if candidate["is_background"]}
        ),
        "ignored_tokens": ignored,
        "layout": asdict(layout),
        "config": asdict(config),
    }


def _provenance(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != "embedding"}


def _values_equal(left: Any, right: Any) -> bool:
    result = left == right
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)
