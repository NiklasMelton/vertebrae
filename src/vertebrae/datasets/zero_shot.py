"""Dataset contracts for frozen zero-shot semantic-alignment evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from vertebrae.cache.fingerprint import hash_json_exact
from vertebrae.datasets.base import BenchmarkDataset
from vertebrae.utils.labels import SINGLE_LABEL_TARGET
from vertebrae.utils.semantic_labels import (
    LABEL_ENCODING,
    portable_json,
    semantic_label_catalog,
    semantic_label_keys,
    strict_json_dumps,
)


@dataclass(frozen=True)
class ZeroShotClassSpec:
    """One declared target class and its fixed text prompts."""

    label: Any
    prompts: Tuple[str, ...]
    template_ids: Optional[Tuple[str, ...]] = None


@dataclass
class ZeroShotDataset:
    """A single-label dataset paired with an explicit fixed prompt protocol.

    The underlying :class:`BenchmarkDataset` supplies raw samples and labels.  Prompt
    text is deliberately part of this dataset rather than hidden in an extractor so
    that it participates in cache keys, reports, and reproducible recipes.
    """

    dataset: BenchmarkDataset
    class_specs: Sequence[ZeroShotClassSpec]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _validated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.validated()

    @classmethod
    def from_dataset(
        cls,
        dataset: BenchmarkDataset,
        class_prompts: Mapping[Any, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ZeroShotDataset":
        """Create a protocol from fully rendered prompts keyed by class label."""

        specs = [
            ZeroShotClassSpec(label=label, prompts=_normalize_prompts(prompts, label))
            for label, prompts in class_prompts.items()
        ]
        return cls(dataset=dataset, class_specs=specs, metadata=dict(metadata or {}))

    @classmethod
    def from_templates(
        cls,
        dataset: BenchmarkDataset,
        templates: Iterable[str],
        *,
        class_names: Optional[Mapping[Any, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ZeroShotDataset":
        """Expand explicit ``{label}`` templates for each observed class."""

        template_values = tuple(templates)
        if not template_values:
            raise ValueError("templates must contain at least one prompt template.")
        for template in template_values:
            if not isinstance(template, str) or not template.strip() or "{label}" not in template:
                raise ValueError(
                    "Every zero-shot prompt template must be a non-empty string containing "
                    "'{label}'."
                )
        dataset.validate()
        labels = _ordered_labels(dataset.y)
        names = dict(class_names or {})
        if class_names is not None and set(names) != set(labels):
            raise ValueError("class_names must contain exactly the observed dataset labels.")
        specs = []
        for label in labels:
            if class_names is None:
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(
                        "class_names is required when zero-shot dataset labels are not "
                        "non-empty strings."
                    )
                name = label
            else:
                name = names[label]
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("class_names values must be non-empty strings.")
            specs.append(
                ZeroShotClassSpec(
                    label=label,
                    prompts=tuple(template.format(label=name) for template in template_values),
                    template_ids=template_values,
                )
            )
        return cls(dataset=dataset, class_specs=specs, metadata=dict(metadata or {}))

    @property
    def samples(self) -> Any:
        """Return source samples in their original deterministic order."""

        return self.dataset.X

    @property
    def labels(self) -> np.ndarray:
        """Return canonical single-label targets."""

        return self.dataset.y

    @property
    def modality(self) -> str:
        return str(self.dataset.modality)

    def validated(self) -> "ZeroShotDataset":
        """Validate the source target and explicit prompt protocol."""

        if self._validated:
            return self
        self.dataset.validate()
        if self.dataset.metadata.get("target_type", "auto") != SINGLE_LABEL_TARGET:
            raise ValueError("ZeroShotDataset supports single-label targets only.")
        labels = _ordered_labels(self.dataset.y)
        for label in labels:
            _ensure_exact_label_identity(label, "dataset labels")
        by_label: Dict[Any, ZeroShotClassSpec] = {}
        all_prompts: set[str] = set()
        template_presence: list[Optional[Tuple[str, ...]]] = []
        for spec in self.class_specs:
            _ensure_hashable(spec.label, "ZeroShotClassSpec.label")
            _ensure_exact_label_identity(spec.label, "ZeroShotClassSpec.label")
            if spec.label in by_label:
                raise ValueError(f"Duplicate zero-shot class specification for {spec.label!r}.")
            prompts = _normalize_prompts(spec.prompts, spec.label)
            if any(prompt in all_prompts for prompt in prompts):
                raise ValueError("Zero-shot prompts must be unique across the complete protocol.")
            all_prompts.update(prompts)
            if spec.template_ids is not None:
                if len(spec.template_ids) != len(prompts) or any(
                    not isinstance(identifier, str) or not identifier
                    for identifier in spec.template_ids
                ):
                    raise ValueError(
                        "template_ids must be non-empty strings aligned one-to-one with prompts."
                    )
                template_presence.append(tuple(spec.template_ids))
            else:
                template_presence.append(None)
            by_label[spec.label] = ZeroShotClassSpec(
                label=spec.label,
                prompts=prompts,
                template_ids=None if spec.template_ids is None else tuple(spec.template_ids),
            )
        if set(by_label) != set(labels):
            raise ValueError(
                "Zero-shot class specifications must contain exactly the observed dataset labels."
            )
        if any(value is None for value in template_presence) and any(
            value is not None for value in template_presence
        ):
            raise ValueError("template_ids must be supplied for every class or for no classes.")
        if template_presence and template_presence[0] is not None and any(
            value != template_presence[0] for value in template_presence
        ):
            raise ValueError(
                "Template-generated zero-shot classes must share the same template IDs."
            )
        try:
            portable_json(self.metadata)
            portable_json(_source_sample_ids(self.dataset, len(self.dataset.y)))
        except TypeError as exc:
            raise ValueError(
                "Zero-shot metadata and sample IDs must be deterministically JSON-serializable."
            ) from exc
        self.class_specs = tuple(by_label[label] for label in labels)
        self._validated = True
        return self

    def prompt_rows(self) -> Tuple[Tuple[str, ...], Tuple[Any, ...], Optional[Tuple[str, ...]]]:
        """Return flattened prompts, their class labels, and optional template IDs."""

        self.validated()
        prompts: list[str] = []
        prompt_labels: list[Any] = []
        template_ids: list[str] = []
        has_templates = all(spec.template_ids is not None for spec in self.class_specs)
        for spec in self.class_specs:
            prompts.extend(spec.prompts)
            prompt_labels.extend([spec.label] * len(spec.prompts))
            if has_templates and spec.template_ids is not None:
                template_ids.extend(spec.template_ids)
        return tuple(prompts), tuple(prompt_labels), tuple(template_ids) if has_templates else None

    def summary(self) -> Dict[str, Any]:
        self.validated()
        prompts, _prompt_labels, template_ids = self.prompt_rows()
        return portable_json(
            {
                "modality": "zero_shot",
                "sample_modality": self.modality,
                "n_samples": len(self.labels),
                "n_classes": len(self.class_specs),
                "n_prompts": len(prompts),
                "prompt_ensemble": any(len(spec.prompts) > 1 for spec in self.class_specs),
                "template_generated": template_ids is not None,
                "protocol": self.protocol_recipe(),
                "source_dataset": self.dataset.summary(),
                "metadata": self.metadata,
            }
        )

    def protocol_recipe(self) -> Dict[str, Any]:
        """Return complete, stable prompt provenance for artifacts and JSON reports."""

        self.validated()
        prompts, prompt_labels, template_ids = self.prompt_rows()
        class_labels = [spec.label for spec in self.class_specs]
        label_catalog = semantic_label_catalog(class_labels)
        class_keys = semantic_label_keys(class_labels)
        sample_labels = semantic_label_keys(self.labels.tolist())
        prompt_keys = semantic_label_keys(prompt_labels)
        sample_ids = portable_json(_source_sample_ids(self.dataset, len(self.labels)))
        recipe = {
            "label_encoding": LABEL_ENCODING,
            "source_dataset_fingerprint": self.dataset.fingerprint(),
            "class_specs": [
                {
                    "label": class_keys[index],
                    "prompts": list(spec.prompts),
                    "template_ids": list(spec.template_ids) if spec.template_ids else None,
                }
                for index, spec in enumerate(self.class_specs)
            ],
            "ordered_labels": class_keys,
            "sample_labels": sample_labels,
            "sample_ids": sample_ids,
            "prompts": list(prompts),
            "prompt_labels": prompt_keys,
            "template_ids": list(template_ids) if template_ids is not None else None,
            "label_catalog": label_catalog,
            "metadata": portable_json(self.metadata),
            "n_samples": len(sample_labels),
            "n_prompts": len(prompts),
        }
        recipe["protocol_fingerprint"] = hash_json_exact(recipe)
        strict_json_dumps(recipe)
        return recipe

    def fingerprint(self) -> str:
        return str(self.protocol_recipe()["protocol_fingerprint"])


def _normalize_prompts(value: Any, label: Any) -> Tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"Zero-shot class {label!r} must declare at least one prompt.")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in values):
        raise ValueError(f"Zero-shot prompts for {label!r} must be non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"Zero-shot prompts for {label!r} must not contain duplicates.")
    return tuple(values)


def _ordered_labels(labels: Any) -> Tuple[Any, ...]:
    ordered = []
    seen = set()
    for label in np.asarray(labels, dtype=object).tolist():
        _ensure_hashable(label, "dataset labels")
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return tuple(ordered)


def _ensure_hashable(value: Any, name: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"{name} entries must be hashable.") from exc


def _ensure_exact_label_identity(value: Any, name: str) -> None:
    """Reject labels that cannot participate in reproducible protocol identities."""

    try:
        hash_json_exact(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} entries must have a stable exact identity. Use strings, numbers, "
            "UUIDs, enums, datetimes, Decimals, Fractions, or dataclasses; convert "
            "arbitrary custom labels to one of those forms."
        ) from exc


def _source_sample_ids(dataset: BenchmarkDataset, n_samples: int) -> list[Any]:
    values = dataset.metadata.get("sample_indices")
    if values is None:
        return list(range(n_samples))
    if len(values) != n_samples:
        raise ValueError("Dataset sample_indices metadata must align with zero-shot samples.")
    return list(values)
