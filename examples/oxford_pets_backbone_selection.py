"""Select a frozen vision backbone and head on Oxford-IIIT Pet.

The experiment asks whether frozen representation geometry can screen which
backbone/layer outputs are worth downstream head training. It keeps four roles separate:

* a head-training split fits downstream heads;
* a representation-selection probe supplies OverlapIndex and Separatix evidence;
* a head-validation split checks the fixed head recipes;
* the official test split remains untouched until final accuracy measurement.

Oxford-IIIT Pet trimaps create foreground-only and background-swapped audit
conditions. The backbone/layer ranking uses clean breed OverlapIndex. Separatix runs
on the same clean head-training rows used by the downstream heads and its actual
aligned MLP-versus-linear evidence chooses the head family. Background sensitivity
remains a separate diagnostic rather than changing the head-selection task.

A relational audit reuses the frozen embeddings for same-breed verification with
same-species hard negatives. It diagnoses Separatix after combining the head-train
and validation pairs, reconstructs its exact family recipes, and scores them once on
held-out test pairs. It compares raw endpoint concatenation with explicit
absolute-difference and product interactions; test-near-best stars are retrospective
audit markers, never selection inputs.

Install and run from the repository root:

    poetry install -E backbone-selection
    poetry run python examples/oxford_pets_backbone_selection.py

The first run downloads Oxford-IIIT Pet and the requested pretrained checkpoints.
Use ``--models`` to run a smaller subset; standard model-provider offline settings
can require checkpoints to come from local caches.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from _common import ensure_output_dir

from vertebrae import (
    BenchmarkDataset,
    DatasetIdentity,
    Evaluator,
    OverlapScoringConfig,
    SeparatixConfig,
    StabilityConfig,
)
from vertebrae.config import CacheConfig
from vertebrae.extractors import (
    HFVisionExtractor,
    OpenCLIPExtractor,
    PrecomputedExtractor,
    TimmVisionExtractor,
)
from vertebrae.scoring.separatix import probe_summary_for_result

_MODEL_ORDER = (
    "dinov2-small",
    "deit-tiny",
    "convnext-tiny",
    "mobilenetv3-large",
    "openclip-vit-b-32",
)
_DISPLAY_NAMES = {
    "dinov2-small": "DINOv2-Small",
    "deit-tiny": "DeiT-Tiny",
    "convnext-tiny": "ConvNeXt-Tiny",
    "mobilenetv3-large": "MobileNetV3-Large",
    "openclip-vit-b-32": "OpenCLIP ViT-B/32",
}
_HEAD_FAMILIES = ("linear", "mlp")
_RELATIONAL_HEADS = ("linear", "smooth_poly", "knn", "kernel_approx", "mlp")
_RELATIONAL_FAMILIES = ("linear", "smooth_nonlinear", "local_kernel", "mlp")
_RELATIONAL_HEAD_FAMILY = {
    "linear": "linear",
    "smooth_poly": "smooth_nonlinear",
    "knn": "local_kernel",
    "kernel_approx": "local_kernel",
    "mlp": "mlp",
}
_PAIR_COMPOSITIONS = ("concatenation", "interaction")
_MEASUREMENT_TARGET_LABELS = (
    ("clean", "breed", "Clean breed"),
    ("clean", "species", "Clean species"),
    ("foreground", "breed", "Foreground breed"),
    ("background_swapped", "breed", "Swapped breed"),
    ("background_swapped", "background_species", "Background species"),
)
_HEATMAP_TARGET_LABELS = _MEASUREMENT_TARGET_LABELS[:-1]
_OUTPUT_ORDER = {
    "early_cls": 0,
    "middle_cls": 1,
    "late_cls": 2,
    "final_cls": 3,
    "final": 3,
    "final_image": 3,
}


@dataclass(frozen=True)
class PetSample:
    """One Oxford-IIIT Pet image with aligned trimap and labels."""

    sample_id: str
    image_path: Path
    trimap_path: Path
    breed: str
    species: str
    source_split: str


@dataclass(frozen=True)
class BackgroundSwap:
    """A target pet paired with a separately sampled donor background."""

    target: PetSample
    donor: PetSample
    background_species: str


@dataclass(frozen=True)
class VerificationPair:
    """One source-disjoint same-breed verification example."""

    left_index: int
    right_index: int
    target: int
    species: str


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(exc)
        print("Install the experiment dependencies with: poetry install -E backbone-selection")
        return

    output_dir = ensure_output_dir()
    figure_dir = args.figure_dir or output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.replot_from is not None:
        try:
            figure_paths = _replot_saved_results(args.replot_from, figure_dir, plt)
        except ValueError as exc:
            print(f"Could not replot saved results: {exc}")
            return
        for path in figure_paths:
            print(f"Figure written to {path}")
        return

    try:
        from torchvision.datasets import OxfordIIITPet
    except ImportError as exc:
        print(exc)
        print("Install the experiment dependencies with: poetry install -E backbone-selection")
        return

    try:
        trainval_source = OxfordIIITPet(
            root=str(args.data_dir),
            split="trainval",
            target_types=("category", "binary-category", "segmentation"),
            download=not args.no_download,
        )
        test_source = OxfordIIITPet(
            root=str(args.data_dir),
            split="test",
            target_types=("category", "binary-category", "segmentation"),
            download=not args.no_download,
        )
    except (OSError, RuntimeError) as exc:
        print(f"Could not prepare Oxford-IIIT Pet: {exc}")
        print("Remove --no-download for the first run or point --data-dir at an existing cache.")
        return

    trainval_samples = _records_from_torchvision(trainval_source, "trainval")
    test_samples = _records_from_torchvision(test_source, "test")
    head_train, selection, validation = _stratified_trainval_split(
        trainval_samples,
        head_train_per_breed=args.head_train_per_breed,
        selection_per_breed=args.selection_per_breed,
        validation_per_breed=args.validation_per_breed,
        seed=args.seed,
    )
    test = _stratified_subset(test_samples, args.test_per_breed, args.seed + 1)
    selection_swaps = _balanced_background_donors(selection, args.seed + 2)
    test_swaps = _balanced_background_donors(test, args.seed + 3)
    relational_pairs = {
        "head_train": _same_breed_verification_pairs(head_train, args.seed + 11),
        "validation": _same_breed_verification_pairs(validation, args.seed + 12),
        "test": _same_breed_verification_pairs(test, args.seed + 13),
    }

    print(
        "Prepared Oxford-IIIT Pet protocol: "
        f"head train={len(head_train)}, selection={len(selection)}, "
        f"validation={len(validation)}, test={len(test)}."
    )
    print("Rendering foreground and background interventions.")
    selection_foreground = [_render_foreground(sample) for sample in selection]
    selection_background_swapped = [_render_background_swap(pair) for pair in selection_swaps]
    test_background_swapped = [_render_background_swap(pair) for pair in test_swaps]

    requested_models = _parse_model_names(args.models)
    extractors = _build_extractors(
        requested_models,
        batch_size=args.embedding_batch_size,
        device=args.device,
    )
    original_samples = [*head_train, *selection, *validation, *test]
    original_inputs = [str(sample.image_path) for sample in original_samples]
    intervention_inputs = [
        *selection_foreground,
        *selection_background_swapped,
        *test_background_swapped,
    ]
    original_slices = _named_slices(
        (
            ("head_train", len(head_train)),
            ("selection", len(selection)),
            ("validation", len(validation)),
            ("test", len(test)),
        )
    )
    intervention_slices = _named_slices(
        (
            ("selection_foreground", len(selection)),
            ("selection_swapped", len(selection)),
            ("test_swapped", len(test)),
        )
    )

    metric_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    relational_rows: List[Dict[str, Any]] = []
    relational_head_rows: List[Dict[str, Any]] = []
    relational_results: Dict[str, Any] = {}
    vertebrae_results: Dict[str, Any] = {}
    for model_name, extractor in zip(requested_models, extractors):
        print(f"Extracting {_DISPLAY_NAMES[model_name]} representations.")
        try:
            original_outputs = _transform_many(extractor, original_inputs)
            intervention_outputs = _transform_many(extractor, intervention_inputs)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"Could not run {_DISPLAY_NAMES[model_name]}: {exc}")
            print("Check the requested optional extra, checkpoint access, and local model cache.")
            return
        intervention_by_name = {output.name: output for output in intervention_outputs}
        for original_output in original_outputs:
            if original_output.name not in intervention_by_name:
                raise ValueError(
                    f"Extractor {extractor.name!r} changed output names across conditions."
                )
            intervention_output = intervention_by_name[original_output.name]
            representation = f"{_DISPLAY_NAMES[model_name]} · {_layer_label(original_output.name)}"
            original = _split_embeddings(original_output.embeddings, original_slices)
            intervention = _split_embeddings(
                intervention_output.embeddings,
                intervention_slices,
            )
            print(f"Scoring {representation}.")
            measurements, serialized = _representation_measurements(
                representation=representation,
                model_name=model_name,
                output_name=original_output.name,
                head_train_embeddings=original["head_train"],
                selection_embeddings=original["selection"],
                foreground_embeddings=intervention["selection_foreground"],
                swapped_embeddings=intervention["selection_swapped"],
                head_train=head_train,
                selection=selection,
                selection_swaps=selection_swaps,
                overlap_k=args.overlap_k,
                stability_repeats=args.stability_repeats,
                near_optimal_margin=args.near_optimal_margin,
                mlp_min_improvement=args.mlp_min_improvement,
                mlp_trigger_skill_threshold=args.mlp_trigger_skill_threshold,
                seed=args.seed,
            )
            metric_rows.extend(measurements)
            serialized["source_extractor_recipe"] = extractor.recipe()
            vertebrae_results[representation] = serialized
            head_evidence = _separatix_mlp_evidence(serialized["head_selection"])
            selected_head, selection_reason = _select_head(
                head_evidence,
                min_improvement=args.mlp_min_improvement,
            )
            run_rows = _evaluate_head_families(
                representation=representation,
                model_name=model_name,
                output_name=original_output.name,
                train_embeddings=original["head_train"],
                validation_embeddings=original["validation"],
                clean_test_embeddings=original["test"],
                swapped_test_embeddings=intervention["test_swapped"],
                train_labels=_breeds(head_train),
                validation_labels=_breeds(validation),
                test_labels=_breeds(test),
                selected_head=selected_head,
                head_result=serialized["head_selection"],
                repeats=args.head_repeats,
                seed=args.seed,
            )
            head_rows.extend(run_rows)
            candidate_rows.append(
                _candidate_summary(
                    representation=representation,
                    model_name=model_name,
                    output_name=original_output.name,
                    measurements=measurements,
                    head_rows=run_rows,
                    selected_head=selected_head,
                    selection_reason=selection_reason,
                    head_evidence=head_evidence,
                    mlp_min_improvement=args.mlp_min_improvement,
                    mlp_trigger_skill_threshold=args.mlp_trigger_skill_threshold,
                )
            )
            print(f"Evaluating same-breed relational compositions for {representation}.")
            composition_rows, composition_head_rows, composition_results = (
                _evaluate_relational_compositions(
                    representation=representation,
                    model_name=model_name,
                    output_name=original_output.name,
                    train_embeddings=original["head_train"],
                    validation_embeddings=original["validation"],
                    test_embeddings=original["test"],
                    train_pairs=relational_pairs["head_train"],
                    validation_pairs=relational_pairs["validation"],
                    test_pairs=relational_pairs["test"],
                    overlap_k=args.overlap_k,
                    near_optimal_margin=args.near_optimal_margin,
                    mlp_min_improvement=args.mlp_min_improvement,
                    mlp_trigger_skill_threshold=args.mlp_trigger_skill_threshold,
                    seed=args.seed,
                )
            )
            relational_rows.extend(composition_rows)
            relational_head_rows.extend(composition_head_rows)
            relational_results[representation] = composition_results

    candidate_rows = _rank_candidates(candidate_rows)
    head_choice_audit = _head_choice_audit_summary(candidate_rows)
    relational_audit = _relational_audit_summary(relational_rows)
    _write_csv(output_dir / "oxford_pets_backbone_metrics.csv", metric_rows)
    _write_csv(output_dir / "oxford_pets_head_runs.csv", head_rows)
    _write_csv(output_dir / "oxford_pets_backbone_selection.csv", candidate_rows)
    _write_csv(output_dir / "oxford_pets_head_choice_audit.csv", [head_choice_audit])
    _write_csv(output_dir / "oxford_pets_relational_composition.csv", relational_rows)
    _write_csv(output_dir / "oxford_pets_relational_head_runs.csv", relational_head_rows)
    _write_csv(output_dir / "oxford_pets_relational_audit.csv", [relational_audit])
    _write_json(
        output_dir / "oxford_pets_backbone_selection.json",
        {
            "schema_version": 2,
            "protocol": _protocol_payload(
                args,
                requested_models,
                head_train,
                selection,
                validation,
                test,
                relational_pairs,
            ),
            "vertebrae_results": vertebrae_results,
            "metrics": metric_rows,
            "head_runs": head_rows,
            "candidate_selection": candidate_rows,
            "head_choice_audit": head_choice_audit,
            "relational_composition": relational_rows,
            "relational_head_runs": relational_head_rows,
            "relational_results": relational_results,
            "relational_audit": relational_audit,
        },
    )
    figure_paths = [
        *_plot_overlap_heatmap(metric_rows, figure_dir, plt),
        *_plot_overlap_accuracy_scatter(metric_rows, head_rows, candidate_rows, figure_dir, plt),
        *_plot_selection_budget(candidate_rows, head_rows, figure_dir, plt),
        *_plot_head_choice_audit(candidate_rows, figure_dir, plt),
        *_plot_background_shift_effect(candidate_rows, figure_dir, plt),
        *_plot_relational_composition(relational_rows, figure_dir, plt),
    ]
    print("\nSelection-probe ranking (test accuracy was not used):")
    for row in candidate_rows:
        print(
            f"{int(row['selection_rank']):>2}. {row['representation']}: "
            f"clean OI={row['clean_breed_overlap']:.3f}, "
            f"head={row['selected_head']}"
        )
    print(
        "Head-choice audit: "
        f"material agreement={head_choice_audit['material_agreement_count']}/"
        f"{head_choice_audit['candidate_count']}, "
        "mean validation regret="
        f"{_format_optional_metric(head_choice_audit['mean_validation_regret'])}."
    )
    print(
        "Relational composition audit: "
        f"near-optimal recommendations={relational_audit['near_optimal_count']}/"
        f"{relational_audit['recommendation_count']}, "
        "mean held-out test regret="
        f"{_format_optional_metric(relational_audit['mean_test_regret'])}."
    )
    print(f"\nReports written to {output_dir}")
    for path in figure_paths:
        print(f"Figure written to {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(_MODEL_ORDER),
        help=f"Comma-separated model keys: {', '.join(_MODEL_ORDER)}.",
    )
    parser.add_argument("--head-train-per-breed", type=int, default=24)
    parser.add_argument("--selection-per-breed", type=int, default=12)
    parser.add_argument("--validation-per-breed", type=int, default=12)
    parser.add_argument("--test-per-breed", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--overlap-k", type=int, default=2)
    parser.add_argument("--stability-repeats", type=int, default=5)
    parser.add_argument("--head-repeats", type=int, default=3)
    parser.add_argument(
        "--near-optimal-margin",
        type=float,
        default=0.02,
        help=(
            "Retrospective accuracy margin used to call a family near-best; it "
            "does not affect Separatix selection."
        ),
    )
    parser.add_argument(
        "--mlp-min-improvement",
        type=float,
        default=0.02,
        help="Minimum paired Separatix MLP improvement required for an override.",
    )
    parser.add_argument(
        "--mlp-trigger-skill-threshold",
        type=float,
        default=1.0,
        help="Run optional MLP probes when simpler normalized skill is below this value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("examples/data"))
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument(
        "--replot-from",
        type=Path,
        default=None,
        help="Regenerate figures from an existing experiment JSON without rerunning models.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require Oxford-IIIT Pet to exist in the local torchvision cache.",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in (
        "head_train_per_breed",
        "selection_per_breed",
        "validation_per_breed",
        "test_per_breed",
        "embedding_batch_size",
        "overlap_k",
        "stability_repeats",
        "head_repeats",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1.")
    if args.selection_per_breed < 2 * args.overlap_k:
        parser.error("--selection-per-breed must be at least twice --overlap-k.")
    if args.selection_per_breed % 2 or args.test_per_breed % 2:
        parser.error("--selection-per-breed and --test-per-breed must be even.")
    if not 0.0 <= args.near_optimal_margin <= 1.0:
        parser.error("--near-optimal-margin must be between 0 and 1.")
    if not 0.0 <= args.mlp_min_improvement <= 1.0:
        parser.error("--mlp-min-improvement must be between 0 and 1.")
    if not 0.0 <= args.mlp_trigger_skill_threshold <= 1.0:
        parser.error("--mlp-trigger-skill-threshold must be between 0 and 1.")
    _parse_model_names(args.models, parser=parser)


def _parse_model_names(
    value: str,
    *,
    parser: Optional[argparse.ArgumentParser] = None,
) -> List[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(_MODEL_ORDER))
    if not names or unknown or len(names) != len(set(names)):
        message = "--models must contain unique supported keys"
        if unknown:
            message += f"; unknown: {', '.join(unknown)}"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return names


def _records_from_torchvision(dataset: Any, source_split: str) -> List[PetSample]:
    required = ("_images", "_segs", "_labels", "_bin_labels", "classes", "bin_classes")
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise ValueError(f"OxfordIIITPet object is missing expected fields: {missing}.")
    return [
        PetSample(
            sample_id=Path(image_path).stem,
            image_path=Path(image_path),
            trimap_path=Path(trimap_path),
            breed=str(dataset.classes[int(breed_index)]),
            species=str(dataset.bin_classes[int(species_index)]),
            source_split=source_split,
        )
        for image_path, trimap_path, breed_index, species_index in zip(
            dataset._images,
            dataset._segs,
            dataset._labels,
            dataset._bin_labels,
        )
    ]


def _stratified_trainval_split(
    samples: Sequence[PetSample],
    *,
    head_train_per_breed: int,
    selection_per_breed: int,
    validation_per_breed: int,
    seed: int,
) -> Tuple[List[PetSample], List[PetSample], List[PetSample]]:
    required = head_train_per_breed + selection_per_breed + validation_per_breed
    grouped = _group_by_breed(samples)
    rng = np.random.default_rng(seed)
    splits: Tuple[List[PetSample], List[PetSample], List[PetSample]] = ([], [], [])
    for breed in sorted(grouped):
        values = grouped[breed]
        if len(values) < required:
            raise ValueError(
                f"Breed {breed!r} has {len(values)} trainval rows; {required} are required."
            )
        selected = [values[index] for index in rng.permutation(len(values))[:required]]
        boundaries = (
            head_train_per_breed,
            head_train_per_breed + selection_per_breed,
        )
        splits[0].extend(selected[: boundaries[0]])
        splits[1].extend(selected[boundaries[0] : boundaries[1]])
        splits[2].extend(selected[boundaries[1] :])
    return splits


def _stratified_subset(
    samples: Sequence[PetSample],
    per_breed: int,
    seed: int,
) -> List[PetSample]:
    grouped = _group_by_breed(samples)
    rng = np.random.default_rng(seed)
    selected: List[PetSample] = []
    for breed in sorted(grouped):
        values = grouped[breed]
        if len(values) < per_breed:
            raise ValueError(f"Breed {breed!r} has {len(values)} rows; {per_breed} are required.")
        selected.extend(values[index] for index in rng.permutation(len(values))[:per_breed])
    return selected


def _group_by_breed(samples: Sequence[PetSample]) -> Dict[str, List[PetSample]]:
    grouped: Dict[str, List[PetSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.breed].append(sample)
    return {
        breed: sorted(values, key=lambda sample: sample.sample_id)
        for breed, values in grouped.items()
    }


def _same_breed_verification_pairs(
    samples: Sequence[PetSample],
    seed: int,
) -> List[VerificationPair]:
    """Build balanced same-breed pairs with hard, same-species negatives.

    Each source image appears in at most one pair. Per breed, half of the retained
    rows form positive pairs and half enter the negative pool. This makes the
    species distribution identical for both targets and prevents a classifier from
    solving breed verification with the easier cat-versus-dog distinction.
    """

    indexed_by_breed: Dict[str, List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indexed_by_breed[sample.breed].append(index)
    if len(indexed_by_breed) < 2:
        raise ValueError("Same-breed verification requires at least two breeds.")

    rng = np.random.default_rng(seed)
    positives: List[VerificationPair] = []
    negative_pools: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
    for breed in sorted(indexed_by_breed):
        indices = np.asarray(indexed_by_breed[breed], dtype=int)
        pair_count = len(indices) // 4
        if pair_count < 1:
            raise ValueError(
                f"Breed {breed!r} needs at least four rows for relational verification."
            )
        shuffled = indices[rng.permutation(len(indices))]
        positive_indices = shuffled[: 2 * pair_count]
        negative_indices = shuffled[2 * pair_count : 4 * pair_count]
        species = samples[int(shuffled[0])].species
        if any(samples[int(index)].species != species for index in shuffled):
            raise ValueError(f"Breed {breed!r} contains inconsistent species labels.")
        for offset in range(0, len(positive_indices), 2):
            positives.append(
                VerificationPair(
                    left_index=int(positive_indices[offset]),
                    right_index=int(positive_indices[offset + 1]),
                    target=1,
                    species=species,
                )
            )
        negative_pools[species][breed] = [int(index) for index in negative_indices]

    negatives: List[VerificationPair] = []
    for species in sorted(negative_pools):
        negatives.extend(
            _pair_different_breeds(
                negative_pools[species],
                species=species,
                rng=rng,
            )
        )
    if len(positives) != len(negatives):
        raise ValueError(
            "Relational pairing could not balance same-breed and different-breed rows."
        )

    pairs = [*positives, *negatives]
    randomized: List[VerificationPair] = []
    for pair in pairs:
        if rng.random() < 0.5:
            pair = VerificationPair(
                left_index=pair.right_index,
                right_index=pair.left_index,
                target=pair.target,
                species=pair.species,
            )
        randomized.append(pair)
    order = rng.permutation(len(randomized))
    return [randomized[int(index)] for index in order]


def _pair_different_breeds(
    pools: Mapping[str, Sequence[int]],
    *,
    species: str,
    rng: np.random.Generator,
) -> List[VerificationPair]:
    """Consume equally supported breed pools into different-breed pairs."""

    import heapq

    work: Dict[str, List[int]] = {}
    for breed, values in pools.items():
        shuffled = np.asarray(values, dtype=int)
        shuffled = shuffled[rng.permutation(len(shuffled))]
        work[breed] = [int(value) for value in shuffled]
    heap = [(-len(values), breed) for breed, values in work.items() if values]
    heapq.heapify(heap)
    pairs: List[VerificationPair] = []
    while len(heap) >= 2:
        _, left_breed = heapq.heappop(heap)
        _, right_breed = heapq.heappop(heap)
        left_index = work[left_breed].pop()
        right_index = work[right_breed].pop()
        pairs.append(
            VerificationPair(
                left_index=left_index,
                right_index=right_index,
                target=0,
                species=species,
            )
        )
        if work[left_breed]:
            heapq.heappush(heap, (-len(work[left_breed]), left_breed))
        if work[right_breed]:
            heapq.heappush(heap, (-len(work[right_breed]), right_breed))
    remaining = sum(len(values) for values in work.values())
    if remaining:
        raise ValueError(
            f"Could not form different-breed pairs for species {species!r}; "
            f"{remaining} source rows remained."
        )
    return pairs


def _balanced_background_donors(
    samples: Sequence[PetSample],
    seed: int,
) -> List[BackgroundSwap]:
    species = sorted({sample.species for sample in samples})
    if len(species) != 2:
        raise ValueError(f"Background balancing expects two species; found {species}.")
    grouped = _group_by_breed(samples)
    rng = np.random.default_rng(seed)
    assignments: Dict[str, BackgroundSwap] = {}
    for breed in sorted(grouped):
        targets = grouped[breed]
        candidates = {
            species_name: [
                sample
                for sample in samples
                if sample.species == species_name and sample.breed != breed
            ]
            for species_name in species
        }
        for species_name, values in candidates.items():
            if not values:
                raise ValueError(
                    f"No {species_name!r} background donors are available outside breed {breed!r}."
                )
            rng.shuffle(values)
        offsets = {species_name: 0 for species_name in species}
        for index, target in enumerate(targets):
            donor_species = species[index % 2]
            values = candidates[donor_species]
            donor = values[offsets[donor_species] % len(values)]
            offsets[donor_species] += 1
            assignments[target.sample_id] = BackgroundSwap(
                target=target,
                donor=donor,
                background_species=donor_species,
            )
    return [assignments[sample.sample_id] for sample in samples]


def _render_foreground(sample: PetSample, background_value: int = 127) -> Any:
    from PIL import Image

    with Image.open(sample.image_path) as image_source, Image.open(sample.trimap_path) as trimap:
        image = image_source.convert("RGB")
        mask = _foreground_mask(trimap, image.size)
        background = Image.new("RGB", image.size, color=(background_value,) * 3)
        return Image.composite(image, background, mask)


def _render_background_swap(pair: BackgroundSwap, blur_radius: Optional[float] = None) -> Any:
    from PIL import Image, ImageFilter

    with (
        Image.open(pair.target.image_path) as target_source,
        Image.open(pair.target.trimap_path) as target_trimap,
        Image.open(pair.donor.image_path) as donor_source,
        Image.open(pair.donor.trimap_path) as donor_trimap,
    ):
        target = target_source.convert("RGB")
        donor = donor_source.convert("RGB")
        donor_mask = np.asarray(donor_trimap.convert("L"))
        donor_values = np.asarray(donor, dtype=np.uint8).copy()
        background_pixels = donor_values[donor_mask == 2]
        fill = (
            np.median(background_pixels, axis=0).astype(np.uint8)
            if len(background_pixels)
            else np.median(donor_values.reshape(-1, 3), axis=0).astype(np.uint8)
        )
        donor_values[donor_mask != 2] = fill
        canvas = Image.fromarray(donor_values)
        radius = max(2.0, min(canvas.size) / 32.0) if blur_radius is None else float(blur_radius)
        if radius > 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=radius))
        canvas = canvas.resize(target.size, resample=Image.Resampling.BILINEAR)
        target_mask = _foreground_mask(target_trimap, target.size)
        return Image.composite(target, canvas, target_mask)


def _foreground_mask(trimap: Any, size: Tuple[int, int]) -> Any:
    from PIL import Image

    values = trimap.convert("L")
    if values.size != size:
        values = values.resize(size, resample=Image.Resampling.NEAREST)
    mask = (np.asarray(values) != 2).astype(np.uint8) * 255
    return Image.fromarray(mask)


def _build_extractors(
    model_names: Sequence[str],
    *,
    batch_size: int,
    device: Optional[str],
) -> List[Any]:
    extractors = []
    for model_name in model_names:
        if model_name == "dinov2-small":
            extractors.append(
                HFVisionExtractor(
                    name=model_name,
                    model_id="facebook/dinov2-small",
                    outputs=[
                        {"name": "early_cls", "hidden_layer": 3, "pooling": "cls"},
                        {"name": "middle_cls", "hidden_layer": 6, "pooling": "cls"},
                        {"name": "late_cls", "hidden_layer": 9, "pooling": "cls"},
                        {"name": "final_cls", "hidden_layer": -1, "pooling": "cls"},
                    ],
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                    processor_kwargs={"use_fast": False},
                )
            )
        elif model_name == "deit-tiny":
            extractors.append(
                HFVisionExtractor(
                    name=model_name,
                    model_id="facebook/deit-tiny-patch16-224",
                    outputs=[
                        {"name": "early_cls", "hidden_layer": 3, "pooling": "cls"},
                        {"name": "middle_cls", "hidden_layer": 6, "pooling": "cls"},
                        {"name": "late_cls", "hidden_layer": 9, "pooling": "cls"},
                        {"name": "final_cls", "hidden_layer": -1, "pooling": "cls"},
                    ],
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                    processor_kwargs={"use_fast": False},
                    model_kwargs={"add_pooling_layer": False},
                )
            )
        elif model_name == "convnext-tiny":
            extractors.append(
                TimmVisionExtractor(
                    name=model_name,
                    model_name="convnext_tiny",
                    pretrained=True,
                    outputs=[{"name": "final"}],
                    model_kwargs={"num_classes": 0},
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                )
            )
        elif model_name == "mobilenetv3-large":
            extractors.append(
                TimmVisionExtractor(
                    name=model_name,
                    model_name="mobilenetv3_large_100",
                    pretrained=True,
                    outputs=[{"name": "final"}],
                    model_kwargs={"num_classes": 0},
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                )
            )
        elif model_name == "openclip-vit-b-32":
            extractors.append(
                OpenCLIPExtractor(
                    name=model_name,
                    model_name="ViT-B-32",
                    pretrained="laion2b_s34b_b79k",
                    input_modalities={"image": "image"},
                    outputs=[{"name": "final_image", "source": "image"}],
                    batch_size=batch_size,
                    image_mode="rgb",
                    device=device,
                )
            )
        else:  # pragma: no cover - validated by _parse_model_names
            raise ValueError(f"Unsupported model {model_name!r}.")
    return extractors


def _transform_many(extractor: Any, images: Sequence[Any]) -> List[Any]:
    if getattr(extractor, "extractor_type", None) == "openclip":
        return extractor.transform_many({"image": list(images)})
    return extractor.transform_many(list(images))


def _named_slices(parts: Sequence[Tuple[str, int]]) -> Dict[str, slice]:
    slices: Dict[str, slice] = {}
    start = 0
    for name, size in parts:
        slices[name] = slice(start, start + size)
        start += size
    return slices


def _split_embeddings(values: Any, slices: Mapping[str, slice]) -> Dict[str, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    expected = max(item.stop or 0 for item in slices.values())
    if len(matrix) != expected:
        raise ValueError(
            f"Embedding rows changed order or count; expected {expected}, got {len(matrix)}."
        )
    return {name: matrix[item] for name, item in slices.items()}


def _representation_measurements(
    *,
    representation: str,
    model_name: str,
    output_name: str,
    head_train_embeddings: np.ndarray,
    selection_embeddings: np.ndarray,
    foreground_embeddings: np.ndarray,
    swapped_embeddings: np.ndarray,
    head_train: Sequence[PetSample],
    selection: Sequence[PetSample],
    selection_swaps: Sequence[BackgroundSwap],
    overlap_k: int,
    stability_repeats: int,
    near_optimal_margin: float,
    mlp_min_improvement: float,
    mlp_trigger_skill_threshold: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    targets = {
        ("clean", "breed"): (selection_embeddings, _breeds(selection), True),
        ("clean", "species"): (selection_embeddings, _species(selection), False),
        ("foreground", "breed"): (foreground_embeddings, _breeds(selection), False),
        ("background_swapped", "breed"): (swapped_embeddings, _breeds(selection), True),
        ("background_swapped", "background_species"): (
            swapped_embeddings,
            np.asarray([pair.background_species for pair in selection_swaps]),
            False,
        ),
    }
    rows: List[Dict[str, Any]] = []
    serialized: Dict[str, Any] = {}
    for condition, target, _ in _MEASUREMENT_TARGET_LABELS:
        embeddings, labels, stability = targets[(condition, target)]
        measurement, result = _score_embeddings(
            embeddings,
            labels,
            name=f"{representation}:{condition}:{target}",
            overlap_k=overlap_k,
            stability_repeats=stability_repeats if stability else 0,
            run_separatix=False,
            groups=None,
            mlp_min_improvement=mlp_min_improvement,
            mlp_trigger_skill_threshold=mlp_trigger_skill_threshold,
            seed=seed,
        )
        rows.append(
            {
                "representation": representation,
                "model": model_name,
                "output": output_name,
                "condition": condition,
                "target": target,
                **measurement,
            }
        )
        serialized[f"{condition}:{target}"] = result

    head_measurement, head_result = _score_embeddings(
        head_train_embeddings,
        _breeds(head_train),
        name=f"{representation}:head-selection",
        overlap_k=overlap_k,
        stability_repeats=0,
        run_separatix=True,
        groups=None,
        mlp_min_improvement=mlp_min_improvement,
        mlp_trigger_skill_threshold=mlp_trigger_skill_threshold,
        seed=seed,
    )
    rows.append(
        {
            "representation": representation,
            "model": model_name,
            "output": output_name,
            "condition": "clean_head_train",
            "target": "breed_head_selection",
            **head_measurement,
        }
    )
    serialized["head_selection"] = head_result
    return rows, serialized


def _score_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    overlap_k: int,
    stability_repeats: int,
    run_separatix: bool,
    groups: Optional[np.ndarray],
    mlp_min_improvement: float,
    mlp_trigger_skill_threshold: float,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    identity = DatasetIdentity.from_manifest(
        "oxford-pets-backbone-selection-probe",
        {"name": name, "rows": int(len(labels)), "seed": int(seed)},
    )
    dataset = BenchmarkDataset.from_embeddings(
        embeddings,
        labels,
        identity=identity,
        metadata={"example": "oxford_pets_backbone_selection", "probe": name},
    )
    if groups is not None:
        dataset = dataset.with_groups(groups, name="source_image")
    result = Evaluator(
        dataset=dataset,
        extractor=PrecomputedExtractor(name=name, cache_embeddings=False),
        scoring_config=OverlapScoringConfig(
            k=overlap_k,
            min_k=overlap_k,
            max_k=overlap_k,
            min_samples_per_cluster=2,
            kmeans_kwargs={"random_state": seed},
        ),
        stability_config=(
            StabilityConfig(repeats=stability_repeats, random_state=seed)
            if stability_repeats
            else StabilityConfig(enabled=False)
        ),
        separatix_config=(
            SeparatixConfig(
                enabled=True,
                overlap_threshold=0.0,
                random_state=seed,
                budget="standard",
                mlp_probes=True,
                mlp_trigger_skill_threshold=mlp_trigger_skill_threshold,
                mlp_min_improvement=mlp_min_improvement,
            )
            if run_separatix
            else SeparatixConfig(enabled=False)
        ),
        cache_config=CacheConfig(enabled=False),
    ).run()
    item = result.extractor_results[0]
    frame_row = result.to_dataframe(include_invalid=True).iloc[0].to_dict()
    probe_summary = probe_summary_for_result(item.separatix)
    measurement = {
        "overlap_macro": float(frame_row["overlap_macro"]),
        "stability_lower": _optional_float(frame_row.get("stability_interval_lower")),
        "stability_upper": _optional_float(frame_row.get("stability_interval_upper")),
        "probe_linear_score": _optional_float(frame_row.get("probe_linear_score")),
        "probe_nonlinear_score": _optional_float(frame_row.get("probe_nonlinear_score")),
        "probe_nonlinear_delta": _optional_float(frame_row.get("probe_nonlinear_delta")),
        "probe_grouped": bool(frame_row.get("probe_grouped")) if run_separatix else None,
    }
    payload = result.to_dict()
    payload["probe_summary"] = probe_summary
    return measurement, payload


def _separatix_mlp_evidence(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the optional aligned MLP-versus-linear evidence from one result."""

    extractor_results = result.get("extractor_results") or []
    extractor_result = extractor_results[0] if extractor_results else {}
    separatix = extractor_result.get("separatix") or {}
    report = separatix.get("report") or {}
    metrics = report.get("metrics") or {}
    trigger = metrics.get("mlp_trigger_evidence") or {}
    mlp = metrics.get("mlp_recommendation_evidence") or {}
    comparison = (mlp.get("pairwise_comparisons") or {}).get("linear") or {}
    aligned = mlp.get("aligned_comparators") or {}
    linear = aligned.get("linear") or {}
    best = mlp.get("best_architecture") or {}
    architecture_payload = (metrics.get("mlp_probes") or {}).get("architectures", [])
    if isinstance(architecture_payload, Mapping):
        architecture_payload = list(architecture_payload.values())
    architecture_results = {
        str(item.get("probe_name")): item
        for item in architecture_payload
        if isinstance(item, Mapping) and item.get("probe_name")
    }
    best_result = architecture_results.get(str(best.get("probe_name")), {})
    aligned_payload = (metrics.get("mlp_probes") or {}).get("aligned_comparators", {})
    if isinstance(aligned_payload, Mapping):
        linear_payload = aligned_payload.get("linear") or {}
    elif isinstance(aligned_payload, Sequence) and not isinstance(aligned_payload, (str, bytes)):
        linear_payload = next(
            (
                item
                for item in aligned_payload
                if isinstance(item, Mapping) and item.get("probe_name") == "linear"
            ),
            {},
        )
    else:
        linear_payload = {}
    linear_recipe = linear.get("probe_recipe") or linear_payload.get("probe_recipe")
    mlp_recipe = (
        best_result.get("probe_recipe")
        or best.get("probe_recipe")
        or _find_probe_recipe(metrics, best.get("probe_name"))
    )
    return {
        "status": mlp.get("status"),
        "trigger_status": trigger.get("status"),
        "trigger_good_enough": trigger.get("good_enough"),
        "trigger_threshold": trigger.get("threshold"),
        "trigger_reason": trigger.get("reason"),
        "recommendation": report.get("recommendation"),
        "recommendation_override": bool(mlp.get("recommendation_override")),
        "override_reason": mlp.get("override_reason") or mlp.get("reason"),
        "best_architecture": best.get("probe_name"),
        "mlp_score": _optional_float(best.get("balanced_accuracy")),
        "linear_score": _optional_float(linear.get("balanced_accuracy")),
        "evaluation_mode": linear.get("evaluation_mode"),
        "mean_delta": _optional_float(comparison.get("mean_delta")),
        "lower_95": _optional_float(comparison.get("lower_95")),
        "upper_95": _optional_float(comparison.get("upper_95")),
        "clear_advantage": comparison.get("clear_advantage"),
        "absolute_skill": _optional_float(mlp.get("absolute_skill")),
        "linear_recipe": linear_recipe,
        "linear_recipe_id": _recipe_id(linear_recipe),
        "mlp_recipe": mlp_recipe,
        "mlp_recipe_id": _recipe_id(mlp_recipe),
        "mlp_recipe_status": (best_result.get("probe_recipe_status") or {}).get("status"),
    }


def _select_head(
    evidence: Mapping[str, Any],
    *,
    min_improvement: float,
) -> Tuple[str, str]:
    if evidence.get("recommendation_override"):
        delta = _optional_float(evidence.get("mean_delta"))
        rendered = "unavailable" if delta is None else f"{delta:.3f}"
        return (
            "mlp",
            f"Separatix's aligned MLP override was active (MLP−linear={rendered}; "
            f"minimum improvement {min_improvement:.3f}).",
        )
    if evidence.get("status") == "not_triggered":
        threshold = _optional_float(evidence.get("trigger_threshold"))
        rendered = "configured" if threshold is None else f"{threshold:.3f}"
        return (
            "linear",
            f"Simpler probes cleared the {rendered} normalized-skill threshold, "
            "so Separatix did not run an MLP probe.",
        )
    if evidence.get("status") == "completed":
        delta = _optional_float(evidence.get("mean_delta"))
        lower = _optional_float(evidence.get("lower_95"))
        upper = _optional_float(evidence.get("upper_95"))
        if delta is not None and lower is not None and upper is not None:
            comparison = f"MLP−linear={delta:.3f}, 95% interval [{lower:.3f}, {upper:.3f}]"
        else:
            comparison = "an aligned MLP−linear comparison was unavailable"
        return (
            "linear",
            f"Separatix ran MLP probes but did not activate its override ({comparison}).",
        )
    reason = evidence.get("override_reason") or evidence.get("trigger_reason")
    suffix = "" if reason is None else f" {reason}"
    return "linear", f"Separatix produced no actionable MLP override.{suffix}"


def _compose_pair_embeddings(
    embeddings: np.ndarray,
    pairs: Sequence[VerificationPair],
    composition: str,
) -> np.ndarray:
    """Compose normalized endpoints without fitting a task-aware transform."""

    if composition not in _PAIR_COMPOSITIONS:
        raise ValueError(f"Unknown relational composition {composition!r}.")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Relational composition requires a two-dimensional embedding matrix.")
    if not pairs:
        raise ValueError("Relational composition requires at least one pair.")
    indices = np.asarray(
        [(pair.left_index, pair.right_index) for pair in pairs],
        dtype=int,
    )
    if indices.min() < 0 or indices.max() >= len(matrix):
        raise ValueError("A relational pair index falls outside the embedding matrix.")
    normalized = _l2_normalize_rows(matrix)
    left = normalized[indices[:, 0]]
    right = normalized[indices[:, 1]]
    if composition == "concatenation":
        composed = np.hstack([left, right])
    else:
        composed = np.hstack([np.abs(left - right), left * right])
    return _l2_normalize_rows(composed).astype(np.float32, copy=False)


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


def _pair_targets(pairs: Sequence[VerificationPair]) -> np.ndarray:
    return np.asarray([pair.target for pair in pairs], dtype=np.int64)


def _separatix_family_evidence(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the primary family recommendation plus any conditional MLP override."""

    extractor_results = result.get("extractor_results") or []
    extractor_result = extractor_results[0] if extractor_results else {}
    separatix = extractor_result.get("separatix") or {}
    report = separatix.get("report") or {}
    metrics = report.get("metrics") or {}
    recommendation = metrics.get("recommendation_evidence") or {}
    families = recommendation.get("families") or {}
    mlp = metrics.get("mlp_recommendation_evidence") or {}
    guidance = separatix.get("family_guidance") or {}
    mlp_guidance = guidance.get("mlp_override")
    if isinstance(mlp_guidance, Mapping):
        mlp_override = bool(
            mlp_guidance.get("active", mlp_guidance.get("recommendation_override", False))
        )
    elif mlp_guidance is not None:
        mlp_override = bool(mlp_guidance)
    else:
        mlp_override = bool(mlp.get("recommendation_override"))
    selected_family = guidance.get("selected_family")
    if selected_family is None:
        selected_family = "mlp" if mlp_override else recommendation.get("recommended_family")
    minimum_family = guidance.get("minimum_recommended_family")
    if minimum_family is None:
        minimum_family = recommendation.get("recommended_family")
    plausible_families = list(guidance.get("plausible_families") or [])
    if not plausible_families:
        plausible = (recommendation.get("plausible_family_set") or {}).get("plausible_families")
        plausible_families = list(plausible or [])
    mlp_best_name = (
        mlp_guidance.get("probe_name") if isinstance(mlp_guidance, Mapping) else None
    ) or (mlp.get("best_architecture") or {}).get("probe_name")
    family_probes: Dict[str, Optional[str]] = {}
    family_recipes: Dict[str, Any] = {}
    for family in _RELATIONAL_FAMILIES:
        if family == "mlp":
            probe_name = mlp_best_name
        else:
            family_payload = families.get(family) or {}
            probe_name = family_payload.get("best_probe")
        family_probes[family] = str(probe_name) if probe_name else None
        family_recipes[family] = _find_probe_recipe(metrics, probe_name)
    selected_payload = families.get(selected_family) or {}
    if selected_family == "mlp":
        selected_payload = mlp.get("best_architecture") or {}
    selected_probe = guidance.get("selected_probe") or (
        mlp_best_name if selected_family == "mlp" else selected_payload.get("best_probe")
    )
    selected_recipe = _find_probe_recipe(metrics, selected_probe)
    probe_summary = result.get("probe_summary") or {}
    evaluation = probe_summary.get("evaluation") or {}
    paired = guidance.get("paired") or {}
    if not paired:
        paired = report.get("metrics", {}).get("paired_probe_comparisons") or {}
    return {
        "recommendation": report.get("recommendation"),
        "confidence": report.get("confidence"),
        "raw_best_family": recommendation.get("raw_best_family"),
        "recommended_family": selected_family,
        "minimum_recommended_family": minimum_family,
        "plausible_families": plausible_families,
        "recommended_probe": selected_probe,
        "recommended_recipe": selected_recipe,
        "recommended_recipe_id": _recipe_id(selected_recipe),
        "family_probes": family_probes,
        "family_recipes": family_recipes,
        "best_clearly_beats_dummy": recommendation.get("best_clearly_beats_dummy"),
        "mlp_status": mlp.get("status"),
        "mlp_override": mlp_override,
        "paired": {
            "status": paired.get("status"),
            "method": paired.get("method"),
            "evaluation_plan_id": paired.get("evaluation_plan_id"),
        },
        "effective_train_size": evaluation.get("effective_train_size_summary"),
        "decision_method": guidance.get("decision_method")
        or (recommendation.get("plausible_family_set") or {}).get("decision_method"),
    }


def _recipe_id(recipe: Any) -> Optional[str]:
    if isinstance(recipe, Mapping) and recipe.get("recipe_id"):
        return str(recipe["recipe_id"])
    return None


def _find_probe_recipe(metrics: Mapping[str, Any], probe_name: Any) -> Any:
    """Find an emitted Separatix recipe by probe name or recipe id."""

    if not probe_name:
        return None
    wanted = str(probe_name)
    candidates: List[Mapping[str, Any]] = []
    probes = metrics.get("probes") or {}
    if isinstance(probes, Mapping):
        candidates.extend(value for value in probes.values() if isinstance(value, Mapping))
    elif isinstance(probes, Sequence) and not isinstance(probes, (str, bytes)):
        candidates.extend(value for value in probes if isinstance(value, Mapping))
    mlp_probes = metrics.get("mlp_probes") or {}
    if isinstance(mlp_probes, Mapping):
        for key in ("architectures", "aligned_comparators"):
            values = mlp_probes.get(key) or {}
            if isinstance(values, Mapping):
                candidates.extend(value for value in values.values() if isinstance(value, Mapping))
            elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                candidates.extend(value for value in values if isinstance(value, Mapping))
    for payload in candidates:
        recipe = payload.get("probe_recipe")
        if not isinstance(recipe, Mapping):
            continue
        probe = recipe.get("probe") or {}
        if wanted in {str(payload.get("probe_name")), str(probe.get("name")), _recipe_id(recipe)}:
            return recipe
    return None


class _RecipeUnavailable(RuntimeError):
    """A Separatix recipe cannot be reconstructed in this runtime."""


def _make_recipe_estimator(recipe: Any) -> Any:
    if not isinstance(recipe, Mapping):
        raise _RecipeUnavailable("Separatix did not emit a reconstructable probe recipe.")
    try:
        from separatix import make_probe_estimator
    except (ImportError, AttributeError) as exc:
        raise _RecipeUnavailable(
            "Separatix 0.1.1 with make_probe_estimator is required for aligned heads."
        ) from exc
    try:
        return make_probe_estimator(recipe, version_policy="error")
    except Exception as exc:
        raise _RecipeUnavailable(f"Could not reconstruct Separatix probe recipe: {exc}") from exc


def _evaluate_relational_compositions(
    *,
    representation: str,
    model_name: str,
    output_name: str,
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_pairs: Sequence[VerificationPair],
    validation_pairs: Sequence[VerificationPair],
    test_pairs: Sequence[VerificationPair],
    overlap_k: int,
    near_optimal_margin: float,
    mlp_min_improvement: float,
    mlp_trigger_skill_threshold: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate the same verification task under two fixed pair compositions."""

    rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}
    targets = {
        "train": _pair_targets(train_pairs),
        "validation": _pair_targets(validation_pairs),
        "test": _pair_targets(test_pairs),
    }
    for composition in _PAIR_COMPOSITIONS:
        composed = {
            "train": _compose_pair_embeddings(train_embeddings, train_pairs, composition),
            "validation": _compose_pair_embeddings(
                validation_embeddings,
                validation_pairs,
                composition,
            ),
            "test": _compose_pair_embeddings(test_embeddings, test_pairs, composition),
        }
        development_embeddings = np.vstack([composed["train"], composed["validation"]])
        development_targets = np.concatenate([targets["train"], targets["validation"]])
        _, serialized = _score_embeddings(
            development_embeddings,
            development_targets,
            name=f"{representation}:same-breed:{composition}",
            overlap_k=overlap_k,
            stability_repeats=0,
            run_separatix=True,
            groups=None,
            mlp_min_improvement=mlp_min_improvement,
            mlp_trigger_skill_threshold=mlp_trigger_skill_threshold,
            seed=seed,
        )
        evidence = _separatix_family_evidence(serialized)
        runs = _evaluate_relational_heads(
            representation=representation,
            model_name=model_name,
            output_name=output_name,
            composition=composition,
            development_embeddings=development_embeddings,
            test_embeddings=composed["test"],
            development_labels=development_targets,
            test_labels=targets["test"],
            family_evidence=evidence,
            seed=seed,
        )
        head_rows.extend(runs)
        rows.append(
            _relational_composition_summary(
                representation=representation,
                model_name=model_name,
                output_name=output_name,
                composition=composition,
                head_rows=runs,
                head_evidence=evidence,
                near_optimal_margin=near_optimal_margin,
                train_pairs=train_pairs,
                validation_pairs=validation_pairs,
                test_pairs=test_pairs,
            )
        )
        results[composition] = serialized
    return rows, head_rows, results


def _evaluate_relational_heads(
    *,
    representation: str,
    model_name: str,
    output_name: str,
    composition: str,
    development_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    development_labels: np.ndarray,
    test_labels: np.ndarray,
    family_evidence: Mapping[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    from sklearn.metrics import balanced_accuracy_score

    rows: List[Dict[str, Any]] = []
    family_recipes = family_evidence.get("family_recipes") or {}
    selected_family = family_evidence.get("recommended_family")
    for family in _RELATIONAL_FAMILIES:
        recipe = family_recipes.get(family)
        probe_name = (family_evidence.get("family_probes") or {}).get(family)
        base = {
            "representation": representation,
            "model": model_name,
            "output": output_name,
            "task": "same_breed_verification",
            "composition": composition,
            "head": family,
            "family": family,
            "probe": probe_name,
            "recipe_id": _recipe_id(recipe),
            "selected_by_separatix": family == selected_family,
            "repeat": 0,
            "seed": seed,
            "development_pair_count": int(len(development_labels)),
            "test_pair_count": int(len(test_labels)),
            "validation_balanced_accuracy": None,
            "test_balanced_accuracy": None,
            "status": "unavailable",
            "unavailable_reason": None,
        }
        try:
            estimator = _make_recipe_estimator(recipe)
            estimator.fit(development_embeddings, development_labels)
            score = balanced_accuracy_score(
                test_labels,
                estimator.predict(test_embeddings),
            )
            base["test_balanced_accuracy"] = float(score)
            base["status"] = "completed"
        except _RecipeUnavailable as exc:
            base["unavailable_reason"] = str(exc)
        except (ValueError, RuntimeError, TypeError) as exc:
            base["unavailable_reason"] = f"Exact Separatix probe failed: {exc}"
        rows.append(base)
    return rows


def _relational_composition_summary(
    *,
    representation: str,
    model_name: str,
    output_name: str,
    composition: str,
    head_rows: Sequence[Mapping[str, Any]],
    head_evidence: Mapping[str, Any],
    near_optimal_margin: float,
    train_pairs: Sequence[VerificationPair],
    validation_pairs: Sequence[VerificationPair],
    test_pairs: Sequence[VerificationPair],
) -> Dict[str, Any]:
    family_scores: Dict[str, Dict[str, Any]] = {}
    for family in _RELATIONAL_FAMILIES:
        candidates = [row for row in head_rows if row.get("family") == family]
        row = candidates[0] if candidates else {}
        family_scores[family] = {
            "head": family,
            "probe": row.get("probe"),
            "recipe_id": row.get("recipe_id"),
            "test": _optional_float(row.get("test_balanced_accuracy")),
            "status": row.get("status", "unavailable"),
        }
    available = {
        family: scores["test"]
        for family, scores in family_scores.items()
        if scores["test"] is not None
    }
    best_test = max(available.values()) if available else None
    empirical_family = None
    if best_test is not None:
        empirical_family = next(
            family
            for family in _RELATIONAL_FAMILIES
            if family in available and available[family] >= best_test - near_optimal_margin
        )
    recommended_family = head_evidence.get("recommended_family")
    selected = family_scores.get(str(recommended_family)) if recommended_family else None
    selected_test = selected.get("test") if selected else None
    plausible = [str(item) for item in head_evidence.get("plausible_families", [])]
    plausible_coverage = empirical_family in plausible if empirical_family is not None else None
    return {
        "representation": representation,
        "model": model_name,
        "output": output_name,
        "task": "same_breed_verification",
        "negative_pair_policy": "different breed, same species",
        "composition": composition,
        "near_optimal_margin": float(near_optimal_margin),
        "separatix_recommendation": head_evidence.get("recommendation"),
        "separatix_confidence": head_evidence.get("confidence"),
        "separatix_raw_best_family": head_evidence.get("raw_best_family"),
        "separatix_minimum_recommended_family": head_evidence.get("minimum_recommended_family"),
        "separatix_recommended_family": recommended_family,
        "separatix_selected_family": recommended_family,
        "separatix_recommended_probe": head_evidence.get("recommended_probe"),
        "separatix_recommended_recipe_id": head_evidence.get("recommended_recipe_id"),
        "separatix_family_probes": head_evidence.get("family_probes", {}),
        "separatix_family_recipe_ids": {
            family: _recipe_id(recipe)
            for family, recipe in (head_evidence.get("family_recipes") or {}).items()
        },
        "separatix_plausible_families": plausible,
        "separatix_decision_method": head_evidence.get("decision_method"),
        "separatix_paired_status": (head_evidence.get("paired") or {}).get("status"),
        "separatix_paired_method": (head_evidence.get("paired") or {}).get("method"),
        "separatix_evaluation_plan_id": (head_evidence.get("paired") or {}).get(
            "evaluation_plan_id"
        ),
        "separatix_effective_train_size": head_evidence.get("effective_train_size"),
        "separatix_effective_train_size_mean": _optional_float(
            (head_evidence.get("effective_train_size") or {}).get("mean")
            if isinstance(head_evidence.get("effective_train_size"), Mapping)
            else None
        ),
        "separatix_development_cohort_size": len(train_pairs) + len(validation_pairs),
        "separatix_mlp_status": head_evidence.get("mlp_status"),
        "separatix_mlp_override": head_evidence.get("mlp_override"),
        "empirical_simplest_near_best_family": empirical_family,
        "test_near_optimal": (
            selected_test is not None
            and best_test is not None
            and selected_test >= best_test - near_optimal_margin
        ),
        "selected_test_balanced_accuracy": selected_test,
        "best_observed_test_balanced_accuracy": best_test,
        "selected_test_regret": (
            best_test - selected_test
            if best_test is not None and selected_test is not None
            else None
        ),
        "plausible_family_coverage": plausible_coverage,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "test_pair_count": len(test_pairs),
        **{
            f"{family}_test_balanced_accuracy": scores["test"]
            for family, scores in family_scores.items()
        },
        "family_scores": family_scores,
        "local_kernel_selected_head": family_scores["local_kernel"].get("probe"),
    }


def _relational_audit_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty relational audit.")
    recommended = [row for row in rows if row.get("separatix_recommended_family") is not None]
    test_regrets = [
        float(row["selected_test_regret"])
        for row in recommended
        if row.get("selected_test_regret") is not None
    ]
    family_counts = Counter(str(row["separatix_recommended_family"]) for row in recommended)
    coverage = [
        bool(row["plausible_family_coverage"])
        for row in recommended
        if row.get("plausible_family_coverage") is not None
    ]
    return {
        "case_count": len(rows),
        "recommendation_count": len(recommended),
        "near_optimal_count": sum(
            bool(row.get("test_near_optimal", row.get("recommendation_near_optimal")))
            for row in recommended
        ),
        "near_optimal_rate": (
            float(
                sum(
                    bool(row.get("test_near_optimal", row.get("recommendation_near_optimal")))
                    for row in recommended
                )
                / len(recommended)
            )
            if recommended
            else None
        ),
        "mean_test_regret": (float(np.mean(test_regrets)) if test_regrets else None),
        "plausible_family_coverage_count": sum(coverage),
        "plausible_family_coverage_rate": (float(np.mean(coverage)) if coverage else None),
        "test_near_optimal_count": sum(
            bool(row.get("test_near_optimal", row.get("recommendation_near_optimal")))
            for row in recommended
        ),
        "test_near_optimal_rate": (
            float(
                sum(
                    bool(row.get("test_near_optimal", row.get("recommendation_near_optimal")))
                    for row in recommended
                )
                / len(recommended)
            )
            if recommended
            else None
        ),
        "recommendation_family_counts": dict(sorted(family_counts.items())),
    }


def _evaluate_head_families(
    *,
    representation: str,
    model_name: str,
    output_name: str,
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    clean_test_embeddings: np.ndarray,
    swapped_test_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    selected_head: str,
    head_result: Mapping[str, Any],
    repeats: int,
    seed: int,
) -> List[Dict[str, Any]]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    train_targets, validation_targets, test_targets = _encode_head_labels(
        train_labels,
        validation_labels,
        test_labels,
    )
    rows = []
    combined_embeddings = np.vstack([train_embeddings, validation_embeddings])
    combined_targets = np.concatenate([train_targets, validation_targets])
    evidence = _separatix_mlp_evidence(head_result)
    recipes = {
        "linear": evidence.get("linear_recipe"),
        "mlp": evidence.get("mlp_recipe"),
    }
    # Recipes encode Separatix's resolved training policy.  Replaying the exact recipe
    # keeps estimator alignment intact; repeated rows are deterministic reproductions,
    # not independent seed-based uncertainty estimates.
    for repeat in range(repeats):
        repeat_seed = seed + 101 * repeat
        for family in _HEAD_FAMILIES:
            row = {
                "representation": representation,
                "model": model_name,
                "output": output_name,
                "head": family,
                "selected_by_separatix": family == selected_head,
                "repeat": repeat,
                "seed": repeat_seed,
                "repeat_policy": "emitted_recipe_replay",
                "recipe_id": _recipe_id(recipes[family]),
                "recipe_alignment_status": "unavailable",
                "recipe_unavailable_reason": None,
                "validation_accuracy": None,
                "validation_balanced_accuracy": None,
                "clean_test_accuracy": None,
                "background_swapped_test_accuracy": None,
            }
            try:
                validation_head = _make_recipe_estimator(recipes[family])
                validation_head.fit(train_embeddings, train_targets)
                validation_predictions = validation_head.predict(validation_embeddings)
                row["validation_accuracy"] = float(
                    accuracy_score(validation_targets, validation_predictions)
                )
                row["validation_balanced_accuracy"] = float(
                    balanced_accuracy_score(validation_targets, validation_predictions)
                )
                final_head = _make_recipe_estimator(recipes[family])
                final_head.fit(combined_embeddings, combined_targets)
                row["clean_test_accuracy"] = float(
                    accuracy_score(test_targets, final_head.predict(clean_test_embeddings))
                )
                row["background_swapped_test_accuracy"] = float(
                    accuracy_score(test_targets, final_head.predict(swapped_test_embeddings))
                )
                row["recipe_alignment_status"] = "aligned"
            except _RecipeUnavailable as exc:
                row["recipe_unavailable_reason"] = str(exc)
            except (ValueError, RuntimeError, TypeError) as exc:
                row["recipe_unavailable_reason"] = f"Exact Separatix probe failed: {exc}"
            rows.append(row)
    return rows


def _encode_head_labels(
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode semantic labels before scikit-learn's internal MLP validation split.

    Scikit-learn 1.5's MLP early-stopping scorer applies numeric finite-value checks
    to predictions. Integer encoding avoids a TypeError for ordinary string breed
    names while fitting the catalog exclusively on the head-training split.
    """

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder().fit(train_labels)
    try:
        return (
            encoder.transform(train_labels),
            encoder.transform(validation_labels),
            encoder.transform(test_labels),
        )
    except ValueError as exc:
        raise ValueError(
            "Validation and test labels must be represented in the head-training split."
        ) from exc


def _make_head(family: str, seed: int) -> Any:
    """Build the legacy reference head used only by the encoding unit test.

    Experiment audits must use emitted Separatix recipes through
    :func:`_make_recipe_estimator`; this helper is intentionally not used by
    the production experiment and is retained only for a small local smoke test.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    if family == "linear":
        estimator = LogisticRegression(
            C=1.0,
            max_iter=1_000,
            random_state=seed,
        )
    elif family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown head family {family!r}.")
    return make_pipeline(Normalizer(norm="l2"), StandardScaler(), estimator)


def _candidate_summary(
    *,
    representation: str,
    model_name: str,
    output_name: str,
    measurements: Sequence[Mapping[str, Any]],
    head_rows: Sequence[Mapping[str, Any]],
    selected_head: str,
    selection_reason: str,
    head_evidence: Mapping[str, Any],
    mlp_min_improvement: float,
    mlp_trigger_skill_threshold: float,
) -> Dict[str, Any]:
    clean_overlap = _measurement_value(measurements, "clean", "breed")
    swapped_overlap = _measurement_value(measurements, "background_swapped", "breed")
    foreground_overlap = _measurement_value(measurements, "foreground", "breed")
    selected_runs = [row for row in head_rows if row["head"] == selected_head]
    linear_runs = [row for row in head_rows if row["head"] == "linear"]
    mlp_runs = [row for row in head_rows if row["head"] == "mlp"]
    repeat_rows: Dict[int, Dict[str, float]] = defaultdict(dict)
    for row in head_rows:
        value = _optional_float(row.get("validation_balanced_accuracy"))
        if value is not None:
            repeat_rows[int(row["repeat"])][str(row["head"])] = value
    observed_deltas = np.asarray(
        [
            values["mlp"] - values["linear"]
            for values in repeat_rows.values()
            if "mlp" in values and "linear" in values
        ],
        dtype=float,
    )
    selected_validation = _mean_optional(selected_runs, "validation_balanced_accuracy")
    best_validation_values = [
        _mean_optional(linear_runs, "validation_balanced_accuracy"),
        _mean_optional(mlp_runs, "validation_balanced_accuracy"),
    ]
    best_validation = (
        max(value for value in best_validation_values if value is not None)
        if any(value is not None for value in best_validation_values)
        else None
    )
    return {
        "representation": representation,
        "model": model_name,
        "output": output_name,
        "robust_breed_overlap": min(clean_overlap, swapped_overlap),
        "clean_breed_overlap": clean_overlap,
        "foreground_breed_overlap": foreground_overlap,
        "background_swapped_breed_overlap": swapped_overlap,
        "selected_head": selected_head,
        "selection_reason": selection_reason,
        "mlp_min_improvement": float(mlp_min_improvement),
        "mlp_trigger_skill_threshold": float(mlp_trigger_skill_threshold),
        "mlp_probe_status": head_evidence.get("status"),
        "mlp_trigger_status": head_evidence.get("trigger_status"),
        "mlp_trigger_good_enough": head_evidence.get("trigger_good_enough"),
        "mlp_recommendation_override": head_evidence.get("recommendation_override"),
        "mlp_probe_architecture": head_evidence.get("best_architecture"),
        "aligned_linear_recipe_id": head_evidence.get("linear_recipe_id"),
        "aligned_mlp_recipe_id": head_evidence.get("mlp_recipe_id"),
        "aligned_mlp_recipe_status": head_evidence.get("mlp_recipe_status"),
        "head_recipe_alignment_status": {
            str(head): sorted(
                {
                    str(row.get("recipe_alignment_status"))
                    for row in head_rows
                    if row.get("head") == head
                }
            )
            for head in _HEAD_FAMILIES
        },
        "mlp_probe_balanced_accuracy": head_evidence.get("mlp_score"),
        "aligned_linear_balanced_accuracy": head_evidence.get("linear_score"),
        "mlp_vs_linear_delta": head_evidence.get("mean_delta"),
        "mlp_vs_linear_lower_95": head_evidence.get("lower_95"),
        "mlp_vs_linear_upper_95": head_evidence.get("upper_95"),
        "mlp_probe_evaluation_mode": head_evidence.get("evaluation_mode"),
        "validation_linear_balanced_accuracy": _mean_optional(
            linear_runs, "validation_balanced_accuracy"
        ),
        "validation_mlp_balanced_accuracy": _mean_optional(
            mlp_runs, "validation_balanced_accuracy"
        ),
        "validation_mlp_advantage": (
            float(observed_deltas.mean()) if observed_deltas.size else None
        ),
        "validation_mlp_advantage_std": (
            float(observed_deltas.std(ddof=0)) if observed_deltas.size else None
        ),
        "selected_head_validation_regret": (
            float(best_validation - selected_validation)
            if best_validation is not None and selected_validation is not None
            else None
        ),
        "selected_head_validation_accuracy": _mean_optional(selected_runs, "validation_accuracy"),
        "selected_head_validation_balanced_accuracy": selected_validation,
        "selected_head_clean_test_accuracy": _mean_optional(selected_runs, "clean_test_accuracy"),
        "selected_head_swapped_test_accuracy": _mean_optional(
            selected_runs,
            "background_swapped_test_accuracy",
        ),
    }


def _rank_candidates(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ranked = [dict(row) for row in rows]
    ranked.sort(
        key=lambda row: (
            float(row["clean_breed_overlap"]),
            float(row["robust_breed_overlap"]),
            row["representation"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
    return ranked


def _head_choice_audit_summary(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty head-choice audit.")
    margin = float(rows[0].get("mlp_min_improvement", 0.02))
    material_agreement = 0
    sign_agreement = 0
    aligned_rows = []
    regrets = []
    for row in rows:
        observed_delta = _optional_float(row.get("validation_mlp_advantage"))
        regret = _optional_float(row.get("selected_head_validation_regret"))
        if observed_delta is None:
            continue
        selected_mlp = row["selected_head"] == "mlp"
        materially_mlp = observed_delta >= margin
        material_agreement += int(selected_mlp == materially_mlp)
        if regret is not None:
            regrets.append(regret)
        aligned_delta = _optional_float(row.get("mlp_vs_linear_delta"))
        if aligned_delta is not None:
            aligned_rows.append((aligned_delta, observed_delta))
            sign_agreement += int((aligned_delta > 0.0) == (observed_delta > 0.0))
    correlation = (
        _spearman_correlation(
            [item[0] for item in aligned_rows],
            [item[1] for item in aligned_rows],
        )
        if aligned_rows
        else float("nan")
    )
    return {
        "candidate_count": len(rows),
        "mlp_min_improvement": margin,
        "mlp_probe_completed_count": sum(
            row.get("mlp_probe_status") == "completed" for row in rows
        ),
        "mlp_probe_not_triggered_count": sum(
            row.get("mlp_probe_status") == "not_triggered" for row in rows
        ),
        "mlp_selected_count": sum(row["selected_head"] == "mlp" for row in rows),
        "material_agreement_count": material_agreement,
        "material_agreement_rate": float(material_agreement / len(rows)),
        "aligned_comparison_count": len(aligned_rows),
        "aligned_sign_agreement_count": sign_agreement,
        "aligned_sign_agreement_rate": (
            float(sign_agreement / len(aligned_rows)) if aligned_rows else None
        ),
        "aligned_delta_spearman": correlation if np.isfinite(correlation) else None,
        "mean_validation_regret": float(np.mean(regrets)) if regrets else None,
        "max_validation_regret": float(np.max(regrets)) if regrets else None,
    }


def _measurement_value(
    rows: Sequence[Mapping[str, Any]],
    condition: str,
    target: str,
) -> float:
    matches = [row for row in rows if row["condition"] == condition and row["target"] == target]
    if len(matches) != 1:
        raise ValueError(f"Expected one {condition}/{target} measurement; found {len(matches)}.")
    return float(matches[0]["overlap_macro"])


def _plot_overlap_heatmap(
    metric_rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    representations = _ordered_representations(metric_rows)
    lookup = {
        (row["representation"], row["condition"], row["target"]): _optional_float(
            row.get("overlap_macro")
        )
        for row in metric_rows
    }
    if not representations or not any(value is not None for value in lookup.values()):
        return _plot_empty_state(
            figure_dir,
            plt,
            "oxford-pets-backbone-overlap-heatmap",
            "Target-specific structure in each frozen representation",
            "No finite OverlapIndex measurements are available for this plot.",
        )
    matrix = np.asarray(
        [
            [
                value
                if (value := lookup.get((representation, condition, target))) is not None
                else np.nan
                for condition, target, _ in _HEATMAP_TARGET_LABELS
            ]
            for representation in representations
        ],
        dtype=float,
    )
    labels = [label for _, _, label in _HEATMAP_TARGET_LABELS]
    with plt.rc_context(_plot_style()):
        figure, axis = plt.subplots(
            figsize=(12.5, max(4.8, 0.58 * len(representations) + 2.4)),
            constrained_layout=True,
        )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_xticks(np.arange(len(labels)), labels=labels)
        axis.set_yticks(np.arange(len(representations)), labels=representations)
        axis.set_title("Target-specific structure in each frozen representation")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                color = (
                    "white" if not np.isfinite(value) or value < 0.38 or value > 0.76 else "black"
                )
                axis.text(
                    column_index,
                    row_index,
                    "—" if not np.isfinite(value) else f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
        colorbar.set_label("OverlapIndex")
        axis.tick_params(axis="x", rotation=20)
        return _save_figure(figure, figure_dir, "oxford-pets-backbone-overlap-heatmap", plt)


def _replot_saved_results(path: Path, figure_dir: Path, plt: Any) -> Tuple[Path, ...]:
    if not path.is_file():
        raise ValueError(f"Saved experiment JSON does not exist: {path}.")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = (
        "metrics",
        "head_runs",
        "candidate_selection",
        "relational_composition",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Saved experiment JSON is missing fields: {missing}.")
    protocol = payload.get("protocol") or {}
    if payload.get("schema_version") != 2 or protocol.get("relational_evidence_schema") != (
        "deployment_family_composition_v2"
    ):
        raise ValueError(
            "Saved results use an older relational deployment schema; rerun the "
            "experiment before replotting."
        )
    if any("mlp_probe_status" not in row for row in payload["candidate_selection"]):
        raise ValueError(
            "Saved results predate the clean aligned-MLP head protocol; rerun the "
            "experiment before replotting."
        )
    return (
        *_plot_overlap_heatmap(payload["metrics"], figure_dir, plt),
        *_plot_overlap_accuracy_scatter(
            payload["metrics"],
            payload["head_runs"],
            payload["candidate_selection"],
            figure_dir,
            plt,
        ),
        *_plot_selection_budget(
            payload["candidate_selection"],
            payload["head_runs"],
            figure_dir,
            plt,
        ),
        *_plot_head_choice_audit(
            payload["candidate_selection"],
            figure_dir,
            plt,
        ),
        *_plot_background_shift_effect(
            payload["candidate_selection"],
            figure_dir,
            plt,
        ),
        *_plot_relational_composition(
            payload["relational_composition"],
            figure_dir,
            plt,
        ),
    )


def _plot_overlap_accuracy_scatter(
    metric_rows: Sequence[Mapping[str, Any]],
    head_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    """Plot the primary representation-screening result with one fixed head family."""

    linear_rows = [
        row
        for row in head_rows
        if row.get("head") == "linear"
        and _optional_float(row.get("clean_test_accuracy")) is not None
        and row.get("status", "completed") != "unavailable"
    ]
    linear_representations = {str(row["representation"]) for row in linear_rows}
    metric_lookup = {
        (row["representation"], row["condition"], row["target"]): row for row in metric_rows
    }
    representations = [
        representation
        for representation in _ordered_representations(metric_rows)
        if representation in linear_representations
        and _optional_float(
            (metric_lookup.get((representation, "clean", "breed")) or {}).get("overlap_macro")
        )
        is not None
    ]
    if not representations:
        return _plot_empty_state(
            figure_dir,
            plt,
            "oxford-pets-overlap-vs-head-accuracy",
            "Frozen breed geometry predicts downstream transfer",
            "No aligned linear-head test scores are available for this plot.",
        )
    model_by_representation = {
        str(row["representation"]): row["model"]
        for row in [*candidate_rows, *metric_rows, *linear_rows]
        if row.get("representation") in representations and row.get("model") is not None
    }
    colors = _model_colors([model_by_representation[item] for item in representations], plt)
    all_accuracies = [
        float(value)
        for row in linear_rows
        if row["representation"] in representations
        and (value := _optional_float(row.get("clean_test_accuracy"))) is not None
    ]
    y_lower = max(0.0, min(all_accuracies) - 0.07)
    y_upper = min(1.0, max(all_accuracies) + 0.07)
    y_span = max(0.1, y_upper - y_lower)
    with plt.rc_context(_plot_style()):
        figure, axis = plt.subplots(figsize=(10.8, 7.0), constrained_layout=True)
        x_values = [
            float(metric_lookup[(representation, "clean", "breed")]["overlap_macro"])
            for representation in representations
        ]
        x_min = min(x_values)
        x_max = max(x_values)
        x_span = max(0.1, x_max - x_min)
        label_entries = []
        plotted_y = []
        for representation, x_value in zip(representations, x_values):
            runs = [row for row in linear_rows if row["representation"] == representation]
            y_values = np.asarray([float(row["clean_test_accuracy"]) for row in runs], dtype=float)
            y_value = float(y_values.mean())
            plotted_y.append(y_value)
            axis.scatter(
                x_value,
                y_value,
                s=78,
                color=colors[model_by_representation[representation]],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            label_entries.append(
                {
                    "label": _compact_representation_label(representation),
                    "point": (x_value, y_value),
                    "value": y_value,
                }
            )
        label_positions = _spread_label_positions(
            [float(entry["value"]) for entry in label_entries],
            lower=y_lower + 0.02 * y_span,
            upper=y_upper - 0.02 * y_span,
            min_gap=0.032 * y_span,
        )
        for entry, label_y in zip(label_entries, label_positions):
            point_x, point_y = entry["point"]
            axis.annotate(
                entry["label"],
                xy=(point_x, point_y),
                xytext=(point_x + 0.018 * x_span, label_y),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#9CA3AF",
                    "linewidth": 0.7,
                    "shrinkA": 1,
                    "shrinkB": 3,
                },
                zorder=4,
            )
        rho = _spearman_correlation(x_values, plotted_y)
        axis.text(
            0.03,
            0.96,
            _correlation_label(rho),
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        axis.set_xlabel("Clean breed OverlapIndex on the selection probe")
        axis.set_ylabel("Clean test accuracy with a standardized linear head")
        axis.grid(True, color="#E5E7EB", linewidth=0.8)
        axis.set_xlim(max(0.0, x_min - 0.05 * x_span), min(1.0, x_max + 0.23 * x_span))
        axis.set_ylim(y_lower, y_upper)
        from matplotlib.lines import Line2D

        model_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=color,
                markeredgecolor=color,
                label=_DISPLAY_NAMES[model],
            )
            for model, color in colors.items()
        ]
        figure.legend(
            handles=model_handles,
            loc="outside lower center",
            ncols=min(5, len(model_handles)),
            frameon=False,
        )
        figure.suptitle(
            "Frozen breed geometry predicts downstream transfer\n"
            "The fixed linear head isolates representation quality from head-family choice",
        )
        return _save_figure(figure, figure_dir, "oxford-pets-overlap-vs-head-accuracy", plt)


def _plot_selection_budget(
    candidate_rows: Sequence[Mapping[str, Any]],
    head_rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    statistics = _selection_budget_statistics(candidate_rows, head_rows)
    if not statistics["budget"]:
        return _plot_empty_state(
            figure_dir,
            plt,
            "oxford-pets-oi-selection-budget",
            "OverlapIndex concentrates head-training budget",
            "No completed linear-head test scores are available for this plot.",
        )
    budgets = np.asarray(statistics["budget"], dtype=int)
    oi_best = np.asarray(statistics["oi_ranked_best"], dtype=float)
    random_mean = np.asarray(statistics["random_mean_best"], dtype=float)
    random_lower = np.asarray(statistics["random_lower_best"], dtype=float)
    random_upper = np.asarray(statistics["random_upper_best"], dtype=float)
    with plt.rc_context(_plot_style()):
        figure, axis = plt.subplots(figsize=(10.0, 6.2), constrained_layout=True)
        axis.fill_between(
            budgets,
            random_lower,
            random_upper,
            color="#9CA3AF",
            alpha=0.22,
            label="Random candidates (10th–90th percentile)",
        )
        axis.plot(
            budgets,
            random_mean,
            color="#6B7280",
            linewidth=2.0,
            linestyle="--",
            label="Random candidates (mean)",
        )
        axis.plot(
            budgets,
            oi_best,
            color="#2563EB",
            linewidth=2.7,
            marker="o",
            markersize=5.5,
            label="Candidates ranked by clean OI",
        )
        best_value = float(max(oi_best))
        first_best = int(budgets[int(np.argmax(oi_best))])
        axis.annotate(
            f"Best linear head found after {first_best} candidates",
            xy=(first_best, best_value),
            xytext=(first_best + 0.45, max(0.0, best_value - 0.12)),
            arrowprops={"arrowstyle": "-", "color": "#6B7280", "linewidth": 0.8},
            ha="left",
        )
        axis.set_xticks(budgets)
        axis.set_xlabel("Linear heads trained")
        axis.set_ylabel("Best clean test accuracy available")
        axis.set_ylim(max(0.0, min(random_lower) - 0.06), min(1.0, best_value + 0.04))
        axis.grid(True, color="#E5E7EB", linewidth=0.8)
        axis.legend(frameon=False, loc="lower right")
        figure.suptitle(
            "OverlapIndex concentrates head-training budget on useful representations\n"
            "Test accuracy evaluates each screening policy; it does not determine the OI order",
        )
        return _save_figure(figure, figure_dir, "oxford-pets-oi-selection-budget", plt)


def _selection_budget_statistics(
    candidate_rows: Sequence[Mapping[str, Any]],
    head_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, List[float]]:
    """Compare clean-OI screening with every equally sized random candidate subset."""

    from itertools import combinations

    linear_accuracy: Dict[str, float] = {}
    for representation in {str(row["representation"]) for row in candidate_rows}:
        values = [
            value
            for row in head_rows
            if row.get("representation") == representation
            and row.get("head") == "linear"
            and row.get("status", "completed") != "unavailable"
            if (value := _optional_float(row.get("clean_test_accuracy"))) is not None
        ]
        if values:
            linear_accuracy[representation] = float(np.mean(values))
    ordered = sorted(
        [
            row
            for row in candidate_rows
            if str(row["representation"]) in linear_accuracy
            and _optional_float(row.get("clean_breed_overlap")) is not None
        ],
        key=lambda row: (float(row["clean_breed_overlap"]), row["representation"]),
        reverse=True,
    )
    result: Dict[str, List[float]] = {
        "budget": [],
        "oi_ranked_best": [],
        "random_mean_best": [],
        "random_lower_best": [],
        "random_upper_best": [],
    }
    if not ordered:
        return result
    ordered_accuracy = [linear_accuracy[row["representation"]] for row in ordered]
    all_accuracy = [linear_accuracy[row["representation"]] for row in ordered]
    for budget in range(1, len(ordered) + 1):
        random_best = np.asarray(
            [max(values) for values in combinations(all_accuracy, budget)],
            dtype=float,
        )
        result["budget"].append(float(budget))
        result["oi_ranked_best"].append(float(max(ordered_accuracy[:budget])))
        result["random_mean_best"].append(float(random_best.mean()))
        result["random_lower_best"].append(float(np.quantile(random_best, 0.10)))
        result["random_upper_best"].append(float(np.quantile(random_best, 0.90)))
    return result


def _plot_head_choice_audit(
    candidate_rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    rows = []
    for candidate in candidate_rows:
        probe_delta = _optional_float(candidate.get("mlp_vs_linear_delta"))
        if probe_delta is None:
            continue
        observed_delta = _optional_float(candidate.get("validation_mlp_advantage"))
        observed_std = _optional_float(candidate.get("validation_mlp_advantage_std"))
        if observed_delta is None or observed_std is None:
            continue
        rows.append(
            {
                "representation": str(candidate["representation"]),
                "model": str(candidate["model"]),
                "probe_delta": probe_delta,
                "probe_lower": _optional_float(candidate.get("mlp_vs_linear_lower_95")),
                "probe_upper": _optional_float(candidate.get("mlp_vs_linear_upper_95")),
                "observed_delta": observed_delta,
                "observed_std": observed_std,
                "override": bool(candidate.get("mlp_recommendation_override")),
            }
        )
    untriggered = [
        _compact_representation_label(str(candidate["representation"]))
        for candidate in candidate_rows
        if candidate.get("mlp_probe_status") == "not_triggered"
    ]
    mlp_override_threshold = next(
        (
            value
            for candidate in candidate_rows
            if (value := _optional_float(candidate.get("mlp_min_improvement"))) is not None
        ),
        0.02,
    )
    if not candidate_rows:
        return _plot_empty_state(
            figure_dir,
            plt,
            "oxford-pets-separatix-head-choice-audit",
            "Separatix head-choice audit",
            "No candidate rows are available for this audit.",
        )
    audit_summary = _head_choice_audit_summary(candidate_rows)
    if not rows:
        with plt.rc_context(_plot_style()):
            figure, axis = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
            axis.axis("off")
            if any(
                candidate.get("mlp_probe_status") == "completed" for candidate in candidate_rows
            ):
                message = (
                    "No aligned MLP/linear validation rows are available.\n"
                    "The exact emitted probe recipes were unavailable for this audit."
                )
            else:
                message = (
                    "No optional MLP probes were triggered.\n"
                    "Simpler probes cleared the configured normalized-skill threshold."
                )
            axis.text(
                0.5,
                0.55,
                message,
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            figure.suptitle("Separatix head-choice audit")
            return _save_figure(
                figure,
                figure_dir,
                "oxford-pets-separatix-head-choice-audit",
                plt,
            )
    colors = _model_colors([row["model"] for row in rows], plt)
    x_values = [row["probe_delta"] for row in rows]
    y_values = [row["observed_delta"] for row in rows]
    x_span = max(0.1, max(x_values) - min(x_values))
    y_span = max(0.1, max(y_values) - min(y_values))
    with plt.rc_context(_plot_style()):
        figure, axis = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
        axis.axhline(0.0, color="#6B7280", linewidth=1.0)
        axis.axvline(
            mlp_override_threshold,
            color="#6B7280",
            linewidth=1.0,
            linestyle="--",
        )
        label_positions = _spread_label_positions(
            y_values,
            lower=min(y_values) - 0.14 * y_span,
            upper=max(y_values) + 0.14 * y_span,
            min_gap=0.055 * y_span,
        )
        for row, label_y in zip(rows, label_positions):
            color = colors[row["model"]]
            lower = row["probe_lower"]
            upper = row["probe_upper"]
            xerr = None
            if lower is not None and upper is not None:
                xerr = np.asarray(
                    [
                        [max(0.0, row["probe_delta"] - lower)],
                        [max(0.0, upper - row["probe_delta"])],
                    ],
                    dtype=float,
                )
            axis.errorbar(
                row["probe_delta"],
                row["observed_delta"],
                xerr=xerr,
                yerr=row["observed_std"],
                fmt="^" if row["override"] else "o",
                markersize=9 if row["override"] else 8,
                color=color,
                markeredgecolor="white",
                markeredgewidth=1.0,
                capsize=2.5,
                zorder=3,
            )
            axis.annotate(
                _compact_representation_label(row["representation"]),
                xy=(row["probe_delta"], row["observed_delta"]),
                xytext=(row["probe_delta"] + 0.025 * x_span, label_y),
                fontsize=8,
                ha="left",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#9CA3AF",
                    "linewidth": 0.7,
                    "shrinkA": 1,
                    "shrinkB": 3,
                },
            )
        axis.text(
            mlp_override_threshold,
            axis.get_ylim()[1],
            f"  MLP override threshold = {mlp_override_threshold:.2f}",
            ha="left",
            va="top",
            color="#4B5563",
        )
        axis.text(
            0.03,
            0.96,
            "Material head-choice agreement: "
            f"{audit_summary['material_agreement_count']}/"
            f"{audit_summary['candidate_count']}\n"
            "Mean validation regret: "
            f"{_format_optional_metric(audit_summary['mean_validation_regret'])}",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        axis.set_xlabel("Separatix aligned advantage: MLP − linear balanced accuracy")
        axis.set_ylabel("Observed clean-validation advantage: MLP − linear balanced accuracy")
        axis.grid(True, color="#E5E7EB", linewidth=0.8)
        figure.suptitle(
            "Does Separatix's aligned MLP evidence predict downstream head advantage?\n"
            "Horizontal bars show paired 95% intervals; vertical bars show deterministic replay",
        )
        if untriggered:
            figure.text(
                0.01,
                -0.015,
                "MLP not triggered because simpler probes were sufficient: "
                + ", ".join(untriggered),
                ha="left",
                va="top",
                fontsize=8,
            )
        return _save_figure(figure, figure_dir, "oxford-pets-separatix-head-choice-audit", plt)


def _plot_background_shift_effect(
    candidate_rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    rows = []
    for candidate in candidate_rows:
        clean_overlap = _optional_float(candidate.get("clean_breed_overlap"))
        swapped_overlap = _optional_float(candidate.get("background_swapped_breed_overlap"))
        clean_accuracy = _optional_float(candidate.get("selected_head_clean_test_accuracy"))
        swapped_accuracy = _optional_float(candidate.get("selected_head_swapped_test_accuracy"))
        if (
            clean_overlap is None
            or swapped_overlap is None
            or clean_accuracy is None
            or swapped_accuracy is None
        ):
            continue
        rows.append(
            {
                "representation": str(candidate["representation"]),
                "model": str(candidate["model"]),
                "overlap_delta": swapped_overlap - clean_overlap,
                "accuracy_delta": swapped_accuracy - clean_accuracy,
            }
        )
    if not rows:
        return _plot_empty_state(
            figure_dir,
            plt,
            "oxford-pets-background-shift-effect",
            "Background intervention effect",
            "No completed selected-head test scores are available for this plot.",
        )
    colors = _model_colors([row["model"] for row in rows], plt)
    x_values = [row["overlap_delta"] for row in rows]
    y_values = [row["accuracy_delta"] for row in rows]
    x_span = max(0.05, max(x_values) - min(x_values))
    y_span = max(0.05, max(y_values) - min(y_values))
    with plt.rc_context(_plot_style()):
        figure, axis = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
        axis.axhline(0.0, color="#6B7280", linewidth=1.0)
        axis.axvline(0.0, color="#6B7280", linewidth=1.0)
        label_positions = _spread_label_positions(
            y_values,
            lower=min(y_values) - 0.14 * y_span,
            upper=max(y_values) + 0.14 * y_span,
            min_gap=0.055 * y_span,
        )
        for row, label_y in zip(rows, label_positions):
            color = colors[row["model"]]
            axis.scatter(
                row["overlap_delta"],
                row["accuracy_delta"],
                s=76,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
            axis.annotate(
                _compact_representation_label(row["representation"]),
                xy=(row["overlap_delta"], row["accuracy_delta"]),
                xytext=(row["overlap_delta"] + 0.025 * x_span, label_y),
                fontsize=8,
                ha="left",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#9CA3AF",
                    "linewidth": 0.7,
                    "shrinkA": 1,
                    "shrinkB": 3,
                },
            )
        rho = _spearman_correlation(x_values, y_values)
        axis.text(
            0.03,
            0.88,
            _correlation_label(rho),
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        axis.set_xlabel("Change in breed OverlapIndex: swapped − clean")
        axis.set_ylabel("Change in selected-head accuracy: swapped − clean")
        axis.grid(True, color="#E5E7EB", linewidth=0.8)
        figure.suptitle(
            "Geometry change weakly tracks accuracy change under one background swap\n"
            "Negative values indicate degradation; this intervention remains diagnostic evidence",
        )
        return _save_figure(figure, figure_dir, "oxford-pets-background-shift-effect", plt)


def _plot_relational_composition(
    rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    plt: Any,
) -> Tuple[Path, Path]:
    """Compare head-family utility under raw and interaction-aware pair features."""

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    representations = _ordered_representations(rows)
    lookup = {(str(row["representation"]), str(row["composition"])): row for row in rows}
    family_labels = {
        "linear": "Linear",
        "smooth_nonlinear": "Smooth\nnonlinear",
        "local_kernel": "Local /\nkernel",
        "mlp": "MLP",
    }
    matrices = {}
    for composition in _PAIR_COMPOSITIONS:
        matrices[composition] = np.asarray(
            [
                [
                    (
                        value
                        if (
                            value := _optional_float(
                                lookup.get((representation, composition), {}).get(
                                    f"{family}_test_balanced_accuracy"
                                )
                            )
                        )
                        is not None
                        else np.nan
                    )
                    for family in _RELATIONAL_FAMILIES
                ]
                for representation in representations
            ],
            dtype=float,
        )
    if not representations or not any(np.isfinite(value).any() for value in matrices.values()):
        return _plot_empty_state(
            figure_dir,
            plt,
            "oxford-pets-relational-composition-heads",
            "Relational composition and head complexity",
            "No completed held-out test family scores are available for this plot.",
        )
    with plt.rc_context(_plot_style()):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(13.5, max(5.6, 0.56 * len(representations) + 2.6)),
            sharey=True,
            constrained_layout=True,
        )
        images = []
        titles = {
            "concatenation": "Raw concatenation  [left, right]",
            "interaction": "Interaction-aware  [|left−right|, left×right]",
        }
        for axis, composition in zip(axes, _PAIR_COMPOSITIONS):
            matrix = matrices[composition]
            image = axis.imshow(
                matrix,
                vmin=0.45,
                vmax=1.0,
                cmap="viridis",
                aspect="auto",
            )
            images.append(image)
            axis.set_xticks(
                np.arange(len(_RELATIONAL_FAMILIES)),
                labels=[family_labels[family] for family in _RELATIONAL_FAMILIES],
            )
            axis.set_yticks(np.arange(len(representations)), labels=representations)
            axis.set_title(titles[composition])
            for row_index, representation in enumerate(representations):
                item = lookup.get((representation, composition), {})
                for column_index, family in enumerate(_RELATIONAL_FAMILIES):
                    value = matrix[row_index, column_index]
                    text_color = (
                        "white"
                        if not np.isfinite(value) or value < 0.67 or value > 0.91
                        else "black"
                    )
                    axis.text(
                        column_index,
                        row_index,
                        "—" if not np.isfinite(value) else f"{value:.3f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )
                    selected_family = item.get(
                        "separatix_selected_family",
                        item.get("separatix_recommended_family"),
                    )
                    if selected_family == family:
                        axis.add_patch(
                            Rectangle(
                                (column_index - 0.48, row_index - 0.48),
                                0.96,
                                0.96,
                                fill=False,
                                edgecolor="#F97316",
                                linewidth=2.4,
                            )
                        )
                    if family in (item.get("separatix_plausible_families") or []):
                        axis.scatter(
                            column_index - 0.34,
                            row_index + 0.33,
                            marker="o",
                            s=24,
                            facecolor="#F97316",
                            edgecolor="white",
                            linewidth=0.5,
                            zorder=4,
                        )
                    if item.get("empirical_simplest_near_best_family") == family:
                        axis.scatter(
                            column_index + 0.34,
                            row_index - 0.33,
                            marker="*",
                            s=42,
                            color="#111827",
                            edgecolor="white",
                            linewidth=0.5,
                            zorder=4,
                        )
        colorbar = figure.colorbar(images[0], ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label("Held-out test balanced accuracy")
        figure.legend(
            handles=[
                Patch(
                    facecolor="none",
                    edgecolor="#F97316",
                    linewidth=2.4,
                    label="Selected Separatix deployment family",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="#F97316",
                    markeredgecolor="white",
                    markersize=6,
                    label="Separatix plausible core family",
                ),
                Line2D(
                    [],
                    [],
                    marker="*",
                    linestyle="None",
                    markerfacecolor="#111827",
                    markeredgecolor="white",
                    markersize=9,
                    label="Simplest family within test near-best margin",
                ),
            ],
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        figure.suptitle(
            "Does pair composition change the head complexity Oxford Pets needs?\n"
            "Border = selected train+validation deployment family; "
            "orange dots = plausible core families; "
            "stars = retrospective test near-best",
        )
        return _save_figure(
            figure,
            figure_dir,
            "oxford-pets-relational-composition-heads",
            plt,
        )


def _spearman_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    result = spearmanr(x, y)
    return float(result.statistic)


def _correlation_label(value: float) -> str:
    if np.isfinite(value):
        return f"Descriptive Spearman ρ = {value:.2f}"
    return "Descriptive Spearman ρ unavailable"


def _compact_representation_label(representation: str) -> str:
    replacements = {
        "DINOv2-Small": "DINOv2-S",
        "DeiT-Tiny": "DeiT-T",
        "ConvNeXt-Tiny": "ConvNeXt-T",
        "MobileNetV3-Large": "MobileNetV3-L",
        "OpenCLIP ViT-B/32": "OpenCLIP-B/32",
        "Early Cls": "early",
        "Middle Cls": "mid",
        "Late Cls": "late",
        "Final Cls": "final",
        "Final Image": "final",
        "Final": "final",
    }
    label = representation
    for source, target in replacements.items():
        label = label.replace(source, target)
    return label


def _spread_label_positions(
    values: Sequence[float],
    *,
    lower: float,
    upper: float,
    min_gap: float,
) -> List[float]:
    """Return ordered label positions separated by at least ``min_gap``."""

    if not values:
        return []
    if lower > upper:
        raise ValueError("Label-position lower bound must not exceed the upper bound.")
    if min_gap < 0:
        raise ValueError("Label-position minimum gap must be non-negative.")
    if min_gap * max(0, len(values) - 1) > upper - lower:
        min_gap = (upper - lower) / max(1, len(values) - 1)
    ordered = sorted(enumerate(values), key=lambda item: (float(item[1]), item[0]))
    positions = [max(lower, min(upper, float(value))) for _, value in ordered]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + min_gap)
    if positions[-1] > upper:
        positions[-1] = upper
        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(positions[index], positions[index + 1] - min_gap)
    if positions[0] < lower:
        shift = lower - positions[0]
        positions = [position + shift for position in positions]
    restored = [0.0] * len(values)
    for (original_index, _), position in zip(ordered, positions):
        restored[original_index] = position
    return restored


def _ordered_representations(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    identities = {(row["model"], row["output"], row["representation"]) for row in rows}
    model_rank = {name: index for index, name in enumerate(_MODEL_ORDER)}
    return [
        representation
        for _, _, representation in sorted(
            identities,
            key=lambda item: (
                model_rank.get(item[0], len(model_rank)),
                _OUTPUT_ORDER.get(item[1], len(_OUTPUT_ORDER)),
                item[1],
            ),
        )
    ]


def _model_colors(models: Sequence[str], plt: Any) -> Dict[str, Any]:
    unique = list(dict.fromkeys(models))
    palette = plt.get_cmap("tab10")
    model_rank = {model: index for index, model in enumerate(_MODEL_ORDER)}
    return {model: palette(model_rank.get(model, len(model_rank)) % 10) for model in unique}


def _plot_empty_state(
    figure_dir: Path,
    plt: Any,
    stem: str,
    title: str,
    message: str,
) -> Tuple[Path, Path]:
    """Persist an explicit empty-state figure when a diagnostic has no usable rows."""

    with plt.rc_context(_plot_style()):
        figure, axis = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.52,
            message,
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        figure.suptitle(title)
        return _save_figure(figure, figure_dir, stem, plt)


def _plot_style() -> Dict[str, Any]:
    return {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#9CA3AF",
        "axes.labelcolor": "#111827",
        "axes.titleweight": "bold",
        "font.size": 10,
        "text.color": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "savefig.facecolor": "white",
    }


def _save_figure(figure: Any, directory: Path, stem: str, plt: Any) -> Tuple[Path, Path]:
    png_path = directory / f"{stem}.png"
    svg_path = directory / f"{stem}.svg"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


def _protocol_payload(
    args: argparse.Namespace,
    model_names: Sequence[str],
    head_train: Sequence[PetSample],
    selection: Sequence[PetSample],
    validation: Sequence[PetSample],
    test: Sequence[PetSample],
    relational_pairs: Mapping[str, Sequence[VerificationPair]],
) -> Dict[str, Any]:
    return {
        "dataset": "Oxford-IIIT Pet",
        "models": list(model_names),
        "selection_rule": "rank backbone/layer candidates by clean breed OverlapIndex",
        "head_rule": (
            "run Separatix on clean head-training rows and choose MLP only when its "
            f"paired override clears the {args.mlp_min_improvement:.3f} improvement; "
            "otherwise choose linear"
        ),
        "head_diagnostic": (
            "clean head-training embeddings only; background interventions are excluded"
        ),
        "head_evidence_schema": "aligned_optional_mlp_v2_recipe_aligned",
        "relational_evidence_schema": "deployment_family_composition_v2",
        "head_training": (
            "clean head-training images with L2 normalization and fold-local standardization; "
            "refit on head-train plus validation"
        ),
        "final_test_use": "accuracy reporting only; never used for backbone or head selection",
        "relational_audit": {
            "task": "same-breed verification",
            "positive_pairs": "same breed",
            "negative_pairs": "different breed within the same species",
            "source_reuse": "each source image appears in at most one pair per split",
            "compositions": {
                "concatenation": "[left, right] after endpoint normalization",
                "interaction": "[abs(left-right), left*right] after endpoint normalization",
            },
            "head_families": list(_RELATIONAL_FAMILIES),
            "deployment_rule": (
                "diagnose Separatix on combined head-train plus validation pairs, "
                "fit each emitted family recipe once on that development cohort, "
                "and report held-out test balanced accuracy"
            ),
            "retrospective_rule": (
                "star the simplest family within the configured near-optimal margin "
                "of the best observed held-out test score"
            ),
            "pair_counts": {name: len(pairs) for name, pairs in relational_pairs.items()},
        },
        "split_counts": {
            "head_train": len(head_train),
            "selection": len(selection),
            "validation": len(validation),
            "test": len(test),
        },
        "configuration": {
            "overlap_k": args.overlap_k,
            "stability_repeats": args.stability_repeats,
            "head_repeats": args.head_repeats,
            "near_optimal_margin": args.near_optimal_margin,
            "mlp_min_improvement": args.mlp_min_improvement,
            "mlp_trigger_skill_threshold": args.mlp_trigger_skill_threshold,
            "seed": args.seed,
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}.")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        rendered = float(value)
    except (TypeError, ValueError):
        return None
    return rendered if np.isfinite(rendered) else None


def _format_optional_metric(value: Any) -> str:
    rendered = _optional_float(value)
    return "unavailable" if rendered is None else f"{rendered:.3f}"


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError(f"Cannot aggregate empty head rows for {field!r}.")
    return float(np.mean([float(row[field]) for row in rows]))


def _mean_optional(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    values = [rendered for row in rows if (rendered := _optional_float(row.get(field))) is not None]
    return float(np.mean(values)) if values else None


def _breeds(samples: Sequence[PetSample]) -> np.ndarray:
    return np.asarray([sample.breed for sample in samples])


def _species(samples: Sequence[PetSample]) -> np.ndarray:
    return np.asarray([sample.species for sample in samples])


def _layer_label(output_name: str) -> str:
    return output_name.replace("_", " ").title()


if __name__ == "__main__":
    main()
