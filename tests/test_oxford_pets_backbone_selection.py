import importlib.util
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _load_example_module():
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "oxford_pets_backbone_selection",
            examples_dir / "oxford_pets_backbone_selection.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


def _samples(module, tmp_path, *, per_breed=6):
    samples = []
    for species, breeds in (("Cat", ("Abyssinian", "Birman")), ("Dog", ("Beagle", "Pug"))):
        for breed in breeds:
            for index in range(per_breed):
                samples.append(
                    module.PetSample(
                        sample_id=f"{breed}-{index}",
                        image_path=tmp_path / f"{breed}-{index}.jpg",
                        trimap_path=tmp_path / f"{breed}-{index}.png",
                        breed=breed,
                        species=species,
                        source_split="trainval",
                    )
                )
    return samples


def test_help_does_not_require_model_extras():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "oxford_pets_backbone_selection.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--models" in completed.stdout
    assert "--head-margin" in completed.stdout
    assert "--mlp-trigger-skill-threshold" in completed.stdout
    assert "--replot-from" in completed.stdout
    assert "--no-download" in completed.stdout


def test_trainval_roles_are_balanced_disjoint_and_deterministic(tmp_path):
    module = _load_example_module()
    samples = _samples(module, tmp_path, per_breed=8)

    first = module._stratified_trainval_split(
        samples,
        head_train_per_breed=3,
        selection_per_breed=2,
        validation_per_breed=2,
        seed=7,
    )
    second = module._stratified_trainval_split(
        samples,
        head_train_per_breed=3,
        selection_per_breed=2,
        validation_per_breed=2,
        seed=7,
    )

    assert [[sample.sample_id for sample in split] for split in first] == [
        [sample.sample_id for sample in split] for split in second
    ]
    expected_counts = (3, 2, 2)
    id_sets = []
    for split, expected in zip(first, expected_counts):
        assert set(Counter(sample.breed for sample in split).values()) == {expected}
        id_sets.append({sample.sample_id for sample in split})
    assert id_sets[0].isdisjoint(id_sets[1])
    assert id_sets[0].isdisjoint(id_sets[2])
    assert id_sets[1].isdisjoint(id_sets[2])


def test_background_donors_are_balanced_within_breed_and_never_same_breed(tmp_path):
    module = _load_example_module()
    samples = _samples(module, tmp_path, per_breed=4)

    first = module._balanced_background_donors(samples, seed=11)
    second = module._balanced_background_donors(samples, seed=11)

    assert [(pair.target.sample_id, pair.donor.sample_id) for pair in first] == [
        (pair.target.sample_id, pair.donor.sample_id) for pair in second
    ]
    assert all(pair.target.breed != pair.donor.breed for pair in first)
    by_breed = {}
    for breed in sorted({pair.target.breed for pair in first}):
        by_breed[breed] = Counter(
            pair.background_species for pair in first if pair.target.breed == breed
        )
    assert all(counts == {"Cat": 2, "Dog": 2} for counts in by_breed.values())


def test_relational_pairs_are_balanced_hard_negative_and_source_disjoint(tmp_path):
    module = _load_example_module()
    samples = _samples(module, tmp_path, per_breed=8)

    first = module._same_breed_verification_pairs(samples, seed=17)
    second = module._same_breed_verification_pairs(samples, seed=17)

    assert first == second
    assert Counter(pair.target for pair in first) == {0: 8, 1: 8}
    used = [index for pair in first for index in (pair.left_index, pair.right_index)]
    assert len(used) == len(set(used)) == len(samples)
    for pair in first:
        left = samples[pair.left_index]
        right = samples[pair.right_index]
        if pair.target:
            assert left.breed == right.breed
        else:
            assert left.breed != right.breed
            assert left.species == right.species
        assert left.species == right.species == pair.species
    by_target_species = {
        target: Counter(pair.species for pair in first if pair.target == target)
        for target in (0, 1)
    }
    assert by_target_species[0] == by_target_species[1]


def test_relational_compositions_change_only_the_pair_feature_map():
    module = _load_example_module()
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float32,
    )
    pairs = [module.VerificationPair(0, 1, 0, "Cat")]
    reversed_pairs = [module.VerificationPair(1, 0, 0, "Cat")]

    concatenated = module._compose_pair_embeddings(embeddings, pairs, "concatenation")
    reversed_concatenated = module._compose_pair_embeddings(
        embeddings,
        reversed_pairs,
        "concatenation",
    )
    interaction = module._compose_pair_embeddings(embeddings, pairs, "interaction")
    reversed_interaction = module._compose_pair_embeddings(
        embeddings,
        reversed_pairs,
        "interaction",
    )

    assert concatenated.shape == interaction.shape == (1, 4)
    assert not np.allclose(concatenated, reversed_concatenated)
    assert np.allclose(interaction, reversed_interaction)
    assert np.allclose(np.linalg.norm(concatenated, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(interaction, axis=1), 1.0)


def test_foreground_and_background_swap_preserve_target_pixels(tmp_path):
    module = _load_example_module()
    target_image = np.zeros((4, 4, 3), dtype=np.uint8)
    target_image[:] = (255, 0, 0)
    donor_image = np.zeros((4, 4, 3), dtype=np.uint8)
    donor_image[:] = (0, 0, 255)
    target_trimap = np.full((4, 4), 2, dtype=np.uint8)
    target_trimap[1:3, 1:3] = 1
    donor_trimap = np.full((4, 4), 2, dtype=np.uint8)

    Image.fromarray(target_image).save(tmp_path / "target.jpg", quality=100, subsampling=0)
    Image.fromarray(donor_image).save(tmp_path / "donor.jpg", quality=100, subsampling=0)
    Image.fromarray(target_trimap).save(tmp_path / "target.png")
    Image.fromarray(donor_trimap).save(tmp_path / "donor.png")
    target = module.PetSample(
        "target",
        tmp_path / "target.jpg",
        tmp_path / "target.png",
        "Abyssinian",
        "Cat",
        "test",
    )
    donor = module.PetSample(
        "donor",
        tmp_path / "donor.jpg",
        tmp_path / "donor.png",
        "Beagle",
        "Dog",
        "test",
    )

    foreground = np.asarray(module._render_foreground(target))
    swapped = np.asarray(
        module._render_background_swap(
            module.BackgroundSwap(target, donor, "Dog"),
            blur_radius=0,
        )
    )

    assert np.all(foreground[0, 0] == 127)
    assert foreground[1, 1, 0] > 240
    assert swapped[1, 1, 0] > 240
    assert swapped[0, 0, 2] > 240


def test_separatix_head_rule_uses_actual_mlp_override():
    module = _load_example_module()
    override = {
        "status": "completed",
        "recommendation_override": True,
        "mean_delta": 0.04,
    }
    no_override = {
        "status": "completed",
        "recommendation_override": False,
        "mean_delta": 0.01,
        "lower_95": -0.01,
        "upper_95": 0.03,
    }
    untriggered = {"status": "not_triggered", "trigger_threshold": 0.75}

    assert module._select_head(override, min_improvement=0.02)[0] == "mlp"
    assert module._select_head(no_override, min_improvement=0.02)[0] == "linear"
    assert module._select_head(untriggered, min_improvement=0.02)[0] == "linear"
    assert module._select_head({}, min_improvement=0.02)[0] == "linear"


def test_separatix_mlp_evidence_uses_aligned_optional_probe_payload():
    module = _load_example_module()
    payload = {
        "extractor_results": [
            {
                "separatix": {
                    "report": {
                        "recommendation": "feedforward_mlp_recommended",
                        "metrics": {
                            "mlp_trigger_evidence": {
                                "status": "triggered",
                                "good_enough": False,
                                "threshold": 0.75,
                            },
                            "mlp_recommendation_evidence": {
                                "status": "completed",
                                "recommendation_override": True,
                                "best_architecture": {
                                    "probe_name": "mlp_one_layer_wide",
                                    "balanced_accuracy": 0.64,
                                },
                                "aligned_comparators": {
                                    "linear": {
                                        "balanced_accuracy": 0.58,
                                        "evaluation_mode": "cross_validation",
                                    }
                                },
                                "pairwise_comparisons": {
                                    "linear": {
                                        "mean_delta": 0.06,
                                        "lower_95": 0.03,
                                        "upper_95": 0.09,
                                        "clear_advantage": True,
                                    }
                                },
                            },
                        },
                    }
                }
            }
        ]
    }

    evidence = module._separatix_mlp_evidence(payload)

    assert evidence["recommendation_override"] is True
    assert evidence["best_architecture"] == "mlp_one_layer_wide"
    assert evidence["linear_score"] == 0.58
    assert evidence["mlp_score"] == 0.64
    assert evidence["mean_delta"] == 0.06


def test_separatix_family_evidence_respects_primary_family_and_mlp_override():
    module = _load_example_module()
    payload = {
        "extractor_results": [
            {
                "separatix": {
                    "report": {
                        "recommendation": "smooth_nonlinear_recommended",
                        "confidence": "high",
                        "metrics": {
                            "recommendation_evidence": {
                                "raw_best_family": "local_kernel",
                                "recommended_family": "smooth_nonlinear",
                                "best_clearly_beats_dummy": True,
                                "families": {"smooth_nonlinear": {"best_probe": "smooth_poly"}},
                            },
                            "mlp_recommendation_evidence": {
                                "status": "not_triggered",
                                "recommendation_override": False,
                            },
                        },
                    }
                }
            }
        ]
    }

    evidence = module._separatix_family_evidence(payload)

    assert evidence["recommended_family"] == "smooth_nonlinear"
    assert evidence["recommended_probe"] == "smooth_poly"
    assert evidence["raw_best_family"] == "local_kernel"
    assert evidence["confidence"] == "high"

    metrics = payload["extractor_results"][0]["separatix"]["report"]["metrics"]
    metrics["mlp_recommendation_evidence"] = {
        "status": "completed",
        "recommendation_override": True,
        "best_architecture": {"probe_name": "mlp_one_layer_wide"},
    }
    evidence = module._separatix_family_evidence(payload)
    assert evidence["recommended_family"] == "mlp"
    assert evidence["recommended_probe"] == "mlp_one_layer_wide"


def test_head_diagnostic_uses_clean_head_training_rows_only(tmp_path, monkeypatch):
    module = _load_example_module()
    head_train = _samples(module, tmp_path, per_breed=2)
    selection = _samples(module, tmp_path, per_breed=2)
    swaps = [
        module.BackgroundSwap(sample, selection[(index + 2) % len(selection)], "Dog")
        for index, sample in enumerate(selection)
    ]
    head_embeddings = np.arange(len(head_train) * 3, dtype=np.float32).reshape(-1, 3)
    selection_embeddings = np.ones((len(selection), 3), dtype=np.float32)
    calls = []

    def fake_score(embeddings, labels, **kwargs):
        calls.append((np.asarray(embeddings), np.asarray(labels), kwargs))
        return {
            "overlap_macro": 0.2,
            "stability_lower": None,
            "stability_upper": None,
        }, {"extractor_results": []}

    monkeypatch.setattr(module, "_score_embeddings", fake_score)

    rows, _ = module._representation_measurements(
        representation="Demo",
        model_name="dinov2-small",
        output_name="final_cls",
        head_train_embeddings=head_embeddings,
        selection_embeddings=selection_embeddings,
        foreground_embeddings=selection_embeddings,
        swapped_embeddings=selection_embeddings,
        head_train=head_train,
        selection=selection,
        selection_swaps=swaps,
        overlap_k=2,
        stability_repeats=2,
        head_margin=0.02,
        mlp_trigger_skill_threshold=0.75,
        seed=42,
    )

    diagnostic_embeddings, diagnostic_labels, diagnostic_kwargs = calls[-1]
    assert np.array_equal(diagnostic_embeddings, head_embeddings)
    assert np.array_equal(diagnostic_labels, module._breeds(head_train))
    assert diagnostic_kwargs["run_separatix"] is True
    assert diagnostic_kwargs["groups"] is None
    assert diagnostic_kwargs["mlp_trigger_skill_threshold"] == 0.75
    assert rows[-1]["condition"] == "clean_head_train"


def test_head_labels_are_encoded_before_mlp_internal_validation():
    module = _load_example_module()
    rng = np.random.default_rng(23)
    embeddings = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.1, size=(40, 4)),
            rng.normal(loc=1.0, scale=0.1, size=(40, 4)),
        ]
    )
    labels = np.asarray(["Abyssinian"] * 40 + ["Beagle"] * 40)
    train_targets, validation_targets, test_targets = module._encode_head_labels(
        labels,
        labels[::-1],
        labels,
    )

    head = module._make_head("mlp", seed=5)
    head.fit(embeddings, train_targets)

    assert set(np.unique(train_targets)) == {0, 1}
    assert set(np.unique(validation_targets)) == {0, 1}
    assert set(np.unique(test_targets)) == {0, 1}
    assert set(head.predict(embeddings)) <= {0, 1}
    assert head.named_steps["mlpclassifier"].early_stopping is True
    assert "standardscaler" in head.named_steps


def test_model_recipes_are_lazy_and_expose_expected_layers():
    module = _load_example_module()
    names = ["dinov2-small", "deit-tiny", "convnext-tiny", "openclip-vit-b-32"]

    extractors = module._build_extractors(names, batch_size=4, device="cpu")

    assert [extractor.name for extractor in extractors] == names
    assert [spec.name for spec in extractors[0].output_specs()] == [
        "early_cls",
        "middle_cls",
        "late_cls",
        "final_cls",
    ]
    assert [spec.name for spec in extractors[1].output_specs()] == [
        "early_cls",
        "middle_cls",
        "late_cls",
        "final_cls",
    ]
    assert [spec.hidden_layer for spec in extractors[0].output_specs()] == [3, 6, 9, -1]
    assert [spec.hidden_layer for spec in extractors[1].output_specs()] == [3, 6, 9, -1]
    assert [spec.name for spec in extractors[2].output_specs()] == ["final"]
    assert [spec.name for spec in extractors[3].output_specs()] == ["final_image"]
    assert extractors[0].processor_kwargs == {"use_fast": False}
    assert extractors[1].processor_kwargs == {"use_fast": False}


def test_candidate_ranking_uses_clean_overlap_not_test_accuracy_or_shift():
    module = _load_example_module()
    rows = [
        {
            "representation": "clean geometry",
            "robust_breed_overlap": 0.40,
            "clean_breed_overlap": 0.80,
            "selected_head_clean_test_accuracy": 0.99,
        },
        {
            "representation": "robust geometry",
            "robust_breed_overlap": 0.65,
            "clean_breed_overlap": 0.70,
            "selected_head_clean_test_accuracy": 1.00,
        },
    ]

    ranked = module._rank_candidates(rows)

    assert [row["representation"] for row in ranked] == [
        "clean geometry",
        "robust geometry",
    ]
    assert [row["selection_rank"] for row in ranked] == [1, 2]


def test_scatter_labels_are_compact_and_collision_spread_is_deterministic():
    module = _load_example_module()

    positions = module._spread_label_positions(
        [0.90, 0.91, 0.89, 0.20],
        lower=0.10,
        upper=0.95,
        min_gap=0.04,
    )

    ordered = sorted(positions)
    assert all(right - left >= 0.04 - 1e-12 for left, right in zip(ordered, ordered[1:]))
    assert positions == module._spread_label_positions(
        [0.90, 0.91, 0.89, 0.20],
        lower=0.10,
        upper=0.95,
        min_gap=0.04,
    )
    assert module._compact_representation_label("DINOv2-Small · Final Cls") == ("DINOv2-S · final")
    assert module._compact_representation_label("DINOv2-Small · Early Cls") == ("DINOv2-S · early")
    assert module._compact_representation_label("OpenCLIP ViT-B/32 · Final Image") == (
        "OpenCLIP-B/32 · final"
    )


def test_representations_are_ordered_by_model_then_layer_depth():
    module = _load_example_module()
    rows = [
        {
            "representation": f"DINOv2-Small · {output}",
            "model": "dinov2-small",
            "output": key,
        }
        for key, output in (
            ("final_cls", "Final Cls"),
            ("early_cls", "Early Cls"),
            ("late_cls", "Late Cls"),
            ("middle_cls", "Middle Cls"),
        )
    ]

    assert module._ordered_representations(rows) == [
        "DINOv2-Small · Early Cls",
        "DINOv2-Small · Middle Cls",
        "DINOv2-Small · Late Cls",
        "DINOv2-Small · Final Cls",
    ]


def test_relational_audit_counts_only_actionable_recommendations():
    module = _load_example_module()
    rows = [
        {
            "separatix_recommended_family": "smooth_nonlinear",
            "recommendation_near_optimal": True,
            "selected_validation_regret": 0.01,
            "selected_test_regret": 0.02,
        },
        {
            "separatix_recommended_family": "linear",
            "recommendation_near_optimal": False,
            "selected_validation_regret": 0.04,
            "selected_test_regret": 0.03,
        },
        {
            "separatix_recommended_family": None,
            "recommendation_near_optimal": False,
            "selected_validation_regret": None,
            "selected_test_regret": None,
        },
    ]

    summary = module._relational_audit_summary(rows)

    assert summary["case_count"] == 3
    assert summary["recommendation_count"] == 2
    assert summary["near_optimal_count"] == 1
    assert summary["near_optimal_rate"] == 0.5
    assert summary["mean_validation_regret"] == 0.025
    assert summary["mean_test_regret"] == 0.025
    assert summary["recommendation_family_counts"] == {
        "linear": 1,
        "smooth_nonlinear": 1,
    }


def test_visuals_render_from_protocol_rows(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    module = _load_example_module()
    representations = (
        ("DINOv2-Small · Final Cls", "dinov2-small", "final_cls", 0.62),
        ("ConvNeXt-Tiny · Final", "convnext-tiny", "final", 0.55),
    )
    metric_rows = []
    for representation, model, output, base in representations:
        for offset, (condition, target, _) in enumerate(module._MEASUREMENT_TARGET_LABELS):
            value = base - 0.03 * offset
            metric_rows.append(
                {
                    "representation": representation,
                    "model": model,
                    "output": output,
                    "condition": condition,
                    "target": target,
                    "overlap_macro": value,
                    "stability_lower": value - 0.02 if target == "breed" else None,
                    "stability_upper": value + 0.02 if target == "breed" else None,
                }
            )
    head_rows = []
    for representation, model, output, base in representations:
        for head, lift in (("linear", 0.0), ("mlp", 0.03)):
            for repeat in range(2):
                head_rows.append(
                    {
                        "representation": representation,
                        "model": model,
                        "output": output,
                        "head": head,
                        "repeat": repeat,
                        "validation_accuracy": base + 0.14 + lift + repeat * 0.005,
                        "validation_balanced_accuracy": (base + 0.14 + lift + repeat * 0.005),
                        "clean_test_accuracy": base + 0.18 + lift + repeat * 0.005,
                        "background_swapped_test_accuracy": base + 0.08 + lift + repeat * 0.005,
                    }
                )
    candidate_rows = [
        {
            "representation": representation,
            "model": model,
            "output": output,
            "selected_head": "mlp",
            "clean_breed_overlap": base,
            "robust_breed_overlap": base - 0.09,
            "background_swapped_breed_overlap": base - 0.09,
            "selected_head_clean_test_accuracy": base + 0.21,
            "selected_head_swapped_test_accuracy": base + 0.11,
            "mlp_probe_status": "completed",
            "mlp_recommendation_override": True,
            "mlp_vs_linear_delta": 0.03,
            "mlp_vs_linear_lower_95": 0.01,
            "mlp_vs_linear_upper_95": 0.05,
            "validation_mlp_advantage": 0.03,
            "validation_mlp_advantage_std": 0.005,
            "selected_head_validation_regret": 0.0,
            "head_margin": 0.02,
        }
        for representation, model, output, base in representations
    ]
    relational_rows = []
    for representation, model, output, base in representations:
        for composition, recommended, empirical in (
            ("concatenation", "smooth_nonlinear", "smooth_nonlinear"),
            ("interaction", "linear", "linear"),
        ):
            relational_rows.append(
                {
                    "representation": representation,
                    "model": model,
                    "output": output,
                    "composition": composition,
                    "separatix_recommended_family": recommended,
                    "empirical_simplest_near_best_family": empirical,
                    "linear_test_balanced_accuracy": base + 0.08,
                    "smooth_nonlinear_test_balanced_accuracy": base + 0.14,
                    "local_kernel_test_balanced_accuracy": base + 0.11,
                    "mlp_test_balanced_accuracy": base + 0.12,
                }
            )

    heatmap_paths = module._plot_overlap_heatmap(metric_rows, tmp_path, plt)
    scatter_paths = module._plot_overlap_accuracy_scatter(
        metric_rows,
        head_rows,
        candidate_rows,
        tmp_path,
        plt,
    )
    budget_paths = module._plot_selection_budget(candidate_rows, head_rows, tmp_path, plt)
    head_audit_paths = module._plot_head_choice_audit(
        candidate_rows,
        tmp_path,
        plt,
    )
    shift_paths = module._plot_background_shift_effect(candidate_rows, tmp_path, plt)
    relational_paths = module._plot_relational_composition(
        relational_rows,
        tmp_path,
        plt,
    )
    audit = module._head_choice_audit_summary(candidate_rows)

    assert audit["material_agreement_count"] == 2
    assert audit["mlp_probe_completed_count"] == 2

    for path in (
        *heatmap_paths,
        *scatter_paths,
        *budget_paths,
        *head_audit_paths,
        *shift_paths,
        *relational_paths,
    ):
        assert path.is_file()
        assert path.stat().st_size > 0
