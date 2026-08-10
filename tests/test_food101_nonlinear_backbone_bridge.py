"""Focused, network-free protocol tests for the Food-101 bridge driver.

The Food-101 experiment is intentionally a frozen confirmatory protocol.  The
tests below exercise the protocol helpers with tiny synthetic panels; they do
not download Food-101, construct a torchvision model, or require a GPU.  A
small amount of helper-name tolerance is useful while the example remains a
stand-alone script (and keeps these tests focused on behaviour rather than
private spelling).
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest


def _load_example_module():
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    module_name = "food101_nonlinear_backbone_bridge"
    sys.path.insert(0, str(examples_dir))
    try:
        path = examples_dir / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise AssertionError(f"could not load Food-101 bridge at {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


@pytest.fixture(scope="module")
def experiment():
    return _load_example_module()


def _call_with_supported_kwargs(function, **kwargs):
    """Call a helper while ignoring harmless spelling/optional aliases."""

    signature = inspect.signature(function)
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted = {
        name: value
        for name, value in kwargs.items()
        if accepts_arbitrary or name in signature.parameters
    }
    return function(**accepted)


def _helper(module, *names):
    for name in names:
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    raise AssertionError(f"Food-101 bridge is missing one of {names!r}")


def _constant(module, *names, default=None):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    if default is not None:
        return default
    raise AssertionError(f"Food-101 bridge is missing one of {names!r}")


def _extract(value, *keys, default=None):
    """Extract a value from a mapping using a short list of aliases."""

    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return value[key]
    return default if default is not None else value


def _rows(value):
    if isinstance(value, Mapping):
        value = _extract(value, "samples", "rows", "indices", "ids")
    return list(value)


def _roles(value):
    if isinstance(value, Mapping):
        nested = _extract(value, "roles", "cohorts", "splits")
        if isinstance(nested, Mapping):
            value = nested
        if isinstance(value, Mapping) and all(key in value for key in ("selector", "development")):
            test_key = next(
                (key for key in ("test", "bridge_test", "heldout_test") if key in value),
                None,
            )
            if test_key is not None:
                return {
                    "selector": value["selector"],
                    "development": value["development"],
                    "test": value[test_key],
                }
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return dict(zip(("selector", "development", "test"), value))
    raise AssertionError(f"unrecognised Food-101 cohorts payload: {type(value)!r}")


def _sample_id(row, fallback=None):
    if isinstance(row, Mapping):
        return row.get("sample_id", row.get("id", row.get("index", fallback)))
    return getattr(row, "sample_id", getattr(row, "id", fallback))


def _sample_label(row, fallback=None):
    if isinstance(row, Mapping):
        return row.get(
            "class_name",
            row.get("label", row.get("class", row.get("category", fallback))),
        )
    return getattr(
        row,
        "class_name",
        getattr(row, "label", getattr(row, "category", fallback)),
    )


def _sample_split(row):
    if isinstance(row, Mapping):
        return row.get("split", row.get("official_split"))
    return getattr(row, "split", getattr(row, "official_split", None))


def _food_samples(
    *,
    classes: tuple[str, ...] | None = None,
    per_class: int = 190,
    split: str = "train",
    start: int = 0,
):
    names = classes or tuple(f"food_{index:02d}" for index in range(40))
    rows = []
    for class_index, name in enumerate(names):
        for index in range(per_class):
            rows.append(
                SimpleNamespace(
                    sample_id=f"{split}:{name}:{start + index}",
                    class_name=name,
                    label=name,
                    split=split,
                    image_path=Path(f"{split}_{class_index}_{start + index}.jpg"),
                )
            )
    return rows


def _class_labels(rows):
    return np.asarray([_sample_label(row) for row in rows], dtype=object)


def _square_worker(item):
    """Top-level spawn-safe worker used by checkpoint tests."""

    return {"key": item["key"], "value": item["value"] ** 2}


def _three_arm_rows(*, backbones=5, replicates=5, include_linear=False):
    """Complete Food-101 geometry cells used by bootstrap/factorial tests."""

    arms = (
        ("baseline", 0.70, 0.68),
        ("nonlinearity_full", 0.82, 0.72),
        ("nuisance_full", 0.62, 0.70),
    )
    rows = []
    for backbone_index in range(backbones):
        for replicate in range(replicates):
            # Four of five backbones have a positive direct effect; the last
            # one is deliberately the negative-control backbone for the >=4/5
            # gate.
            direct = 0.12 if backbone_index < max(0, backbones - 1) else -0.02
            for arm, overlap, probe in arms:
                overlap_value = (
                    overlap
                    + (0.01 * backbone_index)
                    + direct * (1.0 if arm == "nonlinearity" else 0.0)
                )
                rows.append(
                    {
                        "backbone": f"backbone-{backbone_index}",
                        "replicate": replicate,
                        "arm": arm,
                        "method": "overlap_cross_fitted",
                        "head": "quadratic",
                        "auc": overlap_value,
                        "score": overlap_value,
                        "status": "complete",
                    }
                )
                rows.append(
                    {
                        "backbone": f"backbone-{backbone_index}",
                        "replicate": replicate,
                        "arm": arm,
                        "method": "linear_probe_oof",
                        "head": "quadratic",
                        "auc": probe,
                        "score": probe,
                        "status": "complete",
                    }
                )
                if include_linear:
                    rows.append(
                        {
                            "backbone": f"backbone-{backbone_index}",
                            "replicate": replicate,
                            "arm": arm,
                            "method": "overlap_cross_fitted",
                            "head": "linear",
                            "auc": overlap_value,
                            "score": overlap_value,
                            "status": "complete",
                        }
                    )
    return rows


def _test_predictions(*, backbones=5, replicates=5, n_per_class=4):
    labels = np.repeat(np.arange(4), n_per_class)
    predictions = []

    def with_correct_count(count):
        output = (labels + 1) % 4
        output[: int(count)] = labels[: int(count)]
        return output

    for backbone_index in range(backbones):
        for replicate in range(replicates):
            for arm in ("baseline", "nonlinearity_full", "nuisance_full"):
                if arm == "baseline":
                    correct = backbone_index
                elif arm == "nonlinearity_full":
                    correct = backbones - 1 - backbone_index
                else:
                    correct = backbones - 1 - backbone_index
                predictions.append(
                    {
                        "backbone": f"backbone-{backbone_index}",
                        "replicate": replicate,
                        "arm": arm,
                        "head": "quadratic",
                        "predictions": with_correct_count(correct).tolist(),
                        "labels": labels.tolist(),
                        "class_labels": labels.tolist(),
                    }
                )
    return predictions, labels


def _install_fake_final_output_extractors(monkeypatch):
    """Patch provider classes so final-output factory tests never load models."""

    from vertebrae.extractors import huggingface_vision, openclip, timm

    records = []

    def fake_class(provider, extractor_type):
        class FakeExtractor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.extractor_type = extractor_type
                records.append((provider, self))

        return FakeExtractor

    monkeypatch.setattr(
        huggingface_vision,
        "HFVisionExtractor",
        fake_class("huggingface", "frozen_pretrained"),
    )
    monkeypatch.setattr(
        openclip,
        "OpenCLIPExtractor",
        fake_class("openclip", "openclip"),
    )
    monkeypatch.setattr(
        timm,
        "TimmVisionExtractor",
        fake_class("timm", "timm"),
    )
    return records


def test_help_is_lazy_and_defaults_are_frozen(experiment):
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "food101_nonlinear_backbone_bridge.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "Traceback" not in completed.stderr
    for option in ("--jobs", "--device", "--resume"):
        assert option in completed.stdout
    assert "food" in completed.stdout.lower()
    assert "extracting" not in completed.stdout.lower()

    quality = tuple(
        _constant(experiment, name) for name in ("_QUALITY_LEVELS",) if hasattr(experiment, name)
    )
    assert quality, "quality ladder must be declared as a module constant"
    assert tuple(float(value) for value in quality[0]) == pytest.approx((1.0,))
    heads = tuple(_constant(experiment, "_HEAD_FAMILIES"))
    assert heads == ("linear", "quadratic", "knn", "rbf")
    methods = tuple(_constant(experiment, "_METHODS"))
    assert "overlap_cross_fitted" in methods
    assert "linear_probe_oof" in methods
    parser = _helper(experiment, "_parser")
    args = parser().parse_args([])
    assert getattr(args, "stage", "all") == "all"
    assert getattr(args, "jobs", "auto") == "auto"
    assert getattr(args, "device", "auto") == "auto"


def test_final_output_factory_builds_one_declared_output_per_frozen_backbone(
    experiment, monkeypatch
):
    records = _install_fake_final_output_extractors(monkeypatch)
    expected = {
        "dinov2-small": {
            "provider": "huggingface",
            "identifier": ("model_id", "facebook/dinov2-small"),
            "output": {"name": "final_cls", "hidden_layer": -1, "pooling": "cls"},
        },
        "deit-tiny": {
            "provider": "huggingface",
            "identifier": ("model_id", "facebook/deit-tiny-patch16-224"),
            "output": {"name": "final_cls", "hidden_layer": -1, "pooling": "cls"},
        },
        "convnext-tiny": {
            "provider": "timm",
            "identifier": ("model_name", "convnext_tiny"),
            "output": {"name": "final"},
        },
        "mobilenetv3-large": {
            "provider": "timm",
            "identifier": ("model_name", "mobilenetv3_large_100"),
            "output": {"name": "final"},
        },
        "openclip-vit-b-32": {
            "provider": "openclip",
            "identifier": ("model_name", "ViT-B-32"),
            "output": {"name": "final_image", "source": "image"},
        },
        "resnet50": {
            "provider": "timm",
            "identifier": ("model_name", "resnet50"),
            "output": {"name": "final"},
        },
        "efficientnet-b0": {
            "provider": "timm",
            "identifier": ("model_name", "efficientnet_b0"),
            "output": {"name": "final"},
        },
        "swin-tiny": {
            "provider": "timm",
            "identifier": ("model_name", "swin_tiny_patch4_window7_224"),
            "output": {"name": "final"},
        },
        "vit-small-16": {
            "provider": "timm",
            "identifier": ("model_name", "vit_small_patch16_224"),
            "output": {"name": "final"},
        },
        "densenet121": {
            "provider": "timm",
            "identifier": ("model_name", "densenet121"),
            "output": {"name": "final"},
        },
    }
    models = tuple(experiment._DEFAULT_MODELS)
    extractors = experiment._build_final_output_extractors(models, batch_size=8, device="cpu")

    assert len(extractors) == len(models) == len(expected)
    assert len(records) == len(models)
    assert [extractor.kwargs["name"] for _, extractor in records] == list(models)
    for provider, extractor in records:
        model = extractor.kwargs["name"]
        spec = expected[model]
        identifier_key, identifier = spec["identifier"]
        assert provider == spec["provider"]
        assert extractor.kwargs[identifier_key] == identifier
        assert extractor.kwargs["batch_size"] == 8
        assert extractor.kwargs["device"] == "cpu"
        outputs = extractor.kwargs["outputs"]
        assert len(outputs) == 1
        assert outputs[0] == spec["output"]

    assert records[1][1].kwargs["model_kwargs"] == {"add_pooling_layer": False}
    assert records[4][1].kwargs["pretrained"] == "laion2b_s34b_b79k"
    assert records[4][1].kwargs["input_modalities"] == {"image": "image"}
    for _, extractor in records:
        if extractor.kwargs["name"] != "deit-tiny":
            assert extractor.kwargs.get("model_kwargs", {}).get("add_pooling_layer") is not False


def test_final_output_transform_uses_openclip_image_mapping(experiment):
    class RecordingExtractor:
        def __init__(self, extractor_type):
            self.extractor_type = extractor_type
            self.inputs = []

        def transform_many(self, inputs):
            self.inputs.append(inputs)
            return []

    images = (SimpleNamespace(name="first"), SimpleNamespace(name="second"))
    openclip = RecordingExtractor("openclip")
    regular = RecordingExtractor("timm")
    assert experiment._transform_final_outputs(openclip, images) == []
    assert experiment._transform_final_outputs(regular, images) == []
    assert openclip.inputs == [{"image": list(images)}]
    assert regular.inputs == [list(images)]


def test_final_output_factory_rejects_unknown_model_without_loading_optional_models(
    experiment, monkeypatch
):
    records = _install_fake_final_output_extractors(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported model"):
        experiment._build_final_output_extractors(
            ["unknown-food-backbone"], batch_size=8, device="cpu"
        )
    assert records == []


def test_class_selection_is_alphabetical_and_deterministic(experiment):
    selector = _helper(experiment, "_first_food101_classes")
    classes = [f"class_{index:02d}" for index in range(45)]
    scrambled = list(reversed(classes))
    first = _call_with_supported_kwargs(
        selector,
        classes=scrambled,
        class_names=scrambled,
        labels=scrambled,
        n_classes=40,
        count=40,
        seed=13,
    )
    second = _call_with_supported_kwargs(
        selector,
        classes=classes,
        class_names=classes,
        labels=classes,
        n_classes=40,
        count=40,
        seed=999,
    )
    first = list(first)
    second = list(second)
    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 40
    assert first == classes[:40]


def test_official_train_test_isolation_and_80_52_52_cohorts(experiment):
    classes = tuple(f"food_{index:02d}" for index in range(40))
    train = _food_samples(classes=classes, per_class=264, split="train")
    official_test = _food_samples(classes=classes, per_class=190, split="test", start=1000)
    splitter = _helper(experiment, "_food101_cohort_splits")
    first = _call_with_supported_kwargs(
        splitter,
        samples=train,
        train_samples=train,
        test_samples=official_test,
        labels=_class_labels(train),
        train_labels=_class_labels(train),
        test_labels=_class_labels(official_test),
        replicates=2,
        repeats=2,
        seed=17,
    )
    second = _call_with_supported_kwargs(
        splitter,
        samples=train,
        train_samples=train,
        test_samples=official_test,
        labels=_class_labels(train),
        train_labels=_class_labels(train),
        test_labels=_class_labels(official_test),
        replicates=2,
        repeats=2,
        seed=17,
    )
    assert repr(first) == repr(second) or str(first) == str(second)
    replicates = (
        [first]
        if isinstance(first, Mapping) and "selector" in first
        else (list(first.values()) if isinstance(first, Mapping) else list(first))
    )
    official_test_ids = {_sample_id(row) for row in official_test}
    train_role_ids = []
    for replicate in replicates:
        roles = _roles(replicate)
        role_ids = {}
        for role, value in roles.items():
            rows = _rows(value)
            ids = {_sample_id(row, index) for index, row in enumerate(rows)}
            role_ids[role] = ids
            counts = {}
            for index, row in enumerate(rows):
                label = _sample_label(row, train[index].class_name if index < len(train) else None)
                counts[label] = counts.get(label, 0) + 1
            expected = {"selector": 80, "development": 52, "test": 52}[role]
            assert set(counts.values()) == {expected}
            # The selected/development roles must be official training rows;
            # a bridge test may be sourced from either explicitly declared
            # held-out test rows or the disjoint train reserve, but never both.
            if role != "test":
                assert all(_sample_split(row) in (None, "train") for row in rows)
        assert role_ids["selector"].isdisjoint(role_ids["development"])
        assert role_ids["selector"].isdisjoint(role_ids["test"])
        assert role_ids["development"].isdisjoint(role_ids["test"])
        # No official test image is allowed to leak into selector/development.
        assert not (role_ids["selector"] | role_ids["development"]) & official_test_ids
        train_role_ids.append(role_ids["selector"] | role_ids["development"])
    for index, current in enumerate(train_role_ids):
        for previous in train_role_ids[:index]:
            assert current.isdisjoint(previous)


def test_split_local_banks_are_independent_and_do_not_alias_or_leak(experiment):
    bank_builder = _helper(experiment, "_paired_split_banks")
    first = np.arange(48, dtype=np.float32).reshape(12, 4)
    second = np.arange(48, 96, dtype=np.float32).reshape(12, 4)
    labels = np.repeat(np.arange(3), 4)
    bank_a = _call_with_supported_kwargs(
        bank_builder, embeddings=first, matrix=first, labels=labels, seed=7
    )
    bank_b = _call_with_supported_kwargs(
        bank_builder, embeddings=second, matrix=second, labels=labels, seed=7
    )
    assert isinstance(bank_a, Mapping) and isinstance(bank_b, Mapping)
    arrays_a = [np.asarray(value) for value in bank_a.values() if isinstance(value, np.ndarray)]
    arrays_b = [np.asarray(value) for value in bank_b.values() if isinstance(value, np.ndarray)]
    assert arrays_a and arrays_b
    assert all(not np.shares_memory(array, first) for array in arrays_a)
    assert all(not np.shares_memory(array, second) for array in arrays_b)
    assert all(not np.shares_memory(left, right) for left in arrays_a for right in arrays_b)
    before = [array.copy() for array in arrays_b]
    arrays_a[0][...] = -999.0
    for original, current in zip(before, arrays_b):
        np.testing.assert_array_equal(original, current)
    # If the implementation materialises source indices, they must remain in
    # the split-local range rather than drawing donors from a global bank.
    for key, value in list(bank_a.items()) + list(bank_b.items()):
        if "index" in str(key).lower() or "source" in str(key).lower():
            indices = np.asarray(value)
            assert np.all((indices >= 0) & (indices < len(first)))


def test_q1_bridge_has_exact_three_arm_geometry_and_unit_rows(experiment):
    arms = _constant(experiment, "_FOOD101_ARMS")
    arm_names = tuple(
        row[0] if isinstance(row, (tuple, list)) else row.get("name", row.get("arm"))
        for row in arms
    )
    assert set(arm_names) == {"baseline", "nonlinearity_full", "nuisance_full"}
    assert len(arm_names) == 3
    transform = _helper(experiment, "_bridge_transform")
    x = np.eye(4, dtype=np.float32)
    donor = np.roll(x, 1, axis=0)
    mode = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    nuisance = np.flip(x, axis=1).astype(np.float32)

    def call(**kwargs):
        return np.asarray(
            _call_with_supported_kwargs(
                transform,
                x=x,
                embeddings=x,
                base=x,
                donor=donor,
                donor_embeddings=donor,
                noise=donor,
                mode=mode,
                nuisance=nuisance,
                quality=1.0,
                q=1.0,
                **kwargs,
            )
        )

    outputs = {}
    for row in arms:
        name = row[0] if isinstance(row, (tuple, list)) else row.get("name", row.get("arm"))
        lam = (
            row[1]
            if isinstance(row, (tuple, list)) and len(row) > 1
            else row.get("lambda", row.get("lam", 0.0))
        )
        nu = (
            row[2]
            if isinstance(row, (tuple, list)) and len(row) > 2
            else row.get("nu", row.get("nuisance", 0.0))
        )
        outputs[name] = call(
            lam=lam,
            lambda_=lam,
            nonlinearity=lam,
            nu=nu,
            nuisance_strength=nu,
        )
    assert len({array.shape for array in outputs.values()}) == 1
    assert outputs["baseline"].shape[0] == len(x)
    assert outputs["baseline"].shape[1] == 3 * x.shape[1] + 1
    for array in outputs.values():
        np.testing.assert_allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-5)
    zeros = np.zeros_like(x)
    expected_baseline = np.concatenate((x, zeros, np.zeros((len(x), 1)), zeros), axis=1)
    expected_nonlinear = np.concatenate((zeros, mode[:, None] * x, mode[:, None], zeros), axis=1)
    expected_nuisance = np.concatenate((x, zeros, np.zeros((len(x), 1)), 1.5 * nuisance), axis=1)
    expected_nonlinear /= np.linalg.norm(expected_nonlinear, axis=1, keepdims=True)
    expected_nuisance /= np.linalg.norm(expected_nuisance, axis=1, keepdims=True)
    np.testing.assert_allclose(outputs["baseline"], expected_baseline, atol=1e-6)
    np.testing.assert_allclose(outputs["nonlinearity_full"], expected_nonlinear, atol=1e-6)
    np.testing.assert_allclose(outputs["nuisance_full"], expected_nuisance, atol=1e-6)
    assert not np.allclose(outputs["baseline"], outputs["nonlinearity_full"])
    assert not np.allclose(outputs["baseline"], outputs["nuisance_full"])
    assert all(
        not np.shares_memory(outputs["baseline"], array)
        for name, array in outputs.items()
        if name != "baseline"
    )


def test_overlap_adapter_uses_exact_k10_five_fold_min5_and_probe(experiment, monkeypatch):
    calls = []

    class FakeScorer:
        def __init__(self, config):
            self.config = config

        def score_cross_fitted(self, embeddings, labels, n_splits, seed=None):
            calls.append((self.config, len(labels), n_splits, seed))
            return SimpleNamespace(
                macro_score=0.73,
                warnings=[],
                k_per_class={0: 10, 1: 10},
                metadata={"score_kind": "classification_overlap_cross_fitted"},
            )

    monkeypatch.setattr(experiment, "OverlapIndexScorer", FakeScorer)
    x = np.arange(120, dtype=np.float32).reshape(60, 2)
    labels = np.repeat(np.asarray(["class_a", "class_b"]), 30)
    scorer = _helper(experiment, "_score_overlap")
    outcome = _call_with_supported_kwargs(
        scorer,
        embeddings=x,
        labels=labels,
        seed=23,
        folds=5,
        n_splits=5,
        k=10,
        overlap_k=10,
        mode="overlap_cross_fitted",
    )
    assert float(_extract(outcome, "score", "macro_score")) == pytest.approx(0.73)
    assert len(calls) == 1
    config, _, folds, seed = calls[0]
    assert config.k == config.min_k == config.max_k == 10
    assert config.min_samples_per_cluster == 5
    assert folds == 5 and seed == 23
    probe = _helper(experiment, "_score_probe")
    probe_result = _call_with_supported_kwargs(
        probe, embeddings=x, labels=labels, seed=23, folds=5, n_splits=5
    )
    assert np.isfinite(float(_extract(probe_result, "score", "accuracy", "auc")))


def test_fixed_head_recipes_have_quadratic_primary(experiment):
    families = tuple(_constant(experiment, "_HEAD_FAMILIES"))
    assert families == ("linear", "quadratic", "knn", "rbf")
    make_head = _helper(experiment, "_make_head_estimator")
    recipe = _helper(experiment, "_head_recipe")
    signature = inspect.signature(make_head)
    assert not any(
        name in signature.parameters for name in ("grid", "search", "select_family", "validation")
    )
    for family in families:
        estimator = _call_with_supported_kwargs(make_head, family=family, seed=11)
        assert "standardscaler" not in repr(estimator).lower()
    recipes = {family: recipe(family) for family in families}
    assert recipes["linear"]["C"] == 1.0
    assert recipes["quadratic"]["kernel"] == "poly"
    assert recipes["quadratic"]["degree"] == 2
    assert recipes["knn"]["n_neighbors"] == 15
    assert recipes["knn"]["weights"] == "distance"
    assert recipes["knn"]["metric"] == "cosine"
    assert recipes["rbf"]["kernel"] == "rbf"
    assert recipes["rbf"]["C"] == 1.0
    protocol = _helper(experiment, "_master_protocol")(_helper(experiment, "_configuration")())
    assert protocol["frozen"]["primary_reference_head"] == "quadratic"


def test_nested_budgets_are_balanced_nested_and_paired(experiment):
    sampler = _helper(experiment, "_nested_stratified_indices")
    labels = np.repeat(np.arange(5), 12)
    budgets = (2, 4, 8)
    result = _call_with_supported_kwargs(
        sampler,
        labels=labels,
        fine_labels=labels,
        budgets=budgets,
        seed=7,
    )
    assert isinstance(result, Mapping)
    previous = set()
    for budget in budgets:
        current = np.asarray(result[budget])
        assert len(current) == budget * 5
        assert len(np.unique(current)) == len(current)
        assert set(np.unique(labels[current], return_counts=True)[1]) == {budget}
        assert previous.issubset(set(current))
        previous = set(current)
    # Every arm must reuse the same split-local paired bank; only its declared
    # lambda/nu factor changes.  This guards against silently drawing a fresh
    # donor or nuisance sample for one arm.
    matrix = np.eye(5, dtype=np.float32)
    labels_small = np.arange(5)
    bank = _helper(experiment, "_paired_split_banks")(matrix, labels_small, seed=13)
    bank_before = {key: np.asarray(value).copy() for key, value in bank.items()}
    transform = _helper(experiment, "_bridge_transform")
    arms = _constant(experiment, "_FOOD101_ARMS")
    transform_bank = {
        key: value for key, value in bank.items() if key in {"donor", "mode", "nuisance"}
    }
    transformed = [
        transform(matrix, **transform_bank, q=1.0, lam=lam, nu=nu) for _, lam, nu in arms
    ]
    assert all(not np.shares_memory(transformed[0], current) for current in transformed[1:])
    for key, value in bank.items():
        np.testing.assert_array_equal(value, bank_before[key])


def test_cross_backbone_ranks_and_normalized_log_auc_are_tie_safe(experiment):
    rank = _helper(experiment, "_rank_metrics")
    metrics = _call_with_supported_kwargs(
        rank,
        candidates={"a": 0.2, "b": 0.8, "c": 0.5},
        reference={"a": 0.1, "b": 0.9, "c": 0.4},
    )
    assert float(metrics["spearman"]) > 0.9
    assert float(metrics["kendall"]) > 0.5
    assert float(metrics["exact_best"]) == pytest.approx(1.0)
    within = metrics.get("within_1pct", metrics.get("within_one_percent"))
    assert float(within) == pytest.approx(1.0)
    auc = _helper(experiment, "_normalized_log_auc")
    value = _call_with_supported_kwargs(auc, budgets=(64, 80, 160), values=(0.5, 0.75, 1.0))
    assert float(value) == pytest.approx(0.8141176993, abs=1e-10)
    selector_rows = []
    reference_rows = []
    for arm, _, _ in experiment._FOOD101_ARMS:
        for method in experiment._METHODS:
            for budget in experiment._FOOD101_BUDGETS:
                for backbone_index in range(2):
                    selector_rows.append(
                        {
                            "replicate": 0,
                            "backbone": f"backbone-{backbone_index}",
                            "arm": arm,
                            "method": method,
                            "budget": budget,
                            "score": float(backbone_index) + budget / 1_000.0,
                        }
                    )
        for head in experiment._HEAD_FAMILIES:
            for backbone_index in range(2):
                reference_rows.append(
                    {
                        "replicate": 0,
                        "backbone": f"backbone-{backbone_index}",
                        "arm": arm,
                        "head": head,
                        "test_accuracy": float(backbone_index),
                    }
                )
    ranking_rows, auc_rows = _helper(experiment, "_cross_backbone_rows")(
        selector_rows, reference_rows, budgets=experiment._FOOD101_BUDGETS
    )
    assert ranking_rows and auc_rows
    assert all(np.isfinite(float(row["auc"])) for row in auc_rows)


def test_bootstrap_direct_interaction_effects_test_sensitivity_gates_and_nuisance(experiment):
    bootstrap = _helper(experiment, "_bootstrap_food101")
    rows = _three_arm_rows()
    auc_rows = [row for row in rows if row["backbone"] == "backbone-0"]
    predictions, test_labels = _test_predictions()
    selector_rows = []
    for row in rows:
        for budget in (64, 68, 72, 80):
            backbone_index = int(str(row["backbone"]).rsplit("-", 1)[-1])
            if row["arm"] == "baseline":
                rank_score = 4 - backbone_index
            elif row["arm"] == "nonlinearity_full":
                rank_score = (
                    4 - backbone_index
                    if row["method"] == "overlap_cross_fitted"
                    else backbone_index
                )
            else:
                rank_score = (
                    backbone_index
                    if row["method"] == "overlap_cross_fitted"
                    else 4 - backbone_index
                )
            selector_rows.append(
                {
                    "backbone": row["backbone"],
                    "replicate": row["replicate"],
                    "arm": row["arm"],
                    "method": row["method"],
                    "budget": budget,
                    "score": rank_score + 0.001 * budget,
                }
            )
    kwargs = {
        "rows": rows,
        "auc_rows": auc_rows,
        "selector_rows": selector_rows,
        "replicates": 5,
        "backbones": 5,
        "n_resamples": 100,
        "bootstrap_resamples": 100,
        "seed": 19,
        "canonical": True,
        "protocol_conformant": True,
        "test_predictions": predictions,
        "test_within_class_predictions": predictions,
        "test_labels": test_labels,
        "test_fine_labels": test_labels,
        "models": [f"backbone-{index}" for index in range(5)],
    }
    result = _call_with_supported_kwargs(bootstrap, **kwargs)
    assert isinstance(result, Mapping)
    for names in (
        ("direct_nonlinear_oi_minus_probe",),
        ("direct_nonlinear_oi_minus_probe_interval_95",),
        ("nonlinear_baseline_interaction",),
        ("nonlinear_baseline_interaction_interval_95",),
        ("nuisance_direct_interval_95",),
    ):
        assert any(name in result for name in names), f"missing bootstrap field {names!r}"
    positive_count = _extract(
        result,
        "replicate_advantages",
    )
    if positive_count is not None:
        assert int(positive_count) >= 4
    positive_fraction = _extract(
        result,
        "replicate_advantage_fraction",
    )
    if positive_fraction is not None:
        assert float(positive_fraction) >= 0.8
    valid_fraction = _extract(result, "valid_fraction", "complete_fraction")
    assert valid_fraction is None or float(valid_fraction) >= 0.90
    nuisance_interval = _extract(result, "nuisance_direct_interval_95")
    if nuisance_interval is not None:
        assert float(nuisance_interval[1]) < 0.0
    assert result["claim_supported"] is False
    assert result["nuisance_diagnostic_only"] is True

    # A test-within-class perturbation must change the direct-effect summary;
    # the bridge cannot derive this diagnostic from development rows alone.
    perturbed = copy.deepcopy(predictions)
    for row in perturbed:
        if row["backbone"] == "backbone-0":
            row["predictions"] = ((test_labels + 1) % 4).tolist()
    changed_kwargs = dict(kwargs)
    changed_kwargs["test_predictions"] = perturbed
    changed_kwargs["test_within_class_predictions"] = perturbed
    changed = _call_with_supported_kwargs(bootstrap, **changed_kwargs)
    summary_keys = (
        "test_sensitivity_interval_95",
        "test_within_class_sensitivity",
        "direct_nonlinear_oi_minus_probe_interval_95",
    )
    assert any(
        result.get(key) != changed.get(key)
        for key in summary_keys
        if key in result or key in changed
    )


def test_full_factorial_validator_rejects_duplicate_missing_and_nonfinite(experiment):
    validator = _helper(experiment, "_validate_factorial_grid")
    rows = _three_arm_rows()

    expected_keys = [
        (f"backbone-{backbone}", replicate, arm, method, "quadratic")
        for backbone in range(5)
        for replicate in range(5)
        for arm in ("baseline", "nonlinearity_full", "nuisance_full")
        for method in ("overlap_cross_fitted", "linear_probe_oof")
    ]

    def call(payload):
        return validator(
            payload,
            key_fields=("backbone", "replicate", "arm", "method", "head"),
            expected_keys=expected_keys,
        )

    call(rows)
    with pytest.raises((AssertionError, ValueError), match="duplicates|factorial|cell"):
        call(rows + [copy.deepcopy(rows[0])])
    with pytest.raises((AssertionError, ValueError), match="missing|factorial|cell"):
        call(rows[1:])
    nonfinite = copy.deepcopy(rows)
    nonfinite[0]["auc"] = np.nan
    with pytest.raises((AssertionError, ValueError), match="nonfinite|factorial|finite"):
        call(nonfinite)


def test_operational_jobs_do_not_change_scientific_or_cache_identity(experiment):
    configuration = _helper(experiment, "_configuration")
    stem = _helper(experiment, "_artifact_stem")
    first = _call_with_supported_kwargs(
        configuration,
        stage="all",
        jobs=1,
        device="cpu",
        seed=42,
    )
    second = _call_with_supported_kwargs(
        configuration,
        stage="all",
        jobs=8,
        device="cuda",
        seed=42,
    )
    first_stem = _call_with_supported_kwargs(stem, configuration=first, config=first)
    second_stem = _call_with_supported_kwargs(stem, configuration=second, config=second)
    assert first_stem == second_stem
    scientific_first = dict(first)
    scientific_second = dict(second)
    for key in ("jobs", "requested_jobs", "worker_count", "device", "device_resolved"):
        scientific_first.pop(key, None)
        scientific_second.pop(key, None)
    assert scientific_first == scientific_second


def test_explicit_mps_requires_an_available_torch_backend(experiment, monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="Apple MPS is unavailable"):
        _helper(experiment, "_resolve_torch_device")("mps")
    assert _helper(experiment, "_resolve_torch_device")("auto") == "cpu"


def test_compact_extraction_panel_preserves_28480_rows_and_test_offset(
    experiment, monkeypatch, tmp_path
):
    classes = tuple(f"food_{index:02d}" for index in range(40))
    train = _food_samples(classes=classes, per_class=660, split="train")
    official_test = _food_samples(classes=classes, per_class=52, split="test", start=1000)
    train_labels = _class_labels(train)
    test_labels = _class_labels(official_test)
    images = [str(row.image_path) for row in train]
    test_images = [str(row.image_path) for row in official_test]
    monkeypatch.setattr(
        experiment,
        "_load_food101_rows",
        lambda _data_dir, _no_download: (
            train,
            train_labels,
            images,
            official_test,
            test_labels,
            test_images,
        ),
    )
    monkeypatch.setattr(experiment, "_resolve_torch_device", lambda value: "cpu")
    captured = {}

    def fake_extract(images, models, **kwargs):
        captured["images"] = list(images)
        captured["models"] = tuple(models)
        matrix = np.zeros((len(images), 2), dtype=np.float32)
        manifest = experiment._write_float32_memmap(
            Path(kwargs["cache_dir"]) / "compact.npy", matrix
        )
        return {models[0]: manifest}, []

    monkeypatch.setattr(experiment, "_extract_final_embeddings", fake_extract)

    def fake_parallel(tasks, **kwargs):
        captured["tasks"] = list(tasks)
        outputs = []
        for task in tasks:
            reference_rows = [
                {
                    "backbone": task["backbone"],
                    "replicate": task["replicate"],
                    "arm": arm,
                    "head": head,
                    "test_accuracy": 0.5,
                }
                for arm, _, _ in experiment._FOOD101_ARMS
                for head in experiment._HEAD_FAMILIES
            ]
            selector_rows = [
                {
                    "backbone": task["backbone"],
                    "replicate": task["replicate"],
                    "arm": arm,
                    "method": method,
                    "budget": budget,
                    "score": 0.5,
                }
                for arm, _, _ in experiment._FOOD101_ARMS
                for method in experiment._METHODS
                for budget in (64, 68, 72, 80)
            ]
            outputs.append(
                {
                    "key": task["key"],
                    "status": "complete",
                    "reference_rows": reference_rows,
                    "selector_rows": selector_rows,
                    "test_prediction_rows": [],
                }
            )
        return outputs

    monkeypatch.setattr(experiment, "_run_parallel_blocks", fake_parallel)
    args = SimpleNamespace(
        jobs=1,
        device="cpu",
        resume=False,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        models=experiment._DEFAULT_MODELS[0],
        embedding_batch_size=2,
        budgets="64,68,72,80",
        seed=42,
        no_download=True,
        replicates=5,
        bootstrap_resamples=1,
    )
    assert _helper(experiment, "_run")(args) == 0
    assert len(captured["images"]) == 28_480
    assert len(captured["images"]) == len(train) + len(official_test)
    assert len(captured["tasks"]) == 5
    role_indices = captured["tasks"][0]["roles"]
    train_count = len(train)
    test_indices = [index for task in captured["tasks"] for index in task["roles"]["test"]]
    assert min(test_indices) == train_count
    assert max(test_indices) == len(captured["images"]) - 1
    assert all(index < train_count for index in role_indices["selector"])
    assert all(index < train_count for index in role_indices["development"])


def test_spawn_parallel_runner_checkpoints_resumes_reports_and_preserves_order(
    experiment, monkeypatch, capsys, tmp_path
):
    runner = _helper(experiment, "_run_parallel_blocks")
    blocks = [{"key": index, "value": index} for index in range(2)]
    progress = []

    spawn_calls = []

    class FakePool:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def imap_unordered(self, function, payloads):
            return (function(payload) for payload in payloads)

    class FakeContext:
        Pool = FakePool

    def fake_get_context(method):
        spawn_calls.append(method)
        return FakeContext()

    monkeypatch.setattr(experiment.mp, "get_context", fake_get_context)
    monkeypatch.setattr(experiment, "_resolve_jobs", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(_square_worker, "__module__", experiment.__name__)

    first = _call_with_supported_kwargs(
        runner,
        blocks=blocks,
        tasks=blocks,
        worker=_square_worker,
        jobs=2,
        start_method="spawn",
        checkpoint_dir=tmp_path,
        output_dir=tmp_path,
        resume=False,
        progress_callback=lambda item: progress.append(item),
        on_progress=lambda item: progress.append(item),
    )
    first_rows = list(first)
    assert [row["key"] for row in first_rows] == list(range(2))
    assert [row["value"] for row in first_rows] == [index * index for index in range(2)]
    assert spawn_calls == ["spawn"]
    assert "Completed block" in capsys.readouterr().out
    assert progress or any(tmp_path.iterdir())

    def fail_if_called(_):
        raise AssertionError("resume must not rerun completed blocks")

    second = _call_with_supported_kwargs(
        runner,
        blocks=blocks,
        tasks=blocks,
        worker=fail_if_called,
        jobs=2,
        start_method="spawn",
        checkpoint_dir=tmp_path,
        output_dir=tmp_path,
        resume=True,
        progress_callback=lambda item: progress.append(item),
        on_progress=lambda item: progress.append(item),
    )
    second_rows = list(second)
    assert second_rows == first_rows


def test_protocol_schema_and_confirmatory_claim_scope_are_explicit(experiment):
    protocol_builder = _helper(experiment, "_master_protocol")
    configuration = _call_with_supported_kwargs(
        _helper(experiment, "_configuration"),
        stage="all",
        jobs="auto",
        seed=42,
    )
    protocol = _call_with_supported_kwargs(
        protocol_builder, configuration=configuration, config=configuration
    )
    assert isinstance(protocol, Mapping)
    assert protocol["schema_version"] == protocol["protocol_version"]
    assert protocol["artifact_status"] in ("planned", "complete", "completed")
    assert protocol["claim_supported"] is False
    claim_text = " ".join(
        str(protocol.get(key, ""))
        for key in ("claim", "claim_text", "claim_scope", "interpretation")
    ).lower()
    assert all(
        token not in claim_text for token in ("all backbones", "universal", "every representation")
    )
    json.dumps(protocol)

    canonical = _helper(experiment, "_is_canonical_configuration")
    defaults = _helper(experiment, "_configuration")(stage="all", jobs="auto", seed=42)
    true_result = _call_with_supported_kwargs(
        canonical, **defaults, configuration=defaults, config=defaults
    )
    if isinstance(true_result, Mapping):
        assert bool(_extract(true_result, "canonical", "is_canonical")) is True
    else:
        assert bool(true_result) is True
    mutations = (
        {"stage": "selector"},
        {"replicates": 4},
        {"seed": 41},
        {"classes": 39},
        {"overlap_k": 5},
        {"cross_fit_folds": 4},
        {"probe_folds": 4},
        {"arms": list(defaults["arms"][:-1])},
        {"methods": ["linear_probe_oof", "overlap_cross_fitted"]},
        {"heads": ["linear", "quadratic", "rbf"]},
        {"protocol_version": 99},
    )
    for override in mutations:
        reduced = dict(defaults)
        reduced.update(override)
        outcome = _call_with_supported_kwargs(
            canonical, **reduced, configuration=reduced, config=reduced
        )
        assert not bool(_extract(outcome, "canonical", "is_canonical", default=outcome))


def test_failed_and_interrupted_artifacts_are_not_valid_completed_results(experiment, tmp_path):
    reader = _helper(experiment, "_read_artifact")
    configuration = _call_with_supported_kwargs(
        _helper(experiment, "_configuration"),
        stage="all",
        jobs=1,
        seed=42,
    )
    digest = _helper(experiment, "_configuration_hash")(configuration)
    protocol = _helper(experiment, "_master_protocol")(configuration)
    protocol["artifact_status"] = "completed"
    protocol["food101_nonlinearity_supported"] = False
    base = {
        "schema_version": 1,
        "protocol_version": 1,
        "study": "food101_nonlinear_backbone_bridge",
        "artifact_status": "completed",
        "configuration": dict(configuration),
        "configuration_hash": digest,
        "claim_supported": False,
        "protocol": protocol,
        "food101_nonlinearity_supported": False,
        "bootstrap": {"food101_nonlinearity_supported": False},
    }
    path = tmp_path / "food101.json"
    for status in ("failed", "interrupted"):
        payload = dict(base)
        payload["artifact_status"] = status
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(
            (AssertionError, ValueError), match="failed|interrupted|status|completed"
        ):
            _call_with_supported_kwargs(reader, path=path, source=path)
    path.write_text(json.dumps(base), encoding="utf-8")
    loaded = _call_with_supported_kwargs(reader, path=path, source=path)
    assert loaded is not None

    invalid_payloads = []
    missing_protocol = dict(base)
    missing_protocol.pop("protocol")
    invalid_payloads.append(missing_protocol)
    bad_hash = copy.deepcopy(base)
    bad_hash["configuration_hash"] = "0" * 64
    invalid_payloads.append(bad_hash)
    bad_protocol_hash = copy.deepcopy(base)
    bad_protocol_hash["protocol"]["configuration_hash"] = "0" * 64
    invalid_payloads.append(bad_protocol_hash)
    missing_narrow = copy.deepcopy(base)
    missing_narrow.pop("food101_nonlinearity_supported")
    invalid_payloads.append(missing_narrow)
    bad_bootstrap = copy.deepcopy(base)
    bad_bootstrap["bootstrap"]["food101_nonlinearity_supported"] = True
    invalid_payloads.append(bad_bootstrap)
    missing_claim = copy.deepcopy(base)
    missing_claim.pop("claim_supported")
    invalid_payloads.append(missing_claim)
    true_claim = copy.deepcopy(base)
    true_claim["claim_supported"] = True
    invalid_payloads.append(true_claim)
    for payload in invalid_payloads:
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(
            (AssertionError, ValueError), match="artifact|protocol|claim|hash|support"
        ):
            _call_with_supported_kwargs(reader, path=path, source=path)
