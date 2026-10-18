import hashlib
import importlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def suite():
    examples = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples))
    try:
        yield importlib.import_module("tiny_shakespeare_transformer_visual_suite")
    finally:
        sys.path.remove(str(examples))


def test_help_does_not_import_optional_visual_dependencies():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "tiny_shakespeare_transformer_visual_suite.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--monitor-every" in completed.stdout
    assert "--profile {fast,quality}" in completed.stdout
    assert "--minimum-macro-support" in completed.stdout
    assert "--device {auto,cpu,cuda,mps,xpu}" in completed.stdout
    assert "--no-download" in completed.stdout


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--steps", "-1"], "--steps must be >= 0"),
        (["--context-length", "0"], "--context-length must be >= 1"),
        (["--probe-per-class-cap", "0"], "--probe-per-class-cap must be >= 1"),
    ],
)
def test_argument_validation_precedes_optional_imports(arguments, message):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "tiny_shakespeare_transformer_visual_suite.py"),
            *arguments,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_corpus_cache_checksum_and_offline_behavior(suite, tmp_path, monkeypatch):
    payload = b"To be, or not to be\n"
    checksum = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(suite, "_CORPUS_SHA256", checksum)
    monkeypatch.setattr(suite, "_CORPUS_BYTES", len(payload))

    with pytest.raises(FileNotFoundError, match="not cached"):
        suite._ensure_corpus(tmp_path, download=False)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(suite.urllib.request, "urlopen", lambda *args, **kwargs: Response(payload))
    path = suite._ensure_corpus(tmp_path, download=True)
    assert path.read_bytes() == payload
    assert suite._ensure_corpus(tmp_path, download=False) == path

    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        suite._ensure_corpus(tmp_path, download=False)


def test_corpus_split_is_contiguous_disjoint_and_vocabulary_is_training_only(suite):
    text = "a" * 90 + "b" * 5 + "c" * 5
    splits = suite._split_corpus(text, checksum="digest")

    assert splits.train + splits.validation + splits.test == text
    assert splits.boundaries == (90, 95)
    assert len(splits.train) == 90
    assert len(splits.validation) == len(splits.test) == 5
    with pytest.raises(ValueError, match="absent from the training-only vocabulary"):
        suite._validate_split_vocabulary(splits, {"a": 0})


def test_probe_uses_preceding_context_exact_target_and_deterministic_full_coverage(suite):
    text = "abcaabbbccaaabbbbccccddx"
    mapping = {character: index for index, character in enumerate(sorted(set(text)))}
    ids = suite._encode(text, mapping)
    kwargs = {
        "context_length": 4,
        "per_class_cap": 3,
        "minimum_macro_support": 5,
        "seed": 17,
        "source_offset": 100,
    }

    first = suite._build_probe(text, ids, **kwargs)
    second = suite._build_probe(text, ids, **kwargs)

    assert np.array_equal(first.contexts, second.contexts)
    assert np.array_equal(first.source_offsets, second.source_offsets)
    assert set(first.labels) == set(first.eligible_classes)
    assert "x" in first.excluded_singletons
    assert set(first.low_support_classes).issubset(first.eligible_classes)
    for context, label, token_id, absolute_offset in zip(
        first.contexts,
        first.labels,
        first.token_ids,
        first.source_offsets,
    ):
        position = int(absolute_offset) - 100
        assert np.array_equal(context, ids[position - 4 : position])
        assert label == text[position]
        assert token_id == ids[position]


def test_low_support_classes_are_only_overlap_macro_exclusions(suite):
    config = suite._scoring_config(("!", "?"), seed=9)

    assert config.k == "auto"
    assert config.min_k == 10
    assert config.max_k == 50
    assert config.min_samples_per_cluster == 5
    assert config.exclude_classes == ["!", "?"]


def test_quality_profile_resolves_larger_training_and_model_defaults(suite):
    parser = suite._parser()
    args = parser.parse_args(["--profile", "quality"])
    profile = suite._apply_profile_defaults(args)

    assert args.steps == 10_000
    assert args.context_length == 256
    assert args.train_batch_size == 32
    assert profile.n_layers == 6
    assert profile.n_heads == 8
    assert profile.width == 256
    assert profile.mlp_width == 1_024
    assert profile.dropout == pytest.approx(0.1)
    assert profile.output_order == (
        "token_position",
        "block_2",
        "block_4",
        "block_6_final",
    )
    metadata = suite._profile_metadata(profile, args)
    assert metadata["tokens_per_step"] == 8_192
    assert metadata["sampled_training_tokens"] == 81_920_000


def test_explicit_training_arguments_override_profile_defaults(suite):
    parser = suite._parser()
    args = parser.parse_args(
        [
            "--profile",
            "quality",
            "--steps",
            "7",
            "--context-length",
            "32",
            "--train-batch-size",
            "4",
        ]
    )
    suite._apply_profile_defaults(args)

    assert args.steps == 7
    assert args.context_length == 32
    assert args.train_batch_size == 4


def test_auto_device_falls_through_failed_accelerators_and_explicit_fails(suite, monkeypatch):
    attempts = []

    monkeypatch.setattr(
        suite,
        "_device_available",
        lambda _torch, device: device in {"cuda", "mps", "cpu"},
    )

    def smoke(_torch, device):
        attempts.append(device)
        if device != "cpu":
            raise RuntimeError("broken backend")

    monkeypatch.setattr(suite, "_device_smoke_test", smoke)
    assert suite._resolve_device(object(), "auto") == "cpu"
    assert attempts == ["cuda", "mps", "cpu"]
    with pytest.raises(RuntimeError, match="Requested Torch device 'cuda' is not usable"):
        suite._resolve_device(object(), "cuda")


def test_model_is_causal_tied_and_has_expected_output_shapes(suite):
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    model = suite._build_model(torch, vocabulary_size=19, context_length=8)
    model.eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    changed_future = torch.tensor([[1, 2, 3, 4, 12, 13, 14, 15]])

    with torch.inference_mode():
        outputs = model(first)
        changed = model(changed_future)

    assert model.lm_head.weight is model.token_embedding.weight
    assert tuple(outputs["logits"].shape) == (1, 8, 19)
    for name in suite._OUTPUT_ORDER:
        assert tuple(outputs[name].shape) == (1, 8, 128)
        assert torch.allclose(outputs[name][:, :4], changed[name][:, :4], atol=1e-6)


def test_torch_extractor_has_four_ordered_outputs_and_restores_mode(suite):
    torch = pytest.importorskip("torch")
    model = suite._build_model(torch, vocabulary_size=19, context_length=8)
    extractor = suite._multi_output_extractor(model, torch, "cpu", context_length=8, seed=4)
    model.train()

    outputs = extractor.transform_many(np.arange(24).reshape(3, 8) % 19)

    assert [output.name for output in outputs] == list(suite._OUTPUT_ORDER)
    assert [output.embeddings.shape for output in outputs] == [(3, 128)] * 4
    assert model.training is True


def test_quality_model_and_extractor_use_profile_geometry(suite):
    torch = pytest.importorskip("torch")
    profile = suite._PROFILES["quality"]
    model = suite._build_model(
        torch,
        vocabulary_size=19,
        context_length=16,
        n_layers=profile.n_layers,
        n_heads=profile.n_heads,
        width=profile.width,
        mlp_width=profile.mlp_width,
        dropout=profile.dropout,
        output_layers=profile.output_layers,
        profile_name=profile.name,
    )
    extractor = suite._multi_output_extractor(
        model,
        torch,
        "cpu",
        context_length=16,
        seed=4,
    )
    model.train()

    outputs = extractor.transform_many(np.arange(32).reshape(2, 16) % 19)

    assert model.lm_head.weight is model.token_embedding.weight
    assert model.final_output_name == "block_6_final"
    assert [output.name for output in outputs] == list(profile.output_order)
    assert [output.embeddings.shape for output in outputs] == [(2, 256)] * 4
    assert sum(parameter.numel() for parameter in model.parameters()) > 4_000_000
    assert model.training is True


def test_snapshot_model_state_is_detached_and_independent(suite):
    torch = pytest.importorskip("torch")
    model = suite._build_model(torch, vocabulary_size=7, context_length=8)
    snapshot = suite._snapshot_model_state(model)
    original = snapshot["token_embedding.weight"].clone()

    with torch.no_grad():
        model.token_embedding.weight.add_(1.0)

    assert torch.equal(snapshot["token_embedding.weight"], original)
    model.load_state_dict(snapshot)
    assert torch.equal(model.token_embedding.weight, original)


def test_final_compression_output_is_extracted_in_bounded_batches(suite):
    calls = []

    class Output:
        def __init__(self, name, embeddings):
            self.name = name
            self.embeddings = embeddings

    class Extractor:
        def transform_many(self, values):
            calls.append(len(values))
            return [
                Output("ignored", np.zeros((len(values), 2))),
                Output("block_6_final", np.asarray(values)[:, :2]),
            ]

    values = np.arange(50).reshape(10, 5)
    extracted = suite._extract_output_in_batches(
        Extractor(),
        values,
        output_name="block_6_final",
        batch_size=3,
    )

    assert calls == [3, 3, 3, 1]
    assert np.array_equal(extracted, values[:, :2])


def test_generation_learning_rate_and_pareto_helpers_are_deterministic(suite):
    torch = pytest.importorskip("torch")
    torch.manual_seed(21)
    vocabulary = tuple("\n:EMOR")
    mapping = {character: index for index, character in enumerate(vocabulary)}
    model = suite._build_model(torch, len(vocabulary), context_length=8)
    kwargs = {
        "torch": torch,
        "device": "cpu",
        "seed": 5,
        "generated_characters": 12,
        "top_k": 4,
    }

    first = suite._generate(model, "ROMEO:\n", mapping, vocabulary, **kwargs)
    second = suite._generate(model, "ROMEO:\n", mapping, vocabulary, **kwargs)

    assert first == second
    assert len(first) == len("ROMEO:\n") + 12
    assert suite._learning_rate(1, 5_000) == pytest.approx(1e-5)
    assert suite._learning_rate(100, 5_000) == pytest.approx(1e-3)
    assert suite._learning_rate(5_000, 5_000) == pytest.approx(1e-4)
    assert suite._pareto_frontier_indices(
        [8.0, 16.0, 32.0, 64.0],
        [0.80, 0.78, 0.90, 0.90],
    ) == [0, 2]


def test_readme_tiny_shakespeare_assets_are_present_and_renderable():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    stems = (
        "tiny-shakespeare-representation-monitoring",
        "tiny-shakespeare-compression-frontier",
        "tiny-shakespeare-next-token-heatmap",
    )

    for stem in stems:
        relative_png = Path("img") / "visuals" / f"{stem}.png"
        png_path = root / relative_png
        assert relative_png.as_posix() in readme
        assert png_path.with_suffix(".svg").is_file()
        with Image.open(png_path) as image:
            image.verify()
            assert image.width >= 1_000
            assert image.height >= 500
