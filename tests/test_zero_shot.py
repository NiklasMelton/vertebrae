import json
import pickle
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from scipy import sparse

from vertebrae import (
    BenchmarkDataset,
    CacheConfig,
    DatasetIdentity,
    EmbeddingCompressionConfig,
    EmbeddingConfig,
    OverlapScoringConfig,
    ResourceProfilingConfig,
    ZeroShotBenchmark,
    ZeroShotCandidate,
    ZeroShotCompressionJob,
    ZeroShotConfig,
    ZeroShotDataset,
    ZeroShotEmbeddingShardJob,
    ZeroShotScoringJob,
)
from vertebrae.cache import LocalArtifactStore
from vertebrae.cache.fingerprint import hash_json_exact
from vertebrae.cli import main
from vertebrae.execution import (
    EmbeddingMergeJob,
    compress_zero_shot_embedding_artifacts,
    materialize_zero_shot_embedding_shard,
    materialize_zero_shot_protocol,
    merge_zero_shot_embedding_shards,
    plan_zero_shot_embedding_shard_jobs,
    score_zero_shot_artifact,
    zero_shot_benchmark_result_from_artifacts,
    zero_shot_compression_artifact_key,
    zero_shot_embedding_artifact_key,
    zero_shot_scoring_artifact_key,
)
from vertebrae.execution.jobs import ShardSpec
from vertebrae.extractors import CallableRetrievalExtractor
from vertebrae.scoring import ZeroShotScorer
from vertebrae.utils.semantic_labels import LABEL_KEY_PREFIX, semantic_label_key


def _cli_query(values):
    return np.asarray(
        [[1.0, 0.0] if str(value).startswith("left") else [0.0, 1.0] for value in values]
    )


def _cli_gallery(values):
    return np.asarray([[1.0, 0.0] if value == "left" else [0.0, 1.0] for value in values])


def _alternate_query(values):
    return _cli_query(values)


def _alternate_gallery(values):
    return _cli_gallery(values)


def _materialize_aligned_zero_shot_endpoints(protocol, store):
    extractor = CallableRetrievalExtractor(
        "aligned", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    for side, branch, count in (("samples", "query", 4), ("prompts", "gallery", 2)):
        materialize_zero_shot_embedding_shard(
            ZeroShotEmbeddingShardJob(
                dataset=protocol,
                extractor=extractor,
                side=side,
                branch=branch,
                shard=ShardSpec(),
                output_key=f"{side}/shard",
            ),
            store,
        )
        merge_zero_shot_embedding_shards(
            EmbeddingMergeJob((f"{side}/shard",), side, n_samples=count), store
        )


class _HeterogeneousRetrievalExtractor:
    name = "heterogeneous"
    extractor_type = "test_retrieval"

    def encode_retrieval(self, values, *, branch, modality):
        if branch == "vision" and modality == "image":
            return _cli_query(values)
        if branch == "language" and modality == "text":
            return _cli_gallery(values)
        raise ValueError("unexpected endpoint")

    def recipe(self):
        return {"name": self.name, "extractor_type": self.extractor_type}


class _SemanticLabel(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class _FirstDataclassLabel:
    value: str


@dataclass(frozen=True)
class _SecondDataclassLabel:
    value: str


class _UnsupportedLabel:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, _UnsupportedLabel) and self.value == other.value


def test_zero_shot_dataset_requires_explicit_complete_prompts():
    dataset = BenchmarkDataset.from_arrays(
        ["a0", "a1", "b0", "b1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    with pytest.raises(ValueError, match="exactly"):
        ZeroShotDataset.from_dataset(dataset, {"left": "left label"})
    with pytest.raises(ValueError, match="unique"):
        ZeroShotDataset.from_dataset(dataset, {"left": "same", "right": "same"})
    protocol = ZeroShotDataset.from_templates(
        dataset,
        ["a photo of {label}", "a close-up of {label}"],
    )
    prompts, labels, template_ids = protocol.prompt_rows()
    assert prompts[0] == "a photo of left"
    assert labels == ("left", "left", "right", "right")
    assert template_ids is not None


def test_zero_shot_protocol_and_evaluation_identities_hash_complete_content():
    dataset = BenchmarkDataset.from_arrays(
        ["a0", "a1", "b0", "b1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    left_prompts = [f"left prompt {index}" for index in range(101)]
    right_prompts = [f"right prompt {index}" for index in range(101)]
    first = ZeroShotDataset.from_dataset(dataset, {"left": left_prompts, "right": right_prompts})
    changed_prompts = list(left_prompts)
    changed_prompts[-1] = "left changed after the sampled-hash truncation boundary"
    second = ZeroShotDataset.from_dataset(
        dataset, {"left": changed_prompts, "right": right_prompts}
    )
    extractor = CallableRetrievalExtractor(
        "aligned", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    assert first.protocol_fingerprint() != second.protocol_fingerprint()
    assert zero_shot_embedding_artifact_key(first, extractor, "prompts", "gallery") != (
        zero_shot_embedding_artifact_key(second, extractor, "prompts", "gallery")
    )
    first_config = ZeroShotConfig(top_k=tuple(range(1, 102)))
    second_config = ZeroShotConfig(top_k=tuple(range(1, 101)) + (999,))
    assert zero_shot_scoring_artifact_key("samples", "prompts", "protocol", first_config) != (
        zero_shot_scoring_artifact_key("samples", "prompts", "protocol", second_config)
    )
    assert hash_json_exact({"values": list(range(101))}) != hash_json_exact(
        {"values": list(range(100)) + [999]}
    )
    first_compression = EmbeddingCompressionConfig(
        enabled=True,
        method="pca",
        n_components=1,
        algorithm_kwargs={"markers": list(range(101))},
    )
    second_compression = EmbeddingCompressionConfig(
        enabled=True,
        method="pca",
        n_components=1,
        algorithm_kwargs={"markers": list(range(100)) + [999]},
    )
    assert zero_shot_compression_artifact_key("samples", "prompts", first_compression) != (
        zero_shot_compression_artifact_key("samples", "prompts", second_compression)
    )


def test_exact_hash_preserves_mapping_key_types_and_rejects_unknown_objects():
    assert hash_json_exact({1: "integer"}) != hash_json_exact({"1": "string"})
    assert hash_json_exact({1: "integer", "1": "string"}) != hash_json_exact(
        {1: "string", "1": "integer"}
    )
    assert hash_json_exact({"a": 1, "b": 2}) == hash_json_exact({"b": 2, "a": 1})
    assert hash_json_exact(["a", "b"]) != hash_json_exact(("a", "b"))
    assert hash_json_exact({"a", "b"}) != hash_json_exact(["a", "b"])
    assert hash_json_exact(Path("artifact")) != hash_json_exact("artifact")
    with pytest.raises(TypeError, match="does not support"):
        hash_json_exact(object())
    int_labels = OverlapScoringConfig(k={1: 2, "1": 3})
    string_labels = OverlapScoringConfig(k={"1": 2, 1: 3})
    assert zero_shot_scoring_artifact_key(
        "samples", "prompts", "protocol", scoring_config=int_labels
    ) != zero_shot_scoring_artifact_key(
        "samples", "prompts", "protocol", scoring_config=string_labels
    )


def test_exact_hash_supports_stable_semantic_label_types():
    first_uuid = UUID("00112233-4455-6677-8899-aabbccddeeff")
    second_uuid = UUID("00112233-4455-6677-8899-aabbccddee00")
    assert hash_json_exact(first_uuid) != hash_json_exact(second_uuid)
    assert hash_json_exact(_SemanticLabel.LEFT) != hash_json_exact(_SemanticLabel.RIGHT)
    assert hash_json_exact(date(2026, 7, 12)) != hash_json_exact(time(12, 0))
    assert hash_json_exact(datetime(2026, 7, 12, 12, 0)) == hash_json_exact(
        datetime(2026, 7, 12, 12, 0)
    )
    assert hash_json_exact(Decimal("1.00")) == hash_json_exact(Decimal("1.0"))
    assert hash_json_exact(Fraction(1, 2)) != hash_json_exact(Fraction(2, 3))
    assert hash_json_exact(_FirstDataclassLabel("left")) != hash_json_exact(
        _SecondDataclassLabel("left")
    )
    assert semantic_label_key(1) != semantic_label_key("1")
    reserved = f"{LABEL_KEY_PREFIX}literal"
    assert semantic_label_key(reserved) != reserved


def test_zero_shot_supports_uuid_labels_through_protocol_artifacts_and_scoring(tmp_path):
    left = UUID("00112233-4455-6677-8899-aabbccddeeff")
    right = UUID("00112233-4455-6677-8899-aabbccddee00")
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        [left, left, right, right],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(
        dataset,
        ["a photo of {label}"],
        class_names={left: "left", right: "right"},
    )
    store = LocalArtifactStore(str(tmp_path))
    artifact = materialize_zero_shot_protocol(protocol, store, "uuid-protocol")
    assert artifact["protocol_fingerprint"] == protocol.protocol_fingerprint()
    assert (
        store.get_json("uuid-protocol")["protocol_fingerprint"] == protocol.protocol_fingerprint()
    )
    result = ZeroShotScorer().score(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        dataset.y,
        class_labels=[left, right],
        prompt_labels=[left, right],
    )
    assert result.score == pytest.approx(1.0)
    assert set(result.per_class) == {left, right}


def test_zero_shot_rejects_custom_labels_without_stable_exact_identity():
    left = _UnsupportedLabel("left")
    right = _UnsupportedLabel("right")
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        [left, left, right, right],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    with pytest.raises(ValueError, match="stable exact identity"):
        ZeroShotDataset.from_dataset(dataset, {left: "left", right: "right"})


def test_zero_shot_typed_label_collisions_survive_local_and_artifact_reports(
    tmp_path, fake_overlapindex
):
    uuid_label = UUID("00112233-4455-6677-8899-aabbccddeeff")
    string_label = str(uuid_label)
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        [uuid_label, uuid_label, string_label, string_label],
        modality="image",
        metadata={"sample_indices": [UUID(int=index + 1) for index in range(4)]},
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(
        dataset,
        ["{label}"],
        class_names={uuid_label: "left", string_label: "right"},
    )
    extractor = CallableRetrievalExtractor(
        "aligned", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    local = ZeroShotBenchmark(
        protocol, [extractor], sample_branch="query", text_branch="gallery"
    ).run()
    local_path = tmp_path / "local.json"
    local.save_json(str(local_path))
    local_payload = json.loads(local_path.read_text(encoding="utf-8"))
    local_per_class = local_payload["extractor_results"][0]["zero_shot"]["per_class"]
    assert len(local_per_class) == 2
    markdown = local_path.with_suffix(".md")
    local.save_markdown(str(markdown))
    rendered = markdown.read_text(encoding="utf-8")
    assert "[uuid]" in rendered
    assert "[str]" in rendered

    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    protocol_artifact = materialize_zero_shot_protocol(protocol, store, "protocol")
    assert len(set(protocol_artifact["class_labels"])) == 2
    _materialize_aligned_zero_shot_endpoints(protocol, store)
    score = score_zero_shot_artifact(
        ZeroShotScoringJob(
            sample_embedding_key="samples",
            prompt_embedding_key="prompts",
            protocol_key="protocol",
            output_key="score",
        ),
        store,
    )
    assert len(score["zero_shot"]["per_class"]) == 2
    reconstructed = zero_shot_benchmark_result_from_artifacts(["score"], store)
    assert len(reconstructed.extractor_results[0].zero_shot.per_class) == 2


def test_typed_protocol_recipe_and_score_result_are_strict_json():
    labels = [
        UUID("00112233-4455-6677-8899-aabbccddeeff"),
        _SemanticLabel.LEFT,
        datetime(2026, 7, 12, 12, 0),
        date(2026, 7, 12),
        time(12, 0),
        Decimal("1.25"),
        Fraction(2, 3),
        b"bytes",
        Path("label"),
        _FirstDataclassLabel("value"),
    ]
    embeddings = np.eye(len(labels))
    result = ZeroShotScorer(ZeroShotConfig(worst_samples=len(labels))).score(
        embeddings,
        embeddings,
        labels,
        class_labels=labels,
        prompt_labels=labels,
        sample_ids=[UUID(int=index + 1) for index in range(len(labels))],
    )
    json.dumps(result.to_dict(), allow_nan=False)

    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(
        dataset,
        ["{label}"],
        metadata={
            "typed": {UUID(int=1): b"one", "tuple": (1, 2), "set": {"b", "a"}},
        },
    )
    recipe = protocol.protocol_recipe()
    json.dumps(recipe, allow_nan=False)
    assert recipe == protocol.protocol_recipe()


@pytest.mark.parametrize(
    "field",
    [
        "prompts",
        "labels",
        "label_catalog",
        "template_ids",
        "n_samples",
        "fingerprint",
        "recipe",
    ],
)
def test_zero_shot_protocol_artifact_tampering_is_rejected(tmp_path, fake_overlapindex, field):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    store = LocalArtifactStore(str(tmp_path))
    payload = materialize_zero_shot_protocol(protocol, store, "protocol")
    _materialize_aligned_zero_shot_endpoints(protocol, store)
    changed = deepcopy(payload)
    if field == "prompts":
        changed["prompts"][0] = "tampered"
    elif field == "labels":
        changed["labels"][0] = changed["class_labels"][1]
    elif field == "label_catalog":
        changed["label_catalog"][0]["display"] = "tampered"
    elif field == "template_ids":
        changed["template_ids"][0] = "tampered"
    elif field == "n_samples":
        changed["n_samples"] += 1
    elif field == "fingerprint":
        changed["protocol_fingerprint"] = "tampered"
    else:
        changed["protocol"]["prompts"][0] = "tampered"
    store.put_json("protocol", changed)
    with pytest.raises(ValueError, match="protocol"):
        score_zero_shot_artifact(
            ZeroShotScoringJob(
                sample_embedding_key="samples",
                prompt_embedding_key="prompts",
                protocol_key="protocol",
                output_key="score",
            ),
            store,
        )


@pytest.mark.parametrize("encoding", [None, "vertebrae.semantic-label/v0"])
def test_zero_shot_protocol_requires_current_label_encoding(tmp_path, fake_overlapindex, encoding):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    store = LocalArtifactStore(str(tmp_path))
    payload = materialize_zero_shot_protocol(protocol, store, "protocol")
    if encoding is None:
        payload.pop("label_encoding")
    else:
        payload["label_encoding"] = encoding
    store.put_json("protocol", payload)
    _materialize_aligned_zero_shot_endpoints(protocol, store)
    with pytest.raises(ValueError, match="must use label encoding"):
        score_zero_shot_artifact(
            ZeroShotScoringJob(
                sample_embedding_key="samples",
                prompt_embedding_key="prompts",
                protocol_key="protocol",
                output_key="invalid-score",
            ),
            store,
        )


def test_zero_shot_scorer_rejects_unsupported_direct_labels():
    labels = [_UnsupportedLabel("left"), _UnsupportedLabel("right")]
    with pytest.raises(ValueError, match="stable semantic identities"):
        ZeroShotScorer().score(
            np.eye(2),
            np.eye(2),
            labels,
            class_labels=labels,
            prompt_labels=labels,
        )


def test_zero_shot_scorer_reports_metrics_ensembles_and_ties():
    scorer = ZeroShotScorer(ZeroShotConfig(top_k=(1, 2)))
    result = scorer.score(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        np.asarray([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]]),
        ["left", "left", "right", "right"],
        class_labels=["left", "right"],
        prompt_labels=["left", "left", "right", "right"],
        template_ids=["a {label}", "b {label}", "a {label}", "b {label}"],
    )
    assert result.score == pytest.approx(1.0)
    assert result.metrics["macro_f1"] == pytest.approx(1.0)
    assert result.metrics["top_k_accuracy@2"] == pytest.approx(1.0)
    assert "per_template_metrics" in result.diagnostics
    tied = scorer.score(
        np.asarray([[1.0, 1.0], [1.0, -1.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ["left", "right"],
        class_labels=["left", "right"],
        prompt_labels=["left", "right"],
    )
    assert tied.diagnostics["n_top_score_ties"] == 1
    assert any("tie" in warning for warning in tied.warnings)


def test_zero_shot_squared_l2_ranks_nearest_prototypes_across_batches():
    samples = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]], dtype=np.float64)
    prompts = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    labels = ["left", "left", "right", "right"]
    oracle_scores = -np.sum((samples[:, None, :] - prompts[None, :, :]) ** 2, axis=2)
    assert np.argmax(oracle_scores, axis=1).tolist() == [0, 0, 1, 1]

    result = ZeroShotScorer(
        ZeroShotConfig(
            similarity="squared_l2",
            top_k=(1, 2),
            sample_batch_size=1,
        )
    ).score(
        samples,
        prompts,
        labels,
        class_labels=["left", "right"],
        prompt_labels=["left", "right"],
    )

    assert result.score == pytest.approx(1.0)
    assert result.metrics["top_k_accuracy@1"] == pytest.approx(1.0)
    assert result.metrics["top_k_accuracy@2"] == pytest.approx(1.0)
    assert result.confusion_matrix == [[2, 0], [0, 2]]
    assert result.diagnostics["correct_class_margin"]["min"] > 0.0


@pytest.mark.parametrize("similarity", ["cosine", "dot", "squared_l2"])
def test_zero_shot_blockwise_scoring_matches_single_block(similarity):
    samples = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.8, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    prompts = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
        ]
    )
    labels = ["a", "a", "b", "b", "c", "c"]
    prompt_labels = ["a", "a", "b", "b", "c", "c"]
    template_ids = ["first", "second", "first", "second", "first", "second"]
    kwargs = {
        "class_labels": ["a", "b", "c"],
        "prompt_labels": prompt_labels,
        "template_ids": template_ids,
        "sample_ids": [f"sample-{index}" for index in range(len(samples))],
    }
    single = ZeroShotScorer(
        ZeroShotConfig(
            similarity=similarity,
            top_k=(1, 2, 3),
            sample_batch_size=len(samples),
            worst_samples=len(samples),
        )
    ).score(samples, prompts, labels, **kwargs)
    blocked = ZeroShotScorer(
        ZeroShotConfig(
            similarity=similarity,
            top_k=(1, 2, 3),
            sample_batch_size=2,
            worst_samples=len(samples),
        )
    ).score(samples, prompts, labels, **kwargs)

    assert blocked.score == pytest.approx(single.score)
    assert blocked.metrics == pytest.approx(single.metrics)
    assert blocked.per_class == single.per_class
    assert blocked.confusion_matrix == single.confusion_matrix
    assert blocked.diagnostics == single.diagnostics
    assert blocked.warnings == single.warnings


def test_zero_shot_memory_budget_allows_block_but_rejects_full_matrix():
    samples = np.tile(np.eye(4), (5, 1))
    prompts = np.eye(4)
    labels = [index % 4 for index in range(len(samples))]
    kwargs = {
        "class_labels": [0, 1, 2, 3],
        "prompt_labels": [0, 1, 2, 3],
    }

    blocked = ZeroShotScorer(ZeroShotConfig(sample_batch_size=2, max_dense_bytes=2_500)).score(
        samples, prompts, labels, **kwargs
    )
    assert blocked.metrics["accuracy"] == pytest.approx(1.0)

    with pytest.raises(MemoryError, match="sample_batch_size"):
        ZeroShotScorer(ZeroShotConfig(sample_batch_size=len(samples), max_dense_bytes=2_500)).score(
            samples, prompts, labels, **kwargs
        )


@pytest.mark.parametrize("as_sparse", [False, True])
def test_zero_shot_memory_budget_guards_dense_and_sparse_inputs(as_sparse):
    samples = np.eye(2)
    prompts = np.eye(2)
    if as_sparse:
        samples = sparse.csr_matrix(samples)
        prompts = sparse.csc_matrix(prompts)

    with pytest.raises(MemoryError, match="ZeroShotConfig.max_dense_bytes"):
        ZeroShotScorer(ZeroShotConfig(max_dense_bytes=1)).score(
            samples,
            prompts,
            ["left", "right"],
            class_labels=["left", "right"],
            prompt_labels=["left", "right"],
        )


def test_zero_shot_config_rejects_invalid_sample_batch_size():
    with pytest.raises(ValueError, match="sample_batch_size"):
        ZeroShotConfig(sample_batch_size=0)


def test_zero_shot_benchmark_caches_compresses_and_reports_overlap(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])

    def query(values):
        return np.asarray(
            [
                [1.0, 0.0, 0.2] if str(value).startswith("left") else [0.0, 1.0, 0.2]
                for value in values
            ]
        )

    def gallery(values):
        return np.asarray(
            [[1.0, 0.0, 0.2] if value == "left" else [0.0, 1.0, 0.2] for value in values]
        )

    extractor = CallableRetrievalExtractor(
        "aligned",
        query_fn=query,
        gallery_fn=gallery,
        query_modality="image",
        gallery_modality="text",
        cache_identity="aligned-v1",
    )
    kwargs = {
        "sample_branch": "query",
        "text_branch": "gallery",
        "cache_config": CacheConfig(cache_dir=str(tmp_path)),
        "embedding_config": EmbeddingConfig(batch_size=2),
        "resource_profiling_config": ResourceProfilingConfig(enabled=True),
        "compression_configs": [
            EmbeddingCompressionConfig(),
            EmbeddingCompressionConfig(
                enabled=True,
                method="prefix_truncate",
                n_components=2,
                assume_matryoshka=True,
                dtype="float32",
            ),
        ],
    }
    first = ZeroShotBenchmark(protocol, [extractor], **kwargs).run()
    second = ZeroShotBenchmark(protocol, [extractor], **kwargs).run()
    assert len(first.extractor_results) == 2
    assert first.ranked_results()[0].zero_shot.metrics["accuracy"] == pytest.approx(1.0)
    compressed_variant = next(
        item for item in first.extractor_results if item.compression_metadata["applied"]
    )
    assert compressed_variant.compression_metadata["dtype"] == "float32"
    assert first.extractor_results[0].overlap.kind == "overlap_index"
    assert all(item.cache_metadata["samples"]["hit"] for item in second.extractor_results)
    cached_profiles = second.extractor_results[0].resource_profiles
    assert cached_profiles["samples"].status == "not_measured_cache_hit"
    assert cached_profiles["prompts"].status == "not_measured_cache_hit"
    assert cached_profiles["samples"].embedding.raw_persisted.status == "measured"
    assert "samples_resource_profile_scope" in second.to_dataframe().columns
    assert "sample_cache_hit" not in second.extractor_results[0].recipe
    report = tmp_path / "zero-shot.md"
    payload = tmp_path / "zero-shot.json"
    first.save_markdown(str(report))
    first.save_json(str(payload))
    assert "semantic text alignment" in report.read_text(encoding="utf-8")
    assert json.loads(payload.read_text(encoding="utf-8"))["extractor_results"]


def test_zero_shot_rejects_multilabel_and_zero_embeddings():
    dataset = BenchmarkDataset.from_arrays(
        ["a", "b", "c", "d"],
        [("x",), ("x",), ("y",), ("y",)],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    with pytest.raises(ValueError, match="single-label"):
        ZeroShotDataset.from_dataset(dataset, {"x": "x", "y": "y"})
    with pytest.raises(ValueError, match="zero-norm"):
        ZeroShotScorer().score(
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            ["x", "y"],
            class_labels=["x", "y"],
            prompt_labels=["x", "y"],
        )
    with pytest.raises(ValueError, match="at least two"):
        ZeroShotScorer().score(
            np.ones((2, 2)),
            np.ones((1, 2)),
            ["x", "x"],
            class_labels=["x"],
            prompt_labels=["x"],
        )


def test_zero_shot_artifact_round_trip(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    extractor = CallableRetrievalExtractor(
        "aligned",
        query_fn=_cli_query,
        gallery_fn=_cli_gallery,
        query_modality="image",
        gallery_modality="text",
    )
    store = LocalArtifactStore(str(tmp_path))
    protocol_artifact = materialize_zero_shot_protocol(protocol, store, "protocol")
    keys = {}
    for side, branch, count in (("samples", "query", 4), ("prompts", "gallery", 2)):
        shards = []
        for index in range(2):
            key = f"{side}/shard/{index}"
            materialize_zero_shot_embedding_shard(
                ZeroShotEmbeddingShardJob(
                    dataset=protocol,
                    extractor=extractor,
                    side=side,
                    branch=branch,
                    shard=ShardSpec(total_shards=2, shard_index=index),
                    output_key=key,
                ),
                store,
            )
            shards.append(key)
        output = side
        merge_zero_shot_embedding_shards(
            EmbeddingMergeJob(tuple(shards), output, n_samples=count), store
        )
        keys[side] = output
    compressed = compress_zero_shot_embedding_artifacts(
        ZeroShotCompressionJob(
            sample_embedding_key=keys["samples"],
            prompt_embedding_key=keys["prompts"],
            sample_output_key="compressed/samples",
            prompt_output_key="compressed/prompts",
            compression_config=EmbeddingCompressionConfig(
                enabled=True,
                method="quantize",
                precision="float16",
            ),
        ),
        store,
    )
    assert compressed["compression_metadata"]["fit_side"] == "samples"
    assert store.get_json(compressed["output_key"])["sample_output_key"] == "compressed/samples"
    compressed_samples = store.get_json("compressed/samples")
    compressed_prompts = store.get_json("compressed/prompts")
    assert compressed_samples["compression_pair_id"] == compressed_prompts["compression_pair_id"]
    assert compressed_samples["dtype"] == "float16"
    artifact = score_zero_shot_artifact(
        ZeroShotScoringJob(
            sample_embedding_key="compressed/samples",
            prompt_embedding_key="compressed/prompts",
            protocol_key=protocol_artifact["output_key"],
            output_key="score",
        ),
        store,
    )
    assert artifact["artifact_type"] == "zero_shot_evaluation"
    assert artifact["zero_shot"]["metrics"]["accuracy"] == pytest.approx(1.0)
    reconstructed = zero_shot_benchmark_result_from_artifacts(["score"], store, "report")
    assert reconstructed.extractor_results[0].zero_shot.score == pytest.approx(1.0)
    assert reconstructed.dataset_summary == protocol_artifact["dataset_summary"]
    assert store.get_json("report")["extractor_results"]


def test_zero_shot_squared_l2_artifact_scoring_ranks_nearest_prototypes(
    tmp_path, fake_overlapindex
):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    store = LocalArtifactStore(str(tmp_path))
    protocol_key = materialize_zero_shot_protocol(protocol, store, "protocol")["output_key"]
    _materialize_aligned_zero_shot_endpoints(protocol, store)

    artifact = score_zero_shot_artifact(
        ZeroShotScoringJob(
            sample_embedding_key="samples",
            prompt_embedding_key="prompts",
            protocol_key=protocol_key,
            output_key="score",
            zero_shot_config=ZeroShotConfig(
                similarity="squared_l2",
                top_k=(1, 2),
                sample_batch_size=1,
            ),
        ),
        store,
    )

    assert artifact["zero_shot"]["metrics"]["accuracy"] == pytest.approx(1.0)
    assert artifact["zero_shot"]["confusion_matrix"] == [[2, 0], [0, 2]]
    assert artifact["zero_shot"]["diagnostics"]["correct_class_margin"]["min"] > 0.0


def test_zero_shot_cli_round_trip(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    extractor = CallableRetrievalExtractor(
        "aligned",
        query_fn=_cli_query,
        gallery_fn=_cli_gallery,
        query_modality="image",
        gallery_modality="text",
    )
    dataset_path = tmp_path / "protocol.pkl"
    extractor_path = tmp_path / "extractor.pkl"
    with dataset_path.open("wb") as stream:
        pickle.dump(protocol, stream)
    with extractor_path.open("wb") as stream:
        pickle.dump(extractor, stream)
    plan_path = tmp_path / "plan.json"
    cache_dir = tmp_path / "cache"
    common = ["--cache-dir", str(cache_dir)]
    assert (
        main(
            [
                "plan-zero-shot",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--total-shards",
                "1",
                "--sample-branch",
                "query",
                "--text-branch",
                "gallery",
                "--output-json",
                str(plan_path),
                *common,
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for side in ("samples", "prompts"):
        assert (
            main(
                [
                    "embed-zero-shot-shard",
                    "--dataset-pickle",
                    str(dataset_path),
                    "--extractor-pickle",
                    str(extractor_path),
                    "--side",
                    side,
                    "--branch",
                    plan["endpoints"][side]["branch"],
                    "--total-shards",
                    "1",
                    "--shard-index",
                    "0",
                    *common,
                ]
            )
            == 0
        )
        assert (
            main(
                [
                    "merge-zero-shot-embeddings",
                    "--plan-json",
                    str(plan_path),
                    "--side",
                    side,
                    *common,
                ]
            )
            == 0
        )
    protocol_path = tmp_path / "protocol.json"
    assert (
        main(
            [
                "write-zero-shot-protocol",
                "--dataset-pickle",
                str(dataset_path),
                "--output-json",
                str(protocol_path),
                *common,
            ]
        )
        == 0
    )
    protocol_key = json.loads(protocol_path.read_text(encoding="utf-8"))["output_key"]
    score_path = tmp_path / "score.json"
    assert (
        main(
            [
                "score-zero-shot",
                "--sample-embedding-key",
                plan["endpoints"]["samples"]["output_key"],
                "--prompt-embedding-key",
                plan["endpoints"]["prompts"]["output_key"],
                "--protocol-key",
                protocol_key,
                "--output-json",
                str(score_path),
                *common,
            ]
        )
        == 0
    )
    score = json.loads(score_path.read_text(encoding="utf-8"))
    assert score["zero_shot"]["metrics"]["accuracy"] == 1.0
    reconstructed_json = tmp_path / "reconstructed.json"
    reconstructed_markdown = tmp_path / "reconstructed.md"
    assert (
        main(
            [
                "zero-shot-from-artifacts",
                "--score-key",
                score["output_key"],
                "--json-output",
                str(reconstructed_json),
                "--markdown-output",
                str(reconstructed_markdown),
                *common,
            ]
        )
        == 0
    )
    assert json.loads(reconstructed_json.read_text(encoding="utf-8"))["extractor_results"]
    assert "semantic text alignment" in reconstructed_markdown.read_text(encoding="utf-8")


def test_zero_shot_planning_caps_endpoint_shards_and_cli_uses_plan(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    extractor = CallableRetrievalExtractor(
        "aligned", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    assert (
        len(
            plan_zero_shot_embedding_shard_jobs(
                protocol, extractor, 4, side="samples", branch="query"
            )
        )
        == 4
    )
    prompts = plan_zero_shot_embedding_shard_jobs(
        protocol, extractor, 4, side="prompts", branch="gallery"
    )
    assert [job.shard.total_shards for job in prompts] == [2, 2]
    dataset_path, extractor_path = tmp_path / "dataset.pkl", tmp_path / "extractor.pkl"
    dataset_path.write_bytes(pickle.dumps(protocol))
    extractor_path.write_bytes(pickle.dumps(extractor))
    plan_path, cache_dir = tmp_path / "plan.json", tmp_path / "cache"
    assert (
        main(
            [
                "plan-zero-shot",
                "--dataset-pickle",
                str(dataset_path),
                "--extractor-pickle",
                str(extractor_path),
                "--total-shards",
                "4",
                "--sample-branch",
                "query",
                "--text-branch",
                "gallery",
                "--output-json",
                str(plan_path),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text())
    for entry in plan["endpoints"]["prompts"]["shards"]:
        assert (
            main(
                [
                    "embed-zero-shot-shard",
                    "--dataset-pickle",
                    str(dataset_path),
                    "--extractor-pickle",
                    str(extractor_path),
                    "--side",
                    "prompts",
                    "--plan-json",
                    str(plan_path),
                    "--shard-index",
                    str(entry["shard"]["shard_index"]),
                    "--cache-dir",
                    str(cache_dir),
                ]
            )
            == 0
        )


def test_zero_shot_callable_cache_identity_and_candidate_branches(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    first = CallableRetrievalExtractor(
        "same", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    second = CallableRetrievalExtractor(
        "same",
        _alternate_query,
        _alternate_gallery,
        query_modality="image",
        gallery_modality="text",
    )
    assert first.recipe()["cache_safe"] and second.recipe()["cache_safe"]
    assert first.recipe()["query_callable"] != second.recipe()["query_callable"]
    first_key = zero_shot_embedding_artifact_key(protocol, first, "samples", "query")
    second_key = zero_shot_embedding_artifact_key(protocol, second, "samples", "query")
    assert first_key != second_key
    unsafe = CallableRetrievalExtractor(
        "unsafe",
        lambda values: _cli_query(values),
        lambda values: _cli_gallery(values),
        query_modality="image",
        gallery_modality="text",
    )
    run = ZeroShotBenchmark(
        protocol,
        [unsafe],
        sample_branch="query",
        text_branch="gallery",
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
    ).run()
    assert not run.extractor_results[0].cache_metadata["samples"]["enabled"]
    assert any(
        "Skipped zero-shot embedding cache" in warning
        for warning in run.extractor_results[0].warnings
    )
    with pytest.raises(ValueError, match="cache_identity"):
        zero_shot_embedding_artifact_key(protocol, unsafe, "samples", "query")
    restored = CallableRetrievalExtractor(
        "restored",
        lambda values: _cli_query(values),
        lambda values: _cli_gallery(values),
        query_modality="image",
        gallery_modality="text",
        cache_identity="restored-v1",
    )
    assert restored.recipe()["cache_safe"]
    first_run = ZeroShotBenchmark(
        protocol,
        [restored],
        sample_branch="query",
        text_branch="gallery",
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
    ).run()
    second_run = ZeroShotBenchmark(
        protocol,
        [restored],
        sample_branch="query",
        text_branch="gallery",
        cache_config=CacheConfig(cache_dir=str(tmp_path)),
    ).run()
    assert not first_run.extractor_results[0].cache_metadata["samples"]["hit"]
    assert second_run.extractor_results[0].cache_metadata["samples"]["hit"]
    heterogeneous = ZeroShotBenchmark(
        protocol,
        [
            ZeroShotCandidate(first, "query", "gallery"),
            ZeroShotCandidate(_HeterogeneousRetrievalExtractor(), "vision", "language"),
        ],
    ).run()
    assert {
        item.embedding_metadata["sample_branch"] for item in heterogeneous.extractor_results
    } == {"query", "vision"}


def test_zero_shot_protocol_provenance_variants_and_sample_ids(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        metadata={"sample_indices": [10, 11, 20, 21]},
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["look at {label}"])
    extractor = CallableRetrievalExtractor(
        "aligned", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    result = ZeroShotBenchmark(
        protocol,
        [extractor],
        sample_branch="query",
        text_branch="gallery",
        compression_configs=[
            EmbeddingCompressionConfig(enabled=True, method="pca", n_components=1),
            EmbeddingCompressionConfig(enabled=True, method="pca", n_components=2),
            EmbeddingCompressionConfig(enabled=True, method="pca", n_components=3),
            EmbeddingCompressionConfig(enabled=True, method="quantize", precision="float16"),
            EmbeddingCompressionConfig(enabled=True, method="quantize", precision="int8"),
        ],
    ).run()
    assert {item.name for item in result.extractor_results} == {
        "aligned[pca_1]",
        "aligned[pca_2]",
        "aligned[pca_3]",
        "aligned[quantize_float16]",
        "aligned[quantize_int8]",
    }
    metadata_by_name = {item.name: item.compression_metadata for item in result.extractor_results}
    assert metadata_by_name["aligned[pca_2]"]["compressed_dim"] == 2
    assert metadata_by_name["aligned[pca_3]"]["compressed_dim"] == 2
    assert not metadata_by_name["aligned[pca_2]"]["applied"]
    assert not metadata_by_name["aligned[pca_3]"]["applied"]
    sample_id = result.extractor_results[0].zero_shot.diagnostics["worst_samples"][0]["sample_id"]
    assert sample_id in {10, 11, 20, 21}
    json_path, markdown_path = tmp_path / "result.json", tmp_path / "result.md"
    result.save_json(str(json_path))
    result.save_markdown(str(markdown_path))
    assert "look at left" in json_path.read_text()
    assert "look at left" not in markdown_path.read_text()
    with pytest.raises(ValueError, match="one-dimensional"):
        ZeroShotScorer().score(
            np.ones((2, 2)),
            np.ones((2, 2)),
            [["left"], ["right"]],
            class_labels=["left", "right"],
            prompt_labels=["left", "right"],
        )


def test_zero_shot_sample_artifacts_are_reusable_across_prompt_protocols(tmp_path):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    first = ZeroShotDataset.from_templates(dataset, ["a {label}"])
    second = ZeroShotDataset.from_templates(dataset, ["a photo of {label}"])
    extractor = CallableRetrievalExtractor(
        "aligned", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    assert zero_shot_embedding_artifact_key(first, extractor, "samples", "query") == (
        zero_shot_embedding_artifact_key(second, extractor, "samples", "query")
    )
    assert zero_shot_embedding_artifact_key(first, extractor, "prompts", "gallery") != (
        zero_shot_embedding_artifact_key(second, extractor, "prompts", "gallery")
    )
    store = LocalArtifactStore(str(tmp_path))
    manifest = materialize_zero_shot_embedding_shard(
        ZeroShotEmbeddingShardJob(
            dataset=first,
            extractor=extractor,
            side="samples",
            branch="query",
            shard=ShardSpec(),
            output_key="samples",
        ),
        store,
    )
    assert manifest["protocol_fingerprint"] is None


def test_zero_shot_rejects_mismatched_compression_pairs(tmp_path, fake_overlapindex):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    extractor = CallableRetrievalExtractor(
        "aligned", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    store = LocalArtifactStore(str(tmp_path))
    protocol_key = materialize_zero_shot_protocol(protocol, store, "protocol")["output_key"]
    endpoint_keys = {}
    for side, branch, count in (("samples", "query", 4), ("prompts", "gallery", 2)):
        shard_key = f"{side}/shard"
        materialize_zero_shot_embedding_shard(
            ZeroShotEmbeddingShardJob(
                dataset=protocol,
                extractor=extractor,
                side=side,
                branch=branch,
                shard=ShardSpec(),
                output_key=shard_key,
            ),
            store,
        )
        merge_zero_shot_embedding_shards(
            EmbeddingMergeJob((shard_key,), side, n_samples=count), store
        )
        endpoint_keys[side] = side
    config = EmbeddingCompressionConfig(enabled=True, method="quantize", precision="float16")
    compress_zero_shot_embedding_artifacts(
        ZeroShotCompressionJob(
            sample_embedding_key="samples",
            prompt_embedding_key="prompts",
            sample_output_key="compressed-a/samples",
            prompt_output_key="compressed-a/prompts",
            compression_config=config,
        ),
        store,
    )
    compress_zero_shot_embedding_artifacts(
        ZeroShotCompressionJob(
            sample_embedding_key="samples",
            prompt_embedding_key="prompts",
            sample_output_key="compressed-b/samples",
            prompt_output_key="compressed-b/prompts",
            compression_config=config,
        ),
        store,
    )
    with pytest.raises(ValueError, match="both be raw"):
        score_zero_shot_artifact(
            ZeroShotScoringJob(
                sample_embedding_key="samples",
                prompt_embedding_key="compressed-a/prompts",
                protocol_key=protocol_key,
                output_key="invalid-raw-compressed",
            ),
            store,
        )
    with pytest.raises(ValueError, match="compression_pair_id"):
        score_zero_shot_artifact(
            ZeroShotScoringJob(
                sample_embedding_key="compressed-a/samples",
                prompt_embedding_key="compressed-b/prompts",
                protocol_key=protocol_key,
                output_key="invalid-mixed-pairs",
            ),
            store,
        )


def test_zero_shot_score_keys_and_reconstruction_require_one_evaluation_recipe(
    tmp_path, fake_overlapindex
):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    extractor = CallableRetrievalExtractor(
        "aligned", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    default_key = zero_shot_scoring_artifact_key("samples", "prompts", "protocol")
    macro_config = ZeroShotConfig(primary_metric="macro_f1")
    batch_config = ZeroShotConfig(sample_batch_size=1)
    assert default_key != zero_shot_scoring_artifact_key(
        "samples", "prompts", "protocol", macro_config, OverlapScoringConfig()
    )
    assert default_key != zero_shot_scoring_artifact_key(
        "samples", "prompts", "protocol", batch_config, OverlapScoringConfig()
    )
    store = LocalArtifactStore(str(tmp_path))
    protocol_key = materialize_zero_shot_protocol(protocol, store, "protocol")["output_key"]
    for side, branch, count in (("samples", "query", 4), ("prompts", "gallery", 2)):
        materialize_zero_shot_embedding_shard(
            ZeroShotEmbeddingShardJob(
                dataset=protocol,
                extractor=extractor,
                side=side,
                branch=branch,
                shard=ShardSpec(),
                output_key=f"{side}/shard",
            ),
            store,
        )
        merge_zero_shot_embedding_shards(
            EmbeddingMergeJob((f"{side}/shard",), side, n_samples=count), store
        )
    for output_key, config in (("score-accuracy", ZeroShotConfig()), ("score-macro", macro_config)):
        score_zero_shot_artifact(
            ZeroShotScoringJob(
                sample_embedding_key="samples",
                prompt_embedding_key="prompts",
                protocol_key=protocol_key,
                output_key=output_key,
                zero_shot_config=config,
            ),
            store,
        )
    batch_artifact = score_zero_shot_artifact(
        ZeroShotScoringJob(
            sample_embedding_key="samples",
            prompt_embedding_key="prompts",
            protocol_key=protocol_key,
            output_key="score-batch",
            zero_shot_config=batch_config,
        ),
        store,
    )
    assert batch_artifact["evaluation_recipe"]["zero_shot_config"]["sample_batch_size"] == 1
    assert batch_artifact["zero_shot"]["metadata"]["config"]["sample_batch_size"] == 1
    with pytest.raises(ValueError, match="evaluation configuration"):
        zero_shot_benchmark_result_from_artifacts(["score-accuracy", "score-macro"], store)


def test_zero_shot_local_compression_skip_and_callable_portability(
    tmp_path, fake_overlapindex, monkeypatch
):
    dataset = BenchmarkDataset.from_arrays(
        ["left-0", "left-1", "right-0", "right-1"],
        ["left", "left", "right", "right"],
        modality="image",
        identity=DatasetIdentity.ephemeral(),
    )
    protocol = ZeroShotDataset.from_templates(dataset, ["{label}"])
    extractor = CallableRetrievalExtractor(
        "aligned", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    result = ZeroShotBenchmark(
        protocol,
        [extractor],
        sample_branch="query",
        text_branch="gallery",
        compression_config=EmbeddingCompressionConfig(enabled=True, method="pca", n_components=3),
    ).run()
    assert not result.extractor_results[0].compression_metadata["applied"]
    assert any(
        "skipping compression" in warning for warning in result.extractor_results[0].warnings
    )
    monkeypatch.setattr(_cli_query, "__module__", "__main__")
    main_callable = CallableRetrievalExtractor(
        "main", _cli_query, _cli_gallery, query_modality="image", gallery_modality="text"
    )
    assert not main_callable.recipe()["cache_safe"]
    restored = CallableRetrievalExtractor(
        "identified",
        _cli_query,
        _cli_gallery,
        query_modality="image",
        gallery_modality="text",
        cache_identity="identified-v1",
    )
    assert restored.recipe()["cache_safe"]
