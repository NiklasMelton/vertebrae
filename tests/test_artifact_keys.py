import numpy as np
import pytest
from scipy import sparse

from vertebrae import embedding_output_key, embedding_output_shard_key
from vertebrae.cache.fingerprint import hash_json
from vertebrae.cache.keys import (
    named_output_artifact_key,
    named_output_artifact_keys,
    named_output_key_segment,
    validate_artifact_key,
)


def test_named_output_segment_has_readable_slug_and_exact_digest():
    segment = named_output_key_segment("Résumé / Final!")

    assert segment == (
        "output-v1-resume-final--"
        "55541d6becdfa9437b8890f7aec29d1ee9a4410d18aa463b1ffd5889072a5376"
    )


def test_named_output_segment_bounds_slug_and_falls_back_for_non_ascii_names():
    long_segment = named_output_key_segment("A" * 80)
    non_ascii_segment = named_output_key_segment("東京")

    assert long_segment.startswith(f"output-v1-{'a' * 40}--")
    assert non_ascii_segment.startswith("output-v1-output--")


def test_named_output_keys_preserve_exact_name_identity_despite_slug_collisions():
    keys = named_output_artifact_keys("embeddings/demo", ["a/b", "a_b", "Final", "final"])

    assert keys["a/b"] != keys["a_b"]
    assert keys["Final"] != keys["final"]
    assert "/outputs/output-v1-a-b--" in keys["a/b"]
    assert "/outputs/output-v1-a-b--" in keys["a_b"]
    assert named_output_artifact_key("embeddings/demo", "a/b") == keys["a/b"]
    assert embedding_output_key("embeddings/demo", "a/b") == keys["a/b"]
    assert embedding_output_shard_key("embeddings/demo/shards/00000", "a/b").startswith(
        "embeddings/demo/shards/00000/outputs/output-v1-a-b--"
    )


@pytest.mark.parametrize("output_name", ["", 1, None])
def test_named_output_segment_rejects_invalid_names(output_name):
    with pytest.raises(ValueError, match="Output name"):
        named_output_key_segment(output_name)


def test_named_output_collection_rejects_duplicate_names_before_generation():
    with pytest.raises(ValueError, match="unique"):
        named_output_artifact_keys("embeddings/demo", ["same", "same"])


@pytest.mark.parametrize(
    "key",
    [
        "",
        "/absolute",
        "trailing/",
        "double//separator",
        ".",
        "..",
        "parent/../escape",
        "current/./child",
        "windows\\separator",
        "control\x00character",
        "control\ncharacter",
    ],
)
def test_artifact_key_validation_rejects_noncanonical_or_unsafe_keys(key):
    with pytest.raises(ValueError, match="Artifact key"):
        validate_artifact_key(key)


def test_artifact_key_validation_is_lossless_for_benign_double_dots():
    assert validate_artifact_key("runs/a..b/result") == "runs/a..b/result"
    assert validate_artifact_key("runs/a__b/result") == "runs/a__b/result"


def test_hash_json_uses_complete_typed_content_without_sampling():
    first = list(range(101)) + ["left"]
    second = list(range(101)) + ["right"]

    assert hash_json(first) != hash_json(second)
    assert hash_json({1: "value"}) != hash_json({"1": "value"})
    assert hash_json([1, 2]) != hash_json((1, 2))


def test_hash_json_hashes_complete_dense_and_sparse_values():
    dense_a = np.zeros(1_000, dtype=np.float32)
    dense_b = dense_a.copy()
    dense_b[777] = 1
    sparse_a = sparse.csr_matrix(dense_a.reshape(10, 100))
    sparse_b = sparse.csr_matrix(dense_b.reshape(10, 100))

    assert hash_json(dense_a) != hash_json(dense_b)
    assert hash_json(sparse_a) != hash_json(sparse_b)


def test_hash_json_rejects_opaque_repr_fallbacks():
    with pytest.raises(TypeError, match="does not support"):
        hash_json(object())
