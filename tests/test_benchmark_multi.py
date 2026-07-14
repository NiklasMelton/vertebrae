import numpy as np
import pytest

from vertebrae import Benchmark, BenchmarkDataset, DatasetIdentity
from vertebrae.cache import LocalArtifactStore
from vertebrae.config import (
    CacheConfig,
    LabelViewConfig,
    SeparatixConfig,
    StabilityConfig,
    TargetViewConfig,
)
from vertebrae.datasets import TargetView
from vertebrae.extractors import CallableExtractor, MultiOutputExtractor
from vertebrae.extractors.base import EmbeddingOutputSpec

COARSE_TARGETS = [
    "mammal",
    "mammal",
    "mammal",
    "mammal",
    "avian",
    "avian",
    "mammal",
    "mammal",
]


def test_multi_extractor_benchmark(fake_overlapindex):
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.array(["a"] * 10 + ["b"] * 10)
    dataset = BenchmarkDataset.from_arrays(
        X, y, modality="tabular", identity=DatasetIdentity.ephemeral()
    )

    benchmark = Benchmark(
        dataset,
        stability_config=StabilityConfig(repeats=2),
        cache_config=CacheConfig(enabled=False),
    )
    benchmark.add_extractor(CallableExtractor("identity", lambda value: value, modality="tabular"))
    benchmark.add_extractor(
        CallableExtractor("scaled", lambda value: value * 2, modality="tabular")
    )

    result = benchmark.run()

    assert len(result.extractor_results) == 2
    assert list(result.to_dataframe()["rank"]) == [1, 2]
    assert len(fake_overlapindex.calls) == 6


def test_hierarchy_label_views_produce_separate_result_variants(fake_overlapindex):
    X = np.arange(72, dtype=float).reshape(24, 3)
    y = np.array(["husky"] * 6 + ["pug"] * 6 + ["sedan"] * 6 + ["suv"] * 6)
    dataset = BenchmarkDataset.from_arrays(
        X, y, modality="tabular", identity=DatasetIdentity.ephemeral()
    ).with_label_hierarchy(
        [
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
        ],
        level_names=("domain", "family", "leaf"),
    )

    benchmark = Benchmark(
        dataset,
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
        label_view_config=LabelViewConfig(enabled=True, hierarchy_levels=("domain", "family")),
    )
    benchmark.add_extractor(CallableExtractor("identity", lambda value: value, modality="tabular"))

    result = benchmark.run()

    assert [item.label_view["name"] for item in result.extractor_results] == ["domain", "family"]
    assert [item.name for item in result.extractor_results] == [
        "identity[level=domain]",
        "identity[level=family]",
    ]
    assert set(result.to_dataframe()["label_view"]) == {"domain", "family"}


def test_output_levels_route_multi_output_embeddings_to_hierarchy_views(
    tmp_path,
    fake_overlapindex,
):
    dataset = _hierarchical_vehicle_dataset()
    transform_calls = []

    def transform_many(value):
        transform_calls.append("transform_many")
        return {
            "layer_6": np.asarray(value)[:, :2],
            "final": np.asarray(value)[:, 1:3],
            "unused": np.asarray(value)[:, :2] * -1,
        }

    extractor = MultiOutputExtractor(
        name="hf",
        output_specs=[
            EmbeddingOutputSpec("layer_6"),
            EmbeddingOutputSpec("final"),
            EmbeddingOutputSpec("unused"),
        ],
        transform_many_fn=transform_many,
        modality="tabular",
    )

    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
        label_view_config=LabelViewConfig(
            output_levels={"layer_6": "family", "final": "leaf"},
        ),
    ).run()

    markdown_path = tmp_path / "mapped.md"
    result.save_markdown(str(markdown_path))

    assert transform_calls == ["transform_many"]
    assert [item.name for item in result.extractor_results] == [
        "hf:layer_6[level=family]",
        "hf:final[level=leaf]",
    ]
    assert [item.embedding_metadata["output_name"] for item in result.extractor_results] == [
        "layer_6",
        "final",
    ]
    assert [item.label_view["name"] for item in result.extractor_results] == ["family", "leaf"]
    assert set(result.to_dataframe()["label_view"]) == {"family", "leaf"}
    report = markdown_path.read_text(encoding="utf-8")
    assert "hf:layer_6[level=family]" in report
    assert "hf:final[level=leaf]" in report
    assert len(fake_overlapindex.calls) == 2


def test_output_levels_validate_hierarchy_and_output_names(fake_overlapindex):
    dataset = _hierarchical_vehicle_dataset()
    extractor = MultiOutputExtractor(
        name="hf",
        output_specs=[EmbeddingOutputSpec("layer_6")],
        transform_many_fn=lambda value: {"layer_6": np.asarray(value)[:, :2]},
        modality="tabular",
    )

    with pytest.raises(ValueError, match="unknown output names"):
        Benchmark(
            dataset=dataset,
            extractors=[extractor],
            stability_config=StabilityConfig(enabled=False),
            separatix_config=SeparatixConfig(enabled=False),
            cache_config=CacheConfig(enabled=False),
            label_view_config=LabelViewConfig(output_levels={"missing": "family"}),
        ).run()

    plain_dataset = BenchmarkDataset.from_arrays(
        np.arange(72, dtype=float).reshape(24, 3),
        np.array(["a"] * 12 + ["b"] * 12),
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    with pytest.raises(ValueError, match="requires dataset label hierarchy"):
        Benchmark(
            dataset=plain_dataset,
            extractors=[extractor],
            stability_config=StabilityConfig(enabled=False),
            separatix_config=SeparatixConfig(enabled=False),
            cache_config=CacheConfig(enabled=False),
            label_view_config=LabelViewConfig(output_levels={"layer_6": "family"}),
        ).run()


def test_output_levels_skip_invalid_mapped_hierarchy_levels(fake_overlapindex):
    dataset = _hierarchical_vehicle_dataset()
    extractor = MultiOutputExtractor(
        name="hf",
        output_specs=[EmbeddingOutputSpec("layer_6"), EmbeddingOutputSpec("final")],
        transform_many_fn=lambda value: {
            "layer_6": np.asarray(value)[:, :2],
            "final": np.asarray(value)[:, 1:3],
        },
        modality="tabular",
    )

    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
        label_view_config=LabelViewConfig(
            output_levels={"layer_6": "family", "final": "missing"},
            skip_invalid_levels=True,
        ),
    ).run()

    assert [item.name for item in result.extractor_results] == ["hf:layer_6[level=family]"]
    assert result.metadata["label_view_warnings"]

    with pytest.raises(ValueError, match="Unknown hierarchy level"):
        Benchmark(
            dataset=dataset,
            extractors=[extractor],
            stability_config=StabilityConfig(enabled=False),
            separatix_config=SeparatixConfig(enabled=False),
            cache_config=CacheConfig(enabled=False),
            label_view_config=LabelViewConfig(
                output_levels={"layer_6": "family", "final": "missing"},
                skip_invalid_levels=False,
            ),
        ).run()


def test_output_levels_reuse_base_embedding_cache(tmp_path, fake_overlapindex):
    dataset = _hierarchical_vehicle_dataset()
    transform_calls = []

    def transform_many(value):
        transform_calls.append("transform_many")
        return {
            "layer_6": np.asarray(value)[:, :2],
            "final": np.asarray(value)[:, 1:3],
        }

    extractor = MultiOutputExtractor(
        name="hf",
        output_specs=[EmbeddingOutputSpec("layer_6"), EmbeddingOutputSpec("final")],
        transform_many_fn=transform_many,
        modality="tabular",
    )
    config = LabelViewConfig(output_levels={"layer_6": "family", "final": "leaf"})

    first = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path)),
        label_view_config=config,
    ).run()
    second = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=True, cache_dir=str(tmp_path)),
        label_view_config=config,
    ).run()

    family_identity_key = dataset.label_view("family").identity_key()
    assert transform_calls == ["transform_many"]
    assert all(
        dataset.identity_key() in item.embedding_metadata["cache_key"]
        for item in first.extractor_results
    )
    assert all(
        family_identity_key not in item.embedding_metadata["cache_key"]
        for item in first.extractor_results
    )
    assert all(item.embedding_metadata["cache_hit"] for item in second.extractor_results)


def test_multi_output_cache_keeps_formerly_colliding_names_independent(
    tmp_path, fake_overlapindex
):
    values = np.arange(24, dtype=float).reshape(8, 3)
    dataset = BenchmarkDataset.from_arrays(
        values,
        ["a"] * 4 + ["b"] * 4,
        modality="tabular",
        identity=DatasetIdentity.ephemeral(),
    )
    transform_calls = []

    def transform_many(batch):
        transform_calls.append("transform_many")
        return {
            "a/b": np.asarray(batch)[:, :2],
            "a_b": np.asarray(batch)[:, 1:3] + 100,
        }

    extractor = MultiOutputExtractor(
        name="multi",
        output_specs=[EmbeddingOutputSpec("a/b"), EmbeddingOutputSpec("a_b")],
        transform_many_fn=transform_many,
        modality="tabular",
    )
    kwargs = {
        "stability_config": StabilityConfig(enabled=False),
        "separatix_config": SeparatixConfig(enabled=False),
        "cache_config": CacheConfig(enabled=True, cache_dir=str(tmp_path)),
    }

    first = Benchmark(dataset, [extractor], **kwargs).run()
    second = Benchmark(dataset, [extractor], **kwargs).run()

    assert transform_calls == ["transform_many"]
    assert [item.name for item in second.extractor_results] == ["multi:a/b", "multi:a_b"]
    assert all(item.embedding_metadata["cache_hit"] for item in second.extractor_results)
    cache_keys = [item.embedding_metadata["cache_key"] for item in first.extractor_results]
    assert len(set(cache_keys)) == 2

    store = LocalArtifactStore(str(tmp_path))
    assert np.array_equal(store.get_array(cache_keys[0]), values[:, :2])
    assert np.array_equal(store.get_array(cache_keys[1]), values[:, 1:3] + 100)


def test_multimodal_dataset_expands_multi_output_results(fake_overlapindex):
    dataset = BenchmarkDataset.from_multimodal(
        inputs={
            "image": ["a.png", "b.png", "c.png", "d.png"],
            "caption": ["one", "two", "three", "four"],
        },
        labels=["left", "left", "right", "right"],
        modalities={"image": "image", "caption": "text"},
        identity=DatasetIdentity.ephemeral(),
    )
    extractor = MultiOutputExtractor(
        name="fusion",
        output_specs=[
            EmbeddingOutputSpec("image_branch"),
            EmbeddingOutputSpec("fused"),
        ],
        transform_many_fn=lambda value: {
            "image_branch": np.arange(len(value["image"]) * 2, dtype=float).reshape(-1, 2),
            "fused": np.arange(len(value["caption"]) * 3, dtype=float).reshape(-1, 3),
        },
        modality="multimodal",
        streaming_safe=True,
    )

    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    assert result.dataset_summary["modality"] == "multimodal"
    assert [item.name for item in result.extractor_results] == [
        "fusion:image_branch",
        "fusion:fused",
    ]
    assert all(
        item.embedding_metadata["modality"] == "multimodal" for item in result.extractor_results
    )
    assert len(fake_overlapindex.calls) == 2


def test_target_views_produce_separate_result_variants(fake_overlapindex):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24, dtype=float).reshape(8, 3),
        ["cat", "cat", "dog", "dog", "bird", "bird", "fox", "fox"],
        identity=DatasetIdentity.ephemeral(),
    ).with_target_views(
        [
            TargetView(
                name="coarse",
                targets=COARSE_TARGETS,
            ),
            TargetView(
                name="score",
                targets=np.linspace(0.0, 1.0, 8),
                target_type="regression",
                target_names=["quality"],
            ),
        ]
    )

    result = Benchmark(
        dataset=dataset,
        extractors=[CallableExtractor("identity", lambda value: value, modality="embeddings")],
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
        target_view_config=TargetViewConfig(enabled=True),
    ).run()

    assert {item.name for item in result.extractor_results} == {
        "identity[target=coarse]",
        "identity[target=score]",
    }
    assert set(result.to_dataframe()["target_view"]) == {"coarse", "score"}
    assert len(fake_overlapindex.calls) == 1
    assert len(fake_overlapindex.continuous_calls) == 1


def test_output_views_route_multi_output_embeddings_to_target_views(fake_overlapindex):
    dataset = BenchmarkDataset.from_embeddings(
        np.arange(24, dtype=float).reshape(8, 3),
        ["cat", "cat", "dog", "dog", "bird", "bird", "fox", "fox"],
        identity=DatasetIdentity.ephemeral(),
    ).with_target_views(
        [
            TargetView(
                name="coarse",
                targets=COARSE_TARGETS,
            ),
            TargetView(
                name="role",
                targets=[
                    "title",
                    "title",
                    "body",
                    "body",
                    "caption",
                    "caption",
                    "footer",
                    "footer",
                ],
            ),
        ]
    )
    extractor = MultiOutputExtractor(
        name="hf",
        output_specs=[EmbeddingOutputSpec("layer_6"), EmbeddingOutputSpec("final")],
        transform_many_fn=lambda value: {
            "layer_6": np.asarray(value)[:, :2],
            "final": np.asarray(value)[:, 1:3],
        },
        modality="embeddings",
    )

    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        stability_config=StabilityConfig(enabled=False),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
        target_view_config=TargetViewConfig(
            output_views={"layer_6": "coarse", "final": "role"},
        ),
    ).run()

    assert [item.name for item in result.extractor_results] == [
        "hf:layer_6[target=coarse]",
        "hf:final[target=role]",
    ]
    assert [item.target_view["name"] for item in result.extractor_results] == [
        "coarse",
        "role",
    ]
    assert len(fake_overlapindex.calls) == 2


def _hierarchical_vehicle_dataset():
    X = np.arange(72, dtype=float).reshape(24, 3)
    y = np.array(["husky"] * 6 + ["pug"] * 6 + ["sedan"] * 6 + ["suv"] * 6)
    return BenchmarkDataset.from_arrays(
        X, y, modality="tabular", identity=DatasetIdentity.ephemeral()
    ).with_label_hierarchy(
        [
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "husky"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("animal", "dog", "pug"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "sedan"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
            ("vehicle", "car", "suv"),
        ],
        level_names=("domain", "family", "leaf"),
    )
