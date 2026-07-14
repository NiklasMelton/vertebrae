"""Evaluate latent-slot outputs as structured embedding diagnostics."""

import numpy as np
from _common import ensure_cache_dir, ensure_output_dir

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CallableStructuredExtractor,
    DatasetIdentity,
    LatentSlotAdapter,
    LatentSlotAnnotation,
    drop_special_rows,
)
from vertebrae.config import CacheConfig, OverlapScoringConfig, SeparatixConfig, StabilityConfig
from vertebrae.execution import materialize_structured_artifacts
from vertebrae.extractors import StructuredOutputSpec


def main() -> None:
    output_dir = ensure_output_dir()
    cache_dir = ensure_cache_dir()

    dataset = BenchmarkDataset.from_arrays(
        X=np.asarray(["latent_a", "latent_b", "latent_c", "latent_d"], dtype=object),
        y=["scene", "scene", "object", "object"],
        modality="embeddings",
        metadata={"example": "structured_latent_slots"},
        identity=DatasetIdentity.declared("structured-latent-slots-example", "1"),
    )
    dataset = LatentSlotAdapter().attach(
        dataset,
        [
            LatentSlotAnnotation(
                labels=["background", "foreground"],
                slot_ids=["a:bg", "a:fg"],
                source_component_ids=["decoder", "decoder"],
                ordered=True,
            ),
            LatentSlotAnnotation(
                labels=["background", "foreground"],
                slot_ids=["b:bg", "b:fg"],
                source_component_ids=["decoder", "decoder"],
                ordered=True,
            ),
            LatentSlotAnnotation(
                labels=["background", "foreground"],
                slot_ids=["c:bg", "c:fg"],
                source_component_ids=["decoder", "decoder"],
                ordered=True,
            ),
            LatentSlotAnnotation(
                labels=["background", "foreground"],
                slot_ids=["d:bg", "d:fg"],
                source_component_ids=["decoder", "decoder"],
                ordered=True,
            ),
        ],
    )

    embeddings = {
        "latent_a": np.asarray([[9.0, 9.0], [1.0, 0.0], [0.9, 0.1], [8.0, 8.0]]),
        "latent_b": np.asarray([[9.0, 9.0], [1.0, 0.0], [0.9, 0.1], [8.0, 8.0]]),
        "latent_c": np.asarray([[9.0, 9.0], [0.1, 0.9], [0.0, 1.0], [8.0, 8.0]]),
        "latent_d": np.asarray([[9.0, 9.0], [0.1, 0.9], [0.0, 1.0], [8.0, 8.0]]),
    }
    extractor = CallableStructuredExtractor(
        name="latent_slots",
        transform_fn=_lookup_structured_embeddings(embeddings),
        output_specs=[StructuredOutputSpec(name="slots", unit_type="latent_slot")],
        modality="embeddings",
    )
    aligner = drop_special_rows(leading=1, trailing=1)

    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        scoring_config=OverlapScoringConfig(k=2, min_samples_per_cluster=2),
        stability_config=StabilityConfig(repeats=3, random_state=19),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(
            enabled=False,
            cache_dir=str(cache_dir / "structured_latent_slots"),
        ),
        structured_aligners={"slots": aligner},
    ).run()
    result.save_json(str(output_dir / "structured_latent_slots.json"))
    result.save_markdown(str(output_dir / "structured_latent_slots.md"))

    bundle = materialize_structured_artifacts(
        dataset,
        extractor,
        workflow_store("structured_latent_slots"),
        batch_size=2,
        aligners={"slots": aligner},
    )

    print(
        "Latent slots: "
        f"overlap={result.extractor_results[0].overlap.macro_score:.3f} "
        f"alignment={bundle['outputs'][0]['alignment_mode']} "
        f"recipe={bundle['outputs'][0]['alignment_recipe']['name']} "
        f"rows={bundle['outputs'][0]['n_samples']} "
        f"bundle={bundle['output_key']}"
    )
    print(
        "\nThis workflow evaluates labeled slot embeddings after explicitly dropping unmatched "
        "leading and trailing rows from the raw latent output. It does not compute generative "
        "reconstruction quality, FID, or slot discovery metrics."
    )
    print(f"Reports written to {output_dir}")


def _lookup_structured_embeddings(embeddings_by_parent):
    def transform(batch):
        return [embeddings_by_parent[str(item)] for item in np.asarray(batch).tolist()]

    return transform


def workflow_store(name: str):
    from vertebrae.cache.local_store import LocalArtifactStore

    return LocalArtifactStore(ensure_cache_dir() / name)


if __name__ == "__main__":
    main()
