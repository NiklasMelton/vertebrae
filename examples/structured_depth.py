"""Evaluate sampled depth outputs as regression-style embedding diagnostics."""

import numpy as np
from _common import ensure_cache_dir, ensure_output_dir

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    CallableStructuredExtractor,
    DatasetIdentity,
    DepthAdapter,
    DepthAnnotation,
)
from vertebrae.config import CacheConfig, SeparatixConfig, StabilityConfig
from vertebrae.execution import materialize_structured_artifacts
from vertebrae.extractors import StructuredOutputSpec


def main() -> None:
    output_dir = ensure_output_dir()
    cache_dir = ensure_cache_dir()

    dataset = BenchmarkDataset.from_arrays(
        X=np.asarray(["scene_a", "scene_b", "scene_c", "scene_d"], dtype=object),
        y=[0.10, 0.15, 0.80, 0.85],
        modality="image",
        target_type="regression",
        target_names=["scene_depth"],
        metadata={"example": "structured_depth"},
        identity=DatasetIdentity.declared("structured-depth-example", "1"),
    )
    dataset = DepthAdapter().attach(
        dataset,
        [
            DepthAnnotation(
                labels=[1.0, 1.8, 9.5],
                coordinates=[[0, 0], [1, 1], [2, 2]],
                valid=[1, 1, 0],
                unit_ids=["a:0", "a:1", "a:2"],
                depth_units="meters",
                scaling="linear",
            ),
            DepthAnnotation(
                labels=[1.1, 1.9, 9.6],
                coordinates=[[0, 0], [1, 1], [2, 2]],
                valid=[1, 1, 0],
                unit_ids=["b:0", "b:1", "b:2"],
                depth_units="meters",
                scaling="linear",
            ),
            DepthAnnotation(
                labels=[4.8, 5.6, 9.7],
                coordinates=[[0, 0], [1, 1], [2, 2]],
                valid=[1, 1, 0],
                unit_ids=["c:0", "c:1", "c:2"],
                depth_units="meters",
                scaling="linear",
            ),
            DepthAnnotation(
                labels=[4.9, 5.7, 9.8],
                coordinates=[[0, 0], [1, 1], [2, 2]],
                valid=[1, 1, 0],
                unit_ids=["d:0", "d:1", "d:2"],
                depth_units="meters",
                scaling="linear",
            ),
        ],
    )

    embeddings = {
        "scene_a": np.asarray([[1.0, 0.0], [0.9, 0.1]]),
        "scene_b": np.asarray([[1.0, 0.0], [0.9, 0.1]]),
        "scene_c": np.asarray([[0.1, 0.9], [0.0, 1.0]]),
        "scene_d": np.asarray([[0.1, 0.9], [0.0, 1.0]]),
    }
    extractor = CallableStructuredExtractor(
        name="depth_samples",
        transform_fn=_lookup_structured_embeddings(embeddings),
        output_specs=[StructuredOutputSpec(name="depth_cells", unit_type="depth_sample")],
        modality="image",
    )

    result = Benchmark(
        dataset=dataset,
        extractors=[extractor],
        stability_config=StabilityConfig(repeats=3, random_state=17),
        separatix_config=SeparatixConfig(enabled=False),
        cache_config=CacheConfig(enabled=False, cache_dir=str(cache_dir / "structured_depth")),
    ).run()
    result.save_json(str(output_dir / "structured_depth.json"))
    result.save_markdown(str(output_dir / "structured_depth.md"))

    bundle = materialize_structured_artifacts(
        dataset,
        extractor,
        workflow_store("structured_depth"),
        batch_size=2,
    )

    print(
        "Depth samples: "
        f"overlap={result.extractor_results[0].overlap.score:.3f} "
        f"task_family={bundle['outputs'][0]['task_family']} "
        f"rows={bundle['outputs'][0]['n_samples']} "
        f"bundle={bundle['output_key']}"
    )
    print(
        "\nThis workflow evaluates whether depth-sampled unit embeddings preserve a continuous "
        "depth signal. It does not compute RMSE, absolute relative error, or other depth "
        "estimation metrics."
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
