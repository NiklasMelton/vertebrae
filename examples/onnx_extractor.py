"""ONNXExtractor example for a locally exported ONNX model.

This script shows the core ONNX workflow without bundling a model in the repo.
Set ``VERTABRAE_ONNX_MODEL_PATH`` to a local export before running it, or update
``MODEL_PATH`` below to point at your file.

Requires optional dependencies:

    poetry install -E onnx
"""

import os
from pathlib import Path

import numpy as np
from _common import ensure_output_dir, make_separated_blobs, print_ranking

from vertebrae import BenchmarkDataset, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import ONNXExtractor

MODEL_PATH = Path(
    os.environ.get(
        "VERTABRAE_ONNX_MODEL_PATH",
        Path(__file__).resolve().parent / "models" / "replace-with-your-export.onnx",
    )
)


def main() -> None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        print(exc)
        print("Install optional ONNX Runtime support with: poetry install -E onnx")
        return

    if not MODEL_PATH.exists():
        print(f"ONNX model not found at: {MODEL_PATH}")
        print("Export your model to ONNX, then set VERTABRAE_ONNX_MODEL_PATH to that file.")
        print(
            "A common pattern is to export a locally trained encoder and point the "
            "example at it."
        )
        return

    output_dir = ensure_output_dir()
    X, labels = make_separated_blobs(samples_per_class=30, n_features=6, random_state=29)
    dataset = BenchmarkDataset.from_arrays(
        X,
        labels,
        modality="tabular",
        metadata={"example": "onnx_extractor"},
    )

    def input_fn(batch: np.ndarray) -> dict[str, np.ndarray]:
        return {"input_0": np.asarray(batch, dtype=np.float32)}

    def output_fn(raw_outputs):
        return raw_outputs[0]

    extractor = ONNXExtractor(
        name="onnx_export",
        model_path=MODEL_PATH,
        input_fn=input_fn,
        output_fn=output_fn,
        input_names=["input_0"],
        recipe_data={"model_path": str(MODEL_PATH)},
    )

    result = Evaluator(
        dataset=dataset,
        extractor=extractor,
        scoring_config=OverlapScoringConfig(k=3, min_samples_per_cluster=4),
        stability_config=StabilityConfig(repeats=3, random_state=13),
        probe_config=ProbeConfig(enabled=False),
        cache_config=CacheConfig(enabled=False),
    ).run()

    result.save_json(str(output_dir / "onnx_extractor.json"))
    result.save_markdown(str(output_dir / "onnx_extractor.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
