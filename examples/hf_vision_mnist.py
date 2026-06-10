"""Compare vision embeddings on MNIST handwritten digit images.

The images come from MNIST, a 28x28 grayscale handwritten digit dataset. The
dataset and Hugging Face models must be present in local caches or downloadable
on first run.

Install optional dependencies with:

    poetry install -E hf
"""

import numpy as np
from _common import CACHE_DIR, ensure_output_dir, print_ranking
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import HFVisionExtractor, SklearnExtractor

DATASET_ID = "ylecun/mnist"
MODEL_SPECS = (
    {
        "name": "deit_tiny_imagenet_cls",
        "model_id": "facebook/deit-tiny-patch16-224",
        "outputs": [{"name": "cls", "pooling": "cls"}],
        "image_mode": "rgb",
    },
    {
        "name": "mnist_vit_trained",
        "model_id": "farleyknight-org-username/vit-base-mnist",
        "outputs": [
            {"name": "final_cls", "pooling": "cls"},
            {"name": "mid_cls", "pooling": "cls", "hidden_layer": 6},
        ],
        "image_mode": "rgb",
    },
)


def main() -> None:
    output_dir = ensure_output_dir()

    try:
        images, labels = _load_mnist_images(digits=(0, 1, 2, 3), samples_per_digit=64)
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E hf")
        return
    except OSError as exc:
        print(f"Could not load MNIST dataset '{DATASET_ID}': {exc}")
        print("Use a local Hugging Face cache or run with network access for the first download.")
        return

    dataset = BenchmarkDataset.from_arrays(
        images,
        labels,
        modality="image",
        metadata={
            "example": "hf_vision_mnist",
            "source": DATASET_ID,
            "model_ids": [spec["model_id"] for spec in MODEL_SPECS],
        },
    )

    try:
        benchmark = Benchmark(
            dataset=dataset,
            scoring_config=OverlapScoringConfig(k=2, min_samples_per_cluster=6),
            stability_config=StabilityConfig(repeats=3, random_state=29),
            probe_config=ProbeConfig(methods=("nearest_centroid", "knn")),
            cache_config=CacheConfig(cache_dir=str(CACHE_DIR)),
        )
        for spec in MODEL_SPECS:
            benchmark.add_extractor(
                HFVisionExtractor(
                    name=spec["name"],
                    model_id=spec["model_id"],
                    processor_id=spec.get("processor_id"),
                    pooling=spec["outputs"][0]["pooling"],
                    hidden_layer=spec["outputs"][0].get("hidden_layer"),
                    outputs=spec["outputs"],
                    image_mode=spec["image_mode"],
                    batch_size=8,
                )
            )
        benchmark.add_extractor(
            SklearnExtractor(
                name="sklearn_flatten_scale_pca_32",
                pipeline=Pipeline(
                    [
                        ("flatten", FunctionTransformer(_flatten_images, validate=False)),
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=32, random_state=31)),
                    ]
                ),
                extractor_type="classical_image_pipeline",
            )
        )
        result = benchmark.run()
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E hf")
        return
    except OSError as exc:
        print(f"Could not load one of the Hugging Face models: {exc}")
        print("Use local model paths or run with network access for the first download.")
        return

    result.save_json(str(output_dir / "hf_vision_mnist.json"))
    result.save_markdown(str(output_dir / "hf_vision_mnist.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


def _load_mnist_images(
    digits: tuple[int, ...],
    samples_per_digit: int,
) -> tuple[list[object], np.ndarray]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "hf_vision_mnist.py requires Hugging Face datasets, which is included "
            "in the hf extra."
        ) from exc

    data = load_dataset(DATASET_ID, split="train", streaming=True)
    images = []
    labels = []
    counts = {label: 0 for label in digits}
    for row in data:
        label = int(row["label"])
        if label not in counts or counts[label] >= samples_per_digit:
            continue
        images.append(_image_to_uint8(row["image"]))
        labels.append(str(label))
        counts[label] += 1
        if all(count >= samples_per_digit for count in counts.values()):
            break
    missing = {
        label: samples_per_digit - count
        for label, count in counts.items()
        if count < samples_per_digit
    }
    if missing:
        raise ValueError(f"MNIST split did not contain enough samples for digits: {missing}.")
    return images, np.asarray(labels)


def _image_to_uint8(image: object) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return array.astype(np.uint8, copy=False)


def _flatten_images(images: object) -> np.ndarray:
    return np.asarray([_image_to_uint8(image).reshape(-1) for image in images], dtype=np.float32)


if __name__ == "__main__":
    main()
