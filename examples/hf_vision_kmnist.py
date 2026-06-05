"""Compare Hugging Face vision backbones on KMNIST character images.

The images come from Kuzushiji-MNIST, a 28x28 grayscale handwritten Japanese
character dataset. The dataset and models must be present in local Hugging Face
caches or downloadable on first run.

Install optional dependencies with:

    poetry install -E hf
"""

import numpy as np
from _common import CACHE_DIR, ensure_output_dir, print_ranking

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import HFVisionExtractor

DATASET_ID = "tanganke/kmnist"
MODEL_SPECS = (
    {
        "name": "deit_tiny_imagenet_cls",
        "model_id": "facebook/deit-tiny-patch16-224",
        "pooling": "cls",
        "image_mode": "rgb",
    },
    {
        "name": "mnist_resnet_grayscale_pooler",
        "model_id": "fxmarty/resnet-tiny-mnist",
        "pooling": "pooler",
        "image_mode": "grayscale",
    },
)


def main() -> None:
    output_dir = ensure_output_dir()

    try:
        images, labels = _load_kmnist_images(classes=(0, 1, 2, 3), samples_per_class=32)
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E hf")
        return
    except OSError as exc:
        print(f"Could not load KMNIST dataset '{DATASET_ID}': {exc}")
        print("Use a local Hugging Face cache or run with network access for the first download.")
        return

    dataset = BenchmarkDataset.from_arrays(
        images,
        labels,
        modality="image",
        metadata={
            "example": "hf_vision_kmnist",
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
                    pooling=spec["pooling"],
                    image_mode=spec["image_mode"],
                    batch_size=8,
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

    result.save_json(str(output_dir / "hf_vision_kmnist.json"))
    result.save_markdown(str(output_dir / "hf_vision_kmnist.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


def _load_kmnist_images(
    classes: tuple[int, ...],
    samples_per_class: int,
) -> tuple[list[object], np.ndarray]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "hf_vision_kmnist.py requires Hugging Face datasets, which is included "
            "in the hf extra."
        ) from exc

    data = load_dataset(DATASET_ID, split="train", streaming=True)
    images = []
    labels = []
    counts = {label: 0 for label in classes}
    for row in data:
        label = int(row["label"])
        if label not in counts or counts[label] >= samples_per_class:
            continue
        images.append(_image_to_uint8(row["image"]))
        labels.append(f"class_{label}")
        counts[label] += 1
        if all(count >= samples_per_class for count in counts.values()):
            break
    missing = {
        label: samples_per_class - count
        for label, count in counts.items()
        if count < samples_per_class
    }
    if missing:
        raise ValueError(f"KMNIST split did not contain enough samples for classes: {missing}.")
    return images, np.asarray(labels)


def _image_to_uint8(image: object) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return array.astype(np.uint8, copy=False)


if __name__ == "__main__":
    main()
