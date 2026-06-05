"""Compare Hugging Face vision backbones on real handwritten digit images.

The digit images come from ``sklearn.datasets.load_digits`` so the data itself is
available offline. The models are small real Hugging Face vision backbones; they
must be present in the local Hugging Face cache or downloadable on first run.

Install optional dependencies with:

    poetry install -E hf
"""

import numpy as np
from _common import CACHE_DIR, ensure_output_dir, print_ranking
from sklearn.datasets import load_digits

from vertebrae import Benchmark, BenchmarkDataset
from vertebrae.config import CacheConfig, OverlapScoringConfig, ProbeConfig, StabilityConfig
from vertebrae.extractors import HFVisionExtractor

MODEL_SPECS = (
    {
        "name": "deit_tiny_imagenet_cls",
        "model_id": "facebook/deit-tiny-patch16-224",
        "pooling": "cls",
    },
    {
        "name": "resnet18_imagenet_pooler",
        "model_id": "microsoft/resnet-18",
        "pooling": "pooler",
    },
)


def main() -> None:
    output_dir = ensure_output_dir()

    try:
        images, labels = _load_digit_images(digits=(0, 1, 2, 3), samples_per_digit=32)
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E hf")
        return

    dataset = BenchmarkDataset.from_arrays(
        images,
        labels,
        modality="image",
        metadata={
            "example": "hf_vision_digits",
            "source": "sklearn.datasets.load_digits",
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

    result.save_json(str(output_dir / "hf_vision_digits.json"))
    result.save_markdown(str(output_dir / "hf_vision_digits.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


def _load_digit_images(
    digits: tuple[int, ...],
    samples_per_digit: int,
) -> tuple[list[object], np.ndarray]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "hf_vision_digits.py requires Pillow, which is included in the hf extra."
        ) from exc

    data = load_digits()
    images = []
    labels = []
    for digit in digits:
        digit_indices = np.flatnonzero(data.target == digit)[:samples_per_digit]
        for index in digit_indices:
            images.append(_digit_to_rgb_image(data.images[index], Image))
            labels.append(str(digit))
    return images, np.asarray(labels)


def _digit_to_rgb_image(image: np.ndarray, image_module: object) -> object:
    scaled = np.clip((image / 16.0) * 255.0, 0, 255).astype(np.uint8)
    pil_image = image_module.fromarray(scaled, mode="L")
    return pil_image.resize((224, 224), resample=image_module.Resampling.BICUBIC).convert("RGB")


if __name__ == "__main__":
    main()
