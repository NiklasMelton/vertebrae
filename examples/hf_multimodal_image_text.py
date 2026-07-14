"""Hugging Face multi-modal image-text extractor API example.

Requires optional dependencies and a model available locally or from Hugging Face:

    poetry install -E hf
"""

import numpy as np
from _common import CACHE_DIR, ensure_output_dir, print_ranking

from vertebrae import BenchmarkDataset, DatasetIdentity, Evaluator
from vertebrae.config import CacheConfig, OverlapScoringConfig, StabilityConfig
from vertebrae.extractors import HFMultimodalExtractor


def main() -> None:
    output_dir = ensure_output_dir()
    images = [np.full((8, 8, 3), fill_value=index * 32, dtype=np.uint8) for index in range(6)]
    captions = [
        "A clean sneaker on a white background.",
        "A running shoe photographed from the side.",
        "A ceramic mug on a kitchen shelf.",
        "A coffee cup with a simple handle.",
        "A small succulent in a terracotta pot.",
        "A potted plant on a bright windowsill.",
    ]
    labels = ["shoe", "shoe", "mug", "mug", "plant", "plant"]
    dataset = BenchmarkDataset.from_multimodal(
        inputs={"image": images, "caption": captions},
        labels=labels,
        modalities={"image": "image", "caption": "text"},
        metadata={"example": "hf_multimodal_image_text"},
        identity=DatasetIdentity.ephemeral(),
    )

    extractor = HFMultimodalExtractor(
        name="clip_like",
        model_id="openai/clip-vit-base-patch32",
        input_modalities={"image": "image", "caption": "text"},
        outputs=[
            {"name": "image_branch", "source": "image", "model_output": "image_embeds"},
            {"name": "text_branch", "source": "text", "model_output": "text_embeds"},
            {"name": "fused", "source": "fused", "model_output": "pooler_output"},
        ],
        batch_size=2,
    )

    try:
        result = Evaluator(
            dataset=dataset,
            extractor=extractor,
            scoring_config=OverlapScoringConfig(k=1, min_samples_per_cluster=2),
            stability_config=StabilityConfig(repeats=3),
            cache_config=CacheConfig(cache_dir=str(CACHE_DIR)),
        ).run()
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E hf")
        return
    except OSError as exc:
        print(f"Could not load the Hugging Face model: {exc}")
        print("Use a local model path or run with network access for the first download.")
        return

    result.save_markdown(str(output_dir / "hf_multimodal_image_text.md"))
    print_ranking(result)
    print(f"\nReport written to {output_dir / 'hf_multimodal_image_text.md'}")


if __name__ == "__main__":
    main()
