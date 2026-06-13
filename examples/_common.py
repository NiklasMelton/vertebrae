"""Shared helpers for runnable examples."""

import os
from pathlib import Path
from typing import Tuple

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLES_DIR / "output"
CACHE_DIR = EXAMPLES_DIR / ".vertebrae_cache"

os.environ.setdefault("MPLCONFIGDIR", str(EXAMPLES_DIR / ".matplotlib_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def make_separated_blobs(
    samples_per_class: int = 36,
    n_features: int = 8,
    random_state: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    centers = np.array(
        [
            np.linspace(-2.0, 1.5, n_features),
            np.linspace(2.0, -1.0, n_features),
            np.r_[np.repeat(1.5, n_features // 2), np.repeat(-1.5, n_features - n_features // 2)],
        ]
    )
    embeddings = []
    labels = []
    for class_id, center in enumerate(centers):
        embeddings.append(rng.normal(loc=center, scale=0.45, size=(samples_per_class, n_features)))
        labels.extend([f"class_{class_id}"] * samples_per_class)
    return np.vstack(embeddings), np.asarray(labels)


def print_ranking(result: object) -> None:
    frame = result.to_dataframe()
    columns = ["rank", "extractor", "overlap_macro", "weakest_class",
               "recommendation", "separatix_recommendation"]
    print(frame[columns].to_string(index=False))
