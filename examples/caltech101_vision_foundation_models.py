"""Compare vision foundation-model embeddings on a laptop-sized Caltech-101 subset.

Caltech-101 is a naturally single-label image dataset: images live in one
directory per object category. This example downloads the dataset archive when
needed, samples a small balanced subset with a few related category pairs, and
benchmarks the resulting image paths with DINOv2, a tiny supervised ViT baseline,
and a simple pixel PCA baseline.

Install optional dependencies with:

    poetry install -E hf

Run from the repository root:

    poetry run python examples/caltech101_vision_foundation_models.py

Optional environment variables:

    VERTABRAE_CALTECH101_DIR=/path/to/caltech101
    VERTABRAE_CALTECH101_CLASSES=crab,lobster,flamingo,ibis,cougar_body,wild_cat,crocodile,watch
    VERTABRAE_CALTECH101_SAMPLES_PER_CLASS=20
    VERTABRAE_INCLUDE_DINOV3=1

DINOv3 model access on Hugging Face is gated by Meta's model terms. Set
VERTABRAE_INCLUDE_DINOV3=1 only after accepting those terms for your account.

For a harder stress test, try:

    VERTABRAE_CALTECH101_CLASSES=crab,crayfish,lobster,crocodile,crocodile_head,cougar_body,cougar_face,wild_cat
"""

import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.request import urlretrieve

import numpy as np
from _common import CACHE_DIR, EXAMPLES_DIR, ensure_output_dir, print_ranking
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from vertebrae import Benchmark, BenchmarkDataset, DatasetIdentity
from vertebrae.config import (
    CacheConfig,
    EmbeddingConfig,
    OverlapScoringConfig,
    StabilityConfig,
)
from vertebrae.extractors import HFVisionExtractor, SklearnExtractor

CALTECH101_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
OBJECT_CATEGORIES_DIRNAME = "101_ObjectCategories"
DEFAULT_DATA_DIR = EXAMPLES_DIR / "data" / "caltech101"
DEFAULT_CLASSES = (
    "crab",
    "lobster",
    "flamingo",
    "ibis",
    "cougar_body",
    "wild_cat",
    "crocodile",
    "watch",
)
DEFAULT_SAMPLES_PER_CLASS = 20
DEFAULT_RANDOM_STATE = 17
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_HF_MODEL_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "dinov2_small",
        "model_id": "facebook/dinov2-small",
        "outputs": [
            {"name": "final_cls", "pooling": "cls"},
            {"name": "final_mean", "pooling": "mean"},
        ],
    },
    {
        "name": "deit_tiny_imagenet",
        "model_id": "facebook/deit-tiny-patch16-224",
        "outputs": [{"name": "final_cls", "pooling": "cls"}],
    },
)

DINOV3_HF_MODEL_SPEC = {
    "name": "dinov3_vits16",
    "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "outputs": [{"name": "final_cls", "pooling": "cls"}],
}


@dataclass(frozen=True)
class CaltechImageSample:
    label: str
    path: Path


def main() -> None:
    output_dir = ensure_output_dir()
    data_dir = Path(os.environ.get("VERTABRAE_CALTECH101_DIR", str(DEFAULT_DATA_DIR))).expanduser()
    class_names = _class_names_from_env()
    samples_per_class = _int_from_env(
        "VERTABRAE_CALTECH101_SAMPLES_PER_CLASS",
        DEFAULT_SAMPLES_PER_CLASS,
    )
    random_state = _int_from_env("VERTABRAE_CALTECH101_RANDOM_STATE", DEFAULT_RANDOM_STATE)

    try:
        samples = prepare_caltech101_subset(
            data_dir=data_dir,
            class_names=class_names,
            samples_per_class=samples_per_class,
            random_state=random_state,
        )
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"Could not prepare the Caltech-101 subset: {exc}")
        print("Use VERTABRAE_CALTECH101_DIR to point at an existing Caltech-101 download,")
        print("or run with network access so the example can fetch the dataset archive.")
        return

    paths = [str(sample.path) for sample in samples]
    labels = np.asarray([sample.label for sample in samples])
    dataset = BenchmarkDataset.from_image_paths(
        paths,
        labels,
        metadata={
            "example": "caltech101_vision_foundation_models",
            "source": "Caltech-101 subset",
            "data_dir": str(data_dir),
            "classes": list(class_names),
            "samples_per_class": samples_per_class,
            "label_rule": "Caltech-101 object-category directory",
            "default_slice": "moderate related-category pairs for a laptop run",
        },
        identity=DatasetIdentity.ephemeral(),
    )

    benchmark = Benchmark(
        dataset=dataset,
        scoring_config=OverlapScoringConfig(k=4, min_samples_per_cluster=4),
        stability_config=StabilityConfig(repeats=3, random_state=random_state),
        cache_config=CacheConfig(cache_dir=str(CACHE_DIR / "caltech101_vision_models")),
        embedding_config=EmbeddingConfig(batch_size=8),
    )

    for spec in hf_model_specs():
        benchmark.add_extractor(
            HFVisionExtractor(
                name=spec["name"],
                model_id=spec["model_id"],
                outputs=spec["outputs"],
                pooling=spec["outputs"][0]["pooling"],
                image_mode="rgb",
                batch_size=8,
            )
        )
    benchmark.add_extractor(_sklearn_image_baseline(n_samples=len(samples)))

    print(
        "Prepared Caltech-101 subset with "
        f"{len(samples)} images across {len(class_names)} classes in {data_dir}."
    )
    print("Running extractors:", ", ".join(extractor.name for extractor in benchmark.extractors))

    try:
        result = benchmark.run()
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: poetry install -E hf")
        return
    except OSError as exc:
        print(f"Could not load one of the Hugging Face models: {exc}")
        if include_dinov3():
            print("For DINOv3, make sure you accepted Meta's Hugging Face model terms.")
        print("Use local model paths or run with network access for the first model download.")
        return

    result.save_json(str(output_dir / "caltech101_vision_foundation_models.json"))
    result.save_markdown(str(output_dir / "caltech101_vision_foundation_models.md"))
    print_ranking(result)
    print(f"\nReports written to {output_dir}")


def prepare_caltech101_subset(
    data_dir: Path,
    class_names: Sequence[str],
    samples_per_class: int,
    random_state: int,
) -> List[CaltechImageSample]:
    categories_dir = ensure_caltech101_categories(data_dir)
    return select_caltech101_subset(
        categories_dir=categories_dir,
        class_names=class_names,
        samples_per_class=samples_per_class,
        random_state=random_state,
    )


def ensure_caltech101_categories(data_dir: Path) -> Path:
    existing = _find_categories_dir(data_dir)
    if existing is not None:
        return existing

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = _find_existing_archive(data_dir)
    if archive_path is None:
        archive_path = data_dir / "caltech-101.zip"
        print(f"Downloading Caltech-101 to {archive_path}...")
        _download_file(CALTECH101_URL, archive_path)

    _extract_archive(archive_path, data_dir)
    categories_dir = _find_categories_dir(data_dir)
    if categories_dir is not None:
        return categories_dir

    nested_archive = _find_nested_categories_archive(data_dir)
    if nested_archive is not None:
        _extract_archive(nested_archive, data_dir)
        categories_dir = _find_categories_dir(data_dir)
        if categories_dir is not None:
            return categories_dir

    raise ValueError(f"Could not find a {OBJECT_CATEGORIES_DIRNAME} directory in {data_dir}.")


def select_caltech101_subset(
    categories_dir: Path,
    class_names: Sequence[str],
    samples_per_class: int,
    random_state: int,
) -> List[CaltechImageSample]:
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2.")

    requested = _normalize_class_names(class_names)
    category_dirs = _category_dirs_by_normalized_name(categories_dir)
    missing = sorted(
        name for name in requested if _normalize_category_name(name) not in category_dirs
    )
    if missing:
        available = ", ".join(sorted(path.name for path in category_dirs.values())[:12])
        raise ValueError(
            f"Caltech-101 categories not found: {missing}. Available examples: {available}."
        )

    rng = np.random.default_rng(random_state)
    samples: List[CaltechImageSample] = []
    for requested_name in requested:
        category_dir = category_dirs[_normalize_category_name(requested_name)]
        candidates = sorted(
            path
            for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Caltech-101 category '{category_dir.name}' has {len(candidates)} images; "
                f"need {samples_per_class}."
            )
        indices = rng.permutation(len(candidates))[:samples_per_class]
        samples.extend(
            CaltechImageSample(label=category_dir.name, path=candidates[int(index)])
            for index in indices
        )

    return sorted(samples, key=lambda sample: (sample.label, sample.path.name))


def hf_model_specs() -> Tuple[Dict[str, Any], ...]:
    specs = list(DEFAULT_HF_MODEL_SPECS)
    if include_dinov3():
        specs.append(DINOV3_HF_MODEL_SPEC)
    return tuple(specs)


def include_dinov3() -> bool:
    return _truthy(os.environ.get("VERTABRAE_INCLUDE_DINOV3"))


def _sklearn_image_baseline(n_samples: int) -> SklearnExtractor:
    n_components = max(2, min(48, n_samples - 1))
    pipeline = Pipeline(
        [
            ("pixels", FunctionTransformer(_load_resize_flatten, validate=False)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=31)),
        ]
    )
    return SklearnExtractor(
        name=f"rgb64_standard_pca_{n_components}",
        pipeline=pipeline,
        extractor_type="classical_image_pipeline",
    )


def _load_resize_flatten(paths: Any) -> np.ndarray:
    rows = []
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((64, 64))
            rows.append(np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0)
    return np.vstack(rows)


def _find_categories_dir(data_dir: Path) -> Optional[Path]:
    direct = data_dir / OBJECT_CATEGORIES_DIRNAME
    if direct.exists():
        return direct
    for path in data_dir.rglob(OBJECT_CATEGORIES_DIRNAME):
        if path.is_dir():
            return path
    return None


def _find_existing_archive(data_dir: Path) -> Optional[Path]:
    candidates = [
        data_dir / "caltech-101.zip",
        data_dir / "101_ObjectCategories.tar.gz",
        data_dir / "101_ObjectCategories.tar",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_nested_categories_archive(data_dir: Path) -> Optional[Path]:
    for path in data_dir.rglob("101_ObjectCategories.tar*"):
        if path.is_file():
            return path
    return None


def _extract_archive(archive_path: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(destination)
        return
    raise ValueError(f"Unsupported Caltech-101 archive format: {archive_path}.")


def _category_dirs_by_normalized_name(categories_dir: Path) -> Dict[str, Path]:
    return {
        _normalize_category_name(path.name): path
        for path in categories_dir.iterdir()
        if path.is_dir() and path.name.lower() != "background_google"
    }


def _class_names_from_env() -> Tuple[str, ...]:
    raw = os.environ.get("VERTABRAE_CALTECH101_CLASSES")
    if not raw:
        return DEFAULT_CLASSES
    return _normalize_class_names(raw.split(","))


def _normalize_class_names(class_names: Sequence[str]) -> Tuple[str, ...]:
    normalized = []
    seen = set()
    for raw_name in class_names:
        name = str(raw_name).strip()
        key = _normalize_category_name(name)
        if not name or key in seen:
            continue
        normalized.append(name)
        seen.add(key)
    if len(normalized) < 2:
        raise ValueError("At least two Caltech-101 classes are required.")
    return tuple(normalized)


def _normalize_category_name(name: str) -> str:
    return str(name).strip().lower()


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        urlretrieve(url, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
