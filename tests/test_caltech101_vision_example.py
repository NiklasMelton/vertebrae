import importlib.util
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
EXAMPLE_PATH = EXAMPLES_DIR / "caltech101_vision_foundation_models.py"


def _load_example_module():
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location(
        "caltech101_vision_foundation_models",
        EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


caltech_example = _load_example_module()


def test_select_caltech101_subset_uses_directory_names_as_labels(tmp_path):
    categories_dir = tmp_path / "101_ObjectCategories"
    for category in ["airplanes", "Motorbikes"]:
        category_dir = categories_dir / category
        category_dir.mkdir(parents=True)
        for index in range(4):
            (category_dir / f"image_{index:04d}.jpg").write_bytes(b"image")

    samples = caltech_example.select_caltech101_subset(
        categories_dir=categories_dir,
        class_names=("AIRPLANES", "motorbikes"),
        samples_per_class=3,
        random_state=7,
    )

    assert Counter(sample.label for sample in samples) == {"airplanes": 3, "Motorbikes": 3}
    assert all(sample.path.exists() for sample in samples)
    assert samples == sorted(samples, key=lambda sample: (sample.label, sample.path.name))


def test_build_caltech101_dataset_preserves_descriptive_source(tmp_path):
    samples = [
        caltech_example.CaltechImageSample(label="crab", path=tmp_path / "crab_1.jpg"),
        caltech_example.CaltechImageSample(label="crab", path=tmp_path / "crab_2.jpg"),
        caltech_example.CaltechImageSample(label="watch", path=tmp_path / "watch_1.jpg"),
        caltech_example.CaltechImageSample(label="watch", path=tmp_path / "watch_2.jpg"),
    ]

    dataset = caltech_example.build_caltech101_dataset(
        samples=samples,
        data_dir=tmp_path,
        class_names=("crab", "watch"),
        samples_per_class=2,
    )

    assert dataset.metadata["source"] == "image_paths"
    assert dataset.metadata["dataset_source"] == "Caltech-101 subset"


def test_ensure_caltech101_categories_reuses_existing_directory(tmp_path):
    categories_dir = tmp_path / "101_ObjectCategories"
    categories_dir.mkdir()

    assert caltech_example.ensure_caltech101_categories(tmp_path) == categories_dir


def test_ensure_caltech101_categories_extracts_existing_zip(tmp_path):
    zip_path = tmp_path / "caltech-101.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("caltech-101/101_ObjectCategories/watch/image_0001.jpg", b"image")

    categories_dir = caltech_example.ensure_caltech101_categories(tmp_path)

    assert categories_dir == tmp_path / "caltech-101" / "101_ObjectCategories"
    assert (categories_dir / "watch" / "image_0001.jpg").read_bytes() == b"image"


def test_prepare_caltech101_subset_downloads_archive_when_missing(tmp_path, monkeypatch):
    def fake_download(url, destination):
        assert url == caltech_example.CALTECH101_URL
        with zipfile.ZipFile(destination, "w") as archive:
            for category in ["Faces_easy", "watch"]:
                for index in range(2):
                    archive.writestr(
                        f"101_ObjectCategories/{category}/image_{index:04d}.jpg",
                        b"image",
                    )

    monkeypatch.setattr(caltech_example, "_download_file", fake_download)

    samples = caltech_example.prepare_caltech101_subset(
        data_dir=tmp_path,
        class_names=("faces_easy", "WATCH"),
        samples_per_class=2,
        random_state=3,
    )

    assert Counter(sample.label for sample in samples) == {"Faces_easy": 2, "watch": 2}
    assert all(sample.path.exists() for sample in samples)
