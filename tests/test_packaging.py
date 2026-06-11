from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_sdist_includes_release_support_directories():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = pyproject["tool"]["poetry"]["include"]
    include_by_path = {entry["path"]: entry for entry in includes}

    for path in ["tests", "docs", "examples", "img"]:
        assert include_by_path[path]["format"] == "sdist"


def test_package_declares_typed_marker():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = pyproject["tool"]["poetry"]["include"]

    assert (ROOT / "src" / "vertebrae" / "py.typed").exists()
    assert {
        "path": "src/vertebrae/py.typed",
        "format": ["sdist", "wheel"],
    } in includes


def test_package_excludes_macos_metadata_files():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "**/.DS_Store" in pyproject["tool"]["poetry"]["exclude"]


def test_docs_markdown_fences_are_balanced():
    for path in (ROOT / "docs").glob("*.md"):
        fence_count = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("```")
        )
        assert fence_count % 2 == 0, f"{path.name} has an unmatched Markdown code fence."


def test_distributed_docs_describe_current_backends():
    text = (ROOT / "docs" / "distributed_readiness.md").read_text(encoding="utf-8")

    assert "only\n  the local backend implemented" not in text
    assert "local,\n  Ray, and Dask implementations available" in text
