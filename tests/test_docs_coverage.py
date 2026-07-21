from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "_build",
    "build",
    "dist",
    "node_modules",
}


def _project_readmes() -> set[Path]:
    readmes: set[Path] = set()
    for directory, subdirectories, filenames in os.walk(PROJECT_ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in IGNORED_DIRECTORY_NAMES]
        if "README.md" in filenames:
            readmes.add((Path(directory) / "README.md").resolve())
    return readmes


def _included_markdown_files() -> set[Path]:
    included: set[Path] = set()
    pattern = re.compile(r"^```\{include\}\s+(.+?)\s*$", re.MULTILINE)
    for source in DOCS_ROOT.rglob("*.md"):
        if "_build" in source.parts:
            continue
        for match in pattern.finditer(source.read_text(encoding="utf-8")):
            included.add((source.parent / match.group(1)).resolve())
    return included


def _main_toctree_targets() -> set[str]:
    index_text = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    match = re.search(r"```\{toctree\}\n(?P<body>.*?)\n```", index_text, re.DOTALL)
    assert match is not None, "docs/index.md must contain a toctree"

    targets: set[str] = set()
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        explicit_target = re.search(r"<([^>]+)>$", line)
        target = explicit_target.group(1) if explicit_target else line
        targets.add(target.removesuffix(".md"))
    return targets


def test_every_project_readme_is_published_in_the_docs() -> None:
    missing = _project_readmes() - _included_markdown_files()
    assert not missing, (
        "Every project README must be included by a docs page so its content is "
        f"available on Read the Docs; missing: {sorted(map(str, missing))}"
    )


def test_every_guide_is_reachable_from_the_main_navigation() -> None:
    guide_targets = {
        str(path.relative_to(DOCS_ROOT).with_suffix(""))
        for path in DOCS_ROOT.rglob("*.md")
        if path.name != "index.md" and "_build" not in path.parts
    }
    missing = guide_targets - _main_toctree_targets()
    assert not missing, (
        "Every documentation guide must be linked from docs/index.md; "
        f"missing: {sorted(missing)}"
    )
