"""Sphinx configuration for vertebrae documentation."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 local docs builds
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "vertebrae"
copyright = "2026, Niklas Melton"
author = "Niklas Melton"

with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as pyproject_file:
    pyproject = tomllib.load(pyproject_file)

release = pyproject["tool"]["poetry"]["version"]
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autoapi.extension",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["autoapi.python_import_resolution", "ref.python"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "replacements",
    "smartquotes",
]
myst_heading_anchors = 3

autodoc_typehints = "description"
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autoapi_type = "python"
autoapi_dirs = [str(Path(__file__).resolve().parents[1] / "src" / "vertebrae")]
autoapi_root = "api"
autoapi_keep_files = False
autoapi_generate_api_docs = True
autoapi_add_toctree_entry = False
autoapi_python_class_content = "class"
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
]

html_title = "vertebrae"
