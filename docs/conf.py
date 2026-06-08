"""Sphinx configuration for vertebrae documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "vertebrae"
copyright = "2026, Niklas Melton"
author = "Niklas Melton"

try:
    from vertebrae import __version__
except Exception:  # pragma: no cover - import-time docs fallback
    __version__ = "0.1.0"

release = __version__
version = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autoapi.extension",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
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
autoapi_keep_files = True
autoapi_generate_api_docs = True
autoapi_add_toctree_entry = False
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", {}),
}

html_title = "vertebrae"
