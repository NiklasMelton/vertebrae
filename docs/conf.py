"""Sphinx configuration for vertebrae documentation."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

from pygments.lexers import TextLexer
from sphinx.highlighting import lexer_classes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "vertebrae"
copyright = "2026, Niklas Melton"
author = "Niklas Melton"

from vertebrae import __version__  # noqa: E402

release = __version__
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
# README.md is included verbatim in the overview. GitHub renders its native
# ```mermaid fences; Sphinx only needs a plain-text fallback for those blocks.
lexer_classes["mermaid"] = partial(TextLexer, stripnl=False)

suppress_warnings = ["autoapi.python_import_resolution", "ref.python"]

html_theme = "sphinx_rtd_theme"
html_logo = "../img/vertebrae_logo.png"
html_static_path = ["_static"]
html_css_files = ["logo.css"]

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
