"""Sphinx configuration for imcluster."""

from importlib.metadata import version as package_version

project = "imcluster"
copyright = "2022-2026, Robert Turnbull"
author = "Robert Turnbull"
release = package_version("imcluster")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
