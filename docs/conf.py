# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from mosaic.version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mosaic'
copyright = '2024, E3SM Development Team'
author = 'E3SM Development Team'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

autosummary_generate = ['developers_guide/api.md']

templates_path = ['_templates']

exclude_patterns = ["_build", ".DS_Store"]

# -- MyST settings -----------------------------------------------------------
# copided from mache: https://github.com/E3SM-Project/mache
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath'
]
myst_number_code_blocks = ["typescript"]
myst_heading_anchors = 2
myst_footnote_transition = True
myst_dmath_double_inline = True
myst_enable_checkboxes = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/E3SM-Project/mosaic",
    "use_repository_button": True,
    "show_navbar_depth": 3
}

html_static_path = ['_static']
