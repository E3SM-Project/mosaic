# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import mosaic
from mosaic.version import __version__
from cartopy.io.shapereader import natural_earth

def download_meshes(app, env, docnames):
    """ Function to download meshes prior to executing documentation, so
        progress bars don't appear in the rendered docs
    """
    for mesh in ['QU.240km', 'mpasli.AIS8to30']:
        mosaic.datasets.open_dataset(mesh)

def download_coastlines(app, env, docnames):
    """ Function to download coastlines prior to executing documentation, so
        warnings don't appear in the rendered docs
    """
    for scale in ('110m', '50m'):
        natural_earth(resolution=scale, category='physical', name='coastline')

def setup(app):
    app.connect('env-before-read-docs', download_meshes)
    app.connect('env-before-read-docs', download_coastlines)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mosaic'
copyright = '2024, E3SM Development Team'
author = 'E3SM Development Team'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    #'myst_parser', # cannot use `myst_nb` and `myst_parser`, one or the other
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

autosummary_generate = ['developers_guide/api.md']

templates_path = ['_templates']

exclude_patterns = ["_build", ".DS_Store"]

intersphinx_mapping = {
        'cartopy': ('https://scitools.org.uk/cartopy/docs/latest/', None),
        'matplotlib': ('https://matplotlib.org/stable', None),
        'numpy': ('https://numpy.org/doc/stable', None),
        'python': ('https://docs.python.org/3/', None),
        'xarray': ('https://xarray.pydata.org/en/stable', None),
        }

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

autodoc_typehints = "none"

# Napoleon configurations
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "cartopy.crs.Projection": ":class:`cartopy.crs.CRS`",
    # objects without namespace: xarray
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    # matplotlib terms
    "matplotlib axes object": ":py:class:`matplotlib axes object <matplotlib.axes.Axes>`",
    # objects without namespace: numpy
    "ndarray": "~numpy.ndarray",
    "path-like": ":term:`path-like <path-like object>`",
    "string": ":class:`string <str>`",
}
