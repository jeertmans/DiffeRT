# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
from datetime import date

from differt import __version__

project = "DiffeRT"
copyright = f"2023-{date.today().year}, Jérome Eertmans"
author = "Jérome Eertmans"
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Built-in
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # Additional
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    "sphinxcontrib.apidoc",
    "sphinxcontrib.bibtex",
    "sphinxext.opengraph",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_plotly_directive",
    "sphinx_remove_toctrees",
]

templates_path = ["_templates"]
exclude_patterns = []

suppress_warnings = ["mystnb.unknown_mime_type"]

add_module_names = False
add_function_parentheses = False

# -- Intersphinx mapping

intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "optax": ("https://optax.readthedocs.io/en/latest", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "vispy": ("https://vispy.org", None),
}

# -- API docs settings
apidoc_module_dir = "../../python/differt"
apidoc_output_dir = "reference"
apidoc_excluded_paths = ["conftest.py"]
apidoc_separate_modules = True
apidoc_toc_file = False
apidoc_module_first = False
apidoc_extra_args = ["--maxdepth=1", "--templatedir=source/_templates"]

# -- OpenGraph settings

ogp_site_url = "https://eertmans.be/DiffeRT/"
ogp_use_first_image = True

# -- Sphinx autodoc typehints settings

always_document_param_types = True

# -- MyST-nb settings
myst_heading_anchors = 3

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "html_admonition",
]

nb_execution_mode = "off" if os.environ.get("NB_OFF") else "auto"
nb_merge_streams = True

# By default, MyST-nb chooses the Widget output instead of the 2D snapshot
# so we need to change priorities, because the widget cannot work if Python is
# not actively running.

nb_mime_priority_overrides = [
    ("*", "text/html", 0),
]

# TODO: fix JS warnings about html-manager (wrong version?)

# -- Bibtex

bibtex_bibfiles = ["references.bib"]

# Patch for Plotly from https://github.com/spatialaudio/nbsphinx/issues/128#issuecomment-1158712159

html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

# -- Matplotlib directive

plot_pre_code = """
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
"""
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False

# -- Plotly directive

plotly_pre_code = """
import jax
import jax.numpy as jnp
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
"""
plotly_include_source = True
plotly_html_show_source_link = False
plotly_html_show_formats = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "show_toc_level": 2,
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/jeertmans/DiffeRT",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "navigation_with_keys": False,
}

html_logo = "_static/logo_250px.png"
html_favicon = "_static/favicon.png"

autosummary_generate = True
napolean_use_rtype = False

# Patches

# TODO: fix Plotly's Figure not linking to docs with intersphinx.

"""
def fix_signature(app, what, name, obj, options, signature, return_annotation):
    target = "~plotly.graph_objs._figure.Figure"
    sub = ":py:class:`Figure<plotly.graph_objects.Figure>`"
    sub = "~plotly.graph_objects.Figure"
    if return_annotation and target in return_annotation:
        return_annotation = return_annotation.replace(target, sub)
        return signature, return_annotation.replace(target, sub)


def setup(app):
    app.connect("autodoc-process-signature", fix_signature, priority=999)
"""
