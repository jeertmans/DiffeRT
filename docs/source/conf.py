# noqa: INP001
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import inspect
import os
from datetime import date
from typing import Any

from sphinx.application import Sphinx

from differt import __version__

project = "DiffeRT"
copyright = f"2023-{date.today().year}, Jérome Eertmans"  # noqa: A001, DTZ011
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
    "differt_dev.sphinxext.apidoc",
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
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
apidoc_module_dirs = [
    "../../differt/src/differt",
    "../../differt-core/python/differt_core",
]
apidoc_output_dirs = "reference"
apidoc_exclude_patterns = ["conftest.py", "scene/scenes/**"]
apidoc_separate = True
apidoc_no_toc = True
apidoc_max_depth = 1
apidoc_templatedir = "source/_templates"

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

# -- Bibtex

bibtex_bibfiles = ["references.bib"]

# Patch for Plotly from https://github.com/spatialaudio/nbsphinx/issues/128#issuecomment-1158712159

html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js",
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
    "repository_url": "https://github.com/jeertmans/DiffeRT",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "navigation_with_keys": False,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
}

html_logo = "_static/logo_250px.png"
html_favicon = "_static/favicon.png"

autosummary_generate = True
napolean_use_rtype = False

# Patches

# TODO: fix Plotly's Figure not linking to docs with intersphinx,
#   reported here https://github.com/sphinx-doc/sphinx/issues/12360.


def fix_sionna_folder(_app: Sphinx, obj: Any, _bound_method: bool) -> None:  # noqa: FBT001
    """
    Rename the default folder to a more readeable name.
    """
    if obj.__name__.endswith("_sionna_scenes"):
        sig = inspect.signature(obj)
        parameters = []

        for param_name, parameter in sig.parameters.items():
            if param_name == "folder":
                parameter = parameter.replace(default="<path-to-differt>/scene/scenes")  # noqa: PLW2901

            parameters.append(parameter)

        obj.__signature__ = sig.replace(parameters=parameters)


def setup(app: Sphinx) -> None:
    app.connect("autodoc-before-process-signature", fix_sionna_folder)
