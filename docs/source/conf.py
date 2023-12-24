# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import date

project = "DiffeRT"
copyright = f"{date.today().year}, Jérome Eertmans"
author = "Jérome Eertmans"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Built-in
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # Additional
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    "sphinxext.opengraph",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_remove_toctrees",
]

templates_path = ["_templates"]
exclude_patterns = []

suppress_warnings = ["mystnb.unknown_mime_type"]

# -- Intersphinx mapping

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # "numpy": ("https://numpy.org/doc/stable/", None),
    # "matplotlib": ("https://matplotlib.org/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    # "optax": ("https://optax.readthedocs.io/en/latest", None),
}

# -- OpenGraph settings

ogp_site_url = "https://eertmans.be/DiffeRT/"
ogp_use_first_image = True

# -- Sphinx autodoc typehints settings

always_document_param_types = True

# -- nbsphinx settings

nbsphinx_kernel_name = "DiffeRT"

# Patch for Plotly from https://github.com/spatialaudio/nbsphinx/issues/128#issuecomment-1158712159

html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/jeertmans/DiffeRT",
    "use_repository_button": True,  # add a "link to repository" button
    "navigation_with_keys": False,
}

autosummary_generate = True
napolean_use_rtype = False
