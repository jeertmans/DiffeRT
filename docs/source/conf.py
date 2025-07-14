# noqa: INP001
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import inspect
import operator
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.intersphinx import missing_reference

from differt import __version__

project = "DiffeRT"
copyright = f"2023-{date.today().year}, Jérome Eertmans"  # noqa: A001, DTZ011
author = "Jérome Eertmans"
version = __version__
git_ref = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")
conf_dir = Path(__file__).absolute().parent
root_dir = conf_dir.parent.parent

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Built-in
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    # Additional
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

nitpicky = True
nitpick_ignore = (
    ("py:class", "Array"),
    ("py:class", "differt.plotting._utils._Dispatcher"),
    ("py:class", "differt.utils.TypeVarTuple"),
    ("py:class", "jax._src.typing.SupportsDType"),
    ("py:class", "ndarray"),  # From ArrayLike
    ("py:obj", "differt.utils._T"),
    ("py:obj", "differt.rt.utils._T"),
    ("py:obj", "__main__.ArrayType"),
    ("py:class", "setup.<locals>.ArrayType"),
)

linkcheck_ignore = ["https://doi.org/10.1002/2015RS005659"]
linkcheck_report_timeouts_as_broken = False  # Default value in Sphinx >= 8

# -- MathJax settings

mathjax3_config = {
    "loader": {"load": ["[tex]/boldsymbol"]},
    "tex": {"packages": {"[+]": ["boldsymbol"]}},
}

numfig = True

# -- Intersphinx mapping

intersphinx_mapping = {
    "e3x": ("https://e3x.readthedocs.io/stable/", None),
    "flax": ("https://flax-linen.readthedocs.io/en/latest/", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mitsuba": ("https://mitsuba.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "optax": ("https://optax.readthedocs.io/en/latest", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
    "python": ("https://docs.python.org/3", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sionna": ("https://nvlabs.github.io/sionna/", None),
    "vispy": ("https://vispy.org", None),
}

# -- OpenGraph settings

ogp_site_url = "https://eertmans.be/DiffeRT/"
ogp_use_first_image = True

# -- Sphinx autodoc typehints settings

always_document_param_types = False
always_use_bars_union = True

# -- MyST-nb settings
myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "ftp": None,
    "wiki": "https://en.wikipedia.org/wiki/{{path}}#{{fragment}}",
    "doi": "https://doi.org/{{path}}",
    "gh-pr": {
        "url": "https://github.com/jeertmans/DiffeRT/pull/{{path}}#{{fragment}}",
        "title": "PR #{{path}}",
        "classes": ["github"],
    },
    "gh-issue": {
        "url": "https://github.com/jeertmans/DiffeRT/issues/{{path}}#{{fragment}}",
        "title": "Issue #{{path}}",
        "classes": ["github"],
    },
    "gh-user": {
        "url": "https://github.com/{{path}}",
        "title": "@{{path}}",
        "classes": ["github"],
    },
}

myst_heading_anchors = 3

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "html_admonition",
]

nb_execution_mode = "off" if os.environ.get("NB_OFF") else "auto"
nb_execution_timeout = (
    600  # So cells can take a long time, especially when downloading sionna scenes
)
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

# -- Linkcode settings


def linkcode_resolve(domain: str, info: dict[str, Any]) -> str | None:  # noqa: PLR0911
    if domain != "py":
        return None

    if not info["module"]:
        return None
    if not info["fullname"]:
        return None
    if info["module"].split(".", 1)[0] not in {"differt", "differt_core"}:
        return None

    try:
        mod = sys.modules.get(info["module"])
        obj = operator.attrgetter(info["fullname"])(mod)
        if isinstance(obj, property):
            obj: Any = obj.fget
        obj = inspect.unwrap(obj)
        filename = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except (AttributeError, TypeError):
        return None

    if filename is None:
        return None

    filename = os.path.relpath(filename, start=root_dir)
    lines = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    return f"https://github.com/jeertmans/DiffeRT/blob/{git_ref}/{filename}{lines}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/jeertmans/DiffeRT",
    "repository_branch": git_ref,
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


def fix_sionna_folder(_app: Sphinx, obj: Any, _bound_method: bool) -> None:
    """
    Rename the default folder to a more readeable name.
    """
    module = getattr(obj, "__module__", None)
    if module and module.rsplit(".", maxsplit=1)[-1] == "_sionna":
        sig = inspect.signature(obj)
        parameters = []

        for param_name, parameter in sig.parameters.items():
            if param_name == "folder":
                parameter = parameter.replace(default="<path-to-differt>/scene/scenes")  # noqa: PLW2901

            parameters.append(parameter)

        obj.__signature__ = sig.replace(parameters=parameters)


def fix_reference(
    app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: nodes.TextElement
) -> nodes.reference | None:
    """
    Fix some intersphinx references that are broken.
    """
    if node["refdomain"] == "py":
        if node["reftarget"].startswith(
            "equinox"
        ):  # Sphinx fails to find them in the inventory
            if node["reftarget"].endswith("Module"):
                uri = (
                    "https://docs.kidger.site/equinox/api/module/module/#equinox.Module"
                )
            elif node["reftarget"].endswith("tree_at"):
                uri = (
                    "https://docs.kidger.site/equinox/api/manipulation/#equinox.tree_at"
                )
            elif node["reftype"] == "mod":
                uri = "https://docs.kidger.site/equinox/"
            else:
                return None

            newnode = nodes.reference(
                "", "", internal=False, refuri=uri, reftitle="(in equinox)"
            )
            newnode.append(contnode)

            return newnode
        if node["reftarget"].startswith(
            "jaxtyping"
        ):  # Sphinx fails to find them in the inventory
            if node["reftype"] == "class":
                uri = "https://docs.kidger.site/jaxtyping/api/array/#dtype"
            elif node["reftype"] == "mod":
                uri = "https://docs.kidger.site/jaxtyping/"
            else:
                return None

            newnode = nodes.reference(
                "", "", internal=False, refuri=uri, reftitle="(in jaxtyping)"
            )
            newnode.append(contnode)

            return newnode
        if node["reftarget"] == "plotly.graph_objs._figure.Figure":
            node["reftarget"] = "plotly.graph_objects.Figure"
            return missing_reference(app, env, node, contnode)

    return None


def setup(app: Sphinx) -> None:
    import jaxtyping  # noqa: PLC0415
    # Patch to avoid expanding the ArrayLike union type, which takes a lot
    # of space and is less readable.

    class ArrayLike(jaxtyping.Array):
        pass

    jaxtyping.ArrayLike = ArrayLike

    from typing import TypeVar  # noqa: PLC0415

    from differt.scene import download_sionna_scenes  # noqa: PLC0415

    class ArrayType(jaxtyping.Array):
        def __repr__(self) -> str:
            return "ArrayType"

    import differt.plugins._deepmimo_types  # noqa: PLC0415

    differt.plugins._deepmimo_types.ArrayType = TypeVar("ArrayType", bound=ArrayType)  # type: ignore[generalTypeIssue]  # noqa: SLF001

    download_sionna_scenes()  # Put this here so that download does not occur during notebooks execution

    app.connect("autodoc-before-process-signature", fix_sionna_folder)
    app.connect("missing-reference", fix_reference)
