"""
An improved version of ``sphinxcontrib.apidoc``.
"""

from pathlib import Path

from sphinx.application import Sphinx
from sphinx.ext import apidoc
from sphinx.util import logging

logger = logging.getLogger(__name__)


def builder_inited(app: Sphinx) -> None:  # noqa: C901, PLR0912
    module_dirs = app.config.apidoc_module_dirs
    output_dirs = app.config.apidoc_output_dirs
    exclude_patterns = app.config.apidoc_exclude_patterns

    if isinstance(module_dirs, str):
        module_dirs: list[str] = [module_dirs]

    if isinstance(output_dirs, str):
        output_dirs: list[str] = [output_dirs for _ in module_dirs]

    if len(module_dirs) != len(output_dirs):
        msg = (
            "If you provide a list of output directories, "
            "it must have the same length as the module directories."
        )
        raise ValueError(
            msg,
        )

    options = []

    if app.config.apidoc_quiet:
        options.append("-q")

    if app.config.apidoc_force:
        options.append("--force")

    if app.config.apidoc_follow_links:
        options.append("--follow-links")

    if app.config.apidoc_dry_run:
        options.append("--dry-run")

    options.append(f"-s={app.config.apidoc_suffix}")  # noqa: FURB113
    options.append(f"-d={app.config.apidoc_max_depth}")

    if app.config.apidoc_no_toc:
        options.append("--no-toc")

    if app.config.apidoc_separate:
        options.append("--separate")

    if app.config.apidoc_no_headings:
        options.append("--no-headings")

    if app.config.apidoc_private:
        options.append("--private")

    if app.config.apidoc_implicit_namespaces:
        options.append("--implicit-namespaces")

    if app.config.apidoc_module_first:
        options.append("--module-first")

    options.append(f"--templatedir={app.config.apidoc_templatedir}")

    for module_dir_rel, output_dir_rel in zip(module_dirs, output_dirs):
        module_dir = Path(module_dir_rel)
        if not module_dir.is_absolute():
            module_dir = Path(app.srcdir) / module_dir_rel

        output_dir = Path(app.srcdir) / output_dir_rel

        args = [
            *options,
            f"-o={output_dir}",
            str(module_dir),
            *[str(module_dir / exclude_pattern) for exclude_pattern in exclude_patterns],
        ]

        apidoc.main(args)


def setup(app: Sphinx) -> None:
    app.add_config_value("apidoc_module_dirs", [], "html")
    app.add_config_value("apidoc_output_dirs", "apidoc", "html")
    app.add_config_value("apidoc_quiet", False, "")
    app.add_config_value("apidoc_force", False, "html")
    app.add_config_value("apidoc_follow_links", False, "html")
    app.add_config_value("apidoc_dry_run", False, "")
    app.add_config_value("apidoc_suffix", "rst", "html")
    app.add_config_value("apidoc_max_depth", 4, "html")
    app.add_config_value("apidoc_tocfile", "modules", "html")
    app.add_config_value("apidoc_no_toc", False, "html")
    app.add_config_value("apidoc_separate", False, "html")
    app.add_config_value("apidoc_no_headings", False, "html")
    app.add_config_value("apidoc_private", False, "html")
    app.add_config_value("apidoc_implicit_namespaces", False, "html")
    app.add_config_value("apidoc_module_first", False, "html")
    app.add_config_value("apidoc_templatedir", "_templates", "html")
    app.add_config_value("apidoc_exclude_patterns", [], "html")
    app.connect("builder-inited", builder_inited)
