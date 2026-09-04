"""Sphinx extension to automatically generate an ITU radio materials table."""

from typing import Any

from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective


class ITUMaterialsTableDirective(SphinxDirective):
    """A custom Sphinx directive to automatically generate an ITU materials table."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        """Generate the ITU materials table nodes.

        Returns:
            The list of docutils nodes to insert in the document.
        """
        from differt.em import materials  # ruff: ignore[import-outside-top-level]
        from differt.em._material import (  # ruff: ignore[import-outside-top-level,import-private-name]
            _ITU_MATERIALS_TABLE,
        )

        unique_materials = list(dict.fromkeys(materials.values()))

        rst_lines = [
            ".. _itu-materials-table:",
            "",
            ".. rubric:: ITU Radio Materials",
            "",
            "The table below lists all built-in ITU radio materials provided by ``materials``, based on :cite:`itu-r-2040` (Recommendation ITU-R P.2040-4).",
            "",
            "For each material, the relative permittivity is calculated as :math:`\\varepsilon'_r = a f_{\\text{GHz}}^b` and the conductivity as :math:`\\sigma = c f_{\\text{GHz}}^d` within the specified frequency range.",
            "",
            ".. list-table:: ITU Radio Materials (Recommendation ITU-R P.2040-4)",
            "   :header-rows: 1",
            "   :widths: 16 16 16 13 13 13 13",
            "",
            "   * - Material Name",
            "     - Aliases",
            "     - Frequency Range (GHz)",
            "     - :math:`a`",
            "     - :math:`b`",
            "     - :math:`c`",
            "     - :math:`d`",
        ]

        for mat in unique_materials:
            aliases_str = (
                ", ".join(f"``{a}``" for a in mat.aliases) if mat.aliases else "None"
            )
            itu_properties = _ITU_MATERIALS_TABLE.get(mat.name)
            if itu_properties:
                for idx, (a, b, c, d, f_range) in enumerate(itu_properties):
                    name_col = f"**{mat.name}**" if idx == 0 else ""
                    alias_col = aliases_str if idx == 0 else ""
                    if f_range is None:
                        freq_col = "All"
                    else:
                        freq_col = f"[{f_range[0]}, {f_range[1]}]"
                    rst_lines.extend([
                        f"   * - {name_col}",
                        f"     - {alias_col}",
                        f"     - {freq_col}",
                        f"     - {a}",
                        f"     - {b}",
                        f"     - {c}",
                        f"     - {d}",
                    ])

        container = nodes.container()
        container.document = self.state.document
        view = ViewList(rst_lines, source="<itu-materials-table>")
        self.state.nested_parse(view, self.content_offset, container)

        return container.children


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the itu-materials-table directive with Sphinx.

    Returns:
        The Sphinx extension metadata.
    """
    app.add_directive("itu-materials-table", ITUMaterialsTableDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
