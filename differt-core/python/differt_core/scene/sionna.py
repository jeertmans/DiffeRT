"""Fast and low-memory functions to read Sionna scenes.

Fast because written in Rust and uses the extremely performant
`quick_xml <https://github.com/tafia/quick-xml>`_ library.

Low-memory because it only stores the minimal amount of information
to reproduce Sionna scenes, and skips the rest (e.g., display information).

As filepaths to shapes are relative to the initial XML config file,
it is preferred to directly use ``load_xml`` from another scene
class, like :meth:`TriangleScene<differt_core.scene.triangle_scene.TriangleScene>`.
"""

__all__ = ("Material", "Shape", "SionnaScene")

from differt_core import _lowlevel

Material = _lowlevel.scene.sionna.Material
Shape = _lowlevel.scene.sionna.Shape
SionnaScene = _lowlevel.scene.sionna.SionnaScene
