``differt_core.scene`` module
=============================

.. currentmodule:: differt_core.scene

.. automodule:: differt_core.scene

.. rubric:: Triangle scene

.. autosummary::
   :toctree: _autosummary

   TriangleScene

.. rubric:: Sionna compatibility layer

Fast and low-memory functions to read Sionna scenes.

Fast because written in Rust and uses the extremely performant
`quick_xml <https://github.com/tafia/quick-xml>`_ library.

Low-memory because it only stores the minimal amount of information
to reproduce Sionna scenes, and skips the rest (e.g., display information).

As filepaths to shapes are relative to the initial XML config file,
it is preferred to directly use ``load_xml`` from another scene
class, like :class:`TriangleScene<differt_core.scene.TriangleScene>`.

.. autosummary::
   :toctree: _autosummary

   Material
   Shape
   SionnaScene
