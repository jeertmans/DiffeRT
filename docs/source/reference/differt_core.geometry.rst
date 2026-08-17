``differt_core.geometry`` module
================================

.. currentmodule:: differt_core.geometry

.. automodule:: differt_core.geometry

.. autosummary::
   :toctree: _autosummary

   Mesh
   TriangleMesh

.. rubric:: Scene

.. autosummary::
   :toctree: _autosummary

   Scene
   TriangleScene

.. rubric:: Sionna compatibility layer

Fast and low-memory functions to read Sionna scenes.

Fast because written in Rust and uses the extremely performant
`quick_xml <https://github.com/tafia/quick-xml>`_ library.

Low-memory because it only stores the minimal amount of information
to reproduce Sionna scenes, and skips the rest (e.g., display information).

As filepaths to shapes are relative to the initial XML config file,
it is preferred to directly use ``load_xml`` from another scene
class, like :class:`Scene<differt_core.geometry.Scene>`.

.. autosummary::
   :toctree: _autosummary

   Material
   Shape
   SionnaScene

.. rubric:: Graphs

.. autosummary::
   :toctree: _autosummary

   CompleteGraph
   DiGraph

.. rubric:: Iterators

.. autosummary::
   :toctree: _autosummary

   AllPathsFromCompleteGraphIter
   AllPathsFromDiGraphIter
   AllPathsFromCompleteGraphChunksIter
   AllPathsFromDiGraphChunksIter
