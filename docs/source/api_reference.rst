API Reference
=============

DiffeRT comes as two-package project: :mod:`differt`, the main Python module, that
contains most of the features, and :mod:`differt_core`, a lower-level Python module written
in Rust, for performance reason. The second module (:mod:`differt_core`) is a direct dependency
of the former (:mod:`differt`). However, you can also decide to install :mod:`differt_core` directly,
if you only needs its features.

You can find the documentation for both packages by clicking on the links below.

.. toctree::
   :maxdepth: 1

   reference/differt
   reference/differt_core
