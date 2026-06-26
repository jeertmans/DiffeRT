# Changelog

> [!IMPORTANT]
>
> This file uses special syntax that is only rendered properly
> within the documentation, so we recommend reading the changelog
> [here](https://differt.readthedocs.io/latest/changelog.html).

<!-- start changelog-preamble -->

All notable changes to this project will be documented on this page.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
with one *slight* but **important** difference:
- before version `1.0.0`, an increment in the MINOR version number indicates a breaking change, and an increment in the PATCH version number indicates either a new feature or a bug fix (see the [v0.1.0 milestone](https://github.com/jeertmans/DiffeRT/milestone/1));
- after version `1.0.0`, this project follows standard semantic versioning (see the [v1.0.0 milestone](https://github.com/jeertmans/DiffeRT/milestone/2)).

<!-- end changelog-preamble -->

<!-- start changelog -->

## [Unreleased](https://github.com/jeertmans/DiffeRT/compare/v0.9.0...HEAD)

### Added

- Added Warp-accelerated {meth}`TriangleScene.compute_tx_mlm<differt.scene.TriangleScene.compute_tx_mlm>` method to compute the Multipath Lifetime Map (MLM) from transmitter locations using a shooting and bouncing ray (SBR) approach, providing a much faster and lower-memory alternative to the exhaustive ray tracing approach (by <gh-user:jeertmans>, in <gh-pr:483>).

### Fixed

- Fixed a bug in {func}`fibonacci_lattice<differt.geometry.fibonacci_lattice>` when viewing frustum was provided and, in some cases, the sampled rays were not uniformly distributed within that viewing frustum (by <gh-user:jeertmans>, in <gh-pr:483>).
- Fixed {func}`fibonacci_lattice<differt.geometry.fibonacci_lattice>`'s azimuthal angle calculation at large `n` that was previously causing hatching artifacts (i.e., precision was insufficient to represent the small changes in angle) (by <gh-user:jeertmans>, in <gh-pr:483>).

## [0.9.0](https://github.com/jeertmans/DiffeRT/compare/v0.8.2...v0.9.0)

### Added

- Added {meth}`TriangleMesh.dedup_vertices<differt.geometry.TriangleMesh.dedup_vertices>` method to only renumber triangles to refer to the first occurrence of each unique vertex coordinate, thus preserving the original vertices and their ordering (by <gh-user:jeertmans>, in <gh-pr:463>).
- Added {meth}`TriangleMesh.drop_unused_vertices<differt.geometry.TriangleMesh.drop_unused_vertices>` method to remove vertices that are not referenced by any triangle (by <gh-user:jeertmans>, in <gh-pr:463>).
- Added diffraction edge detection properties (`diffraction_edges_mask`, `diffraction_edges`, `wedge_angles`, `wedge_parameters`) on {class}`TriangleMesh<differt.geometry.TriangleMesh>` to support edge adjacency, quad diagonal exclusion, non-manifold edge warnings, and convex/concave/knife-edge wedge angle classification (by <gh-user:jeertmans>, in <gh-pr:463>).
- Added Warp-accelerated methods on {class}`TriangleMesh<differt.geometry.TriangleMesh>` to significantly improve performance of ray-triangle intersections, first hit search, and visibility checks when smoothing is disabled (by <gh-user:jeertmans>, in <gh-pr:467>).

### Changed

- Removed warning message in {meth}`TriangleMesh.keep_all_within<differt.geometry.TriangleMesh.keep_all_within>` and {meth}`TriangleMesh.keep_any_within<differt.geometry.TriangleMesh.keep_any_within>` when `preserve_objects=True` is used, as the feature is fully supported and the previous warning introduced in <gh-pr:452> was unnecessary since the unexpected filtering was caused by merged mesh geometries in scene files rather than the function implementation (by <gh-user:jeertmans>, in <gh-pr:456>).
- Updated {meth}`TriangleMesh.drop_duplicates<differt.geometry.TriangleMesh.drop_duplicates>` to call both {meth}`TriangleMesh.dedup_vertices<differt.geometry.TriangleMesh.dedup_vertices>` and {meth}`TriangleMesh.drop_unused_vertices<differt.geometry.TriangleMesh.drop_unused_vertices>` in sequence (by <gh-user:jeertmans>, in <gh-pr:463>).
- Documented TPU compatibility limitations due to NVIDIA Warp integration: Warp-accelerated methods on `TriangleMesh` and `TriangleScene` do not support TPUs. Added warning notes across the codebase, updated JAX/TPU references in the documentation, and created a dedicated "Note on TPUs" documentation page detailing JAX/TPU alternatives (by <gh-user:jeertmans>, in <gh-pr:467>).

### Fixed

- Updated the `polarization` parameter in {func}`deepmimo.export<differt.plugins.deepmimo.export>` to accept a tuple of `(tx_polarization, rx_polarization)` to specify different transmitter and receiver polarizations independently (by <gh-user:jeertmans>, in <gh-pr:455>).
- Fixed power and phase calculation discrepancies in {func}`deepmimo.export<differt.plugins.deepmimo.export>` compared to Sionna RT by fixing a bug where the `radio_materials` parameter was ignored, incorporating finite-slab double-boundary formulas for ITU materials with finite thickness, correcting the receiver polarization projection in tests, and using a fully vectorized transition matrix calculation (by <gh-user:jeertmans>, in <gh-pr:455>).
- Fixed a bug in `_keep_within` (used by `keep_all_within` and `keep_any_within`) where the calculation of the active triangles count per object counted all triangles instead of active ones when checking if an object was fully kept/removed (by <gh-user:jeertmans>, in <gh-pr:456>).

### Removed

- Temporarily dropped support for free-threaded Python (i.e., `3.13t` and `3.14t`). The main dependency, [NVIDIA Warp](https://github.com/NVIDIA/warp), does not support free-threaded Python builds at this time (by <gh-user:jeertmans>, in <gh-pr:467>).

## [0.8.2](https://github.com/jeertmans/DiffeRT/compare/v0.8.1...v0.8.2)

### Added

- Improved Sionna-compatible XML scene parser to support top-level `<bsdf type="diffuse">` materials in addition to nested structures, enabling support for OSM buildings and other XML formats (by <gh-user:jeertmans>, in <gh-pr:444>).
- Added fallback to black color `[0.0, 0.0, 0.0]` when material `<rgb>` elements are missing, with appropriate warnings logged (by <gh-user:jeertmans>, in <gh-pr:444>).
- Added the {meth}`TriangleMesh.clip<differt.geometry.TriangleMesh.clip>`, {meth}`TriangleMesh.keep_all_within<differt.geometry.TriangleMesh.keep_all_within>`, and {meth}`TriangleMesh.keep_any_within<differt.geometry.TriangleMesh.keep_any_within>` methods to support clipping and filtering triangle meshes by axis-aligned bounds (by <gh-user:jeertmans>, in <gh-pr:445>).
- Added the {meth}`TriangleMesh.center<differt.geometry.TriangleMesh.center>` and {meth}`TriangleMesh.add_ground<differt.geometry.TriangleMesh.add_ground>` methods to support centering and adding a ground plane to the mesh, especially for when the ground plane is removed by any of the filtering methods (by <gh-user:jeertmans>, in <gh-pr:452>).

## Changed

- Added warning message to {meth}`TriangleMesh.keep_all_within<differt.geometry.TriangleMesh.keep_all_within>` and {meth}`TriangleMesh.keep_any_within<differt.geometry.TriangleMesh.keep_any_within>` methods when `preserve_objects=True` is used, as it is not fully supported yet (by <gh-user:jeertmans>, in <gh-pr:452>).

### Chore

- Added tests for the improved Sionna-compatible XML scene parser using OSM building data, ensuring correct parsing of materials and colors (by <gh-user:jeertmans>, in <gh-pr:444>).

### Fixed

- Unused triangle vertices are now properly removed when masking a mesh with {meth}`TriangleMesh.masked<differt.geometry.TriangleMesh.masked>`, fixing a potential plotting bugs as unused vertices may be used to compute the total visible area (by <gh-user:jeertmans>, in <gh-pr:452>).

## [0.8.1](https://github.com/jeertmans/DiffeRT/compare/v0.8.0...v0.8.1)

### Changed

- Added lighting by default when using Plotly for plotting meshes, see <gh-pr:412>  (by <gh-user:jeertmans>, in <gh-pr:432>).

### Chore

- Replaced raw GitHub issue/PR URLs in documentation (Markdown, RST, and Python docstrings) with dedicated Sphinx roles (`gh-pr`, `gh-issue`, `gh-user`, `ext-gh-issue`) (by <gh-user:copilot>, in <gh-pr:437>).

## [0.8.0](https://github.com/jeertmans/DiffeRT/compare/v0.7.0...v0.8.0)

### Added

- Added {func}`set_backend<differt.plotting.set_backend>` function to easily switch between different plotting backends, without using the more verbose {func}`set_defaults<differt.plotting.set_defaults>` function (by <gh-user:jeertmans>, in <gh-pr:387>).
- Improved type annotations for {meth}`TriangleMesh.at<differt.geometry.TriangleMesh.at>` indexing methods (`get`, `set`, `add`, `mul`, `apply`, and the new `sub`, `div`, `pow`, `min`, `max`): replaced `**kwargs: Any` with `**kwargs: Unpack[_GetIndexingKwargs]` {class}`typing.TypedDict`, and typed the `values` argument as `Float[ArrayLike, "3|1|"]` (by <gh-user:copilot> and <gh-user:jeertmans>, in <gh-pr:420>).
- Added {meth}`TriangleMesh.at<differt.geometry.TriangleMesh.at>` indexing methods `sub`, `div`, `pow`, `min`, and `max` as counterparts to JAX's `ndarray.at[...].subtract`, `divide`, `power`, `min`, and `max` (by <gh-user:copilot> and <gh-user:jeertmans>, in <gh-pr:420>).

### Changed

- Changed default options for plotting with Plotly (`aspectmode="data"` and `flatshading=True` on meshes) so that Plotly is now a much better option for large 3D scenes. Edited the tutorial accordingly, showing the improved visualization and the importance of lighting (by <gh-user:jeertmans>, in <gh-pr:412>).
- {meth}`TriangleMesh.at<differt.geometry.TriangleMesh.at>` now raises a {exc}`ValueError` if the array index passed to `at[...]` is not at most one-dimensional. This is a **breaking-change** (by <gh-user:copilot> and <gh-user:jeertmans>, in <gh-pr:420>).

### Chore

- v0.8.0 changes are now displayed on the changelog page. Users looking for the latest released changes should look at the [stable version](https://differt.readthedocs.io/stable/) of the documentation (by <gh-user:jeertmans>, in <gh-pr:391>).
- Renamed `jaxtyped` Pytest marker to `require_typechecker` (by <gh-user:jeertmans>, in <gh-pr:422>).
- Added `require_no_typechecker` Pytest marker to automatically skip tests that cannot work when type checking is enabled (by <gh-user:jeertmans>, in <gh-pr:422>).

### Fixed

- Fixed missing type annotations for {func}`assemble_paths<differt.geometry.assemble_paths>` in the documentation, caused by the `@no_type_check` decorator suppressing `typing.get_type_hints()` (by <gh-user:jeertmans>).

## [0.7.0](https://github.com/jeertmans/DiffeRT/compare/v0.6.2...v0.7.0)

### Added

- Added `polarization` parameter to {func}`deepmimo.export<differt.plugins.deepmimo.export>` (by <gh-user:jeertmans>, in <gh-pr:356>).
- Changed the type annotation of `backend` from `str` to `LiteralString`. This may be reverted in the future is `ty` support inferring literal string from equality tests (by <gh-user:jeertmans>, in <gh-pr:292>).
- Added the {meth}`TriangleMesh.shuffle<differt.geometry.TriangleMesh.shuffle>` method to easily test set-like properties of machine learning models (by <gh-user:jeertmans>, in <gh-pr:220>).

### Chore

- Removed PyOpenGL from macOS dependencies as it is no longer needed to fix VisPy not finding DLL files (by <gh-user:jeertmans>, in <gh-pr:345>).
- Fix anchor link to JAX's documentation (by <gh-user:jeertmans>, in <gh-pr:346>).
- Simplified {func}`deepmimo.export<differt.plugins.deepmimo.export>` to reduce redundant code (by <gh-user:jeertmans>, in <gh-pr:356>).
- Changed type checker from `pyright` to `ty` (by <gh-user:jeertmans>, in <gh-pr:292>).
- Slightly improved code coverage (by <gh-user:jeertmans>, in <gh-pr:362>).
- Bumped minimum required JAX version to [`0.8.1`](https://docs.jax.dev/en/latest/changelog.html#jax-0-8-1-november-18-2025) to use new {func}`jax.jit` syntax as the use of {func}`functools.partial` now raises errors from `ty`, see <ext-gh-issue:jax-ml/jax#34697> (by <gh-user:jeertmans>, in <gh-pr:370>).
- Added a generic type variable for {attr}`mask<differt.geometry.Paths.mask>` (by <gh-user:jeertmans>, in <gh-pr:349>).

### Fixed

- Restricted `ipykernel` version to `<7` to avoid compatibility issues with `jupyter_rfb`, see <ext-gh-issue:vispy/jupyter_rfb#121> (by <gh-user:jeertmans>, in <gh-pr:347>).
- Pinned `sphinx` to `<9` to avoid breakage with `sphinx-autodoc-typehints` and the Sphinx v9 release (by <gh-user:jeertmans>, in <gh-pr:352>).
- Fixed `get` method when indexing mesh with {meth}`TriangleMesh.at<differt.geometry.TriangleMesh.at>` to **not** drop duplicate indices (by <gh-user:jeertmans>, in <gh-pr:362>).

### Removed

- Removed `confidence` attribute in {class}`Paths<differt.geometry.Paths>` as it is now replaced by {attr}`mask<differt.geometry.Paths.mask>`, possibly holding floating point values. This is a **breaking-change** (by <gh-user:jeertmans>, in <gh-pr:349>).
- Removed `jnp.asarray` field converters from all classes, as it would lead to confusing type hints mismatches between the annotations and the actual types accepted by the classes' `__init__` method. This is a **breaking-change** (by <gh-user:jeertmans>, in <gh-pr:383>).

## [0.6.2](https://github.com/jeertmans/DiffeRT/compare/v0.6.1...v0.6.2)

### Changed

- Changed {func}`fermat_path_on_linear_objects<differt.rt.fermat_path_on_linear_objects>` to leverage the `fpt-jax` library {cite}`fpt-eucap2026` for better performance and faster gradient computations (by <gh-user:jeertmans>, in <gh-pr:335>).

### Chore

- Updated CI to fix builds on Windows, and also explicitly build free-threaded wheels (by <gh-user:jeertmans>, in <gh-pr:336>).

## [0.6.1](https://github.com/jeertmans/DiffeRT/compare/v0.6.0...v0.6.1)

### Chore

- Bumped minimum required JAX version to [`0.7.2`](https://docs.jax.dev/en/latest/changelog.html#jax-0-7-2-september-16-2025) as JAX `0.7.0` and `0.7.1` contained bugs (by <gh-user:jeertmans>, in <gh-pr:325>).
- Added Python 3.14 and 3.14t to the list of tested Python versions (by <gh-user:jeertmans>, in <gh-pr:323>).
- Update the macOS runners (by <gh-user:jeertmans>, in <gh-pr:323>).
- Updated PyPI's Trove classifiers to list Python 3.14 and free-threaded Python (by <gh-user:jeertmans>, in <gh-pr:323>).

## [0.6.0](https://github.com/jeertmans/DiffeRT/compare/v0.5.0...v0.6.0)

### Added

- Added {func}`update_defaults<differt.plotting.update_defaults>`, see [below](#fixed-update-defaults) (by <gh-user:jeertmans>, in <gh-pr:312>).
- [Added the possibility to pass {data}`None` for the `batch_size` argument]{#ray-triangle-batch-size-none} of {func}`rays_intersect_any_triangle<differt.rt.rays_intersect_any_triangle>`, {func}`triangles_visible_from_vertices<differt.rt.triangles_visible_from_vertices>`, and {func}`first_triangles_hit_by_rays<differt.rt.first_triangles_hit_by_rays>`, to indicate that no batching should be performed, i.e., all operations are executed in a single {func}`jax.vmap` call (by <gh-user:jeertmans>, in <gh-pr:310>).
- Added a `batch_size` argument to {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>` to allow users to specify the size of the batch used for ray-triangle intersection tests, see [above](#ray-triangle-batch-size-none) (by <gh-user:jeertmans>, in <gh-pr:310>).
- Added the {meth}`DiGraph.filter_by_mask<differt_core.rt.DiGraph.filter_by_mask>` to disconnected nodes based on a mask (by <gh-user:jeertmans>, in <gh-pr:322>).
- Added a `disconnect_inactive_triangles` to {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>` to allow reducing the number of path candidates, at the cost of potential recompilations (by <gh-user:jeertmans>, in <gh-pr:322>).
- Added an `active_vertices` argument to {func}`viewing_frustum<differt.geometry.viewing_frustum>` to allow users to specify which vertices are active (by <gh-user:jeertmans>, in <gh-pr:322>).

### Changed

- Changed {meth}`DiGraph.disconnect_nodes<differt_core.rt.DiGraph.disconnect_nodes>` to raise an {class}`IndexError` when the node indices are out of bounds (by <gh-user:jeertmans>, in <gh-pr:322>).
- Changed the behavior of {func}`first_triangles_hit_by_rays<differt.rt.first_triangles_hit_by_rays>` to select the triangle with the closest center to the ray origin when two or more triangles are hit at the same distance (by <gh-user:jeertmans>, in <gh-pr:322>).
- Changed the behavior of {func}`first_triangles_hit_by_rays<differt.rt.first_triangles_hit_by_rays>` to include the triangle centers in the world vertices when computing the viewing frustum (by <gh-user:jeertmans>, in <gh-pr:322>).

### Chore

- Rephrased the documentation of methods returning shallow copies to clarify that they return new instances, and do not necessarily copy inner arrays (by <gh-user:jeertmans>, in <gh-pr:307>).
- Fixed plotting issue in the coherence example notebook, where the scene in the second row was not plotted correctly, see [below](#fixed-update-defaults) (by <gh-user:jeertmans>, in <gh-pr:312>).
- Added `jaxtyped` Pytest marker to automatically skip tests that require jaxtyping when it is disabled (by <gh-user:jeertmans>, in <gh-pr:321>).
- Bumped minimum required JAX version to [`0.7.0`](https://docs.jax.dev/en/latest/changelog.html#jax-0-7-0-july-22-2025) to use `wrap_negative_indices=False` with {attr}`at<jax.numpy.ndarray.at>` (by <gh-user:jeertmans>, in <gh-pr:310>).
- Dropped Python 3.10 because we need JAX 0.7.0. This is a **breaking-change** (by <gh-user:jeertmans>, in <gh-pr:310>).

### Fixed

- [Fixed the update of default values in context managers]{#fixed-update-defaults} to actually merge the new values with the existing ones, instead of replacing them, allowing for the nesting multiple context manager without any surprise (by <gh-user:jeertmans>, in <gh-pr:312>).
- Fixed a typo in {func}`viewing_frustum<differt.geometry.viewing_frustum>` that led to incorrect behavior (by <gh-user:jeertmans>, in <gh-pr:322>).
- Fixed a typo in {ref}`conventions` where the azimuth angle was incorrectly described to be in  {math}`[0^\circ, 360^\circ]` instead of {math}`[-180^\circ, 180^\circ]` (by <gh-user:jeertmans>, in <gh-pr:322>).

### Perf

- Changed naive indexing by customized {attr}`at<jax.numpy.ndarray.at>` indexing to enable niche optimizations (by <gh-user:jeertmans>, in <gh-pr:308>).

### Removed

- Removed `differt.utils.sorted_array2`, `differt.utils.dot`, `differt.geometry.pairwise_cross`,`differt.geometry.TriangleMesh.sort` to reduce the size of the API by limiting it to RT-related functionalities. This is a **breaking-change** (by <gh-user:jeertmans>, in <gh-pr:309>).

## [0.5.0](https://github.com/jeertmans/DiffeRT/compare/v0.4.1...v0.5.0)

### Chore

- Fixed typos left in documents with the help of Copilot (for most) and LanguageTool (for some) (by <gh-user:jeertmans>, in <gh-pr:304>).

### Fixed

- Fixed a shape (and possibly `dtype`) issue in the fast path of {func}`consecutive_vertices_are_on_same_side_of_mirrors<differt.rt.consecutive_vertices_are_on_same_side_of_mirrors>`, that would raise an error when trying to stack arrays in {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>` with a non-{data}`None` value for `smoothing_factor` (by <gh-user:jeertmans>, in <gh-pr:303>).

### Removed

- Removed `parallel` keyword argument in {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>` as it was no longer supported, and its presence increased the code complexity. Executing code on multiple devices should be automatically handled by {func}`jax.jit`, or manually specified by the end-user. This is a **breaking-change** (by <gh-user:jeertmans>, in <gh-pr:305>).

## [0.4.1](https://github.com/jeertmans/DiffeRT/compare/v0.4.0...v0.4.1)

### Added

- Added `batch_size` optional keyword argument to {func}`rays_intersect_any_triangle<differt.rt.rays_intersect_any_triangle>`, {func}`triangles_visible_from_vertices<differt.rt.triangles_visible_from_vertices>`, and {func}`first_triangles_hit_by_rays<differt.rt.first_triangles_hit_by_rays>`, see [below](#ray-triangle-perf-1) (by <gh-user:jeertmans>, in <gh-pr:300>).

### Chore

- Refactored {func}`image_method<differt.rt.image_method>` and {func}`fermat_path_on_linear_objects<differt.rt.fermat_path_on_linear_objects>` to use {func}`jnp.vectorize<jax.numpy.vectorize>` instead of a custom but complex chain of calls to {func}`jax.vmap`, reducing the code complexity while not affecting performance (by <gh-user:jeertmans>, in <gh-pr:298>).
- Ignored lints PLR091* globally, instead of per-case (by <gh-user:jeertmans>, in <gh-pr:298>).
- Improved code coverage for ray-triangle intersection tests (by <gh-user:jeertmans>, in <gh-pr:301>).
- Refactored benchmarks to reduce the number of benchmarks and avoid depending on JIT compilation (by <gh-user:jeertmans>, in <gh-pr:301>).

### Fixed

- Fixed VisPy plotting utilities by returning early if the data to be drawn is empty, avoiding potential issue when calling {meth}`view.camera.set_range<vispy.scene.cameras.base_camera.BaseCamera.set_range>` (by <gh-user:jeertmans>, in <gh-pr:300>).

### Perf

- [Improved performance for ray-triangle intersection tests]{#ray-triangle-perf-1} (i.e., {func}`rays_intersect_any_triangle<differt.rt.rays_intersect_any_triangle>`, {func}`triangles_visible_from_vertices<differt.rt.triangles_visible_from_vertices>`, and {func}`first_triangles_hit_by_rays<differt.rt.first_triangles_hit_by_rays>`) by implementing a custom, batched, scan-like check. This avoids having to loop over all triangles (or rays) sequentially while preventing out-of-memory issues. A new `batch_size` argument is now available for these functions, allowing users to customize the size of each batch (by <gh-user:jeertmans>, in <gh-pr:300>).

## [0.4.0](https://github.com/jeertmans/DiffeRT/compare/v0.3.1...v0.4.0)

### Added

- Added `sample_objects` to {meth}`TriangleMesh.sample<differt.geometry.TriangleMesh.sample>` to facilitate sampling *realistic* sub-meshes. The new option is compatible with both `by_masking=False` and `by_masking=True`, offering a {func}`jax.jit`-compatible sampling method with the latter (by <gh-user:jeertmans>, in <gh-pr:297>).

### Changed

- Renamed `TriangleMesh.num_objects` to {attr}`TriangleMesh.num_primitives<differt.geometry.TriangleMesh.num_primitives>` to avoid possible confusion with {attr}`TriangleMesh.object_bounds<differt.geometry.TriangleMesh.object_bounds>`, resulting in a **breaking change** (by <gh-user:jeertmans>, in <gh-pr:297>).

### Chore

- Ignored lints C901 and PLR0912 globally, instead of per-case (by <gh-user:jeertmans>, in <gh-pr:297>).

## [0.3.1](https://github.com/jeertmans/DiffeRT/compare/v0.3.0...v0.3.1)

### Added

- Implemented `method = 'hybrid'` for {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>` (by <gh-user:jeertmans>, in <gh-pr:295>).

### Chore

- Reduced computational time of higher-order RT tests by leveraging visibility matrices (by <gh-user:jeertmans>, in <gh-pr:294>).

### Fixed

- Fixed typo (missing `f`-string prefix) in error message inside {func}`deepmimo.export<differt.plugins.deepmimo.export>` (by <gh-user:jeertmans>, in <gh-pr:294>).

## [0.3.0](https://github.com/jeertmans/DiffeRT/compare/v0.2.0...v0.3.0)

### Added

- Added `gil_used = false` to PyO3 module to support free-threaded Python builds (by <gh-user:jeertmans>, in <gh-pr:293>).

### Chore

- Improved testing on free-threaded Python and optional plotting backends (by <gh-user:jeertmans>, in <gh-pr:293>).

### Removed

- Removed `differt.utils.minimize`, see <gh-pr:283>, resulting in a **breaking change** (by <gh-user:jeertmans>, in <gh-pr:291>).

## [0.2.0](https://github.com/jeertmans/DiffeRT/compare/v0.1.2...v0.2.0)

### Added

- Added {attr}`TriangleMesh.mask<differt.geometry.TriangleMesh.mask>` attribute to allow triangles to be selected using a mask instead of dropping the inactive ones. This is useful for generating multiple sub-meshes of a mesh without changing the memory allocated to each sub-mesh, thus enabling efficient stacking (by <gh-user:jeertmans>, in <gh-pr:287>).
- Added a new `by_masking: bool = False` keyword-only parameter to {meth}`TriangleMesh.sample<differt.geometry.TriangleMesh.sample>` to allow sampling sub-meshes by setting the mask array, instead of dropping triangles (by <gh-user:jeertmans>, in <gh-pr:287>).
- Added a new optional `active_triangles: Array | None = None` parameter to {func}`rays_intersect_any_triangle<differt.rt.rays_intersect_any_triangle>`, {func}`triangles_visible_from_vertices<differt.rt.triangles_visible_from_vertices>`, and {func}`first_triangles_hit_by_rays<differt.rt.first_triangles_hit_by_rays>` (by <gh-user:jeertmans>, in <gh-pr:287>).
- Added `__version_info__` tuple to {mod}`differt` and {mod}`differt_core` (by <gh-user:jeertmans>, in <gh-pr:288>).

### Changed

- Simplified {func}`assemble_paths<differt.geometry.assemble_paths>`'s signature to assume a 2- (TX-RX) or 3-argument (TX-PATH-RX) form is actually sufficient, resulting in a **breaking change** (by <gh-user:jeertmans>, in <gh-pr:289>).

### Fixed

- Fixed `__all__` in {mod}`differt` to re-export `__version__` and not `VERSION` (by <gh-user:jeertmans>, in <gh-pr:288>).

## [0.1.2](https://github.com/jeertmans/DiffeRT/compare/v0.1.1...v0.1.2)

### Changed

- Deprecated `differt.utils.minimize` in favor of specialized implementations, see <gh-pr:283> for motivation and migration information (by <gh-user:jeertmans>, in <gh-pr:283>).
- Changed the default `optimizer` used by {func}`fermat_path_on_linear_objects<differt.rt.fermat_path_on_linear_objects>` to be {func}`optax.lbfgs` (by <gh-user:jeertmans>, in <gh-pr:272>).

### Fixed

- Fixed {class}`ValueError` raised when using `parallel` mode in {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>` with `jax>=0.6` by disabling it (see <gh-issue:280>). This is a *soft* **breaking change** as it will raise a warning (by <gh-user:jeertmans>, in <gh-pr:281>). Using JAX v0.6 (and above) is now allowed again.

## [0.1.1](https://github.com/jeertmans/DiffeRT/compare/v0.1.0...v0.1.1)

### Added

- Added support for `confidence` attribute in {attr}`Paths.mask_duplicate_objects<differt.geometry.Paths.mask_duplicate_objects>` (by <gh-user:jeertmans>, in <gh-pr:272>).
- Added the {attr}`Paths.shape<differt.geometry.Paths.shape>` class attribute (by <gh-user:jeertmans>, in <gh-pr:267>).
  The following equality should always hold: `paths.reshape(*batch).shape = batch`.
- Added the {mod}`differt.plugins` package and {mod}`differt.plugins.deepmimo` module (by <gh-user:jeertmans>, in <gh-pr:267>).
- Added export utility to the [DeepMIMO](https://github.com/DeepMIMO) format (by <gh-user:jeertmans>, in <gh-pr:267>).
- Added {meth}`from_mitsuba<differt.scene.TriangleScene.from_mitsuba>` and {meth}`from_sionna<differt.scene.TriangleScene.from_sionna>` methods to the {class}`TriangleScene<differt.scene.TriangleScene>` class (by <gh-user:jeertmans>, in <gh-pr:267>).

### Chore

- Documented how to build from sources without Rust, i.e., without building {mod}`differt_core` (by <gh-user:jeertmans>, in <gh-pr:269>).
- Fixed link issues (`jnp.{minimum,maximum}` and false-positive on DOI check) (by <gh-user:jeertmans>, in <gh-pr:274>).

### Fixed

- Fixed potential {class}`IndexError` in {attr}`TriangleScene.num_{transmitters,receivers}<differt.scene.TriangleScene.num_transmitters>` when the TX/RX arrays have incorrect shape (by <gh-user:jeertmans>, in <gh-pr:272>).
- Fixed potential {class}`IndexError` in {attr}`Paths.num_valid_paths<differt.geometry.Paths.num_valid_paths>` when the {attr}`objects<differt.geometry.Paths.objects>` array has its last axis being of zero size (by <gh-user:jeertmans>, in <gh-pr:273>).

## [0.1.0](https://github.com/jeertmans/DiffeRT/tree/v0.1.0)

This version is the first important release of DiffeRT with the aim to provide
a stable and documented tool to be used by the scientific community.

Features present in this version are various, and cover way more than what is described in the initial draft
of the [v0.1.0 milestone](https://github.com/jeertmans/DiffeRT/milestone/1).

### Chore

- Created this changelog to document notable changes (by <gh-user:jeertmans>, in <gh-pr:252>).

<!-- end changelog -->
