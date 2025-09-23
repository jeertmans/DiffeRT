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

## [Unreleased](https://github.com/jeertmans/DiffeRT/compare/v0.6.1...HEAD)

<!-- start changelog -->

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

- Added support for {attr}`confidence<differt.geometry.Paths.confidence>` attribute in {attr}`Paths.mask_duplicate_objects<differt.geometry.Paths.mask_duplicate_objects>` (by <gh-user:jeertmans>, in <gh-pr:272>).
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
