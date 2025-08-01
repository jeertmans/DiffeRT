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

## [Unreleased](https://github.com/jeertmans/DiffeRT/compare/v0.5.0...HEAD)

<!-- start changelog -->

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
