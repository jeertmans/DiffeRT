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

## [Unreleased](https://github.com/jeertmans/DiffeRT/compare/v0.3.0...HEAD)

<!-- start changelog -->

## [0.3.0](https://github.com/jeertmans/DiffeRT/compare/v0.2.0...v0.3.0)

### Added

- Added `gil_used = false` to PyO3 module to support free-threaded Python builds (by <gh-user:jeertmans>, in <gh-pr:294>).

### Chore

- Improved testing on free-threaded Python and optional plotting backends (by <gh-user:jeertmans>, in <gh-pr:294>).

### Removed

- Removed `differt.utils.minimize`, see <gh-pr:283>, resulting in a **breaking change** (by <gh-user:jeertmans>, in <gh-pr:291>).

## [0.2.0](https://github.com/jeertmans/DiffeRT/compare/v0.1.2...v0.2.0)

### Added

- Added {attr}`TriangleMesh.mask<differt.geometry.TriangleMesh.mask>` attribute has been added to allow triangles to be selected using a mask instead of dropping the inactive ones. This is useful for generating multiple sub-meshes of a mesh without changing the memory allocated to each sub-mesh, thus enabling efficient stacking (by <gh-user:jeertmans>, in <gh-pr:287>).
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
