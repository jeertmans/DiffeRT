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

## [Unreleased](https://github.com/jeertmans/DiffeRT/compare/v0.1.0...HEAD)

### Added

- Added support for `confidence` attribute in `Paths.mask_duplicate_objects` (by <gh-user:jeertmans>, in <gh-pr:272>).

### Chore

- Documented how to build from sources without Rust, i.e., without building `differt_core` (by <gh-user:jeertmans>, in <gh-pr:269>).

### Fixed

- Fixed potential `IndexError` in `TriangleScene.num_{transmitters,receivers}` when the TX/RX arrays have incorrect shape (by <gh-user:jeertmans>, in <gh-pr:272>).
- Fixed potential `IndexError` in `Paths.num_valid_paths` when the `objects` array has its last axis being of zero size (by <gh-user:jeertmans>, in <gh-pr:273>).

<!-- start changelog -->

## [0.1.0](https://github.com/jeertmans/DiffeRT/tree/v0.1.0)

This version is the first important release of DiffeRT with the aim to provide
a stable and documented tool to be used by the scientific community.

Features present in this version are various, and cover way more than what is described in the initial draft
of the [v0.1.0 milestone](https://github.com/jeertmans/DiffeRT/milestone/1).

### Chore

- Created this changelog to document notable changes (by <gh-user:jeertmans>, in <gh-pr:252>).

<!-- end changelog -->
