# DiffeRT Implementation Plan: Non-Specular Interactions, Near-Field Extensibility, Top-Level API

## Progress (as of 2026-09-04)

- **Phase 1 — Geometry API Expansion**: substantially complete (steps 1-4, 6). Step 5 (SBR Warp kernel extension) explicitly deferred — see note below.
- **Phase 2 — EM & Solver Extensibility**: not started.
- **Phase 3 — Top-Level API**: complete.

All work is implemented, tested (2254 tests passing, zero regressions), linted (`just lint` clean), and documented per the repo's checklist, on branch `refactor-architecture`. Nothing has been committed yet.

This plan covers three areas identified in an architectural review of `differt`/`differt-core`:
geometry support for non-specular interactions (diffraction, scattering, transmission),
near-field/wavefront-curvature extensibility in the EM solver, and a curated top-level API.

## Not pursuing: Warp mesh/BVH caching

An earlier draft of this plan included a "Foundation" phase to reintroduce Warp `wp.Mesh`
(BVH) caching across calls — via either a user-managed `MeshBVH.build()`/`refit()` handle,
or a cache living inside the `wp.jax_callable`-wrapped callback (keyed on concrete
shapes/hashes at actual-execution time, avoiding the identity issues of the previous
attempt). This has already been benchmarked and rejected: it doesn't pay for its added
complexity, consistent with `CHANGELOG.md`'s note on commit `6e67c5c` ("proved to offer
little to no performance benefit, both on CPU and GPU platforms, at the cost of code
complexity"). Every `wp.Mesh` construction below stays as-is (rebuilt per call, or per
`jax_callable` invocation for traced meshes). Do not revisit this without new profiling
evidence.

---

## Phase 1 — Geometry API Expansion (non-specular interactions)

**Goal**: let `Scene.trace_paths` generate and solve paths that reflect, diffract,
scatter, or transmit, selected via a single `allowed_interactions: frozenset[InteractionType]`
argument, replacing today's `specular_reflection`/`diffuse_scattering` booleans.

**Why this is mostly wiring, not new physics**: `TracedPaths.interaction_types` already
exists and is typed against `differt.em.InteractionType`
(`differt/src/differt/geometry/_paths.py:213-218`). `GeometricFieldSolver` already
implements REFLECTION, DIFFRACTION (UTD), SCATTERING (Lambertian/directive), and
TRANSMISSION (slab) physics, dispatching per-bounce via a static
`supported_interaction_types` frozenset (`differt/src/differt/em/_solvers.py`). `Mesh`
already computes diffraction edges, wedge angles/parameters, and edge connectivity
(`_mesh.py:1055-1241`). `fermat_path_on_linear_objects`/`fermat_path_on_planar_mirrors`
already solve for an exact, differentiable path through arbitrary linear objects —
1-vector "objects" for edges, 2-vector for planes (`_solver_fermat.py`), via the
`fpt-jax` custom-VJP optimizer. None of this is reachable today because path *tracers*
only ever emit `interaction_types = 0` (reflection), and `diffuse_scattering` is a
documented no-op (`_solvers.py:101-102`).

### Steps

- [x] **Split `geometry/_solvers.py`** (~1950 lines) into a `geometry/solvers/` subpackage
   (`base.py` for `AbstractPathSolver`/`AbstractPathTracer`/`AbstractPathLauncher`,
   `exhaustive.py`, `hybrid.py`, `sbr.py`) before adding the new dispatch logic below —
   purely organizational, no behavior change, done first because the new code needs a
   home.

- [x] **Unify the path-candidate "node" into an interaction-site index.** Rust's
   `CompleteGraph`/`DiGraph` (`differt-core/src/geometry/graph.rs`) already just
   enumerate simple paths over abstract integer node indices — they don't need to
   change. Build a richer node universe in Python instead:

   ```python
   class InteractionSites(eqx.Module):
       """Flat, unified index space of everything a ray can interact with."""
       kind: Int[Array, " num_sites"]        # InteractionType per site
       primitive: Int[Array, " num_sites"]   # index into mesh.triangles or mesh.diffraction_edges

   def build_interaction_sites(
       mesh: Mesh, allowed_interactions: frozenset[InteractionType]
   ) -> InteractionSites:
       """Faces -> {REFLECTION, SCATTERING, TRANSMISSION}; unique edges -> {DIFFRACTION}."""
       ...
   ```

   `CompleteGraph(sites.kind.shape[0])` / `DiGraph.filter_by_mask(...)` then enumerate
   over this site universe exactly as today over `mesh.num_primitives`.

   > Implemented in `differt/src/differt/geometry/_interaction_sites.py`
   > (`InteractionSites`, `build_interaction_sites`, `interaction_sites_valid_mask`,
   > `interaction_sites_mesh_mask`). Diffraction sites use a flat half-edge index
   > (`3 * triangle_index + local_edge_index`, matching `Mesh.wedge_angles`), not
   > deduplicated `mesh.diffraction_edges`, to stay `jax.jit`-compatible.

- [x] **Replace boolean flags with `allowed_interactions`** in
   `AbstractPathTracer.generate_path_candidates` and `Scene.trace_paths`:

   ```python
   from differt.em import SpecularReflection, Diffraction  # aliases, see Phase 3

   paths = scene.trace_paths(
       order=2,
       allowed_interactions=frozenset({SpecularReflection, Diffraction}),
   )
   ```

   `order=0` (line-of-sight) is always generated regardless of `allowed_interactions` —
   it is the trivial candidate, not an interaction. Keep this to a flat `frozenset`
   applied uniformly to every bounce position for now; don't build per-order interaction
   sets (`Mapping[int, frozenset[InteractionType]]`) speculatively — revisit only if a
   concrete use case needs it.

   > Implemented exactly as sketched, threaded through `Scene.trace_paths`,
   > `Scene.trace_fields`, `AbstractPathTracer.{generate_path_candidates,trace_paths}`,
   > `ExhaustivePathTracer`, and `HybridPathTracer`. `trace_path_candidates` also
   > gained an `allowed_interactions` parameter (not in the original sketch) — it
   > needs to know, ahead of any traced computation, whether DIFFRACTION/TRANSMISSION
   > may be present, to pick the right (Python-static) geometric solver branch.

- [x] **Dispatch per-bounce geometric solving** in `_trace_path_candidates`, mirroring the
   static-set dispatch pattern already used by `GeometricFieldSolver.transition_matrices`:

   ```python
   def _solve_bounce(kind, primitive, prev, next_, mesh):
       reflect = lambda: image_method(prev, next_, mesh.triangle_vertices[primitive, 0], mesh.normals[primitive])
       diffract = lambda: fermat_path_on_linear_objects(prev, next_, mesh.diffraction_edges[primitive, 0], edge_dir[primitive])
       transmit = lambda: intersection_of_ray_with_plane(prev, next_ - prev, mesh.triangle_vertices[primitive, 0], mesh.normals[primitive])
       return jax.lax.switch(kind, [reflect, diffract, transmit], ...)
   ```

   REFLECTION and SCATTERING are geometrically identical (bounce off the same mirror
   plane) and both route through `image_method`; DIFFRACTION routes through the
   already-implemented `fermat_path_on_linear_objects`; TRANSMISSION is a direct
   ray-plane intersection with no direction change (current slab model).
   `interaction_types` on the output `TracedPaths` is then read directly from
   `InteractionSites.kind[path_candidates]` instead of being hardcoded to zero.

   > **Deviation from the sketch above**: a literal per-bounce `jax.lax.switch`
   > over `image_method`/`fermat_path_on_linear_objects` is not mathematically
   > sound — both solve a path's bounces *jointly* (each bounce's position
   > depends on the whole sequence), not independently. The actual
   > implementation (`geometry/solvers/_dispatch.py::solve_mixed_interaction_paths`)
   > stably reorders each candidate so bending bounces (REFLECTION/SCATTERING/
   > DIFFRACTION) come first, solves them in one joint call (reusing the
   > existing trailing-placeholder receiver-collapse trick for the rest), then
   > splices each TRANSMISSION bounce's position in afterward as the
   > intersection of the straight segment between its two nearest solved
   > neighbors with its transmissive face (TRANSMISSION does not bend the ray,
   > so it cannot be solved jointly with the bends — this was confirmed with
   > the user before implementing). Validity checks in `_trace_path_candidates`
   > were extended to match: DIFFRACTION gets a new finite-edge-segment
   > membership check (replacing the in-triangle check); the same-side-of-mirror
   > check is skipped for DIFFRACTION/TRANSMISSION. The reflection-only default
   > path is untouched code, verified byte-for-byte identical.

- [ ] **Extend `SBRPathTracer`/`SBRPathLauncher`'s Warp kernels** to optionally continue
   rays through diffraction edges and transmissive faces during shooting-and-bouncing,
   gated by the same `allowed_interactions`.

   > **Deferred.** This is a separate, GPU-kernel-level undertaking (rewriting
   > the Warp `wp.kernel` bounce loop) that can't be meaningfully tested on this
   > CPU-only environment. `SBRPathTracer.generate_path_candidates` now validates
   > `allowed_interactions` and raises `NotImplementedError` (pointing users at
   > `ExhaustivePathTracer`/`HybridPathTracer`) instead of silently ignoring the
   > request, so the gap is explicit rather than a silent no-op.

- [x] **Close the loop end to end**: once step 4 lands, `GeometricFieldSolver` (which
   already consumes non-zero `interaction_types`) starts receiving real
   DIFFRACTION/SCATTERING/TRANSMISSION paths for the first time — add integration tests
   in `differt/tests/geometry/` and `differt/tests/em/` that exercise
   `Scene.trace_paths(..., allowed_interactions=...)` end to end through
   `Scene.trace_fields`.

   > `differt/tests/geometry/test_non_specular_interactions.py`: DIFFRACTION-only,
   > TRANSMISSION-only, and DIFFRACTION-then-TRANSMISSION paths through
   > `Scene.trace_fields`, checked for finite field values; plus an SBR
   > rejection test and a "default is still reflection-only" backward-compat
   > check. Also `test_interaction_sites.py`, `test_dispatch.py`, and
   > `test_mixed_interactions.py` for the lower-level pieces.

---

## Phase 2 — EM & Solver Extensibility (near-field / wavefront curvature)

**Goal**: propagate wavefront curvature per-path, per-bounce, so astigmatic (near-field)
spreading composes correctly across mixed interaction types, and let users plug in new
interaction physics without subclassing the whole solver.

**Current state**: this is further along than it looks from the outside.
`GeometricFieldSolver.tx_wavefront_radii` (planar/spherical/astigmatic),
`AbstractAntenna.wavefront_radii(k_hat)`, and a fully general astigmatic UTD distance
parameter in `_utd.py`'s `L_i` already exist (PR #460). The actual gap: curvature is a
**solver-level constant** fixed at the transmitter, not a value transported along the
path — `GeometricFieldSolver` explicitly raises `NotImplementedError` for astigmatic +
diffraction combined, because there's no mechanism today to carry an evolving curvature
state through mixed interaction types (reflection off a curved surface changes
principal radii; diffraction changes cylindrical spreading; free-space propagation just
adds path length).

> **Correction found while implementing Phase 1**: the astigmatic+diffraction
> restriction is not a literal `raise NotImplementedError` — it's a runtime
> assertion via `equinox.error_if` (JIT-compatible), gated on
> `paths.interaction_types == InteractionType.DIFFRACTION` being actually
> present, in both `diffraction_matrix` and `spreading_factor`
> (`differt/src/differt/em/_solvers.py`). Worth knowing before touching that
> code: Phase 1's new DIFFRACTION paths now reach it for the first time and
> will trip this assertion if combined with an astigmatic `tx_wavefront_radii`
> — expected, not a Phase 1 bug.

**Status: complete.**

### Steps

- [x] **Add `propagate_wavefront`**, a wavefront curvature tracking module (`differt.em._wavefront.py`), bridging geometry → EM without making `Mesh`/path tracers EM-aware:

    ```python
    class WavefrontState(eqx.Module):
        radii: Float[Array, "*batch 2"]
        axes: Float[Array, "*batch 2 3"]
        is_planar: Bool[Array, "*batch 2"]

    class PathWavefront(eqx.Module):
        state: WavefrontState
        incident_radii: Float[Array, "*batch order 3"]
        spreading_factor: Float[Array, " *batch"]
        segment_radii: Float[Array, "*batch num_segments 2"]

    def propagate_wavefront(
        paths: TracedPaths, mesh: Mesh, tx_wavefront: Any = 0.0
    ) -> PathWavefront:
        ...
    ```

    > Implemented in `differt/src/differt/em/_wavefront.py` (`WavefrontState`,
    > `PathWavefront`, `propagate_wavefront`). Supports free-space propagation,
    > planar reflection, transmission, and Kouyoumjian-Pathak/McNamara straight-edge
    > diffraction curvature transport. Unit tested in `test_wavefront.py`.

- [x] **Let `GeometricFieldSolver.compute_fields` accept either** a scalar/tuple constant
    (today's default, unchanged) or a full per-path radii array / `WavefrontState`, for
    genuinely path-dependent near-field spreading.

    > Implemented in `differt/src/differt/em/_solvers.py`, forwarded from
    > `compute_received_fields` and `TracedFields.from_paths`.

- [x] **Remove the astigmatic + diffraction `NotImplementedError`** once curvature
    transport is generalized across interaction types.

    > Removed `error_if` for astigmatic wavefront with diffraction in both
    > `diffraction_matrix` and `spreading_factor`. Astigmatic diffraction now
    > evaluates exact edge-fixed plane curvature $\rho_e^i$ and distance parameter
    > $L_i$, tested in `TestNonPlanarWavefront.test_astigmatic_with_diffraction_computes_fields`.

- [x] **Add a pluggable interaction registry** so new physics (RIS, a research-only
    backscatter model) don't require subclassing `GeometricFieldSolver`, mirroring the
    composition-over-inheritance pattern already used by `Material.scattering_pattern`:

    ```python
    solver = GeometricFieldSolver(
        radio_materials=materials,
        interaction_matrices={
            InteractionType.RIS: my_ris_matrix_fn,
        },
    )
    ```

    > Implemented via `interaction_matrices: Mapping[InteractionType | int, Any] | None = None`
    > on `GeometricFieldSolver`, dynamically merged and dispatched in `transition_matrices`.
    > Tested in `TestGeometricFieldSolver.test_pluggable_interaction_matrices`.

---

## Phase 3 — Top-Level API

**Goal**: let common names be imported directly from `differt`, and give
`allowed_interactions` ergonomic aliases, without introducing an eager circular import
between `differt.geometry` and `differt.em` (which already exists today — see
`Scene.load_xml`/`Scene.trace_fields`'s inline `from differt.em import ...`,
`differt/src/differt/geometry/_scene.py:558,913`).

**Status: complete.**

### Steps

- [x] **Define interaction-type aliases once**, in `differt.em`:

   ```python
   # differt/src/differt/em/_interaction_type.py
   SpecularReflection = InteractionType.REFLECTION
   Diffraction = InteractionType.DIFFRACTION
   Scattering = InteractionType.SCATTERING
   Transmission = InteractionType.TRANSMISSION
   RIS = InteractionType.RIS
   ```

   Re-export from `differt.em.__init__`.

- [x] **Add lazy top-level re-exports** in `differt/__init__.py` via module `__getattr__`
   (PEP 562, the numpy/scipy idiom), which sidesteps the geometry↔em import cycle and
   keeps `import differt` cheap (no `warp`/`fpt_jax` load until a geometry name is
   actually touched):

   ```python
   # differt/src/differt/__init__.py
   from ._version import __version__, __version_info__

   _LAZY = {
       "Scene": ".geometry", "Mesh": ".geometry", "TracedPaths": ".geometry", "LaunchedPaths": ".geometry",
       "Material": ".em", "InteractionType": ".em",
       "SpecularReflection": ".em", "Diffraction": ".em", "Scattering": ".em", "Transmission": ".em", "RIS": ".em",
   }

   def __getattr__(name: str):
       if mod := _LAZY.get(name):
           import importlib
           return getattr(importlib.import_module(mod, __name__), name)
       raise AttributeError(name)
   ```

- [x] **Update the quickstart tutorial** to show the shorter top-level imports alongside
   the existing submodule-qualified ones (additive, non-breaking).

   > Added a "A note on imports" cell pair right after the main imports cell,
   > asserting `dt.Scene is Scene` (aliased import, to avoid colliding with the
   > notebook's fixed Colab-install preamble cell, which also does a bare
   > `import differt`).

- [x] Follow the repo's standard checklist for every step above: jaxtyping-annotated
   signatures, Google-style docstrings with Sphinx cross-refs, matching
   `docs/source/reference/differt.<module>.rst` updates, a `CHANGELOG.md` entry under
   `## [Unreleased]`, and tests mirroring the source layout under `differt/tests/`.

   > Applied throughout, for Phase 1 and Phase 3 alike (see `CHANGELOG.md`'s
   > `### Added` entries and `docs/source/reference/differt.{em,rst}` updates).
   > One deliberate exception: `InteractionSites`/`build_interaction_sites`/
   > `interaction_sites_valid_mask`/`interaction_sites_mesh_mask` were kept
   > *private* (not re-exported from `differt.geometry.__init__`, no rst
   > entry) — internal wiring, not part of the plan's public-API surface;
   > can be made public later without a breaking change if a real use case
   > for direct access shows up.
