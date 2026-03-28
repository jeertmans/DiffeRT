# BVH acceleration for DiffeRT: implementation report

**Date:** 2026-03-28
**Author:** Robin Wydaeghe (UGent), with Claude
**Branch:** `feature/bvh-acceleration` on `rwydaegh/DiffeRT`
**Target:** `jeertmans/DiffeRT` (upstream PR after Jerome's thesis defense)

## Context

DiffeRT's three core intersection functions allocate O(rays * triangles) intermediate arrays in JAX, causing out-of-memory errors on scenes with more than a few thousand triangles. This is issue [#313](https://github.com/jeertmans/DiffeRT/issues/313). Jerome Eertmans (DiffeRT author) tried every pure-JAX approach: `vmap+sum` (OOM), `lax.scan` (slow), `lax.map` (slow), `fori_loop` with batching (best compromise but still 20s+ on GPU). The JAX team confirmed in [jax-ml/jax#30841](https://github.com/jax-ml/jax/issues/30841) that `lax.reduce` cannot close over Tracers due to a StableHLO limitation, and there is no fix coming.

The only viable path is to move the ray-triangle loop out of JAX entirely. Jerome's [extending-jax](https://github.com/jeertmans/extending-jax) repo demonstrates calling Rust from JAX via XLA FFI, but only has a forward-pass PoC with no gradients and no geometry code.

This report describes the complete implementation: a Rust BVH in `differt-core` with both PyO3 and XLA FFI bindings, fully integrated into all three `compute_paths` methods.

## Architecture

### Core design: "Rust for candidate selection, JAX for math"

The Moller-Trumbore intersection math stays in JAX (where it auto-differentiates through sigmoid smoothing). The BVH in Rust handles only the spatial query: given a ray, which triangles are worth testing?

```
Python layer
  TriangleBvh(triangle_vertices)    # PyO3 call -> Rust SAH BVH build
      |
      v
  Two query paths:
    PyO3 (outside JIT):             XLA FFI (inside JIT):
      bvh.nearest_hit()               ffi_nearest_hit()
      bvh.get_candidates()            ffi_get_candidates()
      |                                |
      v                                v
  JAX soft intersection on candidates only  # Existing Moller-Trumbore + sigmoid
      |
      v
  Gradients via JAX autodiff (automatic)    # No custom VJP needed
```

This split means:
- The `custom_vjp` is trivial: candidate indices are integers with zero gradient
- No need to hand-derive Moller-Trumbore VJPs in Rust
- Jerome can review the Rust code independently from the gradient logic
- The backward pass cost drops from O(rays * all_triangles) to O(rays * candidates)

### The "expanded BVH" for differentiable mode

For the soft path (`smoothing_factor` set), every boolean test is replaced with `sigmoid(x * alpha)`. For triangles far from a ray, all sigmoid values are exponentially small. The expansion radius guarantees that all triangles with gradient contribution above `epsilon_grad` are included:

```
r_near = triangle_size * ln(1 / epsilon_grad) / smoothing_factor
```

| smoothing_factor | r_near (1m triangles) | Regime |
|------------------|-----------------------|--------|
| 1 | 16.1m | Very soft, BVH falls back to brute force |
| 10 | 1.61m | Moderate, BVH helps on large scenes |
| 100 | 0.16m | Sharp, BVH very effective |
| 1000 | 0.016m | Near-hard, BVH nearly as fast as hard mode |

When `r_near` exceeds the scene bounding box diagonal, the system automatically falls back to brute force. When candidate counts exceed `max_candidates`, it also falls back with a warning.

### XLA FFI: BVH inside JIT

The XLA FFI pipeline follows Jerome's `extending-jax` pattern:

```
Python: ffi_nearest_hit(origins, dirs, bvh_id=id)
    |
    v
jax.ffi.ffi_call("bvh_nearest_hit", ...)     # JAX traces into HLO
    |
    v
BvhNearestHit(XLA_FFI_CallFrame*)            # C++ handler (src/ffi.cc)
    |  decodes XLA buffers, looks up BVH by ID
    v
bvh_nearest_hit_ffi(bvh_id, origins, dirs, ...)  # Rust via cxx bridge
    |  registry_get(bvh_id) -> Arc<Bvh>
    v
Bvh::nearest_hit(origin, dir)                # Pure Rust BVH traversal
```

The BVH is stored in a global `Mutex<HashMap<u64, Arc<Bvh>>>` registry. Python calls `bvh.register()` to get an integer ID, which is passed as an XLA attribute (compile-time constant). This works with `jax.jit`, `jax.lax.scan`, and `jax.vmap`.

## Implementation

### Rust: `differt-core/src/accel/` (~1,150 lines)

**`bvh.rs` (1,010 lines):**
- **BVH construction:** top-down recursive SAH split with 12-bin binning. O(N log N). Leaf size capped at 4 triangles.
- **Node layout:** `BvhNode { bbox_min, bbox_max, left_or_first, count }`. Internal nodes have `count=0`, leaves have `count>0`.
- **`nearest_hit`:** Standard BVH traversal with slab-method AABB test. Returns (triangle_index, t) per ray.
- **`get_candidates`:** Same traversal but with AABB expanded by `r_near`. Returns all leaf triangles in visited nodes.
- **Moller-Trumbore:** Full implementation in Rust for the hard-boolean nearest-hit path.
- **BVH registry:** `Mutex<HashMap<u64, Arc<Bvh>>>` with atomic ID generation. `register()`/`unregister()` on `TriangleBvh`.
- **PyO3 bindings:** `TriangleBvh` class exposed via `differt_core.accel.bvh`.
- **11 unit tests.**

**`ffi.rs` (135 lines):**
- **cxx bridge:** Declares Rust FFI functions and imports C++ XLA handlers.
- **`bvh_nearest_hit_ffi`:** Looks up BVH by ID, runs `nearest_hit` per ray.
- **`bvh_get_candidates_ffi`:** Looks up BVH by ID, runs `get_candidates` per ray.
- **PyCapsule exports:** `bvh_nearest_hit_capsule()` and `bvh_get_candidates_capsule()` for `jax.ffi.register_ffi_target`.

### C++: `src/ffi.cc` + `include/ffi.h` (110 lines)

- **`BvhNearestHitImpl`:** Decodes XLA buffers, calls Rust `bvh_nearest_hit_ffi` via cxx.
- **`BvhGetCandidatesImpl`:** Decodes XLA buffers, calls Rust `bvh_get_candidates_ffi` via cxx.
- **`XLA_FFI_DEFINE_HANDLER_SYMBOL`:** Generates the XLA-compatible C function symbols.

### Build: `build.rs` + `Cargo.toml` (50 lines)

- Queries the active Python interpreter for JAX's XLA FFI header location.
- Compiles `ffi.cc` via `cxx-build` with C++17 and JAX include paths.
- Gated behind `xla-ffi` Cargo feature (optional deps: `cxx`, `cxx-build`).

### Python: `differt/src/differt/accel/` (~700 lines)

- **`_bvh.py` (195 lines):** `TriangleBvh` wrapper with batch dimension handling, `register()`/`unregister()`.
- **`_accelerated.py` (376 lines):** Drop-in replacements: `bvh_rays_intersect_any_triangle`, `bvh_first_triangles_hit_by_rays`, `bvh_triangles_visible_from_vertices`.
- **`_ffi.py` (135 lines):** JAX FFI wrappers: `ffi_nearest_hit()`, `ffi_get_candidates()`. Handles `jax.ffi.register_ffi_target` registration.

### Scene integration: `_triangle_scene.py` (+80 lines)

- **`build_bvh()`:** Convenience method on `TriangleScene`.
- **`compute_paths(bvh=...)`:** All three methods use BVH when provided:
  - **exhaustive:** BVH FFI replaces blocking check inside `@eqx.filter_jit`.
  - **sbr:** BVH FFI replaces `first_triangles_hit_by_rays` inside `jax.lax.scan`.
  - **hybrid:** BVH for visibility estimation (PyO3, 14x faster) + blocking check (FFI).

### Tests: 11 Rust + 29 Python

**Rust unit tests (11):**
- BVH construction (single triangle, cube, empty, random)
- Nearest-hit correctness (hit, miss, closest selection)
- Candidate queries (no expansion, with expansion)
- BVH vs brute-force comparison on cube scene
- Moller-Trumbore edge cases

**Python integration tests (29):**
- `TestTriangleBvhConstruction` (4): single, cube, random, numpy input
- `TestNearestHit` (5): single triangle, miss, cube multi-ray, random scene 100 rays, fallback
- `TestAnyIntersection` (6): hard mode hit/miss, soft mode at alpha=1/10/100, random scene, fallback
- `TestExpansionRadius` (4): positive, monotonic decrease, scaling, zero smoothing
- `TestVisibility` (5): single triangle, cube, brute-force comparison, fallback, multiple origins
- `TestComputePathsBvh` (5): exhaustive+BVH FFI, SBR+BVH FFI, hybrid+BVH, exhaustive match, SBR match

## Performance

### Hard mode (nearest-hit): the main win

| Scene | Triangles | Rays | BVH Build | BVH Query | Brute Force | Speedup | Agreement |
|-------|-----------|------|-----------|-----------|-------------|---------|-----------|
| Munich | 38,936 | 200 | 136ms | 1ms | 1,054ms | **951x** | 100% |
| Random | 10,000 | 100 | 13ms | 9ms | 545ms | **58x** | 100% |
| Random | 5,000 | 100 | 10ms | 5ms | 745ms | **140x** | 100% |
| Random | 1,000 | 1,000 | 2ms | 10ms | 481ms | **47x** | 100% |
| Random | 100 | 100 | 0.4ms | 0.7ms | 383ms | **556x** | 100% |

The BVH build is a one-time cost (cached per scene). Query time scales as O(rays * log(triangles)).

### XLA FFI vs PyO3

Munich scene (38,936 triangles, 200 rays):

| Path | Time | Notes |
|------|------|-------|
| PyO3 (outside JIT) | 3.5ms | Python-to-Rust roundtrip |
| XLA FFI (outside JIT) | 24ms | First call includes registration overhead |
| XLA FFI (inside JIT, warm) | 2.6ms | After JIT compilation |

The FFI path is slightly faster than PyO3 after JIT warmup, and critically works inside `jax.jit` and `jax.lax.scan`.

### Soft mode (differentiable): depends on smoothing_factor

Munich scene (38,936 triangles, 50 rays):

| smoothing_factor | r_near | BVH Time | BF Time | Speedup | Max Diff | Notes |
|------------------|--------|----------|---------|---------|----------|-------|
| 10 | 8.06m | 845ms | 622ms | 0.7x | 0.000000 | Falls back to BF (too many candidates) |
| 50 | 1.61m | 988ms | 597ms | 0.6x | 0.010115 | BVH works, moderate precision |
| 100 | 0.81m | 233ms | 682ms | **2.9x** | 0.000057 | Good speedup, excellent precision |
| 500 | 0.16m | 252ms | 735ms | **2.9x** | 0.000000 | Exact match |
| 1000 | 0.08m | 271ms | 727ms | **2.7x** | 0.000000 | Exact match |

The soft mode speedup is modest (2-3x) because the JAX soft intersection on candidates still dominates. The real value is **avoiding OOM**: where brute force would allocate a `[rays, 39K, 3]` array and crash, the BVH reduces this to `[rays, ~300, 3]`.

### Visibility estimation (hybrid method)

Munich scene (38,936 triangles, 100K rays):

| Method | Visible tris | Time | Speedup |
|--------|-------------|------|---------|
| BVH | 1,143 | 1.12s | **14x** |
| Brute force | 1,128 | 15.18s | 1x |

### Test suite results

| Suite | Passed | Failed | Notes |
|-------|--------|--------|-------|
| Full DiffeRT (`pytest differt/tests/`) | 1,642 | 4 | All failures are pre-existing vispy headless rendering |
| BVH tests (`differt/tests/accel/`) | 29 | 0 | |
| RT tests (`differt/tests/rt/`) | 245 | 0 | |
| Rust tests (`cargo test -- accel`) | 11 | 0 | |

**Zero regressions from BVH changes.**

## What is not done yet

### Soft mode inside JIT

The soft (differentiable) blocking check in `_compute_paths` still falls back to brute force when `smoothing_factor` is set. The `ffi_get_candidates` FFI call is available but not yet wired into the soft path. This would require gathering candidate vertices inside JIT and running the JAX Moller-Trumbore + sigmoid on the reduced set.

### GPU BVH

The Rust BVH runs on CPU. A GPU implementation (via CUDA/OptiX or a Rust GPU crate) would further accelerate large-scale ray tracing. The JAX FFI supports `platform="gpu"` targets.

## Files changed

| File | Lines | Purpose |
|------|-------|---------|
| `differt-core/src/accel/bvh.rs` | +1,010 | Rust BVH: construction, traversal, queries, registry, tests |
| `differt-core/src/accel/ffi.rs` | +135 | XLA FFI bridge: cxx bridge, FFI entry points, PyCapsules |
| `differt-core/src/accel/mod.rs` | +12 | Module declarations |
| `differt-core/src/ffi.cc` | +95 | C++ XLA FFI handlers |
| `differt-core/include/ffi.h` | +16 | C++ handler declarations |
| `differt-core/build.rs` | +45 | Build script: find JAX headers, compile C++ via cxx-build |
| `differt-core/Cargo.toml` | +5 | xla-ffi feature, cxx + cxx-build deps |
| `differt-core/src/lib.rs` | +2 | Register accel module |
| `differt-core/python/differt_core/accel/__init__.py` | +5 | Python stub |
| `differt-core/python/differt_core/accel/_bvh.py` | +5 | Python re-export |
| `differt/src/differt/accel/__init__.py` | +27 | Package exports |
| `differt/src/differt/accel/_bvh.py` | +195 | TriangleBvh wrapper + register/unregister |
| `differt/src/differt/accel/_accelerated.py` | +376 | Drop-in accelerated functions + visibility |
| `differt/src/differt/accel/_ffi.py` | +135 | JAX FFI wrappers: ffi_nearest_hit, ffi_get_candidates |
| `differt/src/differt/scene/_triangle_scene.py` | +80 | build_bvh(), compute_paths(bvh=), BVH in all methods |
| `differt/tests/accel/__init__.py` | +0 | Test package |
| `differt/tests/accel/test_bvh.py` | +480 | 29 Python tests |
| **Total** | **~2,600** | |

## Usage example

```python
from differt.scene import TriangleScene

scene = TriangleScene.load_xml("munich/munich.xml")
bvh = scene.build_bvh()  # one-time O(N log N) build

# Standalone BVH queries (PyO3, outside JIT)
from differt.accel import bvh_first_triangles_hit_by_rays
idx, t = bvh_first_triangles_hit_by_rays(
    ray_origins, ray_directions,
    scene.mesh.triangle_vertices,
    bvh=bvh,
)

# Differentiable mode with BVH candidate pruning
from differt.accel import bvh_rays_intersect_any_triangle
blocked = bvh_rays_intersect_any_triangle(
    ray_origins, ray_directions,
    scene.mesh.triangle_vertices,
    smoothing_factor=100.0,
    bvh=bvh,
)
# Gradients flow through JAX autodiff on the reduced candidate set

# BVH-accelerated path computation (all methods, BVH inside JIT via XLA FFI)
paths = scene.compute_paths(order=1, method="exhaustive", bvh=bvh)  # BVH blocking check
paths = scene.compute_paths(order=1, method="hybrid", bvh=bvh)      # BVH visibility + blocking
paths = scene.compute_paths(order=2, method="sbr", bvh=bvh)         # BVH in lax.scan bounce loop
```
