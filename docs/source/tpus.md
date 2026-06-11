# Note on TPUs

While DiffeRT is built on top of JAX and generally supports CPU, GPU, and TPU acceleration, some methods have limitations on TPUs due to external dependencies.

Specifically, since version 0.9 and particularly {gh-pr}`467`, DiffeRT uses NVIDIA Warp to accelerate ray tracing tasks inside {class}`TriangleMesh<differt.geometry.TriangleMesh>` and {class}`TriangleScene<differt.scene.TriangleScene>`.

NVIDIA Warp is a high-performance framework for writing GPU/CPU physics kernels in Python. Because Warp does not support Tensor Processing Units (TPUs), any DiffeRT method that relies on Warp will fail when running JAX on a TPU device.

## Affected Methods

The following methods are Warp-accelerated and **do not support** TPU execution:

* {meth}`TriangleMesh.rays_intersect_any_triangle<differt.geometry.TriangleMesh.rays_intersect_any_triangle>`
* {meth}`TriangleMesh.first_triangles_hit_by_rays<differt.geometry.TriangleMesh.first_triangles_hit_by_rays>`
* {meth}`TriangleMesh.triangles_visible_from_vertices<differt.geometry.TriangleMesh.triangles_visible_from_vertices>`
* {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>` (which internally utilizes the above methods unless smoothing is enabled)

## Alternatives

If you must run your code on a TPU, you can use the corresponding non-Warp equivalent functions in {mod}`differt.rt`:

* {func}`differt.rt.rays_intersect_any_triangle`
* {func}`differt.rt.first_triangles_hit_by_rays`
* {func}`differt.rt.triangles_visible_from_vertices`

These functions are written in pure JAX, so they will execute correctly on TPUs. However, please note that they may be less memory-efficient and slower than the Warp-accelerated methods on `TriangleMesh`.
