# Limitations

While DiffeRT is built on top of JAX and generally supports a wide variety of hardware and Python environments, some features have compatibility constraints due to external dependencies---principally NVIDIA Warp.

## GPU and TPU Compatibility

Since version 0.9 and particularly {gh-pr}`467`, DiffeRT uses NVIDIA Warp to accelerate ray tracing tasks inside {class}`TriangleMesh<differt.geometry.TriangleMesh>` and {class}`TriangleScene<differt.scene.TriangleScene>`.

NVIDIA Warp is a high-performance framework for writing GPU/CPU physics kernels in Python. While JAX supports a broad range of accelerator backends, NVIDIA Warp **only supports CPU and NVIDIA CUDA-enabled GPUs**. As a result, when using the Warp-accelerated methods:

1. **Google TPUs are not supported:** Running Warp-accelerated methods on a TPU will fail.
2. **Non-CUDA GPUs are not supported:** Running Warp-accelerated methods on non-CUDA GPU backends (such as AMD ROCm or Apple Silicon Metal) will fail.

Any DiffeRT method that relies on Warp will fail when running JAX on a TPU or a non-CUDA GPU device.

### Affected Methods

The following methods are Warp-accelerated and **require either a CPU or an NVIDIA CUDA GPU** to execute:

* {meth}`TriangleMesh.rays_intersect_any_triangle<differt.geometry.TriangleMesh.rays_intersect_any_triangle>`
* {meth}`TriangleMesh.first_triangles_hit_by_rays<differt.geometry.TriangleMesh.first_triangles_hit_by_rays>`
* {meth}`TriangleMesh.triangles_visible_from_vertices<differt.geometry.TriangleMesh.triangles_visible_from_vertices>`
* {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>` (which internally utilizes the above methods unless smoothing is enabled)

### Alternatives

If you need to run your code on a TPU or a non-CUDA GPU, you should use the corresponding non-Warp equivalent functions in {mod}`differt.rt`:

* {func}`differt.rt.rays_intersect_any_triangle`
* {func}`differt.rt.first_triangles_hit_by_rays`
* {func}`differt.rt.triangles_visible_from_vertices`

These functions are written in pure JAX, so they will execute correctly on any backend supported by JAX. However, please note that they may be less memory-efficient and slower than the Warp-accelerated methods on `TriangleMesh`.

## Free-Threaded Python Compatibility

Python 3.13 introduced support for running without the Global Interpreter Lock (GIL), also known as free-threaded Python.

However, NVIDIA Warp does not (currently) support free-threaded Python and will raise an error in such environments.
Consequently, DiffeRT does not support free-threaded Python either.
