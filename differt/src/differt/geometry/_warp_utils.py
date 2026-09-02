"""Shared Warp infrastructure: mesh caching and multi-threaded CPU dispatch."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import cpu_count
from typing import Any, no_type_check

import warp as wp

# NOTE: Cache meshes to avoid re-creating them over and over.
# A problem with the current implementation is that @eqx.filter_jit
# creates a new Mesh instance every time the function is recompiled,
# which creates cache misses. We could create a 'permanent' id for each mesh,
# e.g., when instantiating the Mesh instance and passing it around,
# only updating it when it changes. However, this also means that we must keep
# track of all Mesh instances that point to the same id.
_WARP_MESHES_CACHE: dict[
    tuple[int, int, int, int, int],
    wp.Mesh,
] = {}


@no_type_check
def _clear_warp_mesh_cache(mesh_id: int | None = None) -> None:
    """Clear cached Warp meshes for ``mesh_id`` (or all if ``None``)."""
    if mesh_id is None:
        _WARP_MESHES_CACHE.clear()
    else:
        mesh_id_int = int(mesh_id)
        keys_to_delete = [k for k in _WARP_MESHES_CACHE if k[0] == mesh_id_int]
        for k in keys_to_delete:
            _WARP_MESHES_CACHE.pop(k, None)


@no_type_check
def _get_warp_mesh(
    mesh_id: int,
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
) -> wp.Mesh:
    """Return the cached Warp mesh for ``mesh_id``, building (and caching) it if absent.

    Args:
        mesh_id: The unique id of the (JAX-side) mesh, e.g., ``id(mesh)``.
        points: The mesh vertices.
        indices: The (flattened) mesh triangle indices.

    Returns:
        The corresponding, cached Warp mesh.
    """
    key = (int(mesh_id), points.ptr, indices.ptr, points.size, indices.size)
    if (wp_mesh := _WARP_MESHES_CACHE.get(key)) is None:
        # Clone points/indices: JAX may later free or reuse this memory,
        # which would otherwise cause segfaults once the mesh is reused.
        wp_mesh = wp.Mesh(points=wp.clone(points), indices=wp.clone(indices))
        _WARP_MESHES_CACHE[key] = wp_mesh
    return wp_mesh


# NOTE: 'wp.launch' on a "cpu" device runs single-threaded: a single call
# only ever uses one CPU core, see https://github.com/NVIDIA/warp/issues/224.
# '_warp_launch' below splits a CPU launch's 'dim' into per-thread chunks and
# dispatches them concurrently instead; correctness (each Python thread issuing
# its own 'wp.launch' call on a disjoint slice of the arrays, including
# concurrent read-only queries against the same 'wp.Mesh' BVH) was verified
# empirically on this exact code path. On CUDA, this is a no-op wrapper
# around a single 'wp.launch', since the GPU already parallelizes 'dim'
# itself.
# All of this might not be necessary once Warp supports multi-threaded
# CPU launches natively, see https://github.com/NVIDIA/warp/issues/1309
_CPU_THREAD_POOL: ThreadPoolExecutor | None = None
# Below this many elements, the overhead of splitting work across threads
# is not worth it; just run a single, un-chunked launch.
_CPU_PARALLEL_MIN_DIM = 1024


def _get_cpu_thread_pool() -> ThreadPoolExecutor:
    global _CPU_THREAD_POOL  # noqa: PLW0603
    if _CPU_THREAD_POOL is None:
        _CPU_THREAD_POOL = ThreadPoolExecutor(
            max_workers=cpu_count() or 1, thread_name_prefix="differt-warp-cpu"
        )
    return _CPU_THREAD_POOL


@dataclass(slots=True)
class _Batched:
    """Marks a kernel argument that must be split into per-thread chunks.

    Wrap any input or output :class:`warp.array` argument whose ``axis``
    corresponds 1-to-1 with the launch's chunked dimension (``dim`` itself,
    or ``dim[chunk_axis]`` for a multi-dimensional launch), see
    :func:`_warp_launch`. Every other argument (mesh handles, shared
    geometry, scalars, or an output whose concurrent writes are safe
    regardless of how the launch is chunked) must be passed as-is.

    ``row_size`` only applies to a flat, 1-D array packing multiple values
    per chunked element (e.g., a ``[dim * row_size]`` buffer reshaped from
    ``[dim, row_size]`` on the caller's side to avoid FFI overhead); use the
    default ``row_size=1`` for a plain per-element array, regardless of its
    dimensionality.
    """

    array: Any
    row_size: int = 1
    axis: int = 0


class _Offset:
    """Placeholder for a scalar ``int`` kernel argument receiving the chunk's starting global index.

    The value is ``0`` when the launch is not split into chunks.

    Use this when a kernel needs to recover the *global* thread index (e.g.,
    to compute a batch index from ``tid``), since ``wp.tid()`` is always
    chunk-local once a launch has been split by :func:`_warp_launch`.
    """

    __slots__ = ()


@no_type_check
def _warp_launch(
    kernel: Any,
    dim: int | tuple[int, ...],
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    device: Any,
    chunk_axis: int = 0,
) -> None:
    """Launch a Warp kernel, splitting ``dim`` across CPU threads if ``device`` is CPU.

    On a CUDA device, this is equivalent to a single :func:`warp.launch` call,
    since the GPU already parallelizes ``dim`` itself.

    Args:
        kernel: The Warp kernel to launch.
        dim: The number of elements (e.g., rays) to launch the kernel over,
            or a tuple thereof for a multi-dimensional launch.
        inputs: Positional kernel inputs; wrap any that should be split
            across threads in :class:`_Batched`, or that should receive the
            chunk's starting global index in :class:`_Offset`.
        outputs: Positional kernel outputs; wrap any that should be split
            across threads in :class:`_Batched`.
        device: The Warp device to launch on.
        chunk_axis: If ``dim`` is a tuple, the axis of ``dim`` (matching
            the ``axis`` of every :class:`_Batched` argument) to split
            across threads. Ignored if ``dim`` is a plain ``int``.
    """
    is_tuple = isinstance(dim, tuple)
    dims = dim if is_tuple else (dim,)
    total = dims[chunk_axis]

    def unwrap(arg: Any) -> Any:
        if isinstance(arg, _Batched):
            return arg.array
        if isinstance(arg, _Offset):
            return 0
        return arg

    num_threads = min(cpu_count() or 1, total) if device.is_cpu else 1

    if num_threads <= 1 or total < _CPU_PARALLEL_MIN_DIM:
        wp.launch(
            kernel,
            dim=dim,
            inputs=[unwrap(arg) for arg in inputs],
            outputs=[unwrap(arg) for arg in outputs],
            device=device,
        )
        return

    chunk_size = -(-total // num_threads)  # ceil division

    def slice_arg(arg: Any, start: int, stop: int) -> Any:
        if isinstance(arg, _Batched):
            if arg.row_size != 1:
                k = arg.row_size
                return arg.array[start * k : stop * k]
            index = [slice(None)] * arg.array.ndim
            index[arg.axis] = slice(start, stop)
            return arg.array[tuple(index)]
        if isinstance(arg, _Offset):
            return start
        return arg

    def launch_chunk(t: int) -> None:
        start = t * chunk_size
        stop = min(start + chunk_size, total)
        chunk_dims = tuple(
            stop - start if i == chunk_axis else d for i, d in enumerate(dims)
        )
        wp.launch(
            kernel,
            dim=chunk_dims if is_tuple else chunk_dims[0],
            inputs=[slice_arg(arg, start, stop) for arg in inputs],
            outputs=[slice_arg(arg, start, stop) for arg in outputs],
            device=device,
        )

    list(_get_cpu_thread_pool().map(launch_chunk, range(num_threads)))
