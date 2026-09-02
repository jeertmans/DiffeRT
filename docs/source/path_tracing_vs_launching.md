# Path Tracing vs Path Launching

In DiffeRT, path finding algorithms are fundamentally split into two main approaches: **Path Tracing** and **Path Launching**. Each solves the problem of finding valid paths between transmitters and receivers, but they operate very differently in terms of speed, accuracy, and algorithmic approach.

To help structure these algorithms, DiffeRT provides two base classes in {mod}`differt.geometry`:
* {class}`AbstractPathTracer<differt.geometry.AbstractPathTracer>`
* {class}`AbstractPathLauncher<differt.geometry.AbstractPathLauncher>`

## Path Tracing

**Path Tracing** (sometimes called the Image Method) is an exact, deterministic approach.

1. **Path Candidates**: The tracer enumerates all possible "path candidates" (sequences of object interactions) up to a specified maximum order.
2. **Path Tracing**: For each candidate, it attempts to find the exact, valid physical path that connects the transmitter to the receiver while undergoing the sequence of interactions.
3. **Blockage Checking**: Finally, the tracer checks if any of the valid paths are blocked by other objects in the scene.

**Pros:**
* **Exact**: Guarantees finding the exact path if one exists for the given sequence of interactions.
* **Deterministic**: Yields the exact same results every time.

**Cons:**
* **Scalability**: The number of path candidates grows exponentially with the number of objects and the maximum interaction order. As a result, exhaustive path tracing becomes computationally infeasible for large scenes or high interaction orders.

**Example Solvers:**
* {class}`ExhaustivePathTracer<differt.geometry.ExhaustivePathTracer>`
* {class}`HybridPathTracer<differt.geometry.HybridPathTracer>` (which uses a heuristic visibility graph to reduce the number of path candidates before applying exact tracing)
* {class}`SBRPathTracer<differt.geometry.SBRPathTracer>` (which *discovers* candidates with shooting-and-bouncing rays, instead of enumerating a graph, before applying exact tracing — see below)

### Combining Multiple Orders

`order` is usually a single integer, but {meth}`AbstractPathTracer.generate_path_candidates()<differt.geometry.AbstractPathTracer.generate_path_candidates>`, {meth}`AbstractPathTracer.trace_paths()<differt.geometry.AbstractPathTracer.trace_paths>`, and {meth}`Scene.trace_paths()<differt.geometry.Scene.trace_paths>` also accept a sequence of orders — a list, a {class}`range`, or a `slice` with a defined `stop` — e.g., `order=[0, 1, 2]` or, equivalently, `order=range(0, 3)`, to generate and trace path candidates of different orders in a single call:

```python
scene.trace_paths(order=range(0, 3))
```

For {class}`ExhaustivePathTracer<differt.geometry.ExhaustivePathTracer>` and {class}`HybridPathTracer<differt.geometry.HybridPathTracer>`, candidates are generated independently for each order, then directly combined into a single array of the correct, known-ahead-of-time size (the sum of each individual order's candidate count): lower-order candidates are padded up to the maximum requested order with `-1` placeholders, marking an "inactive" interaction, and all orders are concatenated together. {class}`SBRPathTracer<differt.geometry.SBRPathTracer>` combines orders differently, see below. Either way, the combined array is passed, in a single call, to {meth}`AbstractPathTracer.trace_path_candidates()<differt.geometry.AbstractPathTracer.trace_path_candidates>`, which correctly resolves a placeholder-padded candidate to a path that reaches the receiver right after its last genuine interaction (and stays there for the padded positions), so the returned {class}`TracedPaths<differt.geometry.TracedPaths>` has a single, consistent `order`.

Path candidates you pass in yourself (e.g., via `Scene.trace_paths(path_candidates=...)`) may also be padded this way, as long as placeholders only ever appear as a trailing suffix of a candidate — see {func}`check_path_candidates<differt.geometry.check_path_candidates>`, which is automatically called (and will raise otherwise) whenever candidates reach a path tracer.

This combination is not supported together with a solver's `chunk_size`.

```{important}
For {class}`SBRPathTracer<differt.geometry.SBRPathTracer>`, whose fixed-size candidate buffer (see below) is rarely fully used, an all-`-1` row is a valid, degenerate line-of-sight candidate, and unused buffer slots are exactly that. As a result, the line-of-sight path may appear as many duplicate valid entries. Use {meth}`TracedPaths.mask_duplicate_objects()<differt.geometry.TracedPaths.mask_duplicate_objects>` if you need an accurate count of *distinct* valid paths.
```

### A Third Option: Bounded Candidate Discovery

Both {class}`ExhaustivePathTracer<differt.geometry.ExhaustivePathTracer>` and {class}`HybridPathTracer<differt.geometry.HybridPathTracer>` generate path candidates by enumerating (a subset of) a complete graph over the scene's primitives. Even after visibility pruning, the number of candidates can still grow exponentially with `order`, because the graph is enumerated exactly, depth by depth.

{class}`SBRPathTracer<differt.geometry.SBRPathTracer>` avoids this altogether: instead of enumerating candidates, it launches a fixed, bounded population of `num_rays` rays from each transmitter and lets them bounce specularly through the scene, exactly like {class}`SBRPathLauncher<differt.geometry.SBRPathLauncher>` does. Whichever sequence of primitives each ray happens to hit becomes one candidate. Because many rays typically converge onto the same sequence, especially at low orders, the discovered sequences are deduplicated into a fixed-size buffer of at most `max_num_candidates` unique entries before being handed off to the same exact image-method solver used by the other two tracers.

As a result:
* The cost of generating candidates only depends on `num_rays` and `max_num_candidates`, not on `order` or the number of primitives in the scene.
* Every returned candidate is still validated and solved *exactly* (it is a path **tracer**, not a path **launcher**): there is no approximation in the geometry of the returned paths, only in *which* candidates are considered.
* The search is not guaranteed to be exhaustive: candidates that subtend a very small solid angle as seen from the transmitters may be missed. Increasing `num_rays` improves coverage at the cost of memory and runtime.

```python
from differt.geometry import SBRPathTracer

scene.trace_paths(
    order=3,
    solver=SBRPathTracer(num_rays=1_000_000, max_num_candidates=100_000),
)
```

When `order` is a sequence of orders, rays are launched only once, up to the maximum requested order, and each ray's own *natural* number of interactions — i.e., how many bounces it completes before exiting the scene — decides which requested order, if any, it is a candidate for. A trajectory whose natural order is not one of the requested orders is simply not written to the buffer, rather than being truncated (or extended) to fit a nearby one. Consequently, unlike {class}`ExhaustivePathTracer<differt.geometry.ExhaustivePathTracer>` and {class}`HybridPathTracer<differt.geometry.HybridPathTracer>`, the combined output stays bounded by `max_num_candidates` *alone*, regardless of how many orders are requested:

```python
scene.trace_paths(
    order=range(0, 6),
    solver=SBRPathTracer(num_rays=1_000_000, max_num_candidates=100_000),
)
```

This computes valid paths for orders `0` through `5` in one call, using the exact same, bounded ray population and candidate buffer that a single-order call would use.

## Path Launching

**Path Launching** (such as Shooting and Bouncing Rays, or SBR) is an approximate, forward-simulation approach.

1. **Ray Launching**: A large number of rays are "shot" out from the transmitter in various directions.
2. **Bouncing**: The rays bounce around the scene up to the maximum order.
3. **Capture**: A receiver "captures" a ray if the ray passes within a certain distance (the capture radius or maximum distance) of the receiver.

**Pros:**
* **Scalability**: Scales much better to complex scenes and higher interaction orders compared to exhaustive tracing. The computational cost is largely determined by the number of rays launched rather than an exponential explosion of candidates.

**Cons:**
* **Approximate**: Because it relies on discrete rays and a capture radius, it may miss valid paths (if the angular resolution of the launched rays is too low) or incorrectly estimate path geometry.
* **Tuning Required**: Requires tuning the number of rays and the capture radius (`max_dist`).

**Example Solvers:**
* {class}`SBRPathLauncher<differt.geometry.SBRPathLauncher>`

## Choosing a Solver

When calling {meth}`Scene.trace_paths()<differt.geometry.Scene.trace_paths>` or {meth}`Scene.launch_paths()<differt.geometry.Scene.launch_paths>`, you configure your solver by directly instantiating the respective solver class.

```{important}
{meth}`Scene.trace_paths()<differt.geometry.Scene.trace_paths>` defaults to `solver="sbr"` (i.e., {class}`SBRPathTracer<differt.geometry.SBRPathTracer>`), which scales well but is not guaranteed to find every valid path. Pass `solver="exhaustive"` (or `"hybrid"`) explicitly whenever you need a deterministic, exhaustive search — e.g., for small scenes, low orders, or reference results.
```

For example, to configure an exhaustive tracer with chunking:

```python
from differt.geometry import ExhaustivePathTracer

scene.trace_paths(
    order=1,
    solver=ExhaustivePathTracer(chunk_size=1000)
)
```

To configure an SBR launcher:

```python
from differt.geometry import SBRPathLauncher

scene.launch_paths(
    order=1,
    solver=SBRPathLauncher(num_rays=10_000, max_dist=1e-3)
)
```

## Customizing Solvers

You are not limited to the built-in solvers! You can customize path generation by creating your own solver subclasses.

By subclassing {class}`AbstractPathTracer<differt.geometry.AbstractPathTracer>` or {class}`AbstractPathLauncher<differt.geometry.AbstractPathLauncher>`, or one of its subclasses, you can implement custom logic for path candidate generation, path tracing, and so on.
