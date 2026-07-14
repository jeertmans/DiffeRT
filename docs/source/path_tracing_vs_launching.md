# Path Tracing vs Path Launching

In DiffeRT, path finding algorithms are fundamentally split into two main approaches: **Path Tracing** and **Path Launching**. Each solves the problem of finding valid paths between transmitters and receivers, but they operate very differently in terms of speed, accuracy, and algorithmic approach.

To help structure these algorithms, DiffeRT provides two base classes in {mod}`differt.scene`:
* {class}`AbstractPathTracer<differt.scene.AbstractPathTracer>`
* {class}`AbstractPathLauncher<differt.scene.AbstractPathLauncher>`

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
* {class}`ExhaustivePathTracer<differt.scene.ExhaustivePathTracer>`
* {class}`HybridPathTracer<differt.scene.HybridPathTracer>` (which uses a heuristic visibility graph to reduce the number of path candidates before applying exact tracing)

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
* {class}`SBRPathLauncher<differt.scene.SBRPathLauncher>`

## Choosing a Solver

When calling {meth}`TriangleScene.trace_paths()<differt.scene.TriangleScene.trace_paths>` or {meth}`TriangleScene.launch_paths()<differt.scene.TriangleScene.launch_paths>`, you configure your solver by directly instantiating the respective solver class.

For example, to configure an exhaustive tracer with chunking:

```python
from differt.scene import ExhaustivePathTracer

scene.trace_paths(
    order=1,
    solver=ExhaustivePathTracer(chunk_size=1000)
)
```

To configure an SBR launcher:

```python
from differt.scene import SBRPathLauncher

scene.launch_paths(
    order=1,
    solver=SBRPathLauncher(num_rays=10_000, max_dist=1e-3)
)
```

## Customizing Solvers

You are not limited to the built-in solvers! You can customize path generation by creating your own solver subclasses.

By subclassing {class}`AbstractPathTracer<differt.scene.AbstractPathTracer>` or {class}`AbstractPathLauncher<differt.scene.AbstractPathLauncher>`, or one of its subclasses, you can implement custom logic for path candidate generation, path tracing, and so on.
