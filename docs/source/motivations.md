# Why use DiffeRT?

Why should you use DiffeRT? For what purpose?

Those are two good questions, and we will try to motivate in this document the reasons to use DiffeRT.

## What is DiffeRT

DiffeRT is a Python, array-oriented, Differentiable Ray Tracing (RT) library that aims to provide fast and easy-to-use tools to model the propagation of radio waves. The long-term objective of DiffeRT is to provide:

- fast methods to load large scenes from various formats;
- a large set of performant RT utilities (e.g., ray launching, {func}`image_method<differt.rt.image_method>`);
- the ability to easily compute electromagnetic fields and relevant metrics (e.g., power delay profile, angular spread);
- and the ability to differentiate any of the previous parts with respect to arbitrary input parameters for optimization or Machine Learning applications.

## History

The development of DiffeRT began around 2021 as a collection of unorganized code projects during a PhD program. Later, a 2D version of DiffeRT, [DiffeRT2d](https://github.com/jeertmans/DiffeRT2d), was created and published in an open-access journal in 2024 {cite}`differt2d`.

While 2D RT is excellent for developing toy examples---especially when leveraging object-oriented programming---it often scales poorly to large scenes, limiting DiffeRT2d's use to fundamental research rather than realistic radio propagation scenarios.

DiffeRT builds on some of the principles behind DiffeRT2d while prioritizing performance and scalability for any scene size. Most utilities provided by DiffeRT work directly on arrays to avoid unnecessary abstractions associated with object-oriented programming[^1].

[^1]: DiffeRT still uses object-oriented programming in some places, but those classes are immutable dataclasses, and JAX-compatible PyTree, which makes them compatible with many of the JAX features.

## DiffeRT vs. Sionna

In terms of features, DiffeRT does not aim to match the extensive functionality of Sionna. Instead, DiffeRT focuses on RT-specific applications similar to what `sionna.rt` offers, but with four main differences:

1. **Public lower-level RT Routines[^2]:** Many internal RT mechanisms in Sionna are hidden or undocumented, making it challenging to modify the pipeline. DiffeRT, on the other hand, ensures that most RT utilities are public and well-documented, enabling users to customize or replace parts of the RT algorithms without re-implementing or copy-pasting code.
2. **JAX Integration:** Unlike Sionna, which uses TensorFlow, DiffeRT leverages JAX for efficient array-based programming. JAX offers powerful features like automatic differentiation, just-in-time (JIT) compilation, and compatibility with GPU/TPU acceleration, making it highly suitable for optimization and Machine Learning tasks.
3. **Minimal Abstraction with Immutable Dataclasses:** Sionna internally represents scenes using Mitsuba, which, while powerful, imposes restrictions on the types of scenes it can handle. Moreover, Sionna's classes are relatively complex, with many hidden attributes. In contrast, DiffeRT uses immutable dataclasses that can be created using simple constructors or convenient class methods (e.g., for reading scenes from files). Following JAX principles, all classes are immutable PyTrees, ensuring compatibility with JAX while avoiding unnecessary memory allocations through JIT optimization.
4. **Lightweight and Broadcastable Design:** DiffeRT's design philosophy prioritizes transparency and usability for RT applications, avoiding the heavier abstractions often seen in other libraries. Classes aim to store as few attributes as possible, and most utilities accept input arrays with arbitrary sized inputs, which makes it very easy, e.g., to compute the same operation for one receiving (RX) antenna, or on a two-dimensional grid of RXs.

[^2]: There are some exceptions, like the internal machinery behind
  {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>`,
  but we then provide detailed tutorials to help the user understand and build their version of the function,
  if they wish to do so, e.g., with {ref}`advanced_path_tracing`.

We acknowledge the work of Sionna, and would recommend users to try both tools, and use the one that best fits their needs! If you want to reuse scene files from Sionna, check out the {meth}`TriangleScene.load_xml<differt.scene.TriangleScene.load_xml>` method, as it supports reading the same file format as the one used by Sionna, i.e., the XML file format used by Mitsuba.

## What's Next?

If you have any question, remark, or recommendation regarding DiffeRT, or its comparison with Sionna, please feel free to reach out on [GitHub discussion](https://github.com/jeertmans/DiffeRT/discussions)!
