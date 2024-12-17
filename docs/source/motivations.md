# Why use DiffeRT?

Why should you use DiffeRT? For what purpose?

Those are two good questions, and we will try to motivate in this document the reasons to use DiffeRT.

## What is DiffeRT

DiffeRT is a Python, array-oriented, Differentiable Ray Tracing (RT) library that aims to provides
fast and easy to use tools for model the propagation of radio waves. The long-term objective of DiffeRT
is to provide:

- fast method to load large scenes, from various formats;
- a large set of performant RT utilities (ray launching, image-method, etc.);
- the ability to easily compute electromagnetic fields and relevant metrics (power delay profile, angular spread, etc.);
- and allow to differentiate any of the previous parts, with respect to arbitrary input parameters, for optimization or Machine Learning applications.

## History

At its origins, the development of DiffeRT emerged around 2021, in the form of unorganized code
projects developed as part of a PhD. Then, a 2D version of DiffeRT,
[DiffeRT2d](https://github.com/jeertmans/DiffeRT2d),
came to life and was published in an open access journal in 2024 {cite}`differt2d`.

While 2D RT is very nice to develop toy-example, especially when taking benefits
from object-oriented programming, this usually scales very poorly to large scenes,
ultimately making DiffeRT2d only usable for fundamental research on RT, and not
for radio propagation in realistic scenes.

DiffeRT builds on some of the principles behind DiffeRT2d, while aiming at good performances
and scalability to any scene size. As a result, most of the utilities provided by DiffeRT
directly work on arrays, to avoid unnecessary abstractions caused by object-oriented programming[^1].

[^1]: DiffeRT stills uses object-oriented programming in some places, but those classes are immutable
  dataclasses, and JAX-compatible PyTree, which makes them compatible with many of the JAX features.

## DiffeRT vs Sionna

<!-- TODO: improve this section, acknowledge Sionna and mention the possibility to load scene files created from Sionna -->

In terms of features, DiffeRT is nowhere near to bringing the same level of features than Sionna.
But this is not the goal of DiffeRT either. Instead, DiffeRT tries to focus on RT-oriented applications,
similar to what `sionna.rt` offers, but with **four main differences**:

- **Most RT functions are public**[^2]: a lot of of the internal RT machinery is hidden (i.e., not documented to the external user) in Sionna. This makes it very convenient for users that don't want to understand of RT works, but makes it particularly hard when one wants to modify some part of the code or the pipeline. Inside DiffeRT, most RT utilities and public and documented, so any user can decide to re-implement their own version of some RT algorithm, without having to re-implement or copy-paste code.
- **We use JAX**: Sionna uses TensforFlow, as well as Mistabu, we efficient array-based programming.
  JAX is a very powerful array library that supports...
- **Less abstraction and immutable dataclasses**: internally, Sionna represents and loads the scene files using Mitsuba. While offering performant high-level features, this restricts quite a lot the types of scenes that Sionna can work with. Instead, all our classes are immutable dataclasses, and can be create with their dataclass constructor, or via more convenient class methods, e.g., when reading a scene from file. As a result of using JAX, we apply the principle that all classes should by immutable PyTrees.
  This means that you always return new class instante whenever you want to modify any of its attributes. While this may
  look like a cause to memory problems, JIT compiliation will usually optimize away uncessary memory allocations.

[^2]: The are some exceptions, like the internal machinery behind
  {meth}`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>`,
  but we then provide detailed tutorials to help the user understand and build their version of the function,
  if they which to do so, e.g., with {ref}`advanced_path_tracing`.