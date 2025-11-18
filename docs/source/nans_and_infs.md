# NaN and Infinite Values

When performing floating-point operations, it's common to encounter Not-a-Number (NaN) or infinite values if you're not careful. In some cases, these values can be deliberately used to convey specific information. For instance, our function {func}`intersection_of_rays_with_planes<differt.rt.intersection_of_rays_with_planes>` returns `jnp.inf` coordinates when rays are parallel to the corresponding planes, since no intersection point exists.

However, infinite values (`jnp.inf`) must be used with caution. They can sometimes lead to `jnp.nan`, for example, when subtracting two infinite values. And once a NaN appears, it can quickly propagate through the computation, corrupting otherwise valid results.

This short guide outlines our policy regarding NaN and infinite values in DiffeRT.

## Use of Infinite Values

As noted above, infinite values are occasionally used in DiffeRT to indicate special cases, such as invalid or undefined outputs. However, they are carefully handled to prevent the unintentional creation of NaNs. For instance, we use appropriate JAX tooling to ensure that gradients of functions returning `jnp.inf` remain well-defined and NaN-free. This aligns with recommendations from the [JAX FAQ](https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where).

If you build custom logic on top of such functions, it is your responsibility to ensure your extended logic doesn't inadvertently produce NaNs. Refer to the JAX documentation for guidance on how to handle these cases safely.

While this protective handling may introduce a small computational overhead, it significantly reduces the risk of difficult-to-debug NaN issues.

## No-NaN Policy

NaN values are strictly avoided within DiffeRT. Although they can theoretically be used to indicate invalid results, NaNs tend to spread uncontrollably, making it difficult to trace their origin. Therefore, NaNs are never permitted in either the function outputs or their gradients.

To achieve this, we take proactive steps—such as using safe division and conditional logic-to avoid scenarios like division by zero. In addition, we utilize JAX's built-in tools for [debugging NaNs](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#debugging-nans-and-infs), which makes it easier to locate and eliminate potential sources of NaNs during development.

If you encounter a case where one of our functions returns a NaN, please report it on [GitHub](https://github.com/jeertmans/DiffeRT/issues/new/choose).
