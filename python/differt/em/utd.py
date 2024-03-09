"""
Uniform Theory of Diffraction (UTD) utilities.

The foundamentals of UTD are described in :cite:`utd-mcnamara`.
"""


def F(x: float) -> float:  # noqa: N802
    r"""
    Evaluate the transition function :cite:`utd-mcnamara{p. 184}` at the given points.

    The transition function is defined as follows:

    .. math::
        F(x) = 2j \sqrt{x} e^{j x} \int\limits_\sqrt{x}^\infty e^{-j u^2} \text{d}u,

    where :math:`j^2 = -1`.

    As detailed in :cite:`utd-mcnamara{p. 164}`, the integral can be expressed in
    terms of Fresnel integrals (:math:`C(x)` and :math:`S(x)`), so that:

    .. math::
        C(x) - j S(x) = \int\limits_\sqrt{x}^\infty e^{-j u^2} \text{d}u.

    Because JAX does not provide a XLA implementation of
    :py:func:`scipy.special.fresnel`, this implementation relies on a
    custom complex-valued implementation of the error function and
    the fact that:

    .. math::
        C(z) - j S(z) = \sqrt{\frac{\pi}{2}}\frac{1-i}{2}\text{erf}\left(\frac{1+i}{\sqrt{2}}z\right).

    Args:
        x: <TODO>.

    Return:
        <TODO>

    Examples:
        .. plot::

            The following example reproduces the same plot as in
            :cite:`utd-mcnamara{fig. 4.16}`.

            >>> from differt.em.utd import F
            >>>
            >>> x = jnp.logspace(-3, 1, 100)
            >>> y = F(x)
            >>>
            >>> A = jnp.abs(y)  # Amplitude of F(x)
            >>> P = jnp.angle(y, deg=True)  # Phase (in deg.) of F(x)
            >>>
            >>> fig, ax1 = plt.subplots()
            >>>
            >>> ax1.semilogx(x, A, "k-")
            >>> ax1.set_ylabel("Magnitude - solid line")
            >>> ax2 = plt.twinx()
            >>> ax2.semilogx(x, P, "k--")
            >>> ax2.set_ylabel("Phase (°) - dashed line")
            >>> plt.tight_layout()
    """
    return x  # TODO
