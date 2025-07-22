(conventions)=
# Conventions

This document details the conventions used in this library.
Hence, unless specified otherwise, the following conventions apply everywhere.

## XYZ coordinates

Those coordinates are **absolute**, and are by default expressed in the three-dimensional right-handed XYZ space,
where units are meters.

Arrays expressing such coordinates are always of the type and shape
{class}`Float[Array, "... 3"]<jaxtyping.Float>`.

{numref}`fig:cartesian` shows how Cartesian coordinates are defined.

```{figure} _static/cartesian.svg
:width: 50%
:align: center
:name: fig:cartesian
:alt: Cartesian coordinates.

By <a href="//commons.wikimedia.org/wiki/User:Jorge_Stolfi" title="User:Jorge Stolfi">Jorge Stolfi</a> - <span class="int-own-work" lang="en">Own work</span>, Public Domain, <a href="https://commons.wikimedia.org/w/index.php?curid=6692547">Link</a>.
```

(spherical-coordinates)=
## Spherical coordinates

Those coordinates are **relative** to some origin, and are by default expressed in the three-dimensional {math}`r\theta\varphi` space,
where units are meters for {math}`r`, and radians for {math}`\theta` and {math}`\varphi`. We follow the physical convention (see {numref}`fig:spherical`), where {math}`\theta` is referred to as the polar angle, and {math}`\varphi` as the azimuth angle.

Arrays expressing such coordinates are **mainly** of the type and shape
{class}`Float[Array, "... 3"]<jaxtyping.Float>`, but sometimes the radial component {math}`r` is omitted
which means {class}`Float[Array, "... 2"]<jaxtyping.Float>` arrays can also be used in some places, usually
as a mean to indicate a radial component equal to 1.

{numref}`fig:spherical` shows how spherical coordinates are defined.

```{figure} _static/spherical.svg
:width: 50%
:align: center
:name: fig:spherical
:alt: Spherical coordinates.

By <a href="//commons.wikimedia.org/w/index.php?title=User:Andeggs&amp;action=edit&amp;redlink=1" class="new" title="User:Andeggs (page does not exist)">Andeggs</a> - <span class="int-own-work" lang="en">Own work</span>, Public Domain, <a href="https://commons.wikimedia.org/w/index.php?curid=7478049">Link</a>.
```

## Coordinate transformations

Below are descriptions of how to transform from one coordinate system to another.

### Cartesian to spherical

Spherical coordinates can be derived from Cartesian coordinates using the following relation:

```{math}
\begin{bmatrix}r \\ \theta \\ \varphi \end{bmatrix} =
\begin{bmatrix}
\sqrt{x^2 + y^2 + z^2} \\ \arccos(z / \sqrt{x^2 + y^2 + z^2}) \\ \arctan(y / x)
\end{bmatrix},\ \ \ 0 \le \theta \le \pi,\ \ \ 0 \le \varphi < 2\pi,
```

and is implemented by {func}`cartesian_to_spherical<differt.geometry.cartesian_to_spherical>`.

### Spherical to Cartesian

Conversely, it is possibly to derive Cartesian coordinates from spherical coordinates using:

```{math}
\begin{bmatrix} x \\ y \\ z \end{bmatrix} =
\begin{bmatrix} r\sin\theta\cos\varphi \\ r\sin\theta\sin\varphi \\ r\cos\theta\end{bmatrix},
```

and is implemented by {func}`spherical_to_cartesian<differt.geometry.spherical_to_cartesian>`.
