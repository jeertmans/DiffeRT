<div align="center">
<img src="https://raw.githubusercontent.com/jeertmans/DiffeRT/main/static/logo_250px.png" alt="DiffeRT logo"></img>
</div>

<div align="center">

# DiffeRT-core

[![Latest Release][pypi-version-badge]][pypi-version-url]
[![Python version][pypi-python-version-badge]][pypi-version-url]
[![Documentation][documentation-badge]][documentation-url]
[![Codecov][codecov-badge]][codecov-url]

</div>

This package contains the core backend of
[DiffeRT](https://pypi.org/project/DiffeRT/),
implemented in Rust for performances.

As a result, both `differt` and `differt-core` will
share the same version, and `differt` directly depends on `differt-core`.

However, you can decide to only install `differt-core`
if you want to use features that are specific to this package.

The installation procedure, contributing guidelines, and documentation,
are shared with the
[main DiffeRT package](https://github.com/jeertmans/DiffeRT).

[pypi-version-badge]: https://img.shields.io/pypi/v/DiffeRT-core?label=DiffeRT-core&color=blueviolet
[pypi-version-url]: https://pypi.org/project/DiffeRT-core/
[pypi-python-version-badge]: https://img.shields.io/pypi/pyversions/DiffeRT-core?color=orange
[documentation-badge]: https://readthedocs.org/projects/differt-core/badge/?version=latest
[documentation-url]: https://differt.readthedocs.io/latest/?badge=latest
[codecov-badge]: https://codecov.io/gh/jeertmans/DiffeRT-core/branch/main/graph/badge.svg?token=8P4DY9JCE4
[codecov-url]: https://codecov.io/gh/jeertmans/DiffeRT-core
