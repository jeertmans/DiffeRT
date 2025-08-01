<div align="center">
<img src="https://raw.githubusercontent.com/jeertmans/DiffeRT/main/static/logo_250px.png" alt="DiffeRT logo"></img>
</div>

<div align="center">

# DiffeRT

Differentiable Ray Tracing toolbox for Radio Propagation powered by the [JAX ecosystem](https://github.com/jax-ml/jax).

[![Latest Release][pypi-version-badge]][pypi-version-url]
[![Python version][pypi-python-version-badge]][pypi-version-url]
[![Documentation][documentation-badge]][documentation-url]
[![DOI][doi-badge]][doi-url]
[![Codecov][codecov-badge]][codecov-url]

</div>

## Usage

> [!WARNING]
> This package is still under important development, see
> the [v1.0.0](https://github.com/jeertmans/DiffeRT/milestone/2) milestone for future
> features, and the [CHANGELOG](https://github.com/jeertmans/DiffeRT/blob/main/CHANGELOG.md)
> for versioning policy.
>
> If you have any suggestion regarding the development of this package,
> please open an [issue](https://github.com/jeertmans/DiffeRT/issues).

The easiest way to install DiffeRT is through pip:

```bash
pip install differt
```

We provide pre-built binaries for most platforms. If you want (or need)
to build the package from the source distribution,
or want to customize the installation (e.g., with GPU support), check out the
[installation guide](https://differt.readthedocs.io/latest/installation.html).

### Reporting an issue

<!-- start reporting-an-issue -->

If you think you found a bug,
an error in the documentation,
or wish there was some feature that is currently missing,
we would love to hear from you!

The best way to reach us is via the
[GitHub issues](https://github.com/jeertmans/DiffeRT/issues?q=is%3Aissue).
If your problem is not covered by an already existing (closed or open) issue,
then we suggest you create a
[new issue](https://github.com/jeertmans/DiffeRT/issues/new/choose).
You can choose from a list of templates, or open a
[blank issue](https://github.com/jeertmans/DiffeRT/issues/new)
if your issue does not fit one of the proposed topics.

The more precise you are in the description of your problem, the faster we will
be able to help you!

If you rather have a question than a problem,
then it is probably best suited to ask it in the
[Q&A category of the discussions](https://github.com/jeertmans/DiffeRT/discussions/categories/q-a).

<!-- end reporting-an-issue -->

## Contributing

All types of contributions are more than welcome!

Please follow the
[contributing guide](https://differt.readthedocs.io/latest/contributing.html)
for a detailed step-by-step procedure.

## Citing

<!-- start citing -->

If you use this software, please cite it as:

```bibtex
@software{Eertmans_Differentiable_Ray_Tracing,
  title   = {{DiffeRT: A Differentiable Ray Tracing Toolbox for Radio Propagation Simulations}},
  author  = {Eertmans, Jérome},
  url     = {https://github.com/jeertmans/DiffeRT},
  license = {MIT},
  version = {v0.5.0}
}
```

For other citation formats, please refer to the [*Cite this repository*](https://github.com/jeertmans/DiffeRT) button the main page of our GitHub repository or to our [Zenodo records](https://doi.org/10.5281/zenodo.11386432).

Thank you for using this software and helping us!

<!-- end citing -->

[pypi-version-badge]: https://img.shields.io/pypi/v/DiffeRT?label=DiffeRT&color=blueviolet
[pypi-version-url]: https://pypi.org/project/DiffeRT/
[pypi-python-version-badge]: https://img.shields.io/pypi/pyversions/DiffeRT?color=orange
[documentation-badge]: https://readthedocs.org/projects/differt/badge/?version=latest
[documentation-url]: https://differt.readthedocs.io/latest/?badge=latest
[doi-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.11386432.svg
[doi-url]: https://doi.org/10.5281/zenodo.11386432
[codecov-badge]: https://codecov.io/gh/jeertmans/DiffeRT/branch/main/graph/badge.svg?token=v63alnTWzu
[codecov-url]: https://codecov.io/gh/jeertmans/DiffeRT
