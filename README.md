<div align="center">
<img src="https://raw.githubusercontent.com/jeertmans/DiffeRT/main/static/logo_250px.png" alt="DiffeRT logo"></img>
</div>

<div align="center">

# DiffeRT

[![Latest Release][pypi-version-badge]][pypi-version-url]
[![Python version][pypi-python-version-badge]][pypi-version-url]
[![Documentation][documentation-badge]][documentation-url]
[![Codecov][codecov-badge]][codecov-url]
[![PDM][pdm-badge]][pdm-url]

</div>

## Usage

> [!WARNING]
> Until this package reaches version `0.1.x`, breaking changes
> should be expected. Checkout the [ROADMAP](./ROADMAP.md) for
> future features.
>
> If you have any suggestion regarding the development of this package,
> please open an [issue](https://github.com/jeertmans/DiffeRT/issues).

## Contributing

> [!IMPORTANT]
> The current documentation is very light and a more complete guide for
> new contributors will be written in the near future.
>
> Until then, do not hesitate to reach me for help with
> [GitHub issues](https://github.com/jeertmans/DiffeRT/issues)!

This project is built using both Python and Rust code, to provide an easy-to-use
but performant program. It also heavily uses the capabilities brought by
[JAX](https://github.com/google/jax) for numerical arrays.

### Requirements

To run build this package locally, you need:

- [Python 3.9](https://www.python.org/) or above;
- [Rust](https://www.rust-lang.org/) stable toolchain;
- [Maturin](https://www.maturin.rs/) for building Python bindings from Rust code;
- and [PDM](https://pdm-project.org) to manage all Python dependencies.

### Building locally

You can build the project locally using:

```bash
pdm install
```

### Documentation

To generate the documentation, please run the following:

```bash
cd docs
pdm run make html
```

Finally, you can open `build/html/index.html` to see the generated docs.

### Testing

Both Rust and Python codebases have their own tests and benchmarks.

#### Testing Rust code

You can very easily test you code using Cargo:

```bash
cargo test
```

or benchmark it:

```bash
cargo bench
```

#### Testing Python code

in the same way, you can very test you code with Pytest:

```bash
pdm run pytest
```

or benchmark it:

```bash
pdm run pytest --benchmark-only
```

[pypi-version-badge]: https://img.shields.io/pypi/v/DiffeRT?label=DiffeRT&color=blueviolet
[pypi-version-url]: https://pypi.org/project/DiffeRT/
[pypi-python-version-badge]: https://img.shields.io/pypi/pyversions/DiffeRT?color=orange
[documentation-badge]: https://readthedocs.org/projects/differt/badge/?version=latest
[documentation-url]: https://differt.readthedocs.io/latest/?badge=latest
[codecov-badge]: https://codecov.io/gh/jeertmans/DiffeRT/branch/main/graph/badge.svg?token=8P4DY9JCE4
[codecov-url]: https://codecov.io/gh/jeertmans/DiffeRT
[pdm-badge]: https://img.shields.io/badge/pdm-managed-blueviolet
[pdm-url]: https://pdm-project.org
