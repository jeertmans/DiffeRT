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
> should be expected.
>
> If you have any suggestion regarding the development of this package,
> please open an [issue](/issues).

## Contributing

> [!IMPORTANT]
> The current documentation is very light and a more complete guide for
> new contributors will be written in the near future.
>
> Until then, do not hesitate to reach me for help with
> [GitHub issues](/issues)!

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

```
pdm install
```

If you need to install development dependencies, we recommend running:

```
pdm install -G:all
```

### Documentation

To generate the documentation, you first need to install an IPython kernel named
`DiffeRT`:

```
pdm run ipython kernel install --user --name=DiffeRT
```

If you want to use another name for your kernel, please also modify the
name in [`docs/source/conf.py`](docs/source/conf.py):

```python
nb_kernel_rgx_aliases = {".*": "DiffeRT"}
```

Then, you can build the docs with:

```
cd docs
pdm run make html
```

Finally, you can open `build/html/index.html` to see the generated docs.

### Testing

Both Rust and Python codebases have their own tests and benchmarks.

#### Testing Rust code

You can very easily test you code using Cargo:

```
cargo test
```

or benchmark it:

```
cargo bench
```

#### Testing Python code

in the same way, you can very test you code with Pytest:

```
pdm run pytest
```

or benchmark it:

```
pdm run pytest --benchmark-only
```

[pypi-version-badge]: https://img.shields.io/pypi/v/DiffeRT?label=DiffeRT&color=blueviolet
[pypi-version-url]: https://pypi.org/project/DiffeRT/
[pypi-python-version-badge]: https://img.shields.io/pypi/pyversions/DiffeRT?color=orange
[documentation-badge]: https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=documentation&up_color=blueviolet&up_message=online&url=https%3A%2F%2Feertmans.be%2FDiffeRT%2F
[documentation-url]: https://eertmans.be/DiffeRT/
[codecov-badge]: https://codecov.io/gh/jeertmans/DiffeRT/branch/main/graph/badge.svg?token=8P4DY9JCE4
[codecov-url]: https://codecov.io/gh/jeertmans/DiffeRT
[pdm-badge]: https://img.shields.io/badge/pdm-managed-blueviolet
[pdm-url]: https://pdm-project.org
