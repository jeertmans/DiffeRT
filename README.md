<div align="center">
<img src="https://raw.githubusercontent.com/jeertmans/DiffeRT/main/static/logo_250px.png" alt="DiffeRT logo"></img>
</div>

<div align="center">

# DiffeRT

[![Latest Release][pypi-version-badge]][pypi-version-url]
[![Python version][pypi-python-version-badge]][pypi-version-url]
[![Documentation][documentation-badge]][documentation-url]
[![DOI][doi-badge]][doi-url]
[![Codecov][codecov-badge]][codecov-url]

</div>

## Usage

> [!WARNING]
> Until this package reaches version `v0.1.0`, breaking changes
> should be expected.
> Checkout the [v0.1.0](https://github.com/jeertmans/DiffeRT/milestone/1)
> and [v1.0.0](https://github.com/jeertmans/DiffeRT/milestone/2) milestones for future
> features.
>
> If you have any suggestion regarding the development of this package,
> please open an [issue](https://github.com/jeertmans/DiffeRT/issues).

The easiest way to install DiffeRT is through pip:

```bash
pip install differt
```

We provide pre-built binaries for most platforms. If you want (or need)
to build the package from the source distribution, check out the
requirements below.

## Contributing

> [!IMPORTANT]
> The current documentation is still incomplete and a more complete guide for
> new contributors will be written in the near future.
>
> Until then, do not hesitate to reach for help through
> [GitHub issues](https://github.com/jeertmans/DiffeRT/issues)!

This project is built using both Python and Rust code, to provide an easy-to-use
but performant program. It also heavily uses the capabilities brought by
[JAX](https://github.com/google/jax) for numerical arrays.

### Requirements

To run build this package locally, you need:

- [Python 3.10](https://www.python.org/) or above;
- [Rust](https://www.rust-lang.org/) stable toolchain;
- any modern C compiler;
- [just](https://github.com/casey/just) to easily run commands listed in `justfile`s;
- [Maturin](https://www.maturin.rs/) for building Python bindings from Rust code;
- and [uv](https://docs.astral.sh/uv/) to manage this project.

This project contains `justfile`s with recipes[^1] for most common
use cases, so feel free to use them instead of the commands listed below/

[^1]: `just` is as alternative tool to Make, that provides more modern
  user experience. Enter `just` to list all available recipes.

## Local development

The following commands assume that you installed
the project locally with:

```bash
uv sync
```

and that you activated the corresponding Python virtual environment:

```bash
. .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### Documentation

To generate the documentation, please run the following:

```bash
just docs/build
```

Finally, you can open `docs/build/html/index.html` to see the generated docs.

Other recipes are available, and you can list them with `just docs/`.

### Testing

Both Rust and Python codebases have their own tests and benchmarks.

#### Testing Rust code

You can test Rust code using Cargo:

```bash
cargo test
```

or benchmark it:

```bash
cargo bench
```

#### Testing Python code

Similarly, you can test Python code with Pytest:

```bash
pytest
```

or benchmark it:

```bash
pytest --benchmark-enable differt/tests/benchmarks
```

## Citing

If you use this software, please cite it as
[detailed here](https://differt.readthedocs.io/latest/citing.html).

[pypi-version-badge]: https://img.shields.io/pypi/v/DiffeRT?label=DiffeRT&color=blueviolet
[pypi-version-url]: https://pypi.org/project/DiffeRT/
[pypi-python-version-badge]: https://img.shields.io/pypi/pyversions/DiffeRT?color=orange
[documentation-badge]: https://readthedocs.org/projects/differt/badge/?version=latest
[documentation-url]: https://differt.readthedocs.io/latest/?badge=latest
[doi-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.13895021.svg
[doi-url]: https://doi.org/10.5281/zenodo.13895021
[codecov-badge]: https://codecov.io/gh/jeertmans/DiffeRT/branch/main/graph/badge.svg?token=v63alnTWzu
[codecov-url]: https://codecov.io/gh/jeertmans/DiffeRT
