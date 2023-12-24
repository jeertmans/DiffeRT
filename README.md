<div align="center">

# DiffeRT

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)

</div>


## Contributing

This project is built on both Python and Rust code.

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
nbsphinx_kernel_name = "DiffeRT"
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
