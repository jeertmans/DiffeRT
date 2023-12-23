<div align="center">

# DiffeRT

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)

</div>


## Contributing

This project is built on both Python and Rust code.

### Requirements

To run build this package locally, you need:

- Python 3.9 or above;
- Rust stable toolchain;
- Maturin for building Python bindings from Rust code;
- and PDM to manage all Python dependencies.

### Building locally

You can build the project locally using:

```
pdm install
```

If you need to install development dependencies, we recommend running:

```
pdm install -G:all
```

### Creating an IPython kernel

```
pdm run ipython kernel install --user --name=DiffeRT
```
