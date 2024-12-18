# Installation

DiffeRT consists of two Python modules, {mod}`differt` and {mod}`differt_core`,
where the latter is direct dependency of the former, and both share the *exact same version*.

The main package, {mod}`differt`, uses Python-only code. The core package, {mod}`differt_core`,
even though only containing a very limited set of utitilies, is written in Rust to offer good performances,
especially when reading large files or generating path candidates (see {ref}`path_candidates`).

Pre-built binaries are available for most platforms, and we recommend users to install DiffeRT with pip.

## Pip Install

The recommended installation procedure is with pip:

```bash
pip install differt
```

:::{important}
If you encounter an error with installing from pip, e.g.,
because it is missing pre-built binaries for your platform,
please report it as
[an issue in GitHub](https://github.com/jeertmans/DiffeRT/issues)!
:::

### About JAX

If you want to leverage the power of your GPU(s) or TPU(s), you may want
to look at [JAX's installation guide](https://jax.readthedocs.io/en/latest/installation.html),
as they provide the necessary information about how to install JAX with support for your target device.

DiffeRT works seamlessly with JAX regardless of the active devices (i.e, CPU, GPU, or TPU).

### Optional dependencies

By default, DiffeRT will only install a limited set of dependencies, and will not include
any plotting backend, for example.

You may want to install those optional features by using *extras*[^1]:

- **Plotting backends:**
  - `matplotlib`: provide Matplotlib plotting backend;
  - `plotly`: provide Plotly plotting backend;
  - `vispy`: provide VisPy plotting backend;
- **VisPy-specific:**
  - `vispy-backend`: provide a default [backend toolkit for VisPy](https://vispy.org/installation.html);
- **Jupyter support:**
  - `jupyter`: provide support for Matplotlib and VisPy interactive plot inside notebooks;
- **Aliases:**
  - `all`: alias to `jupyter,matplotlib,plotly,vispy,vispy-backend`;

[^1]: Extras are installed with `pip install "differt[extra_1,extra_2,...]`.

## Install from source

If you consider contributing to DiffeRT (*thanks!*), or you want to work on your own
local version, you will probably need to build the project from source.

This project is built using both Python and Rust code, to provide an easy-to-use
but performant program. It also heavily uses the capabilities brought by
[JAX](https://github.com/jax-ml/jax) for numerical arrays.

### Requirements

To run build this package locally, you need:

- [Rust](https://www.rust-lang.org/) stable toolchain;
- any modern C compiler;
- [just>=1.38.0](https://github.com/casey/just) to easily run commands listed in `justfile`s;
- and [uv>= 0.4.25](https://docs.astral.sh/uv/) to manage this project.

This project contains `justfile`s with recipes[^2] for most common
use cases, so feel free to use them instead of the commands listed below/

[^2]: `just` is as alternative tool to Make, that provides more modern
  user experience. Enter `just` to list all available recipes.

### Building

Building the packages is pretty simple thanks to uv:

```bash
uv sync
```

This will automatically download an appropriate Python version (if not available)
and install the packages in a virtual environment.

:::{note}
By default, `uv sync` will also install all the packages
listed in the `dev-dependencies` list of the `[tool.uv]` section from the main
`pyproject.toml` file.

You can opt out by specifying the `--no-dev` option.
:::

Alternatively, you can install specific extras[^3] with `uv sync --extra <EXTRA>`.

[^3]: The extras available for development are not the same as the extras of the {mod}`differt`
  packages (defined in `differt/pyproject.toml`). E.g., the `cuda` extra installs a GPU-capable
  JAX dependency (granted that you have a compatible CUDA installation). To specify {mod}`differt`'s
  extras, the easiest is to add a new extra in the `[project.optional-dependencies]` section of the root
  `pyproject.toml` file and specify extras there.
