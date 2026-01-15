# Installation

DiffeRT consists of two Python modules, {mod}`differt` and {mod}`differt_core`,
where the latter is a direct dependency of the former, and both share the *exact same version*.

The main package, {mod}`differt`, uses Python-only code. The core package, {mod}`differt_core`,
even though only containing a very limited set of utilities, is written in Rust to offer good performances,
especially when reading large files or generating path candidates (see {ref}`path_candidates`).

Pre-built binaries are available for most platforms, and we recommend that users install DiffeRT from PyPI.

## Pip Install

The recommended installation procedure is with pip:

```bash
pip install differt
```

:::{important}
If you encounter an error with installing from PyPI, e.g.,
because it is missing pre-built binaries for your platform,
please report it as
[an issue in GitHub](https://github.com/jeertmans/DiffeRT/issues)!
:::

### About JAX

If you want to leverage the power of your GPU(s) or TPU(s), you may want
to look at [JAX's installation guide](https://jax.readthedocs.io/en/latest/installation.html),
which provides the necessary information about how to install JAX with support for your target device.

DiffeRT works seamlessly with JAX regardless of the active devices (i.e., CPU, GPU, or TPU).

### Optional dependencies

By default, DiffeRT installs only a limited set of dependencies and does not include
any plotting backend.

You may want to install those optional features by using *extras*[^1]:

- **Plotting backends:**
  - `matplotlib`: provide Matplotlib plotting backend;
  - `plotly`: provide Plotly plotting backend;
  - `vispy`: provide VisPy plotting backend;
- **VisPy-specific:**
  - `vispy-backend`: provide a default [backend toolkit for VisPy](https://vispy.org/installation.html);
- **Jupyter support:**
  - `jupyter`: provide support for Matplotlib and VisPy interactive plots inside notebooks;
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

To build this package locally, you need:

- [Rust](https://www.rust-lang.org/)\* stable toolchain;
- any modern C compiler\*;
- [just>=1.38.0](https://github.com/casey/just) to easily run commands listed in `justfile`s;
- and [uv>= 0.4.25](https://docs.astral.sh/uv/) to manage this project.

This project contains `justfile`s with recipes[^2] for most common
use cases, so feel free to use them instead of the commands listed below.

:::{note}
Requirements with an asterisk (\*) are only needed if you want to build {mod}`differt_core` from source. If you don't plan on making changes to the Rust code, see [building without Rust](#building-without-rust).
:::

[^2]: `just` is an alternative tool to Make, that provides more modern
  user experience. Enter `just` to list all available recipes.

### Building

Building (and installing) the packages is pretty simple thanks to uv:

```bash
uv sync
```

This will automatically download an appropriate Python version (if not available)
and install the packages in a virtual environment.

:::{note}
By default, `uv sync` will also install all the packages
listed in the `dev` list of the `[dependency-groups]` section from the main
`pyproject.toml` file.

You can opt out by specifying the `--no-dev` option.
:::

Alternatively, you can install specific groups[^3] with `uv sync --group <GROUP>`.

[^3]: The groups available for development are not the same as the extras of the {mod}`differt`
  packages (defined in `differt/pyproject.toml`). E.g., the `cuda` extra installs a GPU-capable
  JAX dependency (granted that you have a compatible CUDA installation). To specify {mod}`differt`'s
  extras, the easiest is to add a new group in the `[project.dependency-groups]` section of the root
  `pyproject.toml` file and specify extras there.

### Building without Rust

Rust (and the C compiler) are only needed if you want to build the {mod}`differt_core` package.
However, as most features are written in the {mod}`differt` package, you might be interested to skip building the core package and, instead, download pre-built binaries from PyPI.

To do so, you need to edit two files: `pyproject.toml` and `differt/pyproject.toml`.

```{code-block} toml
:caption: **Adding** one line to `pyproject.toml`
:emphasize-lines: 3

[tool.uv.workspace]
members = ["differt", "differt-core"]
exclude = ["differt-core"]  # Add this line
```

```{code-block} toml
:caption: **Removing** one line from `differt/pyproject.toml`
:class: text-gradient
:emphasize-lines: 2

[tool.uv.sources]
differt_core = {workspace = true}  # Remove (or comment) this line
```

After that, `uv` will know that it must download {mod}`differt_core` from PyPI and not look at the `differt-core` folder.
