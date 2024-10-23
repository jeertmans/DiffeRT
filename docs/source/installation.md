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

If you consider contributing to DiffeRT (*thanks!*), or you want to on your own
local version, you will probably need to build the project from source.

TODO.
