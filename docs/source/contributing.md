# Contributing

Thank you for your interest in this project!

DiffeRT is mainly a one-person project, developed to support the author's research during his PhD.
However, it is also open source so that *anyone* can use this tool for its own research.

We hope that, by making this tool available to the public, other researchers will be
able to benefit from it!

## Reporting an issue

```{include} ../../README.md
:start-after: <!-- start reporting-an-issue -->
:end-before: <!-- end reporting-an-issue -->
```

## Local development

If you want to develop your own version of this package, e.g., to
later submit a [pull request](#proposing-changes), then you need to
[install this package from source](./installation.md#install-from-source).

Depending on what you aim to do, the following sections may help you.

:::{attention}
If you don't plan on changing the Rust code ({mod}`differt_core` package),
we recommend that you check out the [building without Rust](./installation.md#building-without-rust) section.
:::

### Updating the `differt-core` package

While the {mod}`differt` package is updated automatically (because built in `-e` edit mode)
whenever you update a Python file,
this is not true for the {mod}`differt_core` package and its Rust files.

If you make any change to the Rust code of the latter, you need to rebuild it
so that changes are taken into account:

```bash
uv sync --reinstall-package differt-core
```

Alternatively, you can install an import hook that will recompile the
{mod}`differt_core` package each time it is imported:

```bash
just hook-install
```

While recompiling a package that hasn't changed is relatively fast,
it may be better to disable the import hook when it is not needed.
You can do so by uninstalling it:

```bash
just hook-uninstall
```

### Documentation

To generate the documentation, please run the following (from the root folder):

```bash
just docs/build
```

Finally, you can open `docs/build/html/index.html` to see the generated docs.

Other recipes are available, and you can list them with `just docs/`.

### Testing

Both Rust and Python codebases have their own tests and benchmarks.

The following commands assume that you execute them from the root folder.

#### Testing Rust code

You can test Rust code using Cargo:

```bash
just test-rust
```

or benchmark it:

```bash
just bench-rust
```

#### Testing Python code

Similarly, you can test Python code with Pytest:

```bash
just test-python
```

or benchmark it:

```bash
just bench-python
```

## Proposing changes

Once you feel ready and think your contribution is ready to be reviewed,
create a [pull request](https://github.com/jeertmans/DiffeRT/pulls)
and wait for a reviewer to check your work!

Many thanks to you!

:::{tip}
If you are not familiar with GitHub pull requests, please
check out their
[Hello World tutorial](https://docs.github.com/en/get-started/quickstart/hello-world)
or [Fireship's 100 seconds introduction video](https://www.youtube.com/watch?v=8lGpZkjnkt4&ab_channel=Fireship).
:::
