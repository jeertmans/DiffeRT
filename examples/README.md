# Examples

This directory contains example scripts and projects demonstrating the usage of DiffeRT for various tasks.

You can install the required dependencies to run all these examples using:

```bash
uv sync --group examples
```

For testing, you can install the testing dependencies using:

```bash
uv sync --group test-examples
```

It is also possible to install dependencies for individual examples, see below for details.

## Sampling Paths with Machine Learning

In this example, see [train_path_sampler.py](./train_path_sampler.py), a Machine Learning model is trained to sample [path candidates](https://differt.eertmans.be/latest/notebooks/sampling_paths.html) to avoid exhaustive path tracing.

You can install the required dependencies for this example using:

```bash
uv sync --group example-sampling-paths
```

and train the model using:

```bash
python examples/train_path_sampler.py
```

Append `--help` to see all available options.