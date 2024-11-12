# Default command (list all commands)
default:
  @just --list

# Benchmark code
[group: 'test']
bench: bench-python bench-rust

# Benchmark Python code
[group: 'python']
[group: 'test']
bench-python *ARGS:
  uv run pytest -n0 --benchmark-only {{ARGS}}

# Benchmark Rust code
[group: 'rust']
[group: 'test']
bench-rust *ARGS:
  cargo bench {{ARGS}}

# Build Python package(s)
[group: 'dev']
build *ARGS:
  uv build {{ARGS}}

# Bump packages version
[group: 'dev']
bump +ARGS="patch":
  uv run bump-my-version {{ARGS}}

# Check the code can compile
[group: 'rust']
[group: 'test']
check:
  cargo check

# Clean build artifacts
[group: 'dev']
clean:
  cargo clean
  rm -rf dist

# Force reloading CUDA after suspend, see: https://github.com/ami-iit/jaxsim/issues/50#issuecomment-2022483137
[group: 'dev']
cuda-reload:
  sudo rmmod nvidia_uvm
  sudo modprobe nvidia_uvm

# List JAX's devices
[group: 'python']
[group: 'test']
devices:
  uv run python -c "import jax;print(jax.devices())"

# Build and install Python packages
[group: 'dev']
install *ARGS:
  uv sync {{ARGS}}

# Build and install Python package(s) using 'profiling' profile
[group: 'dev']
install-profiling *ARGS:
  uv sync --config-settings=build-args='--profile profiling' --reinstall-package differt_core {{ARGS}}

# Run code linters and formatters
[group: 'dev']
lint:
  uv run pre-commit run --all-files

alias fmt := lint

# Test code
[group: 'test']
test: test-python test-rust

# Test Python code
[group: 'python']
[group: 'test']
test-python *ARGS:
  uv run pytest {{ARGS}}

# Test Rust code
[group: 'rust']
[group: 'test']
test-rust *ARGS:
  cargo test {{ARGS}}

# Run jupyter-lab with a server that support reconnecting to running sessions
[group: 'dev']
remote-jupyter *ARGS:
  jupyverse --set kernels.require_yjs=true --set jupyterlab.server_side_execution=true --set auth.mode=noauth {{ARGS}}
