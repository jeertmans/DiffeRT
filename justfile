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
install:
  uv sync

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
