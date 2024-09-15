# Default command (list all commands)
default:
  @just --list

# Benchmark code
[group('test')]
bench: bench-python bench-rust

# Benchmark Python code
[group('python')]
[group('test')]
bench-python *ARGS:
  uv run pytest --benchmark-only {{ARGS}}

# Benchmark Rust code
[group('rust')]
[group('test')]
bench-rust *ARGS:
  cargo bench {{ARGS}}

# Build Python package(s)
[group('dev')]
build *ARGS:
  uv build {{ARGS}}

# Bump packages version
[group('dev')]
bump +ARGS="patch":
  uv run bump-my-version {{ARGS}}

# Check the code can compile
[group('rust')]
[group('test')]
check:
  cargo check

# Clean build artifacts
[group('dev')]
clean:
  cargo clean
  rm -rf dist

# Build and install Python packages
[group('dev')]
install:
  uv sync

# Run code linters and formatters
[group('dev')]
lint:
  uv pre-commit run --all-files

alias fmt := lint

# Test code
[group('test')]
test: test-python test-rust

# Test Python code
[group('python')]
[group('test')]
test-python *ARGS:
  uv pytest {{ARGS}}

# Test Rust code
[group('rust')]
[group('test')]
test-rust *ARGS:
  cargo test {{ARGS}}
