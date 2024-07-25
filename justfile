# Run Python commands inside environment
env-run := if env_var("NO_RYE") == "1" { "" } else { "rye run" }

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
  pytest --benchmark-only {{ARGS}}

# Benchmark Rust code
[group('rust')]
[group('test')]
bench-rust *ARGS:
  cargo bench {{ARGS}}

# Build Python package(s)
[group('dev')]
build *ARGS:
  rye build {{ARGS}}

# Bump packages version
[group('dev')]
bump +ARGS="patch":
  {{env-run}} bump-my-version {{ARGS}}

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

# Build documentation
[group('dev')]
doc:
  cd docs

# Build and install Python packages
[group('dev')]
install:
  rye sync

# Run code linters and formatters
[group('dev')]
lint:
  {{env-run}} pre-commit run --all-files

alias fmt := lint

# Test code
[group('test')]
test: test-python test-rust

# Test Python code
[group('python')]
[group('test')]
test-python *ARGS:
  {{env-run}} pytest {{ARGS}}

# Test Rust code
[group('rust')]
[group('test')]
test-rust *ARGS:
  cargo test {{ARGS}}
