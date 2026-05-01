/// Build script for differt-core.
///
/// 1. Queries JAX for XLA FFI header locations
/// 2. Compiles the C++ FFI shim via cxx-build
use std::process::exit;

fn main() {
    // Only build FFI when the feature is enabled
    use std::{env, path::PathBuf, str::from_utf8};

    // Find the Python interpreter:
    // 1. PYTHON env var (covers VIRTUAL_ENV and explicit overrides)
    // 2. pyo3_build_config: the interpreter pyo3 itself was built against
    // 3. Fall back to "python3"
    let python = env::var("PYTHON")
        .ok()
        .or_else(|| pyo3_build_config::get().executable.clone())
        .unwrap_or_else(|| "python3".to_owned());

    // Query JAX for its XLA FFI include directory
    let output = std::process::Command::new(&python)
        .args([
            "-c",
            "from jax.ffi import include_dir; print(include_dir())",
        ])
        .output();

    let include_path = match output {
        Ok(ref out) if out.status.success() => {
            let path = from_utf8(&out.stdout)
                .expect("Invalid UTF-8 from JAX include_dir()")
                .trim()
                .to_string();
            if path.is_empty() {
                None
            } else {
                Some(PathBuf::from(path))
            }
        },
        _ => None,
    };

    if let Some(include_path) = include_path {
        println!("cargo:rerun-if-changed=src/ffi.cc");
        println!("cargo:rerun-if-changed=include/ffi.h");

        cxx_build::bridge("src/accel/ffi.rs")
            .file("src/ffi.cc")
            .std("c++17")
            .include(&include_path)
            .include("include")
            .compile("differt-ffi");
    } else {
        println!(
            "cargo:error=JAX not found or missing jax.ffi.include_dir(). Python interpreter used: \
             {python}"
        );
        exit(1);
    }
}
