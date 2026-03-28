/// Build script for differt-core.
///
/// When the `xla-ffi` feature is enabled, this:
/// 1. Queries JAX for XLA FFI header locations
/// 2. Compiles the C++ FFI shim via cxx-build
use std::env;

fn main() {
    // Only build FFI when the feature is enabled
    #[cfg(feature = "xla-ffi")]
    {
        // Find the Python interpreter
        let python = env::var("PYTHON_SYS_EXECUTABLE")
            .unwrap_or_else(|_| "python3".to_string());

        // Query JAX for its XLA FFI include directory
        let output = std::process::Command::new(&python)
            .args(["-c", "from jax.ffi import include_dir; print(include_dir())"])
            .output()
            .expect("Failed to run python to find JAX include dir. Is JAX installed?");

        let include_path = String::from_utf8(output.stdout)
            .expect("Invalid UTF-8 from JAX include_dir()")
            .trim()
            .to_string();

        if include_path.is_empty() {
            panic!(
                "JAX include directory is empty. JAX >= 0.8.0 is required.\n\
                 stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        println!("cargo:rerun-if-changed=src/ffi.cc");
        println!("cargo:rerun-if-changed=include/ffi.h");

        cxx_build::bridge("src/accel/ffi.rs")
            .file("src/ffi.cc")
            .std("c++17")
            .include(&include_path)
            .include("include")
            .compile("differt-ffi");
    }
}
