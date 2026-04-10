/// Build script for differt-core.
///
/// When the `xla-ffi` feature is enabled, this:
/// 1. Queries JAX for XLA FFI header locations
/// 2. Compiles the C++ FFI shim via cxx-build

fn main() {
    // Only build FFI when the feature is enabled
    #[cfg(feature = "xla-ffi")]
    {
        use std::{env, path::PathBuf, str::from_utf8};

        // Find the Python interpreter, using the same resolution order as Jerome's approach:
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
            .output()
            .expect("failed to execute Python interpreter");

        if !output.status.success() {
            let stdout = from_utf8(&output.stdout).unwrap_or("");
            let stderr = from_utf8(&output.stderr).unwrap_or("");
            eprint!("{stdout}{stderr}");
            panic!(
                "JAX not found or missing jax.ffi.include_dir(). Install JAX >= 0.8.0 to enable \
                 XLA FFI support. Python interpreter used: {python}"
            );
        }

        let include_path = PathBuf::from(
            from_utf8(&output.stdout)
                .expect("Invalid UTF-8 from JAX include_dir()")
                .trim(),
        );

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
