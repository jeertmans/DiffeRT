/// Build script for differt-core.
///
/// When the `xla-ffi` feature is enabled, this:
/// 1. Queries JAX for XLA FFI header locations
/// 2. Compiles the C++ FFI shim via cxx-build

fn main() {
    // Only build FFI when the feature is enabled
    #[cfg(feature = "xla-ffi")]
    {
        use std::env;

        // Find the Python interpreter
        let python = env::var("PYTHON_SYS_EXECUTABLE").unwrap_or_else(|_| "python3".to_string());

        // Query JAX for its XLA FFI include directory
        let output = std::process::Command::new(&python)
            .args([
                "-c",
                "from jax.ffi import include_dir; print(include_dir())",
            ])
            .output();

        let include_path = match output {
            Ok(ref out) if out.status.success() => {
                let path = String::from_utf8(out.stdout.clone())
                    .expect("Invalid UTF-8 from JAX include_dir()")
                    .trim()
                    .to_string();
                if path.is_empty() {
                    None
                } else {
                    Some(path)
                }
            }

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
                "cargo:warning=JAX not found or missing jax.ffi.include_dir(). \
                 XLA FFI shim will not be compiled. \
                 Install JAX >= 0.8.0 to enable XLA FFI support."
            );
        }
    }
}
