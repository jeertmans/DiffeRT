#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

use numpy::PyArray2;
use pyo3::{types::IntoPyDict, Python};

use differt_core::rt::utils as rt_utils;

fn large_visibility_matrix<'py>(py: Python<'py>) -> &'py PyArray2<bool> {
    let np = py.import("numpy").unwrap();
    let locals = [("np", np)].into_py_dict(py);
    py.eval(
        "(np.random.rand(1000, 1000) > 0.5).astype(bool)",
        Some(locals),
        None,
    )
    .unwrap()
    .extract()
    .unwrap()
}

#[bench]
fn bench_rt_utils_where_true(bencher: &mut Bencher) {
    Python::with_gil(|py| {
        let binding = large_visibility_matrix(py).readonly();
        let matrix = binding.as_array();
        bencher.iter(|| rt_utils::where_true(black_box(&matrix)));
    });
}

#[bench]
fn bench_rt_utils_generate_all_path_candidates(bencher: &mut Bencher) {
    Python::with_gil(|py| {
        bencher.iter(|| rt_utils::generate_all_path_candidates(py, black_box(100), black_box(3)));
    });
}
