use numpy::ndarray::{s, Array2};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn generate_path_candidates(
    py: Python<'_>,
    num_primitives: usize,
    order: usize,
) -> &PyArray2<usize> {
    if num_primitives == 0 || order == 0 {
        return Array2::default((0, 0)).into_pyarray(py);
    } else if order == 1 {
        let mut path_candidates = Array2::default((1, num_primitives));

        for i in 0..num_primitives {
            path_candidates[(0, i)] = i;
        }
        return path_candidates.into_pyarray(py);
    }
    let num_choices = num_primitives - 1;
    let num_candidates_per_batch = num_choices.pow((order - 1) as u32);
    let num_candidates = num_primitives * num_candidates_per_batch;

    let mut path_candidates = Array2::default((order, num_candidates));
    let mut batch_size = num_candidates_per_batch;
    let mut fill_value = 0;

    for i in 0..order {
        for j in (0..num_candidates).step_by(batch_size) {
            if i > 0 && fill_value == path_candidates[(i - 1, j)] {
                fill_value = (fill_value + 1) % num_primitives;
            }

            path_candidates
                .slice_mut(s![i, j..(j + batch_size)])
                .fill(fill_value);
            fill_value = (fill_value + 1) % num_primitives;
        }
        batch_size /= num_choices;
    }

    path_candidates.into_pyarray(py)
}

/// Core of DiffeRT module, implemented in Rust.
#[pymodule]
fn differt_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_path_candidates, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;

    use pyo3::{types::IntoPyDict, Python};

    #[rstest] // TODO: FIXME on Windows because uint is not consistent
    #[case(0, 0, "np.empty((0, 0), dtype=np.uint)")]
    #[case(3, 0, "np.empty((0, 0), dtype=np.uint)")]
    #[case(0, 3, "np.empty((0, 0), dtype=np.uint)")]
    #[case(9, 1, "np.arange(9, dtype=np.uint).reshape(1, 9)")]
    #[case(3, 1, "np.array([[0, 1, 2]], dtype=np.uint)")]
    #[case(
        3,
        2,
        "np.array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]], dtype=np.uint).T"
    )]
    #[case(
        3,
        3,
        "np.array(
            [
                [0, 1, 2],
                [0, 1, 0],
                [0, 2, 1],
                [0, 2, 0],
                [1, 0, 1],
                [1, 0, 2],
                [1, 2, 0],
                [1, 2, 1],
                [2, 0, 2],
                [2, 0, 1],
                [2, 1, 2],
                [2, 1, 0],
            ], dtype=np.uint
        ).T"
    )]
    fn test_generate_path_candidates(
        #[case] num_primitives: usize,
        #[case] order: usize,
        #[case] code: &str,
    ) {
        Python::with_gil(|py| {
            let np = py.import("numpy").unwrap();
            let locals = [("np", np)].into_py_dict(py);
            let got = generate_path_candidates(py, num_primitives, order);
            let expected: &PyArray2<usize> = py
                .eval(code, Some(locals), None)
                .unwrap()
                .extract()
                .unwrap();

            assert_eq!(got.to_owned_array(), expected.to_owned_array());
        });
    }
}