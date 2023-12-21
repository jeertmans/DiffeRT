use numpy::ndarray::parallel::prelude::*;
use numpy::ndarray::{s, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Generate an array of all path candidates (assuming fully connected primitives).
#[pyfunction]
pub fn generate_all_path_candidates(
    py: Python<'_>,
    num_primitives: u32,
    order: u32,
) -> &PyArray2<u32> {
    if order == 0 {
        // One path of size 0
        return Array2::default((0, 1)).into_pyarray(py);
    } else if num_primitives == 0 {
        // Zero path of size order
        return Array2::default((order as usize, 0)).into_pyarray(py);
    } else if order == 1 {
        let mut path_candidates = Array2::default((1, num_primitives as usize));

        for j in 0..num_primitives {
            path_candidates[(0, j as usize)] = j;
        }
        return path_candidates.into_pyarray(py);
    }
    let num_choices = (num_primitives - 1) as usize;
    let num_candidates_per_batch = num_choices.pow(order - 1);
    let num_candidates = (num_primitives as usize) * num_candidates_per_batch;

    let mut path_candidates = Array2::default((order as usize, num_candidates));
    let mut batch_size = num_candidates_per_batch;
    let mut fill_value = 0;

    for i in 0..(order as usize) {
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

#[pyclass]
pub struct AllPathCandidates {
    /// Number of primitives.
    num_primitives: u32,
    /// Path order.
    order: u32,
    /// Exact number of path candidates that will be generated.
    num_candidates: usize,
    /// The index of the current path candidate.
    index: usize,
    /// Last path candidate.
    path_candidate: Vec<u32>,
    counter: Vec<u32>,
}

impl AllPathCandidates {
    #[inline]
    fn new(num_primitives: u32, order: u32) -> Self {
        let num_choices = num_primitives.saturating_sub(1) as usize;
        let num_candidates_per_batch = num_choices.pow(order.saturating_sub(1));
        let num_candidates = (num_primitives as usize) * num_candidates_per_batch;

        Self {
            num_primitives,
            order,
            num_candidates,
            index: 0,
            path_candidate: (0..order).collect(),
            counter: vec![2; order as usize],
        }
    }
}

impl Iterator for AllPathCandidates {
    type Item = Vec<u32>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;

        let path_candidate = self.path_candidate.clone();

        let start = self
            .counter
            .iter()
            .rposition(|&count| count < self.num_primitives)?;

        println!("Actual counter: {:?}", self.counter);
        println!("start index:{:?}", start);

        self.counter[start] += 1;
        self.path_candidate[start] = (self.path_candidate[start] + 1) % self.num_primitives;

        for i in (start + 1)..(self.order as usize) {
            self.path_candidate[i] = (self.path_candidate[i - 1] + 1) % self.num_primitives;
            self.counter[i] = 2;
        }
        println!("{:?}", self.counter);

        Some(path_candidate)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.num_candidates - self.index;

        (rem, Some(rem))
    }

    #[inline]
    fn count(self) -> usize {
        self.num_candidates
    }
}

#[pymethods]
impl AllPathCandidates {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(mut slf: PyRefMut<'py, Self>, py: Python<'py>) -> Option<&'py PyArray1<u32>> {
        slf.next().map(|v| PyArray1::from_vec(py, v))
    }
}

/// Variant of eponym function, but return an iterator instead.
#[pyfunction]
pub fn generate_all_path_candidates_iter(
    _py: Python<'_>,
    num_primitives: u32,
    order: u32,
) -> AllPathCandidates {
    AllPathCandidates::new(num_primitives, order)
}

/// Return a list of list of indices where the matrix is true.
pub fn where_true(matrix: &ArrayView2<bool>) -> Vec<Vec<usize>> {
    matrix
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| {
            row.indexed_iter()
                .filter_map(|(index, &item)| if item { Some(index) } else { None })
                .collect()
        })
        .collect()
}

/// Generate an array of all path candidates from a visibility matrix.
#[pyfunction]
pub fn generate_path_candidates_from_visibility_matrix<'py>(
    py: Python<'py>,
    visibility_matrix: PyReadonlyArray2<'py, bool>,
    order: u32,
) -> &'py PyArray2<u32> {
    let num_primitives = visibility_matrix.shape()[0];

    if order <= 1 || num_primitives == 0 {
        return generate_all_path_candidates(py, num_primitives as u32, order);
    }

    let _indices = where_true(&visibility_matrix.as_array());
    todo!()
}

#[pyclass]
pub struct PathCandidates {
    /// Indices.
    indices: Vec<Vec<usize>>,
}

#[pymethods]
impl PathCandidates {
    #[new]
    fn py_new(visibility_matrix: PyReadonlyArray2<'_, bool>) -> Self {
        Self {
            indices: where_true(&visibility_matrix.as_array()),
        }
    }
}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "utils")?;
    m.add_function(wrap_pyfunction!(generate_all_path_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(
        generate_path_candidates_from_visibility_matrix,
        m
    )?)?;

    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{array, Array};
    use rstest::*;

    #[rstest]
    #[case(0, 0, Array::zeros((0, 1)))]
    #[case(3, 0, Array::zeros((0, 1)))]
    #[case(0, 3, Array::zeros((3, 0)))]
    #[case(9, 1, array![[0, 1, 2, 3, 4, 5, 6, 7, 8]])]
    #[case(3, 1, array![[0, 1, 2]])]
    #[case(
        3,
        2,
        array![[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]].t().to_owned()
    )]
    #[case(
        3,
        3,
        array![
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
        ].t().to_owned()
    )]
    fn test_generate_all_path_candidates(
        #[case] num_primitives: u32,
        #[case] order: u32,
        #[case] expected: Array2<u32>,
    ) {
        Python::with_gil(|py| {
            let got = generate_all_path_candidates(py, num_primitives, order);

            assert_eq!(got.to_owned_array(), expected);
        });
    }

    /*
    #[rstest]
    #[should_panic] // Because we do not handle this edge case (empty iterator)
    #[case(0, 0)]
    #[should_panic] // Because we do not handle this edge case (empty iterator)
    #[case(3, 0)]
    #[should_panic] // Because we do not handle this edge case (empty iterator)
    #[case(0, 3)]
    #[case(9, 1)]
    #[case(3, 1)]
    #[case(3, 2)]
    #[case(3, 3)]
    fn test_generate_all_path_candidates_iter(#[case] num_primitives: u32, #[case] order: u32) {
        Python::with_gil(|py| {
            let got: Vec<Vec<_>> =
                generate_all_path_candidates_iter(py, num_primitives, order).collect();
            let expected = generate_all_path_candidates(py, num_primitives, order);

            let got = PyArray2::from_vec2(py, &got).unwrap();

            assert_eq!(got.to_owned_array().t(), expected.to_owned_array());
        });
    }*/

    #[rstest]
    #[case(
        array![
            [false, true, true, true],
            [true, false, true, true],
            [true, true, true, false]
        ],
        vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 2],
        ]
    )]
    #[case(
        array![
            [false, false, false],
        ],
        vec![
            vec![],
        ]
    )]
    fn test_where_true(#[case] matrix: Array2<bool>, #[case] expected: Vec<Vec<usize>>) {
        let got = where_true(&matrix.view());
        assert_eq!(got, expected);
    }
}
