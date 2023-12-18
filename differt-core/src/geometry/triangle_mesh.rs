use std::fs::File;
use std::io::BufReader;


use numpy::ndarray::{s, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

struct TriangleMesh {
}

impl TriangleMesh {
    fn load_obj(filename: &str) -> Result<Self> {
        let input = BufReader::new(File::open(filename)?);
        let obj: obj::Obj = obj::load_obj(input)?;
        obj.try_into()
    }
}

impl TryFrom<RawObj> for TriangleMesh {
    type Error = &'static str;

    fn try_from(raw_obj: RawObj) -> Result<Self, Self::Error> {
        todo!()
    }
}


pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "trangle_mesh")?;
    m.add_function(wrap_pyfunction!(generate_all_path_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(
        generate_path_candidates_from_visibility_matrix,
        m
    )?)?;

    Ok(m)
}
