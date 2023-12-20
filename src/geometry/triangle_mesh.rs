use std::fs::File;
use std::io::BufReader;

use numpy::ndarray::{s, Array2, ArrayView2, Axis};
//use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use obj::raw::object::{parse_obj, RawObj};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass]
struct TriangleMesh {
    /// Array of size [num_vertices 3].
    vertices: Vec<(f32, f32, f32)>,
    /// Array of size [num_triangles 3].
    triangles: Vec<(usize, usize, usize)>,
}

#[pymethods]
impl TriangleMesh {
    #[classmethod]
    fn load_obj(_: &PyType, filename: &str) -> PyResult<Self> {
        let input = BufReader::new(File::open(filename)?);
        let obj: RawObj = parse_obj(input).map_err(|err| {
            PyValueError::new_err(format!("An error occured while reading obj file: {}", err))
        })?;
        //obj.try_into()
        todo!()
    }
}

impl TryFrom<RawObj> for TriangleMesh {
    type Error = PyValueError;

    fn try_from(raw_obj: RawObj) -> Result<Self, Self::Error> {
        /*
        use obj::raw::object::Polygon::*;
        let vertices = raw_obj
            .positions
            .into_iter()
            .map(|(x, y, z, _)| (x, y, z))
            .collect();
        let triangles: Vec<Result<_, Self::Error>> = raw_obj.polygons.into_iter().map(
            |polygon| match polygon {
                P(v) if v.len() == 3 => {Ok((v[0], v[1], v[2]))},
                _ => Err(PyValueError::new_err("Cannot create TriangleMesh from an object that contains something else than triangles")),
            }).collect();

        Ok(Self {
            vertices,
            triangles: triangles?,
        })
        */
        todo!()
    }
}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "trangle_mesh")?;
    m.add_class::<TriangleMesh>()?;

    Ok(m)
}
