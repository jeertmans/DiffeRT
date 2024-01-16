use std::{fs::File, io::BufReader};

use numpy::{Element, PyArray2};
use obj::raw::object::{parse_obj, RawObj};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

#[pyclass]
struct TriangleMesh {
    /// Array of size [num_vertices 3].
    vertices: Vec<(f32, f32, f32)>,
    /// Array of size [num_triangles 3].
    triangles: Vec<(usize, usize, usize)>,
}

#[inline]
fn pyarray2_from_vec_tuple<'py, T: Copy + Element>(
    py: Python<'py>,
    v: &[(T, T, T)],
) -> &'py PyArray2<T> {
    let n = v.len();
    unsafe {
        let arr = PyArray2::<T>::new(py, [n, 3], false);

        for i in 0..n {
            let tup = v.get_unchecked(i);
            arr.uget_raw([i, 0]).write(tup.0);
            arr.uget_raw([i, 1]).write(tup.1);
            arr.uget_raw([i, 2]).write(tup.2);
        }
        arr
    }
}

#[pymethods]
impl TriangleMesh {
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        pyarray2_from_vec_tuple(py, &self.vertices)
    }

    #[getter]
    fn triangles<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
        pyarray2_from_vec_tuple(py, &self.triangles)
    }

    #[classmethod]
    fn load_obj(_: &PyType, filename: &str) -> PyResult<Self> {
        let input = BufReader::new(File::open(filename)?);
        let obj: RawObj = parse_obj(input).map_err(|err| {
            PyValueError::new_err(format!("An error occured while reading obj file: {}", err))
        })?;
        obj.try_into()
    }
}

impl TryFrom<RawObj> for TriangleMesh {
    type Error = PyErr;

    fn try_from(raw_obj: RawObj) -> Result<Self, Self::Error> {
        use obj::raw::object::Polygon::*;

        let vertices = raw_obj
            .positions
            .into_iter()
            .map(|(x, y, z, _)| (x, y, z))
            .collect();

        let mut triangles = Vec::with_capacity(raw_obj.polygons.len());

        for polygon in raw_obj.polygons {
            match polygon {
                P(v) if v.len() == 3 => {
                    triangles.push((v[0], v[1], v[2]));
                },
                PT(v) if v.len() == 3 => {
                    triangles.push((v[0].0, v[1].0, v[2].0));
                },
                PN(v) if v.len() == 3 => {
                    triangles.push((v[0].0, v[1].0, v[2].0));
                },
                PTN(v) if v.len() == 3 => {
                    triangles.push((v[0].0, v[1].0, v[2].0));
                },
                _ => {
                    return Err(PyValueError::new_err(
                        "Cannot create TriangleMesh from an object that contains something else \
                         than triangles",
                    ));
                },
            }
        }

        // TODO: remove duplicate vertices to reduce the size further.
        // Steps:
        // 1. Sort vertices
        // 2. Identify same vertices (consecutive)
        // 3. Remap triangles to point to first occurence
        // 4. Resize the triangles array and renumber from 0 to ...

        Ok(Self {
            vertices,
            triangles,
        })
    }
}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "triangle_mesh")?;
    m.add_class::<TriangleMesh>()?;

    Ok(m)
}
