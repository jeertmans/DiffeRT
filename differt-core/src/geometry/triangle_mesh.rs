use std::{fs::File, io::BufReader};

use numpy::{prelude::*, Element, PyArray2};
use obj::raw::object::{parse_obj, RawObj};
use ply_rs::{parser, ply};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

#[derive(Clone, Default)]
#[pyclass]
pub(crate) struct TriangleMesh {
    /// Array of size [num_vertices 3].
    pub(crate) vertices: Vec<(f32, f32, f32)>,
    /// Array of size [num_triangles 3].
    pub(crate) triangles: Vec<(usize, usize, usize)>,
}

#[inline]
fn pyarray2_from_vec_tuple<'py, T: Copy + Element>(
    py: Python<'py>,
    v: &[(T, T, T)],
) -> Bound<'py, PyArray2<T>> {
    let n = v.len();
    unsafe {
        let arr = PyArray2::<T>::new_bound(py, [n, 3], false);

        for i in 0..n {
            let tup = v.get_unchecked(i);
            arr.uget_raw([i, 0]).write(tup.0);
            arr.uget_raw([i, 1]).write(tup.1);
            arr.uget_raw([i, 2]).write(tup.2);
        }
        arr
    }
}

struct PlyVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl ply::PropertyAccess for PlyVertex {
    fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.x = v as _,
            ("y", ply::Property::Float(v)) => self.y = v as _,
            ("z", ply::Property::Float(v)) => self.z = v as _,
            ("x", ply::Property::Double(v)) => self.x = v as _,
            ("y", ply::Property::Double(v)) => self.y = v as _,
            ("z", ply::Property::Double(v)) => self.z = v as _,
            (k, property) => {
                log::info!("vertex: unexpected key/value combination: {k}/{property:?}")
            },
        }
    }
}

struct PlyFace {
    pub vertex_indices: Vec<usize>,
}

impl ply::PropertyAccess for PlyFace {
    fn new() -> Self {
        Self {
            vertex_indices: Vec::new(),
        }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("vertex_indices", ply::Property::ListUInt(vec)) => {
                self.vertex_indices = vec.iter().map(|&x| x as _).collect()
            },
            ("vertex_indices", ply::Property::ListInt(vec)) => {
                self.vertex_indices = vec.iter().map(|&x| x as _).collect()
            },
            (k, property) => {
                log::info!("face: unexpected key/value combination: {k}/{property:?}")
            },
        }
    }
}

#[pymethods]
impl TriangleMesh {
    pub(crate) fn append(&mut self, other: &mut Self) {
        let offset = self.vertices.len();
        self.vertices.append(&mut other.vertices);

        self.triangles.reserve(other.triangles.len());

        for (v0, v1, v2) in &other.triangles {
            self.triangles.push((v0 + offset, v1 + offset, v2 + offset));
        }

        other.triangles.clear();
    }

    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        pyarray2_from_vec_tuple(py, &self.vertices)
    }

    #[getter]
    fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<usize>> {
        pyarray2_from_vec_tuple(py, &self.triangles)
    }

    #[classmethod]
    pub(crate) fn load_obj(_: &Bound<'_, PyType>, filename: &str) -> PyResult<Self> {
        let input = BufReader::new(File::open(filename)?);
        let obj: RawObj = parse_obj(input).map_err(|err| {
            PyValueError::new_err(format!("An error occurred while reading obj file: {}", err))
        })?;
        Ok(obj.into())
    }

    #[classmethod]
    pub(crate) fn load_ply(_: &Bound<'_, PyType>, filename: &str) -> PyResult<Self> {
        let mut input = BufReader::new(File::open(filename)?);

        let vertex_parser = parser::Parser::<PlyVertex>::new();
        let face_parser = parser::Parser::<PlyFace>::new();

        let header = vertex_parser.read_header(&mut input).map_err(|err| {
            PyValueError::new_err(format!(
                "An error occurred while reading the header of ply file: {}",
                err
            ))
        })?;

        let mut vertices = Vec::new();
        let mut triangles = Vec::new();

        for (_, element) in &header.elements {
            match element.name.as_ref() {
                "vertex" => {
                    vertices.extend(
                        vertex_parser
                            .read_payload_for_element(&mut input, element, &header)
                            .map_err(|err| {
                                PyValueError::new_err(format!(
                                    "An error occurred while reading the vertex elements of ply \
                                     file: {}",
                                    err
                                ))
                            })?
                            .into_iter()
                            .map(|vertex| (vertex.x, vertex.y, vertex.z)),
                    );
                },
                "face" => {
                    triangles.extend(
                        face_parser
                            .read_payload_for_element(&mut input, element, &header)
                            .map_err(|err| {
                                PyValueError::new_err(format!(
                                    "An error occurred while reading the face elements of ply \
                                     file: {}",
                                    err
                                ))
                            })?
                            .into_iter()
                            .filter_map(|face| {
                                if face.vertex_indices.len() == 3 {
                                    let indices = face.vertex_indices;
                                    Some((indices[0], indices[1], indices[2]))
                                } else {
                                    log::info!("Face: skipping because it is not a triangle.");
                                    None
                                }
                            }),
                    );
                },
                name => log::info!("Unexpeced element: {name}, skipping."),
            }
        }

        Ok(Self {
            vertices,
            triangles,
        })
    }
}

impl From<RawObj> for TriangleMesh {
    fn from(raw_obj: RawObj) -> Self {
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
                    log::info!("Skipping a polygon because it is not a triangle.")
                },
            }
        }

        // TODO: remove duplicate vertices to reduce the size further.
        // Steps:
        // 1. Sort vertices
        // 2. Identify same vertices (consecutive)
        // 3. Remap triangles to point to first occurrence
        // 4. Resize the triangles array and renumber from 0 to ...

        Self {
            vertices,
            triangles,
        }
    }
}

#[pymodule]
pub(crate) fn triangle_mesh(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TriangleMesh>()?;
    Ok(())
}
