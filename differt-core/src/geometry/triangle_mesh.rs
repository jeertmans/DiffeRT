use std::{fs::File, io::BufReader, path::PathBuf};

use indexmap::IndexMap;
use numpy::{PyArray1, PyArray2, ndarray::arr2};
use obj::raw::object::{RawObj, parse_obj};
use ply_rs::{parser, ply};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

/// A lower-level mesh made of triangles.
///
/// This class should not be used directly, prefer its higher-level version
/// :class:`differt.geometry.TriangleMesh` instead.
#[derive(Clone, Debug, Default)]
#[pyclass]
pub(crate) struct TriangleMesh {
    vertices: Vec<[f32; 3]>,
    triangles: Vec<[usize; 3]>,
    face_colors: Option<Vec<[f32; 3]>>,
    face_materials: Option<Vec<isize>>,
    #[pyo3(get)]
    /// list[str]: List of material names.
    material_names: Vec<String>,
    object_bounds: Option<Vec<[usize; 2]>>,
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
            ("vertex_indices" | "vertex_index", ply::Property::ListUInt(vec)) => {
                self.vertex_indices = vec.iter().map(|&x| x as _).collect()
            },
            ("vertex_indices" | "vertex_index", ply::Property::ListInt(vec)) => {
                self.vertex_indices = vec.iter().map(|&x| x as _).collect()
            },
            (k, property) => {
                log::info!("face: unexpected key/value combination: {k}/{property:?}")
            },
        }
    }
}

impl TriangleMesh {
    pub fn get_material_index(&mut self, material_name: Option<String>) -> isize {
        if let Some(material_name) = material_name {
            return self
                .material_names
                .iter()
                .position(|name| name == &material_name)
                .unwrap_or_else(|| {
                    let index = self.material_names.len();
                    self.material_names.push(material_name);
                    index
                }) as isize;
        }
        -1
    }

    pub(crate) fn set_face_color(&mut self, color: Option<&[f32; 3]>) {
        let num_triangles = self.triangles.len();
        let face_colors = self
            .face_colors
            .get_or_insert_with(|| Vec::with_capacity(num_triangles));
        face_colors.clear();

        let color = color.unwrap_or(&[-1.0, -1.0, -1.0]);

        for _ in 0..num_triangles {
            face_colors.push(*color);
        }
    }

    pub(crate) fn set_face_material(&mut self, material_name: Option<String>) {
        let material_index = self.get_material_index(material_name);
        let num_triangles = self.triangles.len();
        let face_materials = self
            .face_materials
            .get_or_insert_with(|| Vec::with_capacity(num_triangles));
        face_materials.clear();

        for _ in 0..num_triangles {
            face_materials.push(material_index);
        }
    }
}

#[pymethods]
impl TriangleMesh {
    /// :class:`Float[np.ndarray, 'num_vertices 3']<jaxtyping.Float>`: The array of triangle vertices.
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let array = arr2(&self.vertices);
        PyArray2::from_owned_array(py, array)
    }

    /// :class:`Int[np.ndarray, 'num_triangles 3']<jaxtyping.Int>`: The array of triangle indices.
    #[getter]
    fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<usize>> {
        let array = arr2(&self.triangles);
        PyArray2::from_owned_array(py, array)
    }

    /// :class:`Float[np.ndarray, 'num_vertices 3']<jaxtyping.Float>` | :data:`None`: The array of face colors.
    ///
    /// The array contains the face colors, as RGB triplets,
    /// with a black color used as defaults (if some faces have a color).
    /// This attribute is :data:`None` if all face colors are unset.
    #[getter]
    fn face_colors<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        if let Some(face_colors) = &self.face_colors {
            let array = arr2(face_colors);
            return Some(PyArray2::from_owned_array(py, array));
        }
        None
    }

    /// :class:`Int[np.ndarray, 'num_vertices']<jaxtyping.Int>` | :data:`None`: The array of face materials.
    ///
    /// The array contains the material indices,
    /// with a special placeholder value of ``-1``.
    /// The obtain the name of the material, see :attr:`material_names`.
    /// This attribute is :data:`None` if all face materials are unset.
    #[getter]
    fn face_materials<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<isize>>> {
        if let Some(face_materials) = &self.face_materials {
            return Some(PyArray1::from_slice(py, face_materials.as_slice()));
        }
        None
    }

    /// :class:`Int[np.ndarray, 'num_objects 2']<jaxtyping.Int>` | :data:`None`: The array of object indices.
    ///
    /// If the present mesh contains multiple objects, usually as a result of
    /// appending multiple meshes together, this array contain start end end
    /// indices for each sub mesh.
    #[getter]
    fn object_bounds<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<usize>>> {
        if let Some(object_bounds) = &self.object_bounds {
            let array = arr2(object_bounds);
            return Some(PyArray2::from_owned_array(py, array));
        }
        None
    }

    /// Move all the elements of ``other`` into ``self`` and update :attr:`object_bounds`.
    ///
    /// After calling this method, ``other`` will be empty.
    ///
    /// Args:
    ///     other(TriangleMesh): The mesh to be appended to ``self``.
    pub(crate) fn append(&mut self, other: &mut Self) {
        match (&self.face_colors, &other.face_colors) {
            (None, None) => {},
            (Some(_), None) => {
                other.set_face_color(None);
            },
            (None, Some(_)) => {
                self.set_face_color(None);
            },
            _ => {},
        }

        if let (Some(x), Some(y)) = (self.face_colors.as_mut(), other.face_colors.as_mut()) {
            x.append(y);
        }

        match (&self.face_materials, &other.face_materials) {
            (None, None) => {},
            (Some(_), None) => {
                other.set_face_material(None);
            },
            (None, Some(_)) => {
                self.set_face_material(None);
            },
            (Some(_), Some(_)) => {
                // We need to possibly renumber material indices.
                let mut remap: Vec<isize> = vec![0; other.material_names.len()];

                for (i, material_name) in other.material_names.drain(..).enumerate() {
                    remap[i] = match self
                        .material_names
                        .iter()
                        .position(|name| name == &material_name)
                    {
                        Some(material_index) => material_index,
                        None => {
                            let material_index = self.material_names.len();
                            self.material_names.push(material_name);
                            material_index
                        },
                    } as isize;
                }

                for material_index in other.face_materials.as_mut().unwrap() {
                    if *material_index >= 0 {
                        *material_index = remap[*material_index as usize];
                    }
                }
            },
        }

        if let (Some(x), Some(y)) = (self.face_materials.as_mut(), other.face_materials.as_mut()) {
            x.append(y);
        }

        let index_offset = self.vertices.len();
        let bound_offset = self.triangles.len();
        self.vertices.append(&mut other.vertices);

        self.triangles.reserve(other.triangles.len());

        for [v0, v1, v2] in &other.triangles {
            self.triangles
                .push([v0 + index_offset, v1 + index_offset, v2 + index_offset]);
        }

        other.triangles.clear();

        self.object_bounds
            .get_or_insert_with(|| {
                if bound_offset > 0 {
                    vec![[0, bound_offset]]
                } else {
                    vec![]
                }
            })
            .push([bound_offset, self.triangles.len()]);
    }

    /// Load a triangle mesh from a Wavefront .obj file.
    ///
    /// Currently, only vertices and triangles are loaded. Triangle normals
    /// are ignored because they are computed with
    /// :attr:`differt.geometry.TriangleMesh.normals` using
    /// JAX so that they can be differentiated with respect to triangle
    /// vertices.
    ///
    /// Args:
    ///     file (str): The path to the Wavefront .obj file.
    ///
    /// Returns:
    ///     TriangleMesh: The corresponding mesh containing only triangles.
    #[classmethod]
    pub(crate) fn load_obj(_: &Bound<'_, PyType>, filename: &str) -> PyResult<Self> {
        let input = BufReader::new(File::open(filename)?);
        let mut obj: RawObj = parse_obj(input).map_err(|err| {
            PyValueError::new_err(format!(
                "An error occurred while reading obj file {filename:#?}: {}",
                err
            ))
        })?;

        for material_file in obj.material_libraries.iter_mut() {
            let mut path = PathBuf::from(filename);
            path.set_file_name(&material_file);
            if let Some(path_str) = path.to_str() {
                *material_file = path_str.to_string();
            }
        }

        Ok(obj.into())
    }

    /// Load a triangle mesh from a Stanford PLY .ply file.
    ///
    /// Currently, only vertices and triangles are loaded. Triangle normals
    /// are ignored because they are computed with
    /// :attr:`differt.geometry.TriangleMesh.normals` using
    /// JAX so that they can be differentiated with respect to triangle
    /// vertices.
    ///
    /// Args:
    ///     file (str): The path to the Stanford PLY .ply file.
    ///
    /// Returns:
    ///     TriangleMesh: The corresponding mesh containing only triangles.
    #[classmethod]
    pub(crate) fn load_ply(_: &Bound<'_, PyType>, filename: &str) -> PyResult<Self> {
        let mut input = BufReader::new(File::open(filename)?);

        let vertex_parser = parser::Parser::<PlyVertex>::new();
        let face_parser = parser::Parser::<PlyFace>::new();

        let header = vertex_parser.read_header(&mut input).map_err(|err| {
            PyValueError::new_err(format!(
                "An error occurred while reading the header of ply file {filename:#?}: {}",
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
                                     file {filename:#?}: {}",
                                    err
                                ))
                            })?
                            .into_iter()
                            .map(|vertex| [vertex.x, vertex.y, vertex.z]),
                    );
                },
                "face" => {
                    triangles.extend(
                        face_parser
                            .read_payload_for_element(&mut input, element, &header)
                            .map_err(|err| {
                                PyValueError::new_err(format!(
                                    "An error occurred while reading the face elements of PLY \
                                     file {filename:#?}: {}",
                                    err
                                ))
                            })?
                            .into_iter()
                            .filter_map(|face| {
                                if face.vertex_indices.len() == 3 {
                                    let indices = face.vertex_indices;
                                    Some([indices[0], indices[1], indices[2]])
                                } else {
                                    log::info!("Skipping a face because it is not a triangle.");
                                    None
                                }
                            }),
                    );
                },
                name => log::info!("Unexpected element: {name}, skipping."),
            }
        }

        Ok(Self {
            vertices,
            triangles,
            ..Default::default()
        })
    }
}

impl From<RawObj> for TriangleMesh {
    fn from(mut raw_obj: RawObj) -> Self {
        use obj::raw::object::Polygon::*;

        let vertices = raw_obj
            .positions
            .into_iter()
            .map(|(x, y, z, _)| [x, y, z])
            .collect();

        let mut triangles = Vec::with_capacity(raw_obj.polygons.len());

        for polygon in raw_obj.polygons {
            match polygon {
                P(v) if v.len() == 3 => {
                    triangles.push([v[0], v[1], v[2]]);
                },
                PT(v) if v.len() == 3 => {
                    triangles.push([v[0].0, v[1].0, v[2].0]);
                },
                PN(v) if v.len() == 3 => {
                    triangles.push([v[0].0, v[1].0, v[2].0]);
                },
                PTN(v) if v.len() == 3 => {
                    triangles.push([v[0].0, v[1].0, v[2].0]);
                },
                _ => {
                    log::info!("Skipping a polygon because it is not a triangle.")
                },
            }
        }

        if !raw_obj.material_libraries.is_empty() && !raw_obj.meshes.is_empty() {
            let mut materials = IndexMap::new();

            for material_file in raw_obj.material_libraries {
                match File::open(&material_file) {
                    Ok(file) => {
                        match obj::raw::material::parse_mtl(BufReader::new(file)) {
                            Ok(raw_mat) => {
                                for (material_name, material) in raw_mat.materials {
                                    materials.insert(material_name, material);
                                }
                            },
                            Err(e) => {
                                log::warn!(
                                    "An error occurred when parsing MTL file {material_file:#?}: \
                                     {e}."
                                );
                            },
                        }
                    },
                    Err(e) => {
                        log::warn!(
                            "An error occurred when trying to read MTL file {material_file:#?}: \
                             {e}."
                        );
                    },
                }
            }

            let mut face_colors = vec![[0.0, 0.0, 0.0]; triangles.len()];
            let mut face_materials = vec![-1; triangles.len()];
            let mut material_names = Vec::new();

            for (material_index, (material_name, material)) in materials.into_iter().enumerate() {
                if let Some(group) = raw_obj.meshes.remove(&material_name) {
                    use obj::raw::material::MtlColor;
                    let material_color = if let Some(MtlColor::Rgb(r, g, b)) = material.diffuse {
                        [r, g, b]
                    } else {
                        log::warn!(
                            "Currently, we only support diffuse color in RGB format, but it was \
                             not present."
                        );
                        [0.0, 0.0, 0.0]
                    };
                    for polygon_range in group.polygons {
                        face_colors[(polygon_range.start)..(polygon_range.end)]
                            .fill(material_color);
                        face_materials[(polygon_range.start)..(polygon_range.end)]
                            .fill(material_index as isize);
                    }
                } else {
                    log::warn!(
                        "Material {material_name:#?} was listed in OBJ material libraries, but is \
                         not present is mesh groups."
                    );
                }
                material_names.push(material_name);
            }

            if !raw_obj.meshes.is_empty() {
                log::warn!(
                    "Some materials listed in mesh groups where not listed in material libraries: \
                     {}.",
                    raw_obj.meshes.into_keys().collect::<Vec<_>>().join(", ")
                );
            }

            let face_colors = Some(face_colors);
            let face_materials = Some(face_materials);

            return Self {
                vertices,
                triangles,
                face_colors,
                face_materials,
                material_names,
                ..Default::default()
            };
        }

        Self {
            vertices,
            triangles,
            ..Default::default()
        }
    }
}

#[cfg(not(tarpaulin_include))]
#[pymodule]
pub(crate) fn triangle_mesh(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TriangleMesh>()?;
    Ok(())
}
