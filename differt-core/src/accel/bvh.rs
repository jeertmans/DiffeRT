//! BVH (Bounding Volume Hierarchy) acceleration structure for triangle meshes.
//!
//! Provides SAH-based BVH construction and two query types:
//! - Nearest-hit: find the closest triangle intersected by each ray (for SBR)
//! - Candidate selection: find all triangles whose expanded bounding boxes
//!   intersect each ray (for differentiable mode)

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Geometry primitives
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn from_slice(s: &[f32]) -> Self {
        Self {
            x: s[0],
            y: s[1],
            z: s[2],
        }
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn min_comp(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y), self.z.min(other.z))
    }

    fn max_comp(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y), self.z.max(other.z))
    }
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    fn grow_point(&mut self, p: Vec3) {
        self.min = self.min.min_comp(p);
        self.max = self.max.max_comp(p);
    }

    fn grow_aabb(&mut self, other: &Aabb) {
        self.min = self.min.min_comp(other.min);
        self.max = self.max.max_comp(other.max);
    }

    fn expand(&self, amount: f32) -> Aabb {
        Aabb {
            min: Vec3::new(
                self.min.x - amount,
                self.min.y - amount,
                self.min.z - amount,
            ),
            max: Vec3::new(
                self.max.x + amount,
                self.max.y + amount,
                self.max.z + amount,
            ),
        }
    }

    fn surface_area(&self) -> f32 {
        let d = self.max.sub(self.min);
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    fn centroid(&self) -> Vec3 {
        Vec3::new(
            0.5 * (self.min.x + self.max.x),
            0.5 * (self.min.y + self.max.y),
            0.5 * (self.min.z + self.max.z),
        )
    }

    /// Ray-AABB intersection test (slab method).
    /// Returns true if the ray intersects the box at any t >= 0.
    fn intersects_ray(&self, origin: Vec3, inv_dir: Vec3) -> bool {
        let t1x = (self.min.x - origin.x) * inv_dir.x;
        let t2x = (self.max.x - origin.x) * inv_dir.x;
        let t1y = (self.min.y - origin.y) * inv_dir.y;
        let t2y = (self.max.y - origin.y) * inv_dir.y;
        let t1z = (self.min.z - origin.z) * inv_dir.z;
        let t2z = (self.max.z - origin.z) * inv_dir.z;

        let tmin = t1x.min(t2x).max(t1y.min(t2y)).max(t1z.min(t2z));
        let tmax = t1x.max(t2x).min(t1y.max(t2y)).min(t1z.max(t2z));

        tmax >= tmin.max(0.0)
    }
}

fn axis_component(v: Vec3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

// ---------------------------------------------------------------------------
// Moller-Trumbore ray-triangle intersection (hard boolean, for Rust-side queries)
// ---------------------------------------------------------------------------

const MT_EPSILON: f32 = 1e-8;

/// Returns (t, hit) where t is parametric distance, hit indicates valid intersection.
fn ray_triangle_intersect(origin: Vec3, direction: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> (f32, bool) {
    let edge1 = v1.sub(v0);
    let edge2 = v2.sub(v0);
    let h = direction.cross(edge2);
    let a = edge1.dot(h);

    if a.abs() < MT_EPSILON {
        return (f32::INFINITY, false);
    }

    let f = 1.0 / a;
    let s = origin.sub(v0);
    let u = f * s.dot(h);

    if !(0.0..=1.0).contains(&u) {
        return (f32::INFINITY, false);
    }

    let q = s.cross(edge1);
    let v = f * direction.dot(q);

    if v < 0.0 || u + v > 1.0 {
        return (f32::INFINITY, false);
    }

    let t = f * edge2.dot(q);

    if t > MT_EPSILON {
        (t, true)
    } else {
        (f32::INFINITY, false)
    }
}

// ---------------------------------------------------------------------------
// BVH node
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct BvhNode {
    bounds: Aabb,
    /// For leaves: index of first triangle in the reordered tri_indices array.
    /// For internal nodes: index of the left child (right = left + 1).
    left_or_first: u32,
    /// For leaves: number of triangles. For internal nodes: 0.
    count: u32,
}

impl BvhNode {
    fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

// ---------------------------------------------------------------------------
// BVH
// ---------------------------------------------------------------------------

const NUM_SAH_BINS: usize = 12;
const MAX_LEAF_SIZE: u32 = 4;

struct Bvh {
    nodes: Vec<BvhNode>,
    tri_indices: Vec<u32>,
    /// Triangle vertices: [num_triangles, 3 vertices, 3 coords] flattened
    tri_verts: Vec<[Vec3; 3]>,
    /// Per-triangle bounding boxes (precomputed)
    tri_bounds: Vec<Aabb>,
    /// Per-triangle centroids (precomputed)
    tri_centroids: Vec<Vec3>,
    nodes_used: u32,
}

impl Bvh {
    fn new(vertices: &[[f32; 9]]) -> Self {
        let n = vertices.len();
        let mut tri_verts = Vec::with_capacity(n);
        let mut tri_bounds = Vec::with_capacity(n);
        let mut tri_centroids = Vec::with_capacity(n);
        let tri_indices: Vec<u32> = (0..n as u32).collect();

        for verts in vertices {
            let v0 = Vec3::from_slice(&verts[0..3]);
            let v1 = Vec3::from_slice(&verts[3..6]);
            let v2 = Vec3::from_slice(&verts[6..9]);
            tri_verts.push([v0, v1, v2]);

            let mut bb = Aabb::empty();
            bb.grow_point(v0);
            bb.grow_point(v1);
            bb.grow_point(v2);
            tri_bounds.push(bb);
            tri_centroids.push(bb.centroid());
        }

        // Allocate worst-case node count (2*n - 1 for binary tree)
        let max_nodes = if n > 0 { 2 * n - 1 } else { 1 };
        let mut bvh = Bvh {
            nodes: vec![
                BvhNode {
                    bounds: Aabb::empty(),
                    left_or_first: 0,
                    count: 0,
                };
                max_nodes
            ],
            tri_indices,
            tri_verts,
            tri_bounds,
            tri_centroids,
            nodes_used: 1,
        };

        // Initialize root
        bvh.nodes[0].left_or_first = 0;
        bvh.nodes[0].count = n as u32;
        bvh.update_node_bounds(0);
        bvh.subdivide(0);

        bvh
    }

    fn update_node_bounds(&mut self, node_idx: usize) {
        let node = &self.nodes[node_idx];
        let first = node.left_or_first as usize;
        let count = node.count as usize;
        let mut bounds = Aabb::empty();
        for i in first..first + count {
            let ti = self.tri_indices[i] as usize;
            bounds.grow_aabb(&self.tri_bounds[ti]);
        }
        self.nodes[node_idx].bounds = bounds;
    }

    fn find_best_split(&self, node_idx: usize) -> (usize, f32, f32) {
        let node = &self.nodes[node_idx];
        let first = node.left_or_first as usize;
        let count = node.count as usize;

        // Compute centroid bounds for binning
        let mut centroid_bounds = Aabb::empty();
        for i in first..first + count {
            let ti = self.tri_indices[i] as usize;
            centroid_bounds.grow_point(self.tri_centroids[ti]);
        }

        let mut best_axis = 0;
        let mut best_pos = 0.0f32;
        let mut best_cost = f32::INFINITY;

        for axis in 0..3 {
            let lo = axis_component(centroid_bounds.min, axis);
            let hi = axis_component(centroid_bounds.max, axis);
            if (hi - lo).abs() < 1e-10 {
                continue;
            }

            // Binned SAH
            let mut bins = [Aabb::empty(); NUM_SAH_BINS];
            let mut bin_counts = [0u32; NUM_SAH_BINS];

            let scale = NUM_SAH_BINS as f32 / (hi - lo);

            for i in first..first + count {
                let ti = self.tri_indices[i] as usize;
                let c = axis_component(self.tri_centroids[ti], axis);
                let bin = ((c - lo) * scale).min(NUM_SAH_BINS as f32 - 1.0) as usize;
                bin_counts[bin] += 1;
                bins[bin].grow_aabb(&self.tri_bounds[ti]);
            }

            // Sweep from left
            let mut left_area = [0.0f32; NUM_SAH_BINS - 1];
            let mut left_count_arr = [0u32; NUM_SAH_BINS - 1];
            let mut left_box = Aabb::empty();
            let mut left_sum = 0u32;
            for i in 0..NUM_SAH_BINS - 1 {
                left_box.grow_aabb(&bins[i]);
                left_sum += bin_counts[i];
                left_area[i] = left_box.surface_area();
                left_count_arr[i] = left_sum;
            }

            // Sweep from right
            let mut right_box = Aabb::empty();
            let mut right_sum = 0u32;
            for i in (1..NUM_SAH_BINS).rev() {
                right_box.grow_aabb(&bins[i]);
                right_sum += bin_counts[i];
                let cost =
                    left_count_arr[i - 1] as f32 * left_area[i - 1] + right_sum as f32 * right_box.surface_area();
                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_pos = lo + i as f32 / scale;
                }
            }
        }

        (best_axis, best_pos, best_cost)
    }

    fn subdivide(&mut self, node_idx: usize) {
        let count = self.nodes[node_idx].count;
        if count <= MAX_LEAF_SIZE {
            return;
        }

        let (axis, split_pos, split_cost) = self.find_best_split(node_idx);

        // Compare SAH cost to not-split cost
        let no_split_cost = count as f32 * self.nodes[node_idx].bounds.surface_area();
        if split_cost >= no_split_cost {
            return;
        }

        // Partition triangles
        let first = self.nodes[node_idx].left_or_first as usize;
        let last = first + count as usize;
        let mut i = first;
        let mut j = last;

        while i < j {
            let ti = self.tri_indices[i] as usize;
            if axis_component(self.tri_centroids[ti], axis) < split_pos {
                i += 1;
            } else {
                j -= 1;
                self.tri_indices.swap(i, j);
            }
        }

        let left_count = (i - first) as u32;
        if left_count == 0 || left_count == count {
            return; // degenerate split
        }

        let left_child = self.nodes_used as usize;
        self.nodes_used += 2;

        self.nodes[left_child].left_or_first = first as u32;
        self.nodes[left_child].count = left_count;

        self.nodes[left_child + 1].left_or_first = i as u32;
        self.nodes[left_child + 1].count = count - left_count;

        // Convert current node to internal
        self.nodes[node_idx].left_or_first = left_child as u32;
        self.nodes[node_idx].count = 0;

        self.update_node_bounds(left_child);
        self.update_node_bounds(left_child + 1);
        self.subdivide(left_child);
        self.subdivide(left_child + 1);
    }

    /// Find the nearest triangle hit by a ray. Returns (triangle_index, t) or (-1, inf).
    fn nearest_hit(&self, origin: Vec3, direction: Vec3) -> (i32, f32) {
        let inv_dir = Vec3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);
        let mut stack = Vec::with_capacity(64);
        stack.push(0usize);

        let mut best_t = f32::INFINITY;
        let mut best_idx: i32 = -1;

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            if !node.bounds.intersects_ray(origin, inv_dir) {
                continue;
            }

            if node.is_leaf() {
                let first = node.left_or_first as usize;
                let count = node.count as usize;
                for i in first..first + count {
                    let ti = self.tri_indices[i] as usize;
                    let [v0, v1, v2] = self.tri_verts[ti];
                    let (t, hit) = ray_triangle_intersect(origin, direction, v0, v1, v2);
                    if hit && t < best_t {
                        best_t = t;
                        best_idx = ti as i32;
                    }
                }
            } else {
                let left = node.left_or_first as usize;
                stack.push(left);
                stack.push(left + 1);
            }
        }

        (best_idx, best_t)
    }

    /// Find all candidate triangles whose expanded bounding box intersects a ray.
    fn get_candidates(
        &self,
        origin: Vec3,
        direction: Vec3,
        expansion: f32,
        max_candidates: usize,
    ) -> (Vec<i32>, u32) {
        let inv_dir = Vec3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);
        let mut stack = Vec::with_capacity(64);
        stack.push(0usize);

        let mut candidates = Vec::with_capacity(max_candidates.min(256));
        let mut count = 0u32;

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];
            let expanded = node.bounds.expand(expansion);

            if !expanded.intersects_ray(origin, inv_dir) {
                continue;
            }

            if node.is_leaf() {
                let first = node.left_or_first as usize;
                let leaf_count = node.count as usize;
                for i in first..first + leaf_count {
                    let ti = self.tri_indices[i] as i32;
                    if (count as usize) < max_candidates {
                        candidates.push(ti);
                    }
                    count += 1;
                }
            } else {
                let left = node.left_or_first as usize;
                stack.push(left);
                stack.push(left + 1);
            }
        }

        (candidates, count)
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// BVH acceleration structure for triangle meshes.
///
/// Builds a Bounding Volume Hierarchy using the Surface Area Heuristic (SAH)
/// for fast ray-triangle intersection queries.
///
/// Args:
///     triangle_vertices: Triangle vertices with shape ``(num_triangles, 3, 3)``.
///
/// Examples:
///     >>> import numpy as np
///     >>> from differt_core.accel.bvh import TriangleBvh
///     >>> # A single triangle
///     >>> verts = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
///     >>> bvh = TriangleBvh(verts)
///     >>> bvh.num_triangles
///     1
#[pyclass]
struct TriangleBvh {
    inner: Bvh,
}

#[pymethods]
impl TriangleBvh {
    #[new]
    fn new(triangle_vertices: PyReadonlyArray2<f32>) -> PyResult<Self> {
        let shape = triangle_vertices.shape();
        // Expect shape (num_triangles * 3, 3) or we reshape from (num_triangles, 3, 3)
        // NumPy 3D arrays are passed as 2D with shape (N*3, 3) when using PyReadonlyArray2
        if shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "triangle_vertices must have shape (num_triangles * 3, 3) or (num_triangles, 3, 3)",
            ));
        }

        let data = triangle_vertices.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array must be contiguous: {e}"))
        })?;

        let num_verts_rows = shape[0];
        if num_verts_rows % 3 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "First dimension must be divisible by 3 (num_triangles * 3 vertices)",
            ));
        }

        let num_triangles = num_verts_rows / 3;
        let mut flat_tris: Vec<[f32; 9]> = Vec::with_capacity(num_triangles);

        for i in 0..num_triangles {
            let base = i * 9; // 3 vertices * 3 coords
            flat_tris.push([
                data[base],
                data[base + 1],
                data[base + 2],
                data[base + 3],
                data[base + 4],
                data[base + 5],
                data[base + 6],
                data[base + 7],
                data[base + 8],
            ]);
        }

        Ok(Self {
            inner: Bvh::new(&flat_tris),
        })
    }

    /// Number of triangles in the BVH.
    #[getter]
    fn num_triangles(&self) -> usize {
        self.inner.tri_verts.len()
    }

    /// Number of BVH nodes used.
    #[getter]
    fn num_nodes(&self) -> u32 {
        self.inner.nodes_used
    }

    /// Find the nearest triangle hit by each ray.
    ///
    /// Args:
    ///     ray_origins: Ray origins with shape ``(num_rays, 3)``.
    ///     ray_directions: Ray directions with shape ``(num_rays, 3)``.
    ///
    /// Returns:
    ///     A tuple ``(hit_indices, hit_t)`` where ``hit_indices`` has shape
    ///     ``(num_rays,)`` with the triangle index (``-1`` if no hit) and
    ///     ``hit_t`` has shape ``(num_rays,)`` with the parametric distance.
    ///
    /// Examples:
    ///     >>> import numpy as np
    ///     >>> from differt_core.accel.bvh import TriangleBvh
    ///     >>> verts = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    ///     >>> bvh = TriangleBvh(verts)
    ///     >>> origins = np.array([[0.1, 0.1, 1.0]], dtype=np.float32)
    ///     >>> dirs = np.array([[0, 0, -1]], dtype=np.float32)
    ///     >>> idx, t = bvh.nearest_hit(origins, dirs)
    ///     >>> int(idx[0])
    ///     0
    ///     >>> float(t[0])
    ///     1.0
    fn nearest_hit<'py>(
        &self,
        py: Python<'py>,
        ray_origins: PyReadonlyArray2<f32>,
        ray_directions: PyReadonlyArray2<f32>,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
        let origins = ray_origins.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ray_origins must be contiguous: {e}"))
        })?;
        let dirs = ray_directions.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "ray_directions must be contiguous: {e}"
            ))
        })?;

        let num_rays = ray_origins.shape()[0];
        let mut hit_indices = vec![-1i32; num_rays];
        let mut hit_t = vec![f32::INFINITY; num_rays];

        for i in 0..num_rays {
            let origin = Vec3::from_slice(&origins[i * 3..(i + 1) * 3]);
            let dir = Vec3::from_slice(&dirs[i * 3..(i + 1) * 3]);
            let (idx, t) = self.inner.nearest_hit(origin, dir);
            hit_indices[i] = idx;
            hit_t[i] = t;
        }

        Ok((
            PyArray1::from_vec(py, hit_indices),
            PyArray1::from_vec(py, hit_t),
        ))
    }

    /// Find candidate triangles whose expanded bounding boxes intersect each ray.
    ///
    /// This is used for differentiable mode: the expansion captures all triangles
    /// with non-negligible gradient contribution.
    ///
    /// Args:
    ///     ray_origins: Ray origins with shape ``(num_rays, 3)``.
    ///     ray_directions: Ray directions with shape ``(num_rays, 3)``.
    ///     expansion: Bounding box expansion amount (related to smoothing_factor).
    ///     max_candidates: Maximum number of candidates per ray.
    ///
    /// Returns:
    ///     A tuple ``(candidate_indices, candidate_counts)`` where
    ///     ``candidate_indices`` has shape ``(num_rays, max_candidates)`` padded
    ///     with ``-1``, and ``candidate_counts`` has shape ``(num_rays,)``.
    ///
    /// Examples:
    ///     >>> import numpy as np
    ///     >>> from differt_core.accel.bvh import TriangleBvh
    ///     >>> verts = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    ///     >>> bvh = TriangleBvh(verts)
    ///     >>> origins = np.array([[0.1, 0.1, 1.0]], dtype=np.float32)
    ///     >>> dirs = np.array([[0, 0, -1]], dtype=np.float32)
    ///     >>> idx, counts = bvh.get_candidates(origins, dirs, 0.0, 256)
    ///     >>> int(counts[0])
    ///     1
    ///     >>> int(idx[0, 0])
    ///     0
    fn get_candidates<'py>(
        &self,
        py: Python<'py>,
        ray_origins: PyReadonlyArray2<f32>,
        ray_directions: PyReadonlyArray2<f32>,
        expansion: f32,
        max_candidates: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray1<u32>>)> {
        let origins = ray_origins.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ray_origins must be contiguous: {e}"))
        })?;
        let dirs = ray_directions.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "ray_directions must be contiguous: {e}"
            ))
        })?;

        let num_rays = ray_origins.shape()[0];
        let mut all_indices = vec![-1i32; num_rays * max_candidates];
        let mut all_counts = vec![0u32; num_rays];

        for i in 0..num_rays {
            let origin = Vec3::from_slice(&origins[i * 3..(i + 1) * 3]);
            let dir = Vec3::from_slice(&dirs[i * 3..(i + 1) * 3]);
            let (candidates, count) =
                self.inner
                    .get_candidates(origin, dir, expansion, max_candidates);
            all_counts[i] = count;
            let row_start = i * max_candidates;
            for (j, &idx) in candidates.iter().enumerate() {
                all_indices[row_start + j] = idx;
            }
        }

        let indices_array = numpy::ndarray::Array2::from_shape_vec(
            (num_rays, max_candidates),
            all_indices,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

        Ok((
            PyArray2::from_owned_array(py, indices_array),
            PyArray1::from_vec(py, all_counts),
        ))
    }
}

#[cfg(not(tarpaulin_include))]
#[pymodule(gil_used = false)]
pub(crate) fn bvh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TriangleBvh>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn single_triangle() -> Vec<[f32; 9]> {
        vec![[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    }

    fn cube_triangles() -> Vec<[f32; 9]> {
        // 12 triangles forming a unit cube [0,1]^3
        let faces: Vec<([f32; 3], [f32; 3], [f32; 3])> = vec![
            // Front face (z=1)
            ([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
            ([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]),
            // Back face (z=0)
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]),
            ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            // Top face (y=1)
            ([0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
            ([0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]),
            // Bottom face (y=0)
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            // Right face (x=1)
            ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
            ([1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]),
            // Left face (x=0)
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]),
            ([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]),
        ];
        faces
            .into_iter()
            .map(|(a, b, c)| [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]])
            .collect()
    }

    #[test]
    fn test_bvh_construction_single_triangle() {
        let bvh = Bvh::new(&single_triangle());
        assert_eq!(bvh.tri_verts.len(), 1);
        assert!(bvh.nodes_used >= 1);
    }

    #[test]
    fn test_bvh_construction_cube() {
        let bvh = Bvh::new(&cube_triangles());
        assert_eq!(bvh.tri_verts.len(), 12);
        assert!(bvh.nodes_used >= 1);
    }

    #[test]
    fn test_bvh_construction_empty() {
        let bvh = Bvh::new(&[]);
        assert_eq!(bvh.tri_verts.len(), 0);
    }

    #[test]
    fn test_nearest_hit_single_triangle() {
        let bvh = Bvh::new(&single_triangle());
        // Ray pointing down at (0.1, 0.1)
        let origin = Vec3::new(0.1, 0.1, 1.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let (idx, t) = bvh.nearest_hit(origin, dir);
        assert_eq!(idx, 0);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_nearest_hit_miss() {
        let bvh = Bvh::new(&single_triangle());
        // Ray pointing away
        let origin = Vec3::new(0.1, 0.1, 1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let (idx, _t) = bvh.nearest_hit(origin, dir);
        assert_eq!(idx, -1);
    }

    #[test]
    fn test_nearest_hit_cube() {
        let bvh = Bvh::new(&cube_triangles());
        // Ray from outside hitting front face
        let origin = Vec3::new(0.5, 0.5, 2.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let (idx, t) = bvh.nearest_hit(origin, dir);
        assert!(idx >= 0, "Should hit a front-face triangle");
        assert!((t - 1.0).abs() < 1e-5, "Distance to front face should be 1.0");
    }

    #[test]
    fn test_nearest_hit_picks_closest() {
        let bvh = Bvh::new(&cube_triangles());
        // Ray going through both front and back faces -- should hit front (closer)
        let origin = Vec3::new(0.5, 0.5, 2.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let (idx, t) = bvh.nearest_hit(origin, dir);
        assert!(idx >= 0);
        assert!((t - 1.0).abs() < 1e-5, "Should hit front face at t=1, got t={t}");
    }

    #[test]
    fn test_get_candidates_no_expansion() {
        let bvh = Bvh::new(&single_triangle());
        let origin = Vec3::new(0.1, 0.1, 1.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let (candidates, count) = bvh.get_candidates(origin, dir, 0.0, 256);
        assert!(count >= 1, "Should find at least the hit triangle");
        assert_eq!(candidates[0], 0);
    }

    #[test]
    fn test_get_candidates_with_expansion() {
        // Many distant triangles to force BVH splits (need > MAX_LEAF_SIZE=4)
        let mut tris = Vec::new();
        // 5 triangles near origin
        for i in 0..5 {
            let x = i as f32 * 0.5;
            tris.push([x, 0.0, 0.0, x + 0.4, 0.0, 0.0, x, 0.4, 0.0]);
        }
        // 5 triangles far away (x=100)
        for i in 0..5 {
            let x = 100.0 + i as f32 * 0.5;
            tris.push([x, 0.0, 0.0, x + 0.4, 0.0, 0.0, x, 0.4, 0.0]);
        }
        let bvh = Bvh::new(&tris);

        // Ray aimed at near triangles
        let origin = Vec3::new(0.1, 0.1, 1.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);

        // No expansion: should not include the far-away group
        let (_, count_no_exp) = bvh.get_candidates(origin, dir, 0.0, 256);
        assert!(
            count_no_exp <= 5,
            "Without expansion, should not include far triangles, got {count_no_exp}"
        );

        // Large expansion: should include all
        let (_, count_large_exp) = bvh.get_candidates(origin, dir, 200.0, 256);
        assert_eq!(count_large_exp, 10, "With large expansion, should find all 10");
    }

    #[test]
    fn test_nearest_hit_matches_brute_force() {
        let tris = cube_triangles();
        let bvh = Bvh::new(&tris);

        // Test several rays
        let rays = vec![
            (Vec3::new(0.5, 0.5, 2.0), Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::new(0.5, 0.5, -1.0), Vec3::new(0.0, 0.0, 1.0)),
            (Vec3::new(2.0, 0.5, 0.5), Vec3::new(-1.0, 0.0, 0.0)),
            (Vec3::new(0.5, 2.0, 0.5), Vec3::new(0.0, -1.0, 0.0)),
            (Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, 1.0)), // miss
        ];

        for (origin, dir) in &rays {
            let (bvh_idx, bvh_t) = bvh.nearest_hit(*origin, *dir);

            // Brute force
            let mut bf_idx = -1i32;
            let mut bf_t = f32::INFINITY;
            for (ti, tri) in tris.iter().enumerate() {
                let v0 = Vec3::from_slice(&tri[0..3]);
                let v1 = Vec3::from_slice(&tri[3..6]);
                let v2 = Vec3::from_slice(&tri[6..9]);
                let (t, hit) = ray_triangle_intersect(*origin, *dir, v0, v1, v2);
                if hit && t < bf_t {
                    bf_t = t;
                    bf_idx = ti as i32;
                }
            }

            // Both should agree on hit/miss
            assert_eq!(
                bvh_idx >= 0,
                bf_idx >= 0,
                "Hit/miss mismatch for ray {origin:?} -> {dir:?}: bvh={bvh_idx}, bf={bf_idx}"
            );
            if bf_idx >= 0 {
                // t values must match (indices may differ for coplanar triangles)
                assert!(
                    (bvh_t - bf_t).abs() < 1e-5,
                    "t mismatch: bvh={bvh_t}, bf={bf_t}"
                );
            }
        }
    }

    #[test]
    fn test_ray_triangle_intersect_basic() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        // Hit
        let (t, hit) = ray_triangle_intersect(Vec3::new(0.1, 0.1, 1.0), Vec3::new(0.0, 0.0, -1.0), v0, v1, v2);
        assert!(hit);
        assert!((t - 1.0).abs() < 1e-5);

        // Miss (outside triangle)
        let (_, hit) = ray_triangle_intersect(Vec3::new(2.0, 2.0, 1.0), Vec3::new(0.0, 0.0, -1.0), v0, v1, v2);
        assert!(!hit);

        // Miss (behind ray)
        let (_, hit) = ray_triangle_intersect(Vec3::new(0.1, 0.1, -1.0), Vec3::new(0.0, 0.0, -1.0), v0, v1, v2);
        assert!(!hit);
    }
}
