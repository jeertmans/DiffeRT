//! XLA FFI bridge for BVH acceleration.
//!
//! This module provides the cxx bridge between Rust BVH queries and
//! C++ XLA FFI handlers, enabling BVH queries inside JIT-compiled JAX functions.

use super::bvh::{Vec3, registry_get};
use pyo3::prelude::*;

#[cxx::bridge]
mod ffi_bridge {
    extern "Rust" {
        fn bvh_nearest_hit_ffi(
            bvh_id: u64,
            origins: &[f32],
            directions: &[f32],
            active_mask: &[u8],
            hit_indices: &mut [i32],
            hit_t: &mut [f32],
        );

        fn bvh_get_candidates_ffi(
            bvh_id: u64,
            expansion: f32,
            max_candidates: i32,
            origins: &[f32],
            directions: &[f32],
            candidate_indices: &mut [i32],
            candidate_counts: &mut [i32],
        );
    }

    unsafe extern "C++" {
        include!("ffi.h");

        type XLA_FFI_Error;
        type XLA_FFI_CallFrame;

        unsafe fn BvhNearestHit(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error;
        unsafe fn BvhGetCandidates(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error;
    }
}

/// FFI entry point for nearest-hit queries, called from C++ XLA handler.
///
/// `active_mask` is a byte slice of length `num_triangles` (0 = inactive, nonzero = active).
/// An empty slice means no mask (all triangles active).
fn bvh_nearest_hit_ffi(
    bvh_id: u64,
    origins: &[f32],
    directions: &[f32],
    active_mask: &[u8],
    hit_indices: &mut [i32],
    hit_t: &mut [f32],
) {
    let bvh = match registry_get(bvh_id) {
        Some(b) => b,
        None => {
            hit_indices.fill(-1);
            hit_t.fill(f32::INFINITY);
            return;
        },
    };

    // Convert u8 mask to bool slice (empty = no mask)
    let mask_bools: Vec<bool>;
    let mask_opt = if active_mask.is_empty() {
        None
    } else {
        mask_bools = active_mask.iter().map(|&b| b != 0).collect();
        Some(mask_bools.as_slice())
    };

    let num_rays = hit_indices.len();
    for i in 0..num_rays {
        let origin = Vec3::from_slice(&origins[i * 3..(i + 1) * 3]);
        let dir = Vec3::from_slice(&directions[i * 3..(i + 1) * 3]);
        let (idx, t) = bvh.nearest_hit(origin, dir, mask_opt);
        hit_indices[i] = idx;
        hit_t[i] = t;
    }
}

/// FFI entry point for candidate queries, called from C++ XLA handler.
fn bvh_get_candidates_ffi(
    bvh_id: u64,
    expansion: f32,
    max_candidates: i32,
    origins: &[f32],
    directions: &[f32],
    candidate_indices: &mut [i32],
    candidate_counts: &mut [i32],
) {
    let max_cand = max_candidates as usize;
    let bvh = match registry_get(bvh_id) {
        Some(b) => b,
        None => {
            candidate_indices.fill(-1);
            candidate_counts.fill(0);
            return;
        },
    };

    let num_rays = candidate_counts.len();
    for i in 0..num_rays {
        let origin = Vec3::from_slice(&origins[i * 3..(i + 1) * 3]);
        let dir = Vec3::from_slice(&directions[i * 3..(i + 1) * 3]);
        let (candidates, count) = bvh.get_candidates(origin, dir, expansion, max_cand);
        candidate_counts[i] = count as i32;
        let row_offset = i * max_cand;
        for j in 0..max_cand {
            candidate_indices[row_offset + j] = if j < candidates.len() {
                candidates[j] as i32
            } else {
                -1
            };
        }
    }
}

// ---------------------------------------------------------------------------
// PyCapsule exports
// ---------------------------------------------------------------------------

#[pyfunction]
pub(crate) fn bvh_nearest_hit_capsule(py: Python<'_>) -> PyResult<PyObject> {
    use std::ffi::c_void;
    let fn_ptr: *mut c_void = ffi_bridge::BvhNearestHit as *mut c_void;
    unsafe {
        let capsule = pyo3::ffi::PyCapsule_New(fn_ptr, std::ptr::null(), None);
        if capsule.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create PyCapsule for BvhNearestHit",
            ));
        }
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}

#[pyfunction]
pub(crate) fn bvh_get_candidates_capsule(py: Python<'_>) -> PyResult<PyObject> {
    use std::ffi::c_void;
    let fn_ptr: *mut c_void = ffi_bridge::BvhGetCandidates as *mut c_void;
    unsafe {
        let capsule = pyo3::ffi::PyCapsule_New(fn_ptr, std::ptr::null(), None);
        if capsule.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create PyCapsule for BvhGetCandidates",
            ));
        }
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}
