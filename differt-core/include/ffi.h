#pragma once

#include "xla/ffi/api/ffi.h"

// BVH nearest-hit: for each ray, find the closest active triangle.
// Inputs: ray_origins [num_rays, 3], ray_directions [num_rays, 3],
//         active_mask [num_triangles] (PRED, or [0] for no mask)
// Attrs: bvh_id (u64)
// Outputs: hit_indices [num_rays] (i32), hit_t [num_rays] (f32)
extern "C" XLA_FFI_Error *BvhNearestHit(XLA_FFI_CallFrame *call_frame);

// BVH get-candidates: for each ray, find candidate triangles with expanded AABBs.
// Inputs: ray_origins [num_rays, 3], ray_directions [num_rays, 3]
// Attrs: bvh_id (u64), expansion (f32), max_candidates (i32)
// Outputs: candidate_indices [num_rays, max_candidates] (i32),
//          candidate_counts [num_rays] (i32)
extern "C" XLA_FFI_Error *BvhGetCandidates(XLA_FFI_CallFrame *call_frame);
