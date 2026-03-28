/// XLA FFI handlers for BVH acceleration.
///
/// These thin C++ wrappers decode XLA buffers and call into Rust via cxx.
/// The actual computation happens in Rust (src/accel/bvh.rs).

#include "differt-core/src/accel/ffi.rs.h" // cxx-generated bridge header
#include "ffi.h"

namespace ffi = xla::ffi;

// --- BvhNearestHit ---

ffi::Error BvhNearestHitImpl(uint64_t bvh_id,
                              ffi::Buffer<ffi::F32> origins,
                              ffi::Buffer<ffi::F32> directions,
                              ffi::Buffer<ffi::PRED> active_mask,
                              ffi::ResultBuffer<ffi::S32> hit_indices,
                              ffi::ResultBuffer<ffi::F32> hit_t) {
  auto origins_dims = origins.dimensions();
  auto dirs_dims = directions.dimensions();

  if (origins_dims.size() != 2 || origins_dims[1] != 3) {
    return ffi::Error::InvalidArgument(
        "BvhNearestHit: ray_origins must have shape [num_rays, 3]");
  }
  if (dirs_dims.size() != 2 || dirs_dims[1] != 3) {
    return ffi::Error::InvalidArgument(
        "BvhNearestHit: ray_directions must have shape [num_rays, 3]");
  }

  int64_t num_rays = origins_dims[0];

  // Active mask: shape [num_triangles] or [0] (empty means all active)
  auto mask_dims = active_mask.dimensions();
  size_t mask_len = 1;
  for (size_t i = 0; i < mask_dims.size(); ++i) {
    mask_len *= static_cast<size_t>(mask_dims[i]);
  }
  rust::Slice<const uint8_t> mask_slice{
      reinterpret_cast<const uint8_t *>(active_mask.typed_data()), mask_len};

  rust::Slice<const float> origins_slice{origins.typed_data(),
                                         static_cast<size_t>(num_rays * 3)};
  rust::Slice<const float> dirs_slice{directions.typed_data(),
                                      static_cast<size_t>(num_rays * 3)};
  rust::Slice<int32_t> indices_slice{(*hit_indices).typed_data(),
                                     static_cast<size_t>(num_rays)};
  rust::Slice<float> t_slice{(*hit_t).typed_data(),
                              static_cast<size_t>(num_rays)};

  bvh_nearest_hit_ffi(bvh_id, origins_slice, dirs_slice, mask_slice,
                       indices_slice, t_slice);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BvhNearestHit, BvhNearestHitImpl,
    ffi::Ffi::Bind()
        .Attr<uint64_t>("bvh_id")
        .Arg<ffi::Buffer<ffi::F32>>()   // ray_origins
        .Arg<ffi::Buffer<ffi::F32>>()   // ray_directions
        .Arg<ffi::Buffer<ffi::PRED>>()  // active_mask
        .Ret<ffi::Buffer<ffi::S32>>()   // hit_indices
        .Ret<ffi::Buffer<ffi::F32>>());  // hit_t

// --- BvhGetCandidates ---

ffi::Error BvhGetCandidatesImpl(uint64_t bvh_id, float expansion,
                                 int32_t max_candidates,
                                 ffi::Buffer<ffi::F32> origins,
                                 ffi::Buffer<ffi::F32> directions,
                                 ffi::ResultBuffer<ffi::S32> candidate_indices,
                                 ffi::ResultBuffer<ffi::S32> candidate_counts) {
  auto origins_dims = origins.dimensions();

  if (origins_dims.size() != 2 || origins_dims[1] != 3) {
    return ffi::Error::InvalidArgument(
        "BvhGetCandidates: ray_origins must have shape [num_rays, 3]");
  }

  int64_t num_rays = origins_dims[0];

  rust::Slice<const float> origins_slice{origins.typed_data(),
                                         static_cast<size_t>(num_rays * 3)};
  rust::Slice<const float> dirs_slice{directions.typed_data(),
                                      static_cast<size_t>(num_rays * 3)};
  rust::Slice<int32_t> indices_slice{
      (*candidate_indices).typed_data(),
      static_cast<size_t>(num_rays * max_candidates)};
  rust::Slice<int32_t> counts_slice{(*candidate_counts).typed_data(),
                                     static_cast<size_t>(num_rays)};

  bvh_get_candidates_ffi(bvh_id, expansion, max_candidates, origins_slice,
                          dirs_slice, indices_slice, counts_slice);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BvhGetCandidates, BvhGetCandidatesImpl,
    ffi::Ffi::Bind()
        .Attr<uint64_t>("bvh_id")
        .Attr<float>("expansion")
        .Attr<int32_t>("max_candidates")
        .Arg<ffi::Buffer<ffi::F32>>()  // ray_origins
        .Arg<ffi::Buffer<ffi::F32>>()  // ray_directions
        .Ret<ffi::Buffer<ffi::S32>>()  // candidate_indices
        .Ret<ffi::Buffer<ffi::S32>>()); // candidate_counts
