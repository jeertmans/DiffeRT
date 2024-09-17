# Project roadmap

Below is a list of future features by category.

If you feel something is missing, please create
[an issue for a feature request](https://github.com/jeertmans/DiffeRT/issues).

## Library

- Create higher-level path tracing utilities
- Support Min-Path-Tracing
- Implement diffraction on Fermat-Path-Tracing
- Implement diffraction *and* reflection on the same path
- Implement a function that delegates reflection-only paths
  to `image_method`, and the rest to `fermat`.
- Silently skip non-triangles inside `.obj` file with `TriangleMesh`?
  Or log error messages instead
- Fix performances issues on RX-grid examples
- Improve `differt.em.special.erf` accuracy and speed
- Add Fresnel coefficients utilities
- Add UTD coefficients utilities
- Add antenna / path polarization utilities
- Read optional color from `.obj` file and use it for 3D plots
- Extend mesh support to quad meshes?
- Support point clouds?
- Investigate zoom-issue on Vispy plots with large scenes (camera issue?)
- Support custom dtypes (and remove hard-coded dtypes)

## Documentation

- Document install procedure
- Create quickstart
- Create Docker imager and document it
- Create a contributing guide
- Add optimization tutorial
- Show how to *smooth* edges by applying a soft threshold
  on the intersection test.

## Testing

- Improve plotting tests

## GitHub

- Add CITATION.cff file
