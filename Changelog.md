# Changelog

## [Unreleased] - ReleaseDate

### Added

- Add `slug_rendering::TextBuildResult` new vertex/index count and offset attributes.
- Add `SlugPushConstants::new` and `SlugPushConstants::new_column_major`.

### Changed

- Change `slug_vertex_shader.hlsl` push constants `slug_viewport` to use `float2` (two f32 floats) instead of `float4`.
- Change `SlugRendering` text shaping to no longer auto-infer vertex offsets, instead it is now passed as a parameter.
- Functions that output vertices now have a `center_text` parameter flag that attempts to center new vertices horizontally.
- Change `SlugPushConstants::new_2d` to take viewport_dimensions in array form.

## [0.1.4] - 2026-09-05

Released crate.
