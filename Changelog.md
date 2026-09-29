# Changelog

## [0.1.5] - 2026-09-29

### Added

- Add `SlugRendering::build_text_with_offset` and `SlugRendering::build_lines_with_offset`.
- Add `slug_rendering::TextBuildResult` new vertex/index count and offset attributes.
- Add `SlugPushConstants::new` and `SlugPushConstants::new_column_major`.

### Changed

- Change `slug_vertex_shader.hlsl` push constants `slug_viewport` to use `float2` (two f32 floats) instead of `float4`.
- Functions that output vertices now have a `center_text` parameter flag that attempts to center new vertices horizontally.
- Change `SlugPushConstants::new_2d` to take viewport_dimensions in array form.

## [0.1.4] - 2026-09-05

Released crate.
