# Ash-slug

Library dedicated to process text data for use in [Lengyel's Slug font rendering algorithm](https://github.com/EricLengyel/Slug)
shader implementations. It assists with populating vertex/index buffers and textures used in the shaders.

Although this library was designed for use with [Ash](https://github.com/ash-rs/ash) (a wrapper around Vulkan), the Ash bindings
are actually optional and are included only with the "ash" feature (which is enabled by default). The text processing
code is available for use in other Vulkan wrappers and graphics APIs.

This library depends on [HarfRust](https://github.com/harfbuzz/harfrust) for text shaping and [ttf-parser](https://github.com/harfbuzz/ttf-parser)
for parsing fonts.

## Example

```rust
use ash_slug::{SlugRendering, SlugVertex};

// use font-kit (https://github.com/servo/font-kit) or any other library capable of finding system fonts
// otherwise, read a font file manually
//
// font_index corresponds to the font index in a font collection, 0 otherwise
let (font_bytes, font_index) = load_font();

let font_ref: harfrust::FontRef = harfrust::FontRef::from_index(&font_bytes, font_index)
    .expect("Failed to read font data");
let shaper_data: harfrust::ShaperData = harfrust::ShaperData::new(&font_ref);

let font_face: ttf_parser::Face = ttf_parser::Face::parse(&font_bytes, font_index)
    .expect("Failed to parse font face from font data");

let shaper = shaper_data.shaper(&font_ref).build();
let mut slug = SlugRendering::new(&font_face, shaper);

// add number glyphs to the beginning of the textures
slug.add_glyphs_in_str("0123456789");

let mut vertices: Vec<SlugVertex> = Vec::new();
let mut indices: Vec<u32> = Vec::new();

let font_size = 18;

// a different offset is provided for non ash use
let build_result = slug.build_text(
    "hello, ",
    font_size,
    vk::Offset2D::default(),
    &mut vertices,
    &mut indices,
);
// write "world!" to the right
slug.build_text(
    "world!",
    font_size,
    build_result.end_offset,
    &mut vertices,
    &mut indices,
);

// write multiple lines
let _multiline_result = slug.build_lines(
    &["Welcome", "to", "font", "rendering"],
    font_size,
    vk::Offset2D { x: 0, y: slug.get_line_dist(1.5) * -2},
    1.5, // line distance (depending on font ascender)
    &mut vertices,
    &mut indices,
);

let simulate_result = slug.simulate_build_text("17", font_size, vk::Offset2D::default());
assert!(!simulate_result.new_glyphs);

// unicode support depends on the font, unknown glyphs will be replaced by fonts "Notdef" symbol
slug.build_text("c̷̦̮̀r̸̡̩̲̒a̵̪̺̼̾̆͝z̴̛̘̜y̸̢͖̌̌,  魚", font_size, vk::Offset2D::default(), &mut vertices, &mut indices);

let textures = slug.get_texture_data();
// copy the data to your graphics API buffers in any way you see fit
// ptr::copy_nonoverlapping(
//     textures.curve_tex_data.as_ptr() as *const u8,
//     staging_buffer_ptr.as_ptr(),
//     textures.curve_tex_size() as usize,
// );
```

## Vulkan shaders and changes

The Vulkan version of the shader implementations is available in `./vulkan_shaders`.
The reference version is also available [in the original repository](https://github.com/EricLengyel/Slug).

Vertex shader changes:

- Changed `cbuffer ParamStruct` to Vulkan Push Constants `[[vk::push_constant]] PushConstants pc`, as well as all variables
  that mention it.

Pixel / fragment shader changes:

- Added `[[vk::binding(0, 0)]]` (Vulkan binding 0, set 0) and `[[vk::binding(1, 0)]]` (Vulkan binding 1, set 0) to curveTexture
  and bandTexture Texture2Ds, respectively.

## More information

See https://terathon.com/blog/decade-slug.html, https://github.com/EricLengyel/Slug/

## Roadmap

- Add support for multiple simultaneous fonts
- Add automatic bounds line-breaking

## Contributing

Feel free to open new issues / pull requests for any features that you want to request.

Expect breaking changes between minor versions.

## Acknowledgements

Thank you to diffusionstudio for [providing the initial inspiration for the library](https://github.com/diffusionstudio/slug-webgpu)
and of course a big thank you to Eric Lengyel for creating the Slug algorithm and releasing it into the public domain.
