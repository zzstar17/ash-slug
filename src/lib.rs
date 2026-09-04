//! # Ash-slug
//!
//! Library dedicated to process text data for use in [Lengyel's Slug font rendering algorithm](https://github.com/EricLengyel/Slug)
//! shader implementations. It assists with populating vertex/index buffers and textures used in the shaders.
//!
//! Although this library was designed for use with [Ash](https://github.com/ash-rs/ash) (a wrapper around Vulkan), the Ash bindings
//! are actually optional and are included only with the "ash" feature (which is enabled by default). The text processing
//! code is available for use in other Vulkan wrappers and graphics APIs.
//!
//! This library depends on [HarfRust](https://github.com/harfbuzz/harfrust) for text shaping and
//! [ttf-parser](https://github.com/harfbuzz/ttf-parser) for parsing fonts.
//!
//! ## Example
//!
//! ```
//! use ash_slug::{SlugRendering, SlugVertex};
//! # use ash_lib::vk;
//! #
//! # use font_kit::{
//! #     family_name::FamilyName, handle::Handle, properties::Properties, source::SystemSource,
//! # };
//! # pub fn load_font() -> (Box<[u8]>, u32) {
//! #     let handle = SystemSource::new()
//! #         .select_best_match(&[FamilyName::SansSerif], &Properties::new())
//! #         .expect("Failed to select font");
//! #
//! #     let (font_path, font_index) = match handle {
//! #         Handle::Path { path, font_index } => (path, font_index),
//! #         Handle::Memory { .. } => panic!(),
//! #     };
//! #
//! #     let font_bytes = std::fs::read(font_path)
//! #         .expect("Failed to read font file")
//! #         .into_boxed_slice();
//! #
//! #     (font_bytes, font_index)
//! # }
//!
//! // use font-kit (https://github.com/servo/font-kit) or any other library
//! // capable of finding system fonts.
//! // otherwise, read a font file manually
//! //
//! // font_index corresponds to the font index in a font collection, 0 otherwise
//! let (font_bytes, font_index) = load_font();
//!
//! let font_ref: harfrust::FontRef = harfrust::FontRef::from_index(&font_bytes, font_index)
//!     .expect("Failed to read font data");
//! let shaper_data: harfrust::ShaperData = harfrust::ShaperData::new(&font_ref);
//!
//! let font_face: ttf_parser::Face = ttf_parser::Face::parse(&font_bytes, font_index)
//!     .expect("Failed to parse font face from font data");
//!
//! let shaper = shaper_data.shaper(&font_ref).build();
//! let mut slug = SlugRendering::new(&font_face, shaper);
//!
//! // add number glyphs to the beginning of the textures
//! slug.add_glyphs_in_str("0123456789");
//!
//! let mut vertices: Vec<SlugVertex> = Vec::new();
//! let mut indices: Vec<u32> = Vec::new();
//!
//! let font_size = 18;
//!
//! // a different offset is provided for non ash use
//! let build_result = slug.build_text(
//!     "hello, ",
//!     font_size,
//!     vk::Offset2D::default(),
//!     &mut vertices,
//!     &mut indices,
//! );
//! // write "world!" to the right
//! slug.build_text(
//!     "world!",
//!     font_size,
//!     build_result.end_offset,
//!     &mut vertices,
//!     &mut indices,
//! );
//!
//! // write multiple lines
//! let _multiline_result = slug.build_lines(
//!     &["Welcome", "to", "font", "rendering"],
//!     font_size,
//!     vk::Offset2D { x: 0, y: slug.get_line_dist(1.5) * -2},
//!     1.5, // line distance (depending on font ascender)
//!     &mut vertices,
//!     &mut indices,
//! );
//!
//! let simulate_result = slug.simulate_build_text("17", font_size, vk::Offset2D::default());
//! assert!(!simulate_result.new_glyphs);
//!
//! // unicode support depends on the font, unknown glyphs will be replaced by fonts "Notdef" symbol
//! slug.build_text("c̷̦̮̀r̸̡̩̲̒a̵̪̺̼̾̆͝z̴̛̘̜y̸̢͖̌̌,  魚", font_size, vk::Offset2D::default(), &mut vertices, &mut indices);
//!
//! let textures = slug.get_texture_data();
//! // copy the data to your graphics API buffers in any way you see fit
//! // ptr::copy_nonoverlapping(
//! //     textures.curve_tex_data.as_ptr() as *const u8,
//! //     staging_buffer_ptr.as_ptr(),
//! //     textures.curve_tex_size() as usize,
//! // );
//! ```
//!
//! ## Vulkan shaders and changes
//!
//! The Vulkan version of shaders are available in [in this library's repository](https://github.com/zzstar17/ash-slug/blob/main/shaders).
//! The initial version is also available [in the original reference implementation](https://github.com/EricLengyel/Slug).
//!

use std::fmt::Debug;

#[cfg(feature = "ash")]
use ash_lib::vk;
#[cfg(feature = "ash")]
use std::mem::offset_of;

mod slug_glyph_processor;
/// Shaping and individual glyph storage
pub mod slug_rendering;

pub use slug_glyph_processor::{ProcessedGlyphData, SlugGlyphProcessor};
pub use slug_rendering::SlugRendering;

/// Number of vertices generated for each processed glyph
pub const VERTICES_PER_GLYPH: usize = 4;
/// Number of indices generated for each processed glyph
pub const INDICES_PER_GLYPH: usize = 6;

/// Width of the curve/band texture in texels/pixels
pub const TEX_WIDTH: usize = 4096;

/// Vulkan format of the curves / control point texture
#[cfg(feature = "ash")]
pub const CURVE_TEX_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
/// Vulkan format of the band data texture
#[cfg(feature = "ash")]
pub const BAND_TEX_FORMAT: vk::Format = vk::Format::R32G32B32A32_UINT;

/// R32G32B32A32_SFLOAT
pub type CurveTexel = [f32; 4];
/// R32G32B32A32_UINT
pub type BandTexel = [u32; 4];

/// Hardcoded in the shader
pub const BAND_COUNT: usize = 8;

const LINE_EPSILON: f32 = 0.125;

#[cfg(not(feature = "ash"))]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Offset2D {
  pub x: i32,
  pub y: i32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
/// Represents a Quadratic Bezier Curve
pub struct QuadCurve {
  pub p0: [f32; 2],
  pub p1: [f32; 2],
  pub p2: [f32; 2],
}

#[derive(Copy, Clone, Debug, Default)]
/// Rectangle area defined by two points
pub struct PointRect {
  pub min: [f32; 2],
  pub max: [f32; 2],
}

impl PointRect {
  pub const REVERSED_INFINITY: Self = PointRect {
    min: [f32::INFINITY, f32::INFINITY],
    max: [f32::NEG_INFINITY, f32::NEG_INFINITY],
  };

  pub fn width(&self) -> f32 {
    self.max[0] - self.min[0]
  }

  pub fn height(&self) -> f32 {
    self.max[1] - self.min[1]
  }

  /// Get PointRect that contains both rectangle areas.
  pub fn or(self, other: PointRect) -> Self {
    Self {
      min: [self.min[0].min(other.min[0]), self.min[1].min(other.min[1])],
      max: [self.max[0].max(other.max[0]), self.max[1].max(other.max[1])],
    }
  }

  #[cfg(feature = "ash")]
  pub fn into_vk_extent(self) -> vk::Extent2D {
    vk::Extent2D {
      width: self.width() as u32,
      height: self.height() as u32,
    }
  }
}

impl QuadCurve {
  fn line_to_quadratic(a: [f32; 2], b: [f32; 2]) -> Self {
    let mut mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    let dif = [b[0] - a[0], b[1] - a[1]];

    // Perfectly degenerate quadratics interact badly with Slug's root eligibility
    // logic on diagonal segments, causing scanline dropouts. Keep axis-aligned
    // lines exact, but bow diagonal lines by an imperceptible amount so they
    // behave like ordinary quadratics.
    if dif[0].abs() > 0.1 && dif[1].abs() > 0.1 {
      let length = f32::hypot(mid[0], mid[1]);
      if length > 0.0 {
        let inv_length = LINE_EPSILON / length;
        mid[0] -= dif[1] * inv_length;
        mid[1] += dif[0] * inv_length;
      }
    }

    QuadCurve {
      p0: a,
      p1: mid,
      p2: b,
    }
  }

  fn bounding_box(&self) -> [f32; 4] {
    let [x0, x1, x2] = [self.p0[0], self.p1[0], self.p2[0]];
    let [y0, y1, y2] = [self.p0[1], self.p1[1], self.p2[1]];

    let cxmin = x0.min(x1).min(x2);
    let cxmax = x0.max(x1).max(x2);
    let cymin = y0.min(y1).min(y2);
    let cymax = y0.max(y1).max(y2);

    [cxmin, cymin, cxmax, cymax]
  }

  pub fn max_x(&self) -> f32 {
    self.p0[0].max(self.p1[0]).max(self.p2[0])
  }

  pub fn max_y(&self) -> f32 {
    self.p0[1].max(self.p1[1]).max(self.p2[1])
  }
}

/// Location of glyph data in band texture
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlugVertexGlyphInBandLocation {
  pub x: u16,
  pub y: u16,
}

/// Max band indices (usually equal to BAND_COUNT - 1)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlugVertexMaxBandIndices {
  pub max_band_x: u16,
  pub max_band_y: u16,
}

/// Band scale and offset
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlugVertexBandInfo {
  pub scale_x: f32,
  pub scale_y: f32,
  pub offset_x: f32,
  pub offset_y: f32,
}

/// Layout of a Slug Vertex as stated in the vertex shader
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlugVertex {
  /// Object-space vertex coordinates
  pub obj_space_vertex_coords: [f32; 2],
  /// Object-space normal vector
  pub obj_space_normal_vector: [f32; 2],

  /// Em-space sample coordinates
  pub em_space_sample_coords: [f32; 2],
  /// Location of glyph data in band texture
  pub glyph_in_band_loc: SlugVertexGlyphInBandLocation,
  /// Max band indices (usually equal to BAND_COUNT - 1)
  pub max_band_indices: SlugVertexMaxBandIndices,

  /// Inverse Jacobian matrix entries (00, 01, 10, 11)
  pub jac: [f32; 4],
  /// Band scale and offset
  pub band: SlugVertexBandInfo,
  /// RGBA vertex color
  pub color: [f32; 4],
}

#[cfg(feature = "ash")]
impl SlugVertex {
  const ATTRIBUTE_SIZE: usize = 5;

  pub const fn get_binding_description(binding: u32) -> vk::VertexInputBindingDescription {
    vk::VertexInputBindingDescription {
      binding,
      stride: size_of::<Self>() as u32,
      input_rate: vk::VertexInputRate::VERTEX,
    }
  }

  /// Get attribute descriptions as stated in the shader
  pub const fn get_attribute_descriptions(
    offset: u32,
    binding: u32,
  ) -> [vk::VertexInputAttributeDescription; Self::ATTRIBUTE_SIZE] {
    [
      vk::VertexInputAttributeDescription {
        location: offset,
        binding,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: offset_of!(Self, obj_space_vertex_coords) as u32,
      },
      vk::VertexInputAttributeDescription {
        location: offset + 1,
        binding,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: offset_of!(Self, em_space_sample_coords) as u32,
      },
      vk::VertexInputAttributeDescription {
        location: offset + 2,
        binding,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: offset_of!(Self, jac) as u32,
      },
      vk::VertexInputAttributeDescription {
        location: offset + 3,
        binding,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: offset_of!(Self, band) as u32,
      },
      vk::VertexInputAttributeDescription {
        location: offset + 4,
        binding,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: offset_of!(Self, color) as u32,
      },
    ]
  }
}

/// Vertex shader push constants / uniform buffer parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlugPushConstants {
  // The four rows of the model view projection matrix
  pub mvp_matrix: [[f32; 4]; 4],
  /// Viewport dimensions in texels/pixels
  pub viewport_dimensions: [f32; 4],
}

impl SlugPushConstants {
  /// Create using centered orthographic projection (y up pixel coords)
  pub fn new_2d(
    viewport_dimensions_width: f32,
    viewport_dimensions_height: f32,
    offset: [f32; 2],
  ) -> Self {
    let matrix = [
      [
        2.0 / viewport_dimensions_width,
        0.0,
        0.0,
        offset[0] * 2.0 / viewport_dimensions_width - 1.0,
      ],
      [
        0.0,
        2.0 / viewport_dimensions_height,
        0.0,
        offset[1] * 2.0 / viewport_dimensions_height - 1.0,
      ],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ];
    Self {
      mvp_matrix: matrix,
      viewport_dimensions: [
        viewport_dimensions_width,
        viewport_dimensions_height,
        0.0,
        0.0,
      ],
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::LazyLock;

  #[cfg(not(feature = "ash"))]
  use crate::Offset2D;
  #[cfg(feature = "ash")]
  use ash_lib::vk::Offset2D;
  use font_kit::{
    family_name::FamilyName, handle::Handle, properties::Properties, source::SystemSource,
  };

  use crate::{SlugRendering, slug_rendering::TextBuildResult};

  const FONT_SIZE: usize = 30;

  pub static FONT_BYTES: LazyLock<FontBytes> = LazyLock::new(|| load_font());
  pub static FONT_REF: LazyLock<harfrust::FontRef> = LazyLock::new(|| {
    harfrust::FontRef::from_index(&FONT_BYTES.bytes, FONT_BYTES.font_index)
      .expect("Failed to read font data")
  });
  pub static SHAPER_DATA: LazyLock<harfrust::ShaperData> =
    LazyLock::new(|| harfrust::ShaperData::new(&FONT_REF));
  pub static FONT_FACE: LazyLock<ttf_parser::Face> = LazyLock::new(|| {
    ttf_parser::Face::parse(&FONT_BYTES.bytes, FONT_BYTES.font_index)
      .expect("Failed to parse font face from font data")
  });

  pub struct FontBytes {
    pub bytes: Box<[u8]>,
    pub font_index: u32,
  }

  pub fn load_font() -> FontBytes {
    let handle = SystemSource::new()
      .select_best_match(&[FamilyName::SansSerif], &Properties::new())
      .expect("Failed to select font");

    let (font_path, font_index) = match handle {
      Handle::Path { path, font_index } => (path, font_index),
      Handle::Memory { .. } => panic!(),
    };

    let font_bytes = std::fs::read(font_path)
      .expect("Failed to read font file")
      .into_boxed_slice();

    FontBytes {
      bytes: font_bytes,
      font_index,
    }
  }

  fn assert_result_is_nonzero(result: TextBuildResult) {
    assert!(result.rect.width() > 0.0);
    assert!(result.rect.height() > 0.0);
    assert!(result.end_offset.x > 0);
  }

  #[test]
  fn single_glyph() {
    let shaper = SHAPER_DATA.shaper(&FONT_REF).build();
    let mut slug = SlugRendering::new(&FONT_FACE, shaper);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let result = slug.build_text(
      "a",
      FONT_SIZE,
      Offset2D::default(),
      &mut vertices,
      &mut indices,
    );
    assert_result_is_nonzero(result);
    assert!(result.new_glyphs);

    assert_eq!(vertices.len(), crate::VERTICES_PER_GLYPH);
    assert_eq!(indices.len(), crate::INDICES_PER_GLYPH);
  }

  #[test]
  fn glyphs_stored() {
    let shaper = SHAPER_DATA.shaper(&FONT_REF).build();
    let mut slug = SlugRendering::new(&FONT_FACE, shaper);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    slug.add_glyphs_in_str("testing");

    let result = slug.build_text(
      "test",
      FONT_SIZE,
      Offset2D::default(),
      &mut vertices,
      &mut indices,
    );
    assert_result_is_nonzero(result);
    assert!(!result.new_glyphs);

    let result = slug.build_text(
      ".",
      FONT_SIZE,
      Offset2D::default(),
      &mut vertices,
      &mut indices,
    );
    assert_result_is_nonzero(result);
    assert!(result.new_glyphs);

    assert!(!vertices.is_empty());
    assert!(!indices.is_empty());
  }

  #[test]
  fn simulate() {
    let shaper = SHAPER_DATA.shaper(&FONT_REF).build();
    let mut slug = SlugRendering::new(&FONT_FACE, shaper);

    slug.add_glyphs_in_str("h̶̫̞̭̪̭̮̲̱̟̼͔̳͐̍̇̔̕é̵̡̞͓̤̞͉͔͙̭̞̪̩̝̬ĺ̶͎̗͖̹̳̮̺͕͇͖̩͍͖̏̓̾̈͘l̶̨̹̝̯͕̠̥͔͖̆͜ơ̷̢̤̮̱̤̩̰͍̞͉̳̮̭͕͎͋̌͌̔̓̚ world!");

    let result = slug.simulate_build_text("world", FONT_SIZE, Offset2D::default());
    assert_result_is_nonzero(result);
    assert!(!result.new_glyphs);
  }

  #[test]
  fn multiline_shaping() {
    let shaper = SHAPER_DATA.shaper(&FONT_REF).build();
    let mut slug = SlugRendering::new(&FONT_FACE, shaper);

    slug.add_glyphs_in_str("o world");

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let result = slug.build_lines(
      &["hello", "world!"],
      FONT_SIZE,
      Offset2D { x: 0, y: 0 },
      1.5,
      &mut vertices,
      &mut indices,
    );

    assert!(result.first_line_rect.width() > 0.0);
    assert!(result.first_line_rect.height() > 0.0);
    assert_result_is_nonzero(result.total);
    assert!(result.total.new_glyphs);

    assert!(result.first_line_rect.height() < result.total.rect.height());

    assert!(!vertices.is_empty());
    assert!(!indices.is_empty());
  }

  #[test]
  fn main_example_test() {
    use crate::{SlugRendering, SlugVertex};

    use font_kit::{
      family_name::FamilyName, handle::Handle, properties::Properties, source::SystemSource,
    };
    pub fn load_font() -> (Box<[u8]>, u32) {
      let handle = SystemSource::new()
        .select_best_match(&[FamilyName::SansSerif], &Properties::new())
        .expect("Failed to select font");

      let (font_path, font_index) = match handle {
        Handle::Path { path, font_index } => (path, font_index),
        Handle::Memory { .. } => panic!(),
      };

      let font_bytes = std::fs::read(font_path)
        .expect("Failed to read font file")
        .into_boxed_slice();

      (font_bytes, font_index)
    }

    // use font-kit (https://github.com/servo/font-kit) or any other library capable of finding system fonts
    // otherwise, read a font file manually
    //
    // font_index corresponds to the font index in a font collection, 0 otherwise
    let (font_bytes, font_index) = load_font();

    let font_ref: harfrust::FontRef =
      harfrust::FontRef::from_index(&font_bytes, font_index).expect("Failed to read font data");
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
      Offset2D::default(),
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
      Offset2D {
        x: 0,
        y: slug.get_line_dist(1.5) * -2,
      },
      1.5, // line distance (depending on font ascender)
      &mut vertices,
      &mut indices,
    );

    let simulate_result = slug.simulate_build_text("17", font_size, Offset2D::default());
    assert!(!simulate_result.new_glyphs);

    // unicode support depends on the font, unknown glyphs will be replaced by fonts "Notdef" symbol
    slug.build_text(
      "c̷̦̮̀r̸̡̩̲̒a̵̪̺̼̾̆͝z̴̛̘̜y̸̢͖̌̌,  魚",
      font_size,
      Offset2D::default(),
      &mut vertices,
      &mut indices,
    );

    let _textures = slug.get_texture_data();
    // copy the data to your graphics API buffers in any way you see fit
    // ptr::copy_nonoverlapping(
    //     textures.curve_tex_data.as_ptr() as *const u8,
    //     staging_buffer_ptr.as_ptr(),
    //     textures.curve_tex_size() as usize,
    // );
  }
}
