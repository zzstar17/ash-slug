use ash::vk;
use cgmath::{EuclideanSpace, Point2};
use harfrust::{ShapeOptions, Shaper, UnicodeBuffer};
use std::{collections::HashMap, fmt::Debug, mem::offset_of, ptr};
use ttf_parser::Face;

pub const VERTICES_PER_GLYPH: usize = 4;
pub const INDICES_PER_GLYPH: usize = 6;

// Band count is also hardcoded in the shader
const BAND_COUNT: usize = 8;

const LINE_EPSILON: f32 = 0.125;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
/// Quadratic Bezier curve
pub struct QuadCurve {
  pub p0: Point2<f32>,
  pub p1: Point2<f32>,
  pub p2: Point2<f32>,
}

#[derive(Copy, Clone, Debug)]
pub struct PointRect {
  pub min: Point2<f32>,
  pub max: Point2<f32>,
}

impl PointRect {
  pub const REVERSED_INFINITY: Self = PointRect {
    min: Point2 {
      x: f32::INFINITY,
      y: f32::INFINITY,
    },
    max: Point2 {
      x: f32::NEG_INFINITY,
      y: f32::NEG_INFINITY,
    },
  };

  pub fn width(&self) -> f32 {
    self.max.x - self.min.x
  }

  pub fn height(&self) -> f32 {
    self.max.y - self.min.y
  }

  /// Return PointRect that includes both
  pub fn or(self, other: PointRect) -> Self {
    Self {
      min: Point2 {
        x: self.min.x.min(other.min.x),
        y: self.min.y.min(other.min.y),
      },
      max: Point2 {
        x: self.max.x.max(other.max.x),
        y: self.max.y.max(other.max.y),
      },
    }
  }

  pub fn to_vk_extent(self) -> vk::Extent2D {
    vk::Extent2D {
      width: self.width() as u32,
      height: self.height() as u32,
    }
  }
}

impl QuadCurve {
  fn line_to_quadratic(a: Point2<f32>, b: Point2<f32>) -> Self {
    let mut mid = Point2 {
      x: (a.x + b.x) / 2.0,
      y: (a.y + b.y) / 2.0,
    };
    let dif = b - a;

    // Perfectly degenerate quadratics interact badly with Slug's root eligibility
    // logic on diagonal segments, causing scanline dropouts. Keep axis-aligned
    // lines exact, but bow diagonal lines by an imperceptible amount so they
    // behave like ordinary quadratics.
    if dif.x.abs() > 0.1 && dif.y.abs() > 0.1 {
      let length = f32::hypot(mid.x, mid.y);
      if length > 0.0 {
        let inv_length = LINE_EPSILON / length;
        mid.x -= dif.y * inv_length;
        mid.y += dif.x * inv_length;
      }
    }

    QuadCurve {
      p0: a,
      p1: mid,
      p2: b,
    }
  }

  fn bounding_box(&self) -> [f32; 4] {
    let [x0, x1, x2] = [self.p0.x, self.p1.x, self.p2.x];
    let [y0, y1, y2] = [self.p0.y, self.p1.y, self.p2.y];

    let cxmin = x0.min(x1).min(x2);
    let cxmax = x0.max(x1).max(x2);
    let cymin = y0.min(y1).min(y2);
    let cymax = y0.max(y1).max(y2);

    [cxmin, cymin, cxmax, cymax]
  }

  pub fn max_x(&self) -> f32 {
    self.p0.x.max(self.p1.x).max(self.p2.x)
  }

  pub fn max_y(&self) -> f32 {
    self.p0.y.max(self.p1.y).max(self.p2.y)
  }
}

/// Extract glyph curves
struct SlugCurveExtractor<'a> {
  pub curves: &'a mut Vec<QuadCurve>,
  pub start: Point2<f32>,
  pub cur_location: Point2<f32>,
}

impl<'a> SlugCurveExtractor<'a> {
  pub fn new(curves: &'a mut Vec<QuadCurve>) -> Self {
    Self {
      curves,
      start: Point2 { x: 0.0, y: 0.0 },
      cur_location: Point2 { x: 0.0, y: 0.0 },
    }
  }
}

// see ttf_parser::OutlineBuilder
impl<'a> ttf_parser::OutlineBuilder for SlugCurveExtractor<'a> {
  fn move_to(&mut self, x: f32, y: f32) {
    self.start = Point2 { x, y };
    self.cur_location = self.start;
  }

  fn line_to(&mut self, x: f32, y: f32) {
    let to = Point2 { x, y };
    let diff = to - self.cur_location;
    // ignore vertical / horizontal lines
    if diff.x.abs() > 0.1 || diff.y.abs() > 0.1 {
      self
        .curves
        .push(QuadCurve::line_to_quadratic(self.cur_location, to));
    }
    self.cur_location = to;
  }

  fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
    let to = Point2 { x, y };
    self.curves.push(QuadCurve {
      p0: self.cur_location,
      p1: Point2 { x: x1, y: y1 },
      p2: to,
    });
    self.cur_location = to;
  }

  fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
    let p0 = self.cur_location;
    let p1 = Point2 { x: x1, y: y1 };
    let p2 = Point2 { x: x2, y: y2 };
    let p3 = Point2 { x, y };

    let m01 = p0.midpoint(p1);
    let m12 = p1.midpoint(p2);
    let m23 = p2.midpoint(p3);
    let m012 = m01.midpoint(m12);
    let m123 = m12.midpoint(m23);
    let mid = m012.midpoint(m123);

    self.curves.push(QuadCurve {
      p0,
      p1: m01,
      p2: mid,
    });
    self.curves.push(QuadCurve {
      p0: mid,
      p1: m123,
      p2: p3,
    });

    self.cur_location = p3;
  }

  fn close(&mut self) {
    let full_vec = self.start - self.cur_location;
    // ignore vertical / horizontal lines
    if full_vec.x.abs() > 0.1 || full_vec.y.abs() > 0.1 {
      self
        .curves
        .push(QuadCurve::line_to_quadratic(self.cur_location, self.start));
    }

    self.cur_location = self.start;
  }
}

fn build_glyph_bands(
  curves: &[QuadCurve],
  bounding_box: PointRect,
) -> [Vec<usize>; BAND_COUNT * 2] {
  let PointRect { min, max } = bounding_box;
  let width = max.x - min.x;
  let height = max.y - min.y;

  let mut bands: [Vec<usize>; BAND_COUNT * 2] = Default::default();

  for (c_i, curve) in curves.iter().enumerate() {
    let [cxmin, cymin, cxmax, cymax] = curve.bounding_box();

    // horizontal bands
    {
      let b0 = (((cymin - min.y) / height) * BAND_COUNT as f32) as usize;
      let b1 = (((cymax - min.y) / height) * BAND_COUNT as f32) as usize;
      #[allow(clippy::needless_range_loop)]
      for b in b0..=(b1.min(BAND_COUNT - 1)) {
        bands[b].push(c_i);
      }
    }

    // vertical bands
    {
      let b0 = ((cxmin - min.x) / width * BAND_COUNT as f32) as usize;
      let b1 = ((cxmax - min.x) / width * BAND_COUNT as f32) as usize;
      #[allow(clippy::needless_range_loop)]
      for b in b0..=(b1.min(BAND_COUNT - 1)) {
        bands[BAND_COUNT + b].push(c_i);
      }
    }
  }

  // Sort curves: h-bands by descending max x, v-bands by descending max y
  for curve_indices in bands[0..BAND_COUNT].iter_mut() {
    curve_indices.sort_by(|&a, &b| {
      let curve1_max_x = curves[a].max_x();
      let curve2_max_x = curves[b].max_x();
      // reverse ordering
      curve2_max_x.total_cmp(&curve1_max_x)
    });
  }
  for curve_indices in bands[BAND_COUNT..(BAND_COUNT * 2)].iter_mut() {
    curve_indices.sort_by(|&a, &b| {
      let curve1_max_y = curves[a].max_y();
      let curve2_max_y = curves[b].max_y();
      // reverse ordering
      curve2_max_y.total_cmp(&curve1_max_y)
    });
  }

  bands
}

pub const TEX_WIDTH: usize = 4096;

#[derive(Debug, Clone)]
pub struct SlugGlyphProcessor {
  glyph_curve_buffer: Vec<QuadCurve>,

  pub curve_tex_data: Vec<[f32; 4]>,
  pub band_tex_data: Vec<[u32; 4]>,

  total_curve_texels: usize,
  pub curve_tex_height: usize,
  total_band_texels: usize,
  pub band_tex_height: usize,

  curve_texel_i: usize,
  band_texel_i: usize,
}

impl SlugGlyphProcessor {
  pub fn new() -> Self {
    Self {
      glyph_curve_buffer: Vec::new(),

      curve_tex_data: Vec::new(),
      band_tex_data: Vec::new(),

      total_curve_texels: 0,
      curve_tex_height: 0,
      total_band_texels: 0,
      band_tex_height: 0,

      curve_texel_i: 0,
      band_texel_i: 0,
    }
  }

  fn expand_curve_data_new_glyph(&mut self) {
    self.total_curve_texels += self.glyph_curve_buffer.len() * 2;
    let new_curve_tex_height = (self.total_curve_texels / TEX_WIDTH) + 1;
    if new_curve_tex_height > self.curve_tex_height {
      let expand_len = TEX_WIDTH * (new_curve_tex_height - self.curve_tex_height);
      self.curve_tex_data.reserve_exact(expand_len);
      unsafe {
        let end_ptr = self
          .curve_tex_data
          .as_mut_ptr()
          .add(self.curve_tex_data.len());
        ptr::write_bytes(end_ptr, 0, expand_len);
        self
          .curve_tex_data
          .set_len(self.curve_tex_data.len() + expand_len);
      }
      self.curve_tex_height = new_curve_tex_height;
    }
  }

  fn expand_band_data_new_glyph(
    &mut self,
    glyph_band_curve_indices: &[Vec<usize>; BAND_COUNT * 2],
  ) {
    let header_count = glyph_band_curve_indices.len();
    // Pad to avoid header wrapping at row boundary
    let padded = TEX_WIDTH - (self.total_band_texels % TEX_WIDTH);
    if padded < header_count && padded < TEX_WIDTH {
      self.total_band_texels += padded;
    }
    self.total_band_texels += header_count;
    for indices in glyph_band_curve_indices.iter() {
      self.total_band_texels += indices.len();
    }

    let new_band_tex_height = (self.total_band_texels / TEX_WIDTH) + 1;
    if new_band_tex_height > self.band_tex_height {
      let expand_len = TEX_WIDTH * (new_band_tex_height - self.band_tex_height);
      self.band_tex_data.reserve_exact(expand_len);
      unsafe {
        let end_ptr = self
          .band_tex_data
          .as_mut_ptr()
          .add(self.band_tex_data.len());
        ptr::write_bytes(end_ptr, 0, expand_len);
        self
          .band_tex_data
          .set_len(self.band_tex_data.len() + expand_len);
      }
      self.band_tex_height = new_band_tex_height;
    }
  }

  // --- Curve texture (RGBA32Float, width 4096) ---
  // Each curve = 2 texels: (p0x, p0y, p1x, p1y) and (p2x, p2y, 0, 0)
  fn write_curves_new_glyph(&mut self) {
    self.expand_curve_data_new_glyph();
    for c in self.glyph_curve_buffer.iter() {
      // Texel 0: (p0x, p0y, p1x, p1y)
      let i0 = self.curve_texel_i;
      self.curve_tex_data[i0] = [c.p0.x, c.p0.y, c.p1.x, c.p1.y];

      // Texel 1: (p2x, p2y, 0, 0)
      let i1 = self.curve_texel_i + 1;
      self.curve_tex_data[i1][0] = c.p2.x;
      self.curve_tex_data[i1][1] = c.p2.y;

      self.curve_texel_i += 2;
    }
  }

  // --- Band texture (RGBA32Uint, width 4096) ---
  // Per glyph: [hBand headers...] [vBand headers...] [curve index lists...]
  // Each header texel: (curveCount, offsetFromGlyphLoc, 0, 0)
  // Each curve ref texel: (curveTexX, curveTexY, 0, 0)
  //
  // returns glyph in band location
  fn write_bands_new_glyph(
    &mut self,
    glyph_curve_start_i: usize,
    glyph_band_curve_indices: &[Vec<usize>; BAND_COUNT * 2],
  ) -> (u16, u16) {
    self.expand_band_data_new_glyph(glyph_band_curve_indices);
    let header_count = glyph_band_curve_indices.len();

    // Ensure headers don't straddle a row boundary
    let mut band_loc_x = self.band_texel_i % TEX_WIDTH;
    let mut band_loc_y = self.band_texel_i / TEX_WIDTH;
    if band_loc_x + header_count > TEX_WIDTH {
      band_loc_x = 0;
      band_loc_y += 1;
      self.band_texel_i = band_loc_y * TEX_WIDTH;
    }

    // Write band headers
    let mut curve_list_offset = header_count;
    for (i, band_indices) in glyph_band_curve_indices.iter().enumerate() {
      let texel_offset = self.band_texel_i + i;
      // header texel: (curveCount, offsetFromGlyphLoc, 0, 0)
      self.band_tex_data[texel_offset][0] = band_indices.len() as u32;
      self.band_tex_data[texel_offset][1] = curve_list_offset as u32;

      // Write curve ref texels
      let list_start = self.band_texel_i + curve_list_offset;
      for (j, curve_i) in band_indices.iter().enumerate() {
        let curve_texel = glyph_curve_start_i + curve_i * 2;
        let curve_tex_x = curve_texel % TEX_WIDTH;
        let curve_tex_y = curve_texel / TEX_WIDTH;

        let texel_offset = list_start + j;
        // curve ref texel: (curveTexX, curveTexY, 0, 0)
        self.band_tex_data[texel_offset][0] = curve_tex_x as u32;
        self.band_tex_data[texel_offset][1] = curve_tex_y as u32;
      }

      curve_list_offset += band_indices.len();
    }

    self.band_texel_i += curve_list_offset;

    (band_loc_x as u16, band_loc_y.try_into().unwrap())
  }

  pub fn process_new_glyph(&mut self, face: &Face, glyph_id: u16) -> Option<ProcessedGlyphData> {
    self.glyph_curve_buffer.clear();
    let mut curve_extractor = SlugCurveExtractor::new(&mut self.glyph_curve_buffer);

    // extracts curves here
    let bounding_box = match face.outline_glyph(ttf_parser::GlyphId(glyph_id), &mut curve_extractor)
    {
      Some(outline) => outline,
      None => {
        return None;
      }
    };

    let point_bounding_box = PointRect {
      min: Point2 {
        x: bounding_box.x_min as f32,
        y: bounding_box.y_min as f32,
      },
      max: Point2 {
        x: bounding_box.x_max as f32,
        y: bounding_box.y_max as f32,
      },
    };

    let band_curve_indices = build_glyph_bands(&self.glyph_curve_buffer, point_bounding_box);

    let glyph_curve_start_i = self.curve_texel_i;
    self.write_curves_new_glyph();

    let (band_loc_x, band_loc_y) =
      self.write_bands_new_glyph(glyph_curve_start_i, &band_curve_indices);

    Some(ProcessedGlyphData {
      bounding_box,
      band_loc_x,
      band_loc_y,
    })
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlugVertexGlyphInBandLocation {
  pub x: u16,
  pub y: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlugVertexMaxBandIndices {
  pub max_band_x: u16,
  pub max_band_y: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlugVertexBandInfo {
  pub scale_x: f32,
  pub scale_y: f32,
  pub offset_x: f32,
  pub offset_y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlugVertex {
  // pos
  pub obj_space_vertex_coords: [f32; 2],
  pub obj_space_normal_vector: [f32; 2],

  // tex
  pub em_space_sample_coords: [f32; 2],
  pub glyph_in_band_loc: SlugVertexGlyphInBandLocation,
  pub max_band_indices: SlugVertexMaxBandIndices,

  // jac
  pub jac: [f32; 4],
  // bnd
  pub band: SlugVertexBandInfo,
  // col
  pub color: [f32; 4],
}

impl SlugVertex {
  const ATTRIBUTE_SIZE: usize = 5;

  pub const fn get_binding_description(binding: u32) -> vk::VertexInputBindingDescription {
    vk::VertexInputBindingDescription {
      binding,
      stride: size_of::<Self>() as u32,
      input_rate: vk::VertexInputRate::VERTEX,
    }
  }

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

#[derive(Clone, Copy, Debug)]
pub struct SlugTextureData<'a> {
  /// curve_tex_data.len() == TEX_WIDTH * self.curve_tex_height
  pub curve_tex_data: &'a [[f32; 4]],
  /// band_tex_data.len()  == TEX_WIDTH * self.band_tex_height
  pub band_tex_data: &'a [[u32; 4]],
  pub curve_tex_height: usize,
  pub band_tex_height: usize,
}

impl<'a> SlugTextureData<'a> {
  pub fn curve_tex_size(&self) -> u64 {
    (self.curve_tex_data.len() * size_of::<[f32; 4]>()) as u64
  }

  pub fn band_tex_size(&self) -> u64 {
    (self.band_tex_data.len() * size_of::<[u32; 4]>()) as u64
  }
}

#[derive(Debug, Clone, Copy)]
pub struct ProcessedGlyphData {
  bounding_box: ttf_parser::Rect,
  band_loc_x: u16,
  band_loc_y: u16,
}

pub struct SlugRendering<'a> {
  text_buffer: Option<UnicodeBuffer>,
  pub shaper: Shaper<'a>,
  pub font_face: &'a Face<'a>,
  font_ascender: f32,

  processed_glyph_map: HashMap<u16, Option<ProcessedGlyphData>>,
  glyph_processor: SlugGlyphProcessor,
}

#[derive(Debug, Clone, Copy)]
pub struct MultilineRect {
  pub first_line: PointRect,
  pub total: PointRect,
}

impl MultilineRect {
  pub fn from_line_rects(rects: &[PointRect]) -> Self {
    assert!(!rects.is_empty());
    let first_line = rects[0];
    let mut total = first_line;
    for line in rects[1..].iter() {
      total = total.or(*line);
    }

    Self { first_line, total }
  }
}

pub struct TextBuildBounds {
  // unscaled
  pub offset: vk::Offset2D,
  pub rect: PointRect,
}

impl<'a> SlugRendering<'a> {
  // make sure font_ref and shaper_data target the same font and index
  pub fn new(font_face: &'a Face<'a>, shaper: Shaper<'a>) -> Self {
    let text_buffer = UnicodeBuffer::new();

    // None for glyphs with no bounding box (like empty space)
    let processed_glyph_map: HashMap<u16, Option<ProcessedGlyphData>> = HashMap::new();
    let glyph_processor = SlugGlyphProcessor::new();

    Self {
      text_buffer: Some(text_buffer),
      font_face,
      font_ascender: font_face.ascender() as f32,
      processed_glyph_map,
      glyph_processor,
      shaper,
    }
  }

  // add glyphs to map and textures but not create vertices for them
  pub fn add_glyphs_in_str(&mut self, text: &str) {
    let mut text_buffer = self.text_buffer.take().unwrap();

    text_buffer.push_str(text);

    text_buffer.set_direction(harfrust::Direction::LeftToRight);

    let glyph_buffer = self.shaper.shape(text_buffer, ShapeOptions::new());

    // add new glyphs
    for glyph_info in glyph_buffer.glyph_infos() {
      let glyph_id = glyph_info.glyph_id.try_into().unwrap();
      if self.processed_glyph_map.contains_key(&glyph_id) {
        continue;
      }

      let processed_glyph = self
        .glyph_processor
        .process_new_glyph(self.font_face, glyph_id);
      self.processed_glyph_map.insert(glyph_id, processed_glyph);
    }

    self.text_buffer = Some(glyph_buffer.clear());
  }

  pub fn get_bounding_box_from_char(&self, c: char) -> Option<ttf_parser::Rect> {
    let glyph_id = self.font_face.glyph_index(c).unwrap();
    let opt = self.processed_glyph_map.get(&glyph_id.0).unwrap();
    opt.map(|data| data.bounding_box)
  }

  pub fn build_text(
    &mut self,
    text: &str,
    font_size: usize,
    // unscaled offset
    offset: vk::Offset2D,
    vertices: &mut Vec<SlugVertex>,
    indices: &mut Vec<u32>,
  ) -> TextBuildBounds {
    let mut text_buffer = self.text_buffer.take().unwrap();

    text_buffer.push_str(text);

    text_buffer.set_direction(harfrust::Direction::LeftToRight);

    let glyph_buffer = self.shaper.shape(text_buffer, ShapeOptions::new());
    let scale = font_size as f32 / (self.shaper.units_per_em() as f32);

    let mut cursor_x = offset.x;
    let mut cursor_y = offset.y;
    let mut quad_idx: u32 = (vertices.len() / VERTICES_PER_GLYPH) as u32;
    let mut full_text_bounds = PointRect::REVERSED_INFINITY;
    for (info, pos) in glyph_buffer
      .glyph_infos()
      .iter()
      .zip(glyph_buffer.glyph_positions().iter())
    {
      let glyph_id = info.glyph_id as u16;
      // None if glyph is empty space (has no bounding box)
      let glyph_processed_opt = match self.processed_glyph_map.get(&glyph_id) {
        Some(opt) => *opt,
        None => {
          // add new glyph to map
          let processed_opt = self
            .glyph_processor
            .process_new_glyph(self.font_face, glyph_id);
          self.processed_glyph_map.insert(glyph_id, processed_opt);
          processed_opt
        }
      };
      let glyph_processed_data = match glyph_processed_opt {
        // full data
        Some(values) => values,
        None => {
          // empty glyph -> skip
          cursor_x += pos.x_advance;
          cursor_y += pos.y_advance;
          continue;
        }
      };

      let bbox = glyph_processed_data.bounding_box;

      let width = bbox.x_max - bbox.x_min;
      let height = bbox.y_max - bbox.y_min;

      // Object-space position (Y-up screen pixels)
      let ox: i32 = cursor_x + pos.x_offset;
      let oy = cursor_y + pos.y_offset;
      let x0_unscaled = ox + bbox.x_min as i32;
      let y0_unscaled = oy + bbox.y_min as i32;
      let x1_unscaled = ox + bbox.x_max as i32;
      let y1_unscaled = oy + bbox.y_max as i32;
      let area = PointRect {
        min: Point2 {
          x: x0_unscaled as f32 * scale,
          y: -y1_unscaled as f32 * scale,
        },
        max: Point2 {
          x: x1_unscaled as f32 * scale,
          y: -y0_unscaled as f32 * scale,
        },
      };
      full_text_bounds = full_text_bounds.or(area);

      // Band transform: maps em-space to band indices
      let band_scale_x = if width > 0 {
        BAND_COUNT as f32 / width as f32
      } else {
        0.0
      };
      let band_scale_y = if height > 0 {
        BAND_COUNT as f32 / height as f32
      } else {
        0.0
      };
      let band_offset_x = -bbox.x_min as f32 * band_scale_x;
      let band_offset_y = -bbox.y_min as f32 * band_scale_y;

      let band_max_x = BAND_COUNT - 1;
      let band_max_y = BAND_COUNT - 1;

      let inv_scale = 1.0 / scale;

      let corners = [
        [
          area.min.x,
          area.max.y,
          -1.0,
          -1.0,
          bbox.x_min as f32,
          bbox.y_min as f32,
        ], // bottom-left
        [
          area.max.x,
          area.max.y,
          1.0,
          -1.0,
          bbox.x_max as f32,
          bbox.y_min as f32,
        ], // bottom-right
        [
          area.max.x,
          area.min.y,
          1.0,
          1.0,
          bbox.x_max as f32,
          bbox.y_max as f32,
        ], // top-right
        [
          area.min.x,
          area.min.y,
          -1.0,
          1.0,
          bbox.x_min as f32,
          bbox.y_max as f32,
        ], // top-left
      ];
      for [px, py, nx, ny, ex, ey] in corners {
        let vertex = SlugVertex {
          // pos (location 0): object-space position + normal
          obj_space_vertex_coords: [px, py],
          obj_space_normal_vector: [nx, ny],

          // tex (location 1): em-space coords + packed glyph/band data
          em_space_sample_coords: [ex, ey],
          // all this below could be instance data
          glyph_in_band_loc: SlugVertexGlyphInBandLocation {
            x: glyph_processed_data.band_loc_x,
            y: glyph_processed_data.band_loc_y,
          },
          max_band_indices: SlugVertexMaxBandIndices {
            max_band_x: band_max_x as u16,
            max_band_y: band_max_y as u16,
          },

          // jac (location 2): inverse Jacobian (d(em)/d(obj))
          jac: [inv_scale, 0.0, 0.0, inv_scale],
          // bnd (location 3): band transform (scale + offset)
          band: SlugVertexBandInfo {
            scale_x: band_scale_x,
            scale_y: band_scale_y,
            offset_x: band_offset_x,
            offset_y: band_offset_y,
          },
          color: [0.0, 0.0, 0.0, 1.0],
        };
        vertices.push(vertex);
      }

      let base = quad_idx * VERTICES_PER_GLYPH as u32;
      indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
      cursor_x += pos.x_advance;
      cursor_y += pos.y_advance;
      quad_idx += 1;
    }

    self.text_buffer = Some(glyph_buffer.clear());

    TextBuildBounds {
      offset: vk::Offset2D {
        x: cursor_x,
        y: cursor_y,
      },
      rect: full_text_bounds,
    }
  }

  /// Same as build_text but do not add vertices/indices
  pub fn simulate_build_text(
    &mut self,
    text: &str,
    font_size: usize,
    // unscaled offset
    offset: vk::Offset2D,
  ) -> TextBuildBounds {
    let mut text_buffer = self.text_buffer.take().unwrap();

    text_buffer.push_str(text);

    text_buffer.set_direction(harfrust::Direction::LeftToRight);

    let glyph_buffer = self.shaper.shape(text_buffer, ShapeOptions::new());
    let scale = font_size as f32 / (self.shaper.units_per_em() as f32);

    let mut cursor_x = offset.x;
    let mut cursor_y = offset.y;
    let mut full_text_bounds = PointRect::REVERSED_INFINITY;
    for (info, pos) in glyph_buffer
      .glyph_infos()
      .iter()
      .zip(glyph_buffer.glyph_positions().iter())
    {
      let glyph_id = info.glyph_id as u16;
      // None if glyph is empty space (has no bounding box)
      let glyph_processed_opt = match self.processed_glyph_map.get(&glyph_id) {
        Some(opt) => *opt,
        None => {
          // add new glyph to map
          let processed_opt = self
            .glyph_processor
            .process_new_glyph(self.font_face, glyph_id);
          self.processed_glyph_map.insert(glyph_id, processed_opt);
          processed_opt
        }
      };
      let glyph_processed_data = match glyph_processed_opt {
        // full data
        Some(values) => values,
        None => {
          // empty glyph -> skip
          cursor_x += pos.x_advance;
          cursor_y += pos.y_advance;
          continue;
        }
      };

      let bbox = glyph_processed_data.bounding_box;

      // Object-space position (Y-up screen pixels)
      let ox: i32 = cursor_x + pos.x_offset;
      let oy = cursor_y + pos.y_offset;
      let x0_unscaled = ox + bbox.x_min as i32;
      let y0_unscaled = oy + bbox.y_min as i32;
      let x1_unscaled = ox + bbox.x_max as i32;
      let y1_unscaled = oy + bbox.y_max as i32;
      let area = PointRect {
        min: Point2 {
          x: x0_unscaled as f32 * scale,
          y: -y1_unscaled as f32 * scale,
        },
        max: Point2 {
          x: x1_unscaled as f32 * scale,
          y: -y0_unscaled as f32 * scale,
        },
      };
      full_text_bounds = full_text_bounds.or(area);

      cursor_x += pos.x_advance;
      cursor_y += pos.y_advance;
    }

    self.text_buffer = Some(glyph_buffer.clear());

    TextBuildBounds {
      offset: vk::Offset2D {
        x: cursor_x,
        y: cursor_y,
      },
      rect: full_text_bounds,
    }
  }

  pub fn get_line_dist(&self, mult: f32) -> i32 {
    (self.font_ascender * mult) as i32
  }

  pub fn build_lines(
    &mut self,
    text: &[&str],
    font_size: usize,
    offset: vk::Offset2D,
    line_distance_mult: f32,
    vertices: &mut Vec<SlugVertex>,
    indices: &mut Vec<u32>,
  ) -> MultilineRect {
    assert!(
      !text.is_empty(),
      "Slug build lines text must be at least one line"
    );

    let TextBuildBounds {
      rect: first_rect, ..
    } = self.build_text(text[0], font_size, offset, vertices, indices);
    let line_distance = self.get_line_dist(line_distance_mult);

    let mut total_rect = first_rect;
    let mut line_offset = line_distance;
    for &line in text[1..].iter() {
      let TextBuildBounds {
        rect: line_rect, ..
      } = self.build_text(
        line,
        font_size,
        vk::Offset2D {
          x: offset.x,
          y: offset.y - line_offset,
        },
        vertices,
        indices,
      );

      line_offset += line_distance;
      total_rect = total_rect.or(line_rect);
    }

    MultilineRect {
      first_line: first_rect,
      total: total_rect,
    }
  }

  pub fn get_texture_data(&'a self) -> SlugTextureData<'a> {
    SlugTextureData {
      curve_tex_data: &self.glyph_processor.curve_tex_data,
      band_tex_data: &self.glyph_processor.band_tex_data,
      curve_tex_height: self.glyph_processor.curve_tex_height,
      band_tex_height: self.glyph_processor.band_tex_height,
    }
  }
}
