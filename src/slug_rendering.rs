use std::collections::HashMap;

use harfrust::{GlyphBuffer, ShapeOptions, Shaper, UnicodeBuffer};
use ttf_parser::Face;

#[cfg(not(feature = "ash"))]
use crate::Offset2D;
#[cfg(feature = "ash")]
use ash_lib::vk::Offset2D;

use crate::{
  BAND_COUNT, INDICES_PER_GLYPH, PointRect, ProcessedGlyphData, SlugGlyphProcessor, SlugVertex,
  SlugVertexBandInfo, SlugVertexGlyphInBandLocation, SlugVertexMaxBandIndices, VERTICES_PER_GLYPH,
};

/// Reference to the Curves / Band texture data
pub struct SlugTextureData<'a> {
  /// Control point / curves texture data
  ///
  /// Length will always be equal to TEX_WIDTH * curve_tex_height
  pub curve_tex_data: &'a [[f32; 4]],
  /// Band data texture data
  ///
  /// Length will always be equal to TEX_WIDTH * band_tex_height
  pub band_tex_data: &'a [[u32; 4]],
  pub curve_tex_height: usize,
  pub band_tex_height: usize,
}

impl<'a> SlugTextureData<'a> {
  pub fn curve_tex_size(&self) -> u64 {
    std::mem::size_of_val(self.curve_tex_data) as u64
  }

  pub fn band_tex_size(&self) -> u64 {
    std::mem::size_of_val(self.band_tex_data) as u64
  }
}

/// Shapes and stores processed glyphs in a HashMap
pub struct SlugRendering<'a> {
  text_buffer: Option<UnicodeBuffer>,
  pub shaper: Shaper<'a>,
  pub font_face: &'a Face<'a>,
  font_ascender: f32,

  processed_glyph_map: HashMap<u16, Option<ProcessedGlyphData>>,
  glyph_processor: SlugGlyphProcessor,
}

/// Result of text processing
#[derive(Clone, Copy, Debug)]
pub struct TextBuildResult {
  /// Unscaled final offset at the end of the text
  pub end_offset: Offset2D,
  /// Dimensions and position of the text
  pub rect: PointRect,
  /// True if the operation required updating textures with new glyphs
  pub new_glyphs: bool,
  /// Index of the first vertex added in this block
  pub new_vertex_offset: u32,
  /// Index of the first index added in this block
  pub new_index_offset: u32,
  /// Length of the new added vertices block
  pub new_vertex_count: u32,
  /// Length of the new added indices block
  pub new_index_count: u32,
}

/// Result of text processing
#[derive(Clone, Copy, Debug)]
pub struct SimulateTextBuildResult {
  /// Unscaled final offset at the end of the text
  pub end_offset: Offset2D,
  /// Dimensions and position of the text
  pub rect: PointRect,
  /// True if the operation required updating textures with new glyphs
  pub new_glyphs: bool,
  /// Length of the new added vertices block
  pub new_vertex_count: u32,
  /// Length of the new added indices block
  pub new_index_count: u32,
}

/// Result of multiline text processing
#[derive(Clone, Copy, Debug)]
pub struct MultilineBuildResult {
  // Dimensions and position of the first line
  pub first_line_rect: PointRect,
  pub total: TextBuildResult,
}

impl<'a> SlugRendering<'a> {
  /// Creates a new empty SlugRendering for a specific font.
  ///
  /// Note: make sure font_face and shaper target the same font (and index, if the font file is a collection).
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

  /// Process glyphs in the passed string and adds them to the HashMap / textures,
  /// without returning vertices / indices for the text.
  ///
  /// Useful if you want to make sure the glyphs exist in the textures for later use.
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

  fn get_glyph_buffer_text_horizontal_size(buffer: &GlyphBuffer) -> i32 {
    let mut cur = 0;
    let mut min = 0;
    let mut max = 0;
    for pos in buffer.glyph_positions().iter() {
      cur += pos.x_advance;
      let cur_pos = cur + pos.x_offset;
      if cur_pos < min {
        min = cur_pos;
      }
      if cur_pos > max {
        max = cur_pos;
      }
    }

    max - min
  }

  /// Shape text, process new glyphs and append text glyph data to vertices and indexes.
  ///
  /// `font_size` and `offset`` are set in the font's em scale.
  ///
  /// `vertex_offset` sets what vertex do indices start pointing to. Useful when binding different regions
  /// of the vertex/index buffer and using offsets in vkCmdDrawIndexed.
  ///
  /// `center_text` centers text horizontally by applying an additional negative offset equal to
  /// half of the text horizontal size.
  ///
  /// You can call this function multiple times while keeping text in the same "block"
  /// (rendered in one draw call using one Model-View-Projection matrix) by
  /// setting `vertex_offset` to `vertices.len()` and manually changing `offset` using line distance or offsets from
  /// previous build text function calls.
  pub fn build_text(
    &mut self,
    text: &str,
    font_size: usize,
    // em scale
    offset: Offset2D,
    vertex_offset: u32,
    center_text: bool,
    vertices: &mut Vec<SlugVertex>,
    indices: &mut Vec<u32>,
  ) -> TextBuildResult {
    let mut text_buffer = self.text_buffer.take().unwrap();

    text_buffer.push_str(text);

    text_buffer.set_direction(harfrust::Direction::LeftToRight);

    let glyph_buffer = self.shaper.shape(text_buffer, ShapeOptions::new());
    let scale = font_size as f32 / (self.shaper.units_per_em() as f32);

    let new_vertex_offset = vertices.len() as u32;
    let new_index_offset = indices.len() as u32;
    let mut new_vertex_count = 0;
    let mut new_index_count = 0;

    let mut new_glyphs = false;
    let mut cursor_x = offset.x;
    let mut cursor_y = offset.y;
    let mut vertex_offset: u32 = vertex_offset;
    let mut full_text_bounds = PointRect::REVERSED_INFINITY;

    if center_text {
      let horizontal_size = Self::get_glyph_buffer_text_horizontal_size(&glyph_buffer);
      cursor_x -= horizontal_size / 2;
    }

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
          if processed_opt.is_some() {
            new_glyphs = true;
          }
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
        min: [x0_unscaled as f32 * scale, -y1_unscaled as f32 * scale],
        max: [x1_unscaled as f32 * scale, -y0_unscaled as f32 * scale],
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
          area.min[0],
          area.max[1],
          -1.0,
          -1.0,
          bbox.x_min as f32,
          bbox.y_min as f32,
        ], // bottom-left
        [
          area.max[0],
          area.max[1],
          1.0,
          -1.0,
          bbox.x_max as f32,
          bbox.y_min as f32,
        ], // bottom-right
        [
          area.max[0],
          area.min[1],
          1.0,
          1.0,
          bbox.x_max as f32,
          bbox.y_max as f32,
        ], // top-right
        [
          area.min[0],
          area.min[1],
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
      new_vertex_count += corners.len() as u32;

      let base = vertex_offset;
      let new_indices = [base, base + 1, base + 2, base, base + 2, base + 3];
      indices.extend_from_slice(&new_indices);
      new_index_count += new_indices.len() as u32;

      cursor_x += pos.x_advance;
      cursor_y += pos.y_advance;
      vertex_offset += VERTICES_PER_GLYPH as u32;
    }

    self.text_buffer = Some(glyph_buffer.clear());

    TextBuildResult {
      end_offset: Offset2D {
        x: cursor_x,
        y: cursor_y,
      },
      rect: full_text_bounds,
      new_glyphs,
      new_vertex_offset,
      new_index_offset,
      new_vertex_count,
      new_index_count,
    }
  }

  /// Same as build_text but do not add vertices/indices.
  ///
  /// Note: This still process and adds new glyphs to the HashMap.
  pub fn simulate_build_text(
    &mut self,
    text: &str,
    font_size: usize,
    // em scale
    offset: Offset2D,
    center_text: bool,
  ) -> SimulateTextBuildResult {
    let mut text_buffer = self.text_buffer.take().unwrap();

    text_buffer.push_str(text);

    text_buffer.set_direction(harfrust::Direction::LeftToRight);

    let glyph_buffer = self.shaper.shape(text_buffer, ShapeOptions::new());
    let scale = font_size as f32 / (self.shaper.units_per_em() as f32);

    let mut new_vertex_count = 0;
    let mut new_index_count = 0;

    let mut new_glyphs = false;
    let mut cursor_x = offset.x;
    let mut cursor_y = offset.y;
    let mut full_text_bounds = PointRect::REVERSED_INFINITY;

    if center_text {
      let horizontal_size = Self::get_glyph_buffer_text_horizontal_size(&glyph_buffer);
      cursor_x -= horizontal_size / 2;
    }

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
          if processed_opt.is_some() {
            new_glyphs = true;
          }
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
        min: [x0_unscaled as f32 * scale, -y1_unscaled as f32 * scale],
        max: [x1_unscaled as f32 * scale, -y0_unscaled as f32 * scale],
      };
      full_text_bounds = full_text_bounds.or(area);

      new_vertex_count += VERTICES_PER_GLYPH as u32;
      new_index_count += INDICES_PER_GLYPH as u32;

      cursor_x += pos.x_advance;
      cursor_y += pos.y_advance;
    }

    self.text_buffer = Some(glyph_buffer.clear());

    SimulateTextBuildResult {
      end_offset: Offset2D {
        x: cursor_x,
        y: cursor_y,
      },
      rect: full_text_bounds,
      new_glyphs,
      new_vertex_count,
      new_index_count,
    }
  }

  pub fn get_line_dist(&self, mult: f32) -> i32 {
    (self.font_ascender * mult) as i32
  }

  // todo: write more multiline explanation
  /// Perform build_text on multiple lines.
  ///
  /// All lines will be added as one block, meaning that the indices will point to
  /// all the line's vertices as if they were in the same group.
  ///
  /// Returns PointRect::REVERSED_INFINITY if no lines are specified.
  pub fn build_lines(
    &mut self,
    text: &[&str],
    font_size: usize,
    offset: Offset2D,
    line_distance_mult: f32,
    vertex_offset: u32,
    center_text: bool,
    vertices: &mut Vec<SlugVertex>,
    indices: &mut Vec<u32>,
  ) -> MultilineBuildResult {
    let line_distance = self.get_line_dist(line_distance_mult);

    let new_vertex_offset = vertices.len() as u32;
    let new_index_offset = indices.len() as u32;
    let mut total_vertex_count = 0;
    let mut total_index_count = 0;

    if text.is_empty() {
      return MultilineBuildResult {
        first_line_rect: PointRect::REVERSED_INFINITY,
        total: TextBuildResult {
          end_offset: offset,
          rect: PointRect::REVERSED_INFINITY,
          new_glyphs: false,
          new_vertex_offset,
          new_index_offset,
          new_vertex_count: 0,
          new_index_count: 0,
        },
      };
    }

    let TextBuildResult {
      rect: first_line_rect,
      end_offset: first_offset,
      new_glyphs: first_new_glyphs,
      new_vertex_count,
      new_index_count,
      ..
    } = self.build_text(
      text[0],
      font_size,
      Offset2D {
        x: offset.x,
        y: offset.y,
      },
      vertex_offset,
      center_text,
      vertices,
      indices,
    );
    total_vertex_count += new_vertex_count;
    total_index_count += new_index_count;

    let mut line_offset = line_distance;
    let mut total_rect = first_line_rect;
    let mut last_offset = first_offset;
    let mut new_glyphs = first_new_glyphs;

    for &line in text[1..].iter() {
      let TextBuildResult {
        rect: line_rect,
        end_offset: new_offset,
        new_glyphs: cur_new_glyphs,
        new_vertex_count,
        new_index_count,
        ..
      } = self.build_text(
        line,
        font_size,
        Offset2D {
          x: offset.x,
          y: offset.y - line_offset,
        },
        vertex_offset,
        center_text,
        vertices,
        indices,
      );

      line_offset += line_distance;
      total_rect = total_rect.or(line_rect);
      last_offset = new_offset;
      if cur_new_glyphs {
        new_glyphs = true;
      }
      total_vertex_count += new_vertex_count;
      total_index_count += new_index_count;
    }

    MultilineBuildResult {
      first_line_rect,
      total: TextBuildResult {
        end_offset: last_offset,
        rect: total_rect,
        new_glyphs,
        new_vertex_offset,
        new_index_offset,
        new_vertex_count: total_vertex_count,
        new_index_count: total_index_count,
      },
    }
  }

  /// Get a reference to texture data.
  pub fn get_texture_data(&'a self) -> SlugTextureData<'a> {
    SlugTextureData {
      curve_tex_data: &self.glyph_processor.curve_tex_data,
      band_tex_data: &self.glyph_processor.band_tex_data,
      curve_tex_height: self.glyph_processor.curve_tex_height,
      band_tex_height: self.glyph_processor.band_tex_height,
    }
  }
}
