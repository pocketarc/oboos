//! Framebuffer drawing routines.
//!
//! This module is architecture-independent — it just writes pixels to a
//! memory buffer. The framebuffer address and layout come from the bootloader.

use font8x8::legacy::BASIC_LEGACY;

/// Pixel color in 0x00RRGGBB format.
///
/// # Examples
///
/// ```
/// // Named constants for common colors:
/// let bg = Color::DARK_BLUE;
///
/// // Or construct from a raw RGB value:
/// let red = Color(0x00FF0000);
/// ```
#[derive(Clone, Copy)]
pub struct Color(pub u32);

impl Color {
    /// Bright white — used for primary text.
    pub const WHITE: Color = Color(0x00FFFFFF);
    /// Dark navy — the default background color.
    pub const DARK_BLUE: Color = Color(0x001A1A2E);
    /// Muted gray — used for secondary text.
    pub const LIGHT_GRAY: Color = Color(0x00AAAAAA);
}

// Draw a single character at pixel position (x, y).
//
// Looks up the glyph bitmap from the font8x8 crate's BASIC_LEGACY table
// (128 entries, one per ASCII code point). Each glyph is 8 bytes — one byte
// per row. The font uses LSB-left ordering: bit 0 is the leftmost pixel,
// bit 7 is the rightmost. For each set bit, we write a colored pixel into
// the framebuffer at the corresponding offset.
fn draw_char(fb_ptr: *mut u8, pitch: usize, x: usize, y: usize, c: char, color: Color) {
    let idx = c as usize;
    let glyph = if idx < 128 {
        BASIC_LEGACY[idx]
    } else {
        BASIC_LEGACY[0]
    };
    
    let (r, g, b) = ((color.0 >> 16) as u8, (color.0 >> 8) as u8, color.0 as u8);

    for (row, &bits) in glyph.iter().enumerate() {
        for col in 0..8 {
            if bits & (1 << col) != 0 {
                let offset = (y + row) * pitch + (x + col) * 4;
                unsafe {
                    // BGRA byte order
                    *fb_ptr.add(offset) = b;
                    *fb_ptr.add(offset + 1) = g;
                    *fb_ptr.add(offset + 2) = r;
                }
            }
        }
    }
}

/// Draw a string at pixel position (x, y).
///
/// `pitch` is the number of bytes per row in the framebuffer. This can
/// differ from `width * bytes_per_pixel` because the GPU may pad rows
/// for alignment.
///
/// # Examples
///
/// ```
/// // Center a string horizontally on screen:
/// let msg = "Hello, kernel!";
/// let x = (width - msg.len() * 8) / 2;
/// draw_str(fb_ptr, pitch, x, 100, msg, Color::WHITE);
/// ```
pub fn draw_str(fb_ptr: *mut u8, pitch: usize, x: usize, y: usize, s: &str, color: Color) {
    for (i, c) in s.chars().enumerate() {
        draw_char(fb_ptr, pitch, x + i * 8, y, c, color);
    }
}

/// Fill the entire framebuffer with a solid color.
pub fn clear(fb_ptr: *mut u8, width: usize, height: usize, pitch: usize, color: Color) {
    let (r, g, b) = ((color.0 >> 16) as u8, (color.0 >> 8) as u8, color.0 as u8);

    for y in 0..height {
        for x in 0..width {
            let offset = y * pitch + x * 4;
            unsafe {
                *fb_ptr.add(offset) = b;
                *fb_ptr.add(offset + 1) = g;
                *fb_ptr.add(offset + 2) = r;
            }
        }
    }
}
