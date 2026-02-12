/* Framebuffer drawing routines.
 *
 * This module is architecture-independent — it just writes pixels to a
 * memory buffer. The framebuffer address and layout come from the bootloader.
 */

use font8x8::legacy::BASIC_LEGACY;

// Pixel color in 0x00RRGGBB format.
#[derive(Clone, Copy)]
pub struct Color(pub u32);

impl Color {
    pub const WHITE: Color = Color(0x00FFFFFF);
    pub const DARK_BLUE: Color = Color(0x001A1A2E);
    pub const LIGHT_GRAY: Color = Color(0x00AAAAAA);
}

// Draw a single character at pixel position (x, y).
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

// Draw a string at pixel position (x, y).
pub fn draw_str(fb_ptr: *mut u8, pitch: usize, x: usize, y: usize, s: &str, color: Color) {
    for (i, c) in s.chars().enumerate() {
        draw_char(fb_ptr, pitch, x + i * 8, y, c, color);
    }
}

// Fill the entire framebuffer with a solid color.
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
