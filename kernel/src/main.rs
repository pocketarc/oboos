//! OBOOS — Off By One Operating System.
//!
//! A from-scratch, Rust-based OS built for learning. Boots via the Limine
//! protocol, targets x86_64 first, with a HAL designed for adding aarch64 later.

#![no_std] // No standard library — we ARE the operating system.
#![no_main] // No C runtime, no normal main(). We define our own entry point.
#![feature(abi_x86_interrupt)] // Nightly: lets us write interrupt handlers with proper calling convention.

mod arch;
mod framebuffer;
mod platform;

use platform::{Key, Keyboard, Platform};

// Limine boot protocol requests.
//
// These are static structs with magic numbers baked in. The bootloader scans
// our binary for these markers, processes the requests, and fills in response
// pointers before jumping to our entry point. It's a clever handshake that
// avoids needing any runtime negotiation.

use limine::request::{FramebufferRequest, RequestsEndMarker, RequestsStartMarker};
use limine::BaseRevision;

#[used]
#[unsafe(link_section = ".requests_start_marker")]
static _START_MARKER: RequestsStartMarker = RequestsStartMarker::new();

#[used]
#[unsafe(link_section = ".requests")]
static BASE_REVISION: BaseRevision = BaseRevision::with_revision(3);

#[used]
#[unsafe(link_section = ".requests")]
static FRAMEBUFFER: FramebufferRequest = FramebufferRequest::new();

#[used]
#[unsafe(link_section = ".requests_end_marker")]
static _END_MARKER: RequestsEndMarker = RequestsEndMarker::new();

/// Formatted printing to the serial console.
///
/// Routes through the serial port so output appears in the terminal
/// running QEMU (via `-serial stdio`).
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = write!(crate::arch::Serial, $($arg)*);
    }};
}

/// Like [`print!`], but appends a newline.
#[macro_export]
macro_rules! println {
    () => { $crate::print!("\n") };
    ($($arg:tt)*) => {{
        $crate::print!($($arg)*);
        $crate::print!("\n");
    }};
}

/// Kernel entry point — where the bootloader jumps to.
#[unsafe(no_mangle)]
extern "C" fn kmain() -> ! {
    // Initialize the platform (serial port, etc.)
    let _platform = arch::Arch::init();

    // Verify the bootloader speaks our protocol revision.
    assert!(BASE_REVISION.is_supported());

    println!("=========================");
    println!("  OBOOS v0.0");
    println!("  Off By One Operating System");
    println!("=========================");
    println!();

    // Paint the framebuffer and draw text as proof of life on the QEMU display.
    if let Some(response) = FRAMEBUFFER.get_response() {
        if let Some(fb) = response.framebuffers().next() {
            let ptr = fb.addr() as *mut u8;
            let w = fb.width() as usize;
            let h = fb.height() as usize;
            let pitch = fb.pitch() as usize;

            println!("[ok] Framebuffer: {}x{}, {} bpp", w, h, fb.bpp());

            draw_splash(ptr, w, h, pitch, framebuffer::Color::DARK_BLUE);
            println!("[ok] Press Enter to randomize colors!");
            println!("[ok] Press F to trigger a divide-by-zero fault.");

            // Poll keyboard in a loop. When Enter is pressed, pick a new
            // background color using the CPU cycle counter as entropy.
            loop {
                if let Some(key) = arch::KeyboardDriver::poll() {
                    match key {
                        Key::Enter => {
                            let rand = arch::Arch::entropy();
                            let bg = framebuffer::Color((rand as u32) & 0x00FFFFFF);
                            draw_splash(ptr, w, h, pitch, bg);
                            println!("[ok] Color: #{:06X}", bg.0);
                        }
                        Key::F => {
                            println!("[!!] Triggering test fault...");
                            println!("[!!] IDT installed — expect a panic message.");
                            arch::Arch::trigger_test_fault();
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    println!("No framebuffer available. Halting.");
    loop {
        arch::Arch::halt_until_interrupt();
    }
}

// Paint the splash screen: solid background with centered title text.
fn draw_splash(ptr: *mut u8, w: usize, h: usize, pitch: usize, bg: framebuffer::Color) {
    framebuffer::clear(ptr, w, h, pitch, bg);

    let title = "OBOOS v0.0";
    let subtitle = "Off By One Operating System";
    let hint = "Enter = colors / F = fault";
    let title_x = (w - title.len() * 8) / 2;
    let subtitle_x = (w - subtitle.len() * 8) / 2;
    let hint_x = (w - hint.len() * 8) / 2;
    let center_y = h / 2 - 16;

    framebuffer::draw_str(ptr, pitch, title_x, center_y, title, framebuffer::Color::WHITE);
    framebuffer::draw_str(ptr, pitch, subtitle_x, center_y + 16, subtitle, framebuffer::Color::LIGHT_GRAY);
    framebuffer::draw_str(ptr, pitch, hint_x, center_y + 40, hint, framebuffer::Color::LIGHT_GRAY);
}

/// Panic handler — required by `#![no_std]`.
///
/// Prints the panic message to serial and halts forever.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!();
    println!("!!! KERNEL PANIC !!!");
    println!("{}", info);
    loop {
        arch::Arch::halt_until_interrupt();
    }
}
