#![no_std] // No standard library — we ARE the operating system.
#![no_main] // No C runtime, no normal main(). We define our own entry point.

mod arch;
mod platform;

use platform::Platform;

/* ---------------------------------------------------------------------------
 * Limine boot protocol requests.
 *
 * These are static structs with magic numbers baked in. The bootloader scans
 * our binary for these markers, processes the requests, and fills in response
 * pointers before jumping to our entry point. It's a clever handshake that
 * avoids needing any runtime negotiation.
*/

use limine::BaseRevision;
use limine::request::{FramebufferRequest, RequestsEndMarker, RequestsStartMarker};

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

/*
 * print! / println! macros — the first thing any OS needs.
 * These route through the serial port so output appears in the terminal
 * running QEMU (via -serial stdio).
 */

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = write!(crate::arch::x86_64::serial::Serial, $($arg)*);
    }};
}

#[macro_export]
macro_rules! println {
    () => { $crate::print!("\n") };
    ($($arg:tt)*) => {{
        $crate::print!($($arg)*);
        $crate::print!("\n");
    }};
}

/*
 * Kernel entry point — where the bootloader jumps to.
 */

#[unsafe(no_mangle)]
extern "C" fn kmain() -> ! {
    // Initialize the platform (serial port, etc.)
    let _platform = arch::X86_64::init();

    // Verify the bootloader speaks our protocol revision.
    assert!(BASE_REVISION.is_supported());

    println!("=========================");
    println!("  OBOOS v0.0");
    println!("  Off By One Operating System");
    println!("=========================");
    println!();

    // Try to paint the framebuffer as proof of life on the QEMU display.
    if let Some(response) = FRAMEBUFFER.get_response() {
        if let Some(fb) = response.framebuffers().next() {
            paint_framebuffer(&fb);
            println!(
                "[ok] Framebuffer: {}x{}, {} bpp",
                fb.width(),
                fb.height(),
                fb.bpp()
            );
        }
    }

    println!();
    println!("Kernel initialized. Halting.");

    // Nothing left to do — halt the CPU in a loop.
    // The loop is necessary because interrupts (which we haven't set up yet)
    // would wake us from hlt, and we'd need to halt again.
    loop {
        arch::X86_64::halt_until_interrupt();
    }
}

// Fill the framebuffer with a deep blue color.
fn paint_framebuffer(fb: &limine::framebuffer::Framebuffer) {
    let width = fb.width() as usize;
    let height = fb.height() as usize;
    let pitch = fb.pitch() as usize; // bytes per row (may include padding)
    let bpp = fb.bpp() as usize;
    let bytes_per_pixel = bpp / 8;

    let fb_ptr = fb.addr() as *mut u8;

    // Dark blue: R=0x1A, G=0x1A, B=0x2E (assumes 32-bit BGRA or BGRX layout)
    for y in 0..height {
        for x in 0..width {
            let offset = y * pitch + x * bytes_per_pixel;
            unsafe {
                // BGRA byte order (standard for most framebuffers)
                *fb_ptr.add(offset) = 0x2E; // Blue
                *fb_ptr.add(offset + 1) = 0x1A; // Green
                *fb_ptr.add(offset + 2) = 0x1A; // Red
            }
        }
    }
}

/*
 * Panic handler — required by #![no_std].
 * When the kernel panics, print the message and halt forever.
 */

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!();
    println!("!!! KERNEL PANIC !!!");
    println!("{}", info);
    loop {
        arch::X86_64::halt_until_interrupt();
    }
}
