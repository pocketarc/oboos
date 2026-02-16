//! OBOOS — Off By One Operating System.
//!
//! A from-scratch, Rust-based OS built for learning. Boots via the Limine
//! protocol, targets x86_64 first, with a HAL designed for adding aarch64 later.

#![no_std] // No standard library — we ARE the operating system.
#![no_main] // No C runtime, no normal main(). We define our own entry point.
#![feature(abi_x86_interrupt)] // Nightly: lets us write interrupt handlers with proper calling convention.

extern crate alloc;

mod arch;
mod framebuffer;
mod heap;
mod memory;
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
    // Initialize the platform (serial port, GDT, IDT, PIC, memory, PIT).
    let _platform = arch::Arch::init();

    // Initialize the kernel heap — unlocks Vec, Box, String, etc.
    heap::init();

    // Smoke-test the heap allocator.
    {
        use alloc::boxed::Box;
        use alloc::string::String;
        use alloc::vec::Vec;

        let mut v = Vec::new();
        v.push(1u32);
        v.push(2);
        v.push(3);
        assert_eq!(v.iter().sum::<u32>(), 6);

        let b = Box::new(42u64);
        assert_eq!(*b, 42);

        let mut s = String::from("OBOOS");
        s.push_str(" heap works!");
        assert_eq!(s, "OBOOS heap works!");

        println!("[ok] Heap allocator verified (Vec, Box, String)");
    }

    // Smoke-test the frame allocator: allocate 3 frames, free the middle one,
    // re-allocate (should recycle the freed frame), and report free count.
    {
        let f1 = memory::alloc_frame().expect("alloc frame 1");
        let f2 = memory::alloc_frame().expect("alloc frame 2");
        let f3 = memory::alloc_frame().expect("alloc frame 3");
        println!("[test] Allocated frame: {:#018X}", f1);
        println!("[test] Allocated frame: {:#018X}", f2);
        println!("[test] Allocated frame: {:#018X}", f3);

        memory::free_frame(f2);
        println!("[test] Freed frame:     {:#018X}", f2);

        let f4 = memory::alloc_frame().expect("alloc frame 4");
        println!("[test] Re-allocated:    {:#018X} (recycled: {})", f4, f4 == f2);

        // Clean up — free the test frames so they don't leak.
        memory::free_frame(f1);
        memory::free_frame(f3);
        memory::free_frame(f4);
        println!("[test] Free frames remaining: {}", memory::free_frame_count());
        println!("[ok] Frame allocator verified");
    }

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

            draw_splash(ptr, w, h, pitch, framebuffer::Color::DARK_BLUE, None);
            println!("[ok] Press Enter to randomize colors!");
            println!("[ok] Press F to trigger a divide-by-zero fault.");
            println!("[ok] Press T to show uptime.");

            // Interrupts were disabled during init. Now that all handlers are
            // registered and the PIT is ticking, it's safe to let them through.
            arch::Arch::enable_interrupts();
            println!("[ok] Hardware interrupts enabled");

            // Poll keyboard in a loop. `hlt` sleeps the CPU until the next
            // interrupt (PIT fires every ~1ms), so we wake up, check for a
            // keypress, and sleep again — much better than busy-spinning.
            loop {
                if let Some(key) = arch::KeyboardDriver::poll() {
                    match key {
                        Key::Enter => {
                            let rand = arch::Arch::entropy();
                            let bg = framebuffer::Color((rand as u32) & 0x00FFFFFF);
                            let ms = arch::x86_64::pit::elapsed_ms();
                            draw_splash(ptr, w, h, pitch, bg, Some(ms));
                            println!("[ok] Color: #{:06X}", bg.0);
                        }
                        Key::F => {
                            println!("[!!] Triggering test fault...");
                            println!("[!!] IDT installed — expect a panic message.");
                            arch::Arch::trigger_test_fault();
                        }
                        Key::T => {
                            let ms = arch::x86_64::pit::elapsed_ms();
                            println!("[time] {}.{:03} seconds", ms / 1000, ms % 1000);
                        }
                        _ => {}
                    }
                }
                arch::Arch::halt_until_interrupt();
            }
        }
    }

    println!("No framebuffer available. Halting.");
    loop {
        arch::Arch::halt_until_interrupt();
    }
}

// Paint the splash screen: solid background with centered title text.
// If `uptime_ms` is provided, draw the uptime below the hint line.
fn draw_splash(ptr: *mut u8, w: usize, h: usize, pitch: usize, bg: framebuffer::Color, uptime_ms: Option<u64>) {
    framebuffer::clear(ptr, w, h, pitch, bg);

    let title = "OBOOS v0.0";
    let subtitle = "Off By One Operating System";
    let hint = "Enter = colors / F = fault / T = uptime";
    let title_x = (w - title.len() * 8) / 2;
    let subtitle_x = (w - subtitle.len() * 8) / 2;
    let hint_x = (w - hint.len() * 8) / 2;
    let center_y = h / 2 - 16;

    framebuffer::draw_str(ptr, pitch, title_x, center_y, title, framebuffer::Color::WHITE);
    framebuffer::draw_str(ptr, pitch, subtitle_x, center_y + 16, subtitle, framebuffer::Color::LIGHT_GRAY);
    framebuffer::draw_str(ptr, pitch, hint_x, center_y + 40, hint, framebuffer::Color::LIGHT_GRAY);

    if let Some(ms) = uptime_ms {
        let mut buf = [0u8; 32];
        let uptime_str = fmt_uptime(ms, &mut buf);
        let uptime_x = (w - uptime_str.len() * 8) / 2;
        framebuffer::draw_str(ptr, pitch, uptime_x, center_y + 64, uptime_str, framebuffer::Color::WHITE);
    }
}

/// Format milliseconds as "Uptime: X.XXX s" into a stack buffer.
/// Returns the written slice as a `&str`. We do this by hand because
/// `format!` requires an allocator we don't have yet.
fn fmt_uptime(ms: u64, buf: &mut [u8; 32]) -> &str {
    let prefix = b"Uptime: ";
    buf[..prefix.len()].copy_from_slice(prefix);
    let mut pos = prefix.len();

    // Write the whole-seconds part (variable width).
    let secs = ms / 1000;
    let frac = (ms % 1000) as u32;

    // Convert seconds to decimal digits.
    if secs == 0 {
        buf[pos] = b'0';
        pos += 1;
    } else {
        let mut digits = [0u8; 20];
        let mut n = secs;
        let mut len = 0;
        while n > 0 {
            digits[len] = b'0' + (n % 10) as u8;
            n /= 10;
            len += 1;
        }
        for i in (0..len).rev() {
            buf[pos] = digits[i];
            pos += 1;
        }
    }

    // Write ".XXX s"
    buf[pos] = b'.';
    pos += 1;
    buf[pos] = b'0' + (frac / 100) as u8;
    pos += 1;
    buf[pos] = b'0' + ((frac / 10) % 10) as u8;
    pos += 1;
    buf[pos] = b'0' + (frac % 10) as u8;
    pos += 1;
    buf[pos] = b' ';
    pos += 1;
    buf[pos] = b's';
    pos += 1;

    core::str::from_utf8(&buf[..pos]).unwrap()
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
