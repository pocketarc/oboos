//! OBOOS — Off By One Operating System.
//!
//! A from-scratch, Rust-based OS built for learning. Boots via the Limine
//! protocol, targets x86_64 first, with a HAL designed for adding aarch64 later.

#![no_std] // No standard library — we ARE the operating system.
#![no_main] // No C runtime, no normal main(). We define our own entry point.
#![feature(abi_x86_interrupt)] // Nightly: lets us write interrupt handlers with proper calling convention.

extern crate alloc;

mod arch;
mod executor;
mod framebuffer;
mod heap;
mod memory;
mod platform;
mod scheduler;
mod store;
mod task;
mod timer;
#[cfg(feature = "smoke-test")]
mod tests;

use framebuffer::FramebufferInfo;
use platform::{Key, Platform};

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

    // Pre-allocate the keyboard scancode buffer and sleep registry while
    // the heap is fresh. Order matters: allocate before unmasking IRQs
    // so handlers never need to grow their buffers.
    arch::keyboard::init();
    timer::init();
    arch::interrupts::set_irq_handler(1, arch::keyboard::on_key);

    // Initialize the store registry — reactive state trees for IPC.
    store::init();

    // Initialize the cooperative scheduler — wraps kmain as the bootstrap task.
    scheduler::init();

    // Initialize the async executor — polls futures from kmain's main loop.
    executor::init();

    // Run smoke tests when built with `--features smoke-test` (via `make test`).
    #[cfg(feature = "smoke-test")]
    tests::run_all();

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

            // Store framebuffer info in the global so the async keyboard
            // task can draw without carrying raw pointers.
            framebuffer::FRAMEBUFFER_INFO.call_once(|| framebuffer::FramebufferInfo {
                ptr, width: w, height: h, pitch,
            });

            let fbi = framebuffer::FRAMEBUFFER_INFO.get().unwrap();
            draw_splash(fbi, framebuffer::Color::DARK_BLUE, None);
            println!("[ok] Press Enter to randomize colors!");
            println!("[ok] Press F to trigger a divide-by-zero fault.");
            println!("[ok] Press T to show uptime.");

            // Spawn async tasks, enable interrupts, and hand control to
            // the executor. The keyboard task is IRQ-driven via scancodes;
            // the blink task is PIT-driven via timer::sleep().
            executor::spawn(keyboard_task());
            executor::spawn(blink_task());
            executor::spawn(boot_chime());
            arch::Arch::enable_interrupts();
            println!("[ok] Hardware interrupts enabled");
            executor::run(); // never returns
        }
    }

    println!("No framebuffer available. Halting.");
    loop {
        arch::Arch::halt_until_interrupt();
    }
}

/// Async keyboard task — loops forever reading keys via the IRQ-driven
/// [`arch::keyboard::next_key()`] future and handling them the
/// same way the old polling loop did.
async fn keyboard_task() {
    let fb = framebuffer::FRAMEBUFFER_INFO
        .get()
        .expect("framebuffer not initialized");

    loop {
        let key = arch::keyboard::next_key().await;
        match key {
            Key::Enter => {
                let rand = arch::Arch::entropy();
                let bg = framebuffer::Color((rand as u32) & 0x00FFFFFF);
                let ms = arch::Arch::elapsed_ms();
                draw_splash(fb, bg, Some(ms));
                println!("[ok] Color: #{:06X}", bg.0);
            }
            Key::F => {
                println!("[!!] Triggering test fault...");
                println!("[!!] IDT installed — expect a panic message.");
                arch::Arch::trigger_test_fault();
            }
            Key::T => {
                let ms = arch::Arch::elapsed_ms();
                println!("[time] {}.{:03} seconds", ms / 1000, ms % 1000);
            }
            _ => {}
        }
    }
}

/// Async boot chime — plays the opening riff of Doom's E1M1 ("At Doom's
/// Gate") as a nod to the kernel-mode Doom milestone on the roadmap.
///
/// The riff is a chromatic descent from E: a continuous E4 drone with
/// melody notes (E5→D4→C4→Bb3→B3→C4) punching through — no silence
/// between notes, just like the original palm-muted guitar line.
async fn boot_chime() {
    // E1M1 "At Doom's Gate" opening riff — two phrases.
    // No gaps between notes: repeated pedal tones merge into a drone
    // and the chromatic descent notes are heard as pitch changes.
    let t = arch::speaker::play_tone;

    // Phrase 1 (E pedal): E E E' E E D  E E C  E E Bb E E B  C
    t(330, 120).await;  // E4
    t(330, 120).await;  // E4
    t(659, 120).await;  // E5 (octave)
    t(330, 120).await;  // E4
    t(330, 120).await;  // E4
    t(294, 120).await;  // D4
    t(330, 120).await;  // E4
    t(330, 120).await;  // E4
    t(262, 120).await;  // C4
    t(330, 120).await;  // E4
    t(330, 120).await;  // E4
    t(233, 120).await;  // Bb3
    t(330, 120).await;  // E4
    t(330, 120).await;  // E4
    t(247, 120).await;  // B3
    t(262, 120).await;  // C4

    // Phrase 2 (A pedal): same pattern shifted up a fourth.
    t(440, 120).await;  // A4
    t(440, 120).await;  // A4
    t(880, 120).await;  // A5 (octave)
    t(440, 120).await;  // A4
    t(440, 120).await;  // A4
    t(392, 120).await;  // G4
    t(440, 120).await;  // A4
    t(440, 120).await;  // A4
    t(349, 120).await;  // F4
    t(440, 120).await;  // A4
    t(440, 120).await;  // A4
    t(311, 120).await;  // Eb4
    t(440, 120).await;  // A4
    t(440, 120).await;  // A4
    t(330, 120).await;  // E4
    t(349, 120).await;  // F4
}

/// Async blink task — toggles a block cursor on the splash screen at 2 Hz.
///
/// This runs concurrently with [`keyboard_task()`], proving the executor
/// handles two hardware-driven async tasks: one woken by keyboard IRQ,
/// one woken by PIT ticks via [`timer::sleep()`].
async fn blink_task() {
    let fb = framebuffer::FRAMEBUFFER_INFO
        .get()
        .expect("framebuffer not initialized");

    let cursor_x = fb.width / 2;
    let cursor_y = fb.height / 2 + 56;

    loop {
        // Draw the cursor (█ full block).
        framebuffer::draw_str(fb.ptr, fb.pitch, cursor_x, cursor_y, "\u{2588}", framebuffer::Color::WHITE);
        timer::sleep(500).await;

        // Erase the cursor by drawing the same glyph in the background color.
        framebuffer::draw_str(fb.ptr, fb.pitch, cursor_x, cursor_y, "\u{2588}", framebuffer::Color::DARK_BLUE);
        timer::sleep(500).await;
    }
}

// Paint the splash screen: solid background with centered title text.
// If `uptime_ms` is provided, draw the uptime below the hint line.
fn draw_splash(fb: &FramebufferInfo, bg: framebuffer::Color, uptime_ms: Option<u64>) {
    framebuffer::clear(fb.ptr, fb.width, fb.height, fb.pitch, bg);

    let title = "OBOOS v0.0";
    let subtitle = "Off By One Operating System";
    let hint = "Enter = colors / F = fault / T = uptime";
    let title_x = (fb.width - title.len() * 8) / 2;
    let subtitle_x = (fb.width - subtitle.len() * 8) / 2;
    let hint_x = (fb.width - hint.len() * 8) / 2;
    let center_y = fb.height / 2 - 16;

    framebuffer::draw_str(fb.ptr, fb.pitch, title_x, center_y, title, framebuffer::Color::WHITE);
    framebuffer::draw_str(fb.ptr, fb.pitch, subtitle_x, center_y + 16, subtitle, framebuffer::Color::LIGHT_GRAY);
    framebuffer::draw_str(fb.ptr, fb.pitch, hint_x, center_y + 40, hint, framebuffer::Color::LIGHT_GRAY);

    if let Some(ms) = uptime_ms {
        let mut buf = [0u8; 32];
        let uptime_str = fmt_uptime(ms, &mut buf);
        let uptime_x = (fb.width - uptime_str.len() * 8) / 2;
        framebuffer::draw_str(fb.ptr, fb.pitch, uptime_x, center_y + 64, uptime_str, framebuffer::Color::WHITE);
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
