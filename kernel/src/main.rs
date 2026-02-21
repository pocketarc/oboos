//! OBOOS — Off By One Operating System.
//!
//! A from-scratch, Rust-based OS built for learning. Boots via the Limine
//! protocol, targets x86_64 first, with a HAL designed for adding aarch64 later.

#![no_std] // No standard library — we ARE the operating system.
#![no_main] // No C runtime, no normal main(). We define our own entry point.
#![feature(abi_x86_interrupt)] // Nightly: lets us write interrupt handlers with proper calling convention.

extern crate alloc;

mod arch;
mod elf;
mod executor;
mod framebuffer;
mod heap;
mod memory;
mod process;
mod platform;
mod scheduler;
mod store;
pub mod store_handle;
mod task;
mod timer;
mod userspace;
#[cfg(feature = "smoke-test")]
mod tests;

use framebuffer::FramebufferInfo;
use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};
use platform::{Key, Platform};
use store::StoreId;

// Limine boot protocol requests.
//
// These are static structs with magic numbers baked in. The bootloader scans
// our binary for these markers, processes the requests, and fills in response
// pointers before jumping to our entry point. It's a clever handshake that
// avoids needing any runtime negotiation.

 use limine::request::{FramebufferRequest, MpRequest, RequestsEndMarker, RequestsStartMarker};
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
#[unsafe(link_section = ".requests")]
static MP_REQUEST: MpRequest = MpRequest::new();

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
        let _guard = crate::arch::serial::SERIAL_LOCK.lock();
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

// ---------------------------------------------------------------------------
// Store schemas
// ---------------------------------------------------------------------------

/// System monitor schema — uptime and free memory.
struct SystemMonitorSchema;

impl StoreSchema for SystemMonitorSchema {
    fn name() -> &'static str { "SystemMonitor" }
    fn fields() -> &'static [FieldDef] {
        &[
            FieldDef { name: "uptime_ms", kind: FieldKind::U64 },
            FieldDef { name: "free_kb", kind: FieldKind::U64 },
        ]
    }
}

/// App schema — holds the current background color, driven by keyboard input.
struct AppSchema;

impl StoreSchema for AppSchema {
    fn name() -> &'static str { "App" }
    fn fields() -> &'static [FieldDef] {
        &[
            FieldDef { name: "bg_color", kind: FieldKind::U32 },
        ]
    }
}

// ---------------------------------------------------------------------------
// Status bar — footer-style strip at the bottom of the screen.
// 8px top padding + 16px text + 8px bottom padding = 32px total.
// ---------------------------------------------------------------------------

const STATUS_BAR_HEIGHT: usize = 32;

/// Kernel entry point — where the bootloader jumps to.
#[unsafe(no_mangle)]
extern "C" fn kmain() -> ! {
    // Verify the bootloader speaks our protocol revision before touching
    // any Limine responses. If this fails, response pointers may be null
    // or malformed — better to panic here than crash deep in init.
    assert!(BASE_REVISION.is_supported());

    // Initialize the platform (serial port, GDT, IDT, PIC, memory, PIT).
    let _platform = arch::Arch::init();

    // Save the boot CR3 as the kernel PML4 — used as the template when
    // creating per-process page tables.
    arch::paging::init();

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

    // Initialize the process table — tracks userspace processes and their
    // lifecycle stores. Depends on the store registry for creating process stores.
    process::init();

    // Initialize the cooperative scheduler — wraps kmain as the bootstrap task.
    scheduler::init();

    // Initialize the async executor — polls futures from kmain's main loop.
    executor::init();

    // Bring up additional CPU cores. APs initialize their own GDT/TSS,
    // load the shared IDT, and park in hlt. Later phases give them work.
    if let Some(mp_response) = MP_REQUEST.get_response() {
        arch::smp::init(mp_response);
    }

    // Run smoke tests when built with `--features smoke-test` (via `make test`).
    #[cfg(feature = "smoke-test")]
    tests::run_all();

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
            draw_splash(fbi, framebuffer::Color::DARK_BLUE);
            println!("[ok] Press Enter to randomize colors!");
            println!("[ok] Press F to trigger a divide-by-zero fault.");
            println!("[ok] Press H to launch hello (interactive console).");
            println!("[ok] Press T to show uptime.");

            // Create the system monitor store — updated every second, watched
            // by the renderer to draw the status bar.
            let sysmon_id = store::create::<SystemMonitorSchema>(&[
                ("uptime_ms", Value::U64(0)),
                ("free_kb", Value::U64(0)),
            ]).expect("create sysmon store");

            // Create the app store — keyboard writes bg_color, display task
            // watches it and repaints.
            let app_id = store::create::<AppSchema>(&[
                ("bg_color", Value::U32(framebuffer::Color::DARK_BLUE.0)),
            ]).expect("create app store");

            // Spawn async tasks, enable interrupts, and hand control to
            // the executor. The keyboard task is IRQ-driven via scancodes;
            // the blink task is PIT-driven via timer::sleep().
            // Run the updater on CPU 1 when available — exercises the full
            // cross-core IPC path: timer::sleep on CPU 1, store::set wakes
            // the renderer on CPU 0 via IPI, proving end-to-end SMP async.
            if arch::smp::cpu_count() > 1 {
                executor::spawn_on(1, sysmon_updater(sysmon_id));
            } else {
                executor::spawn(sysmon_updater(sysmon_id));
            }
            executor::spawn(sysmon_renderer(sysmon_id, app_id));
            executor::spawn(keyboard_task(app_id));
            executor::spawn(display_task(app_id));
            executor::spawn(blink_task());
            executor::spawn(boot_chime());
            executor::enable_work_stealing();
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

// ---------------------------------------------------------------------------
// System monitor tasks
// ---------------------------------------------------------------------------

/// Async task that updates the system monitor store every second.
///
/// Reads uptime and free memory from the hardware/allocator and writes
/// both fields atomically. The renderer watches both fields, so it
/// wakes once per update cycle.
async fn sysmon_updater(store: StoreId) {
    loop {
        timer::sleep(1000).await;

        let uptime_ms = arch::Arch::elapsed_ms();
        // Each frame is 4 KiB, so free_frames * 4 = free KiB.
        let free_kb = (memory::free_frame_count() as u64) * 4;

        let _ = store::set(store, &[
            ("uptime_ms", Value::U64(uptime_ms)),
            ("free_kb", Value::U64(free_kb)),
        ]);
    }
}

/// Async task that watches the system monitor store and draws the status bar.
///
/// Watches both `uptime_ms` and `free_kb` — wakes when either changes.
/// Since the updater writes both atomically, this fires once per cycle.
/// Reads `bg_color` from the app store to tint the bar background.
async fn sysmon_renderer(sysmon: StoreId, app: StoreId) {
    loop {
        if store::watch(sysmon, &["uptime_ms", "free_kb"]).await.is_err() {
            return;
        }

        let uptime_ms = match store::get(sysmon, "uptime_ms") {
            Ok(Value::U64(v)) => v,
            _ => return,
        };
        let free_kb = match store::get(sysmon, "free_kb") {
            Ok(Value::U64(v)) => v,
            _ => return,
        };
        let bg = match store::get(app, "bg_color") {
            Ok(Value::U32(v)) => framebuffer::Color(v),
            _ => framebuffer::Color::DARK_BLUE,
        };

        let fb = framebuffer::FRAMEBUFFER_INFO.get().unwrap();
        draw_status_bar(fb, bg, uptime_ms, free_kb);
        let secs = uptime_ms / 1000;
        let frac = uptime_ms % 1000;
        println!("[sysmon] render: {}.{:03}s, {} KiB free", secs, frac, free_kb);
    }
}

// ---------------------------------------------------------------------------
// Interactive demo tasks
// ---------------------------------------------------------------------------

/// Async keyboard task — reads keys and writes state changes to the app store.
///
/// Enter randomizes the background color by writing to the store (the
/// [`display_task`] watches the store and repaints). F and T are direct
/// actions that don't go through the store.
async fn keyboard_task(app_store: StoreId) {
    loop {
        let key = arch::keyboard::next_key().await;
        match key {
            Key::Enter => {
                let rand = arch::Arch::entropy();
                let bg = (rand as u32) & 0x00FFFFFF;
                let _ = store::set(app_store, &[("bg_color", Value::U32(bg))]);
                println!("[ok] Color: #{:06X}", bg);
            }
            Key::F => {
                println!("[!!] Triggering test fault...");
                println!("[!!] IDT installed — expect a panic message.");
                arch::Arch::trigger_test_fault();
            }
            Key::H => {
                println!("[ok] Launching hello program...");
                userspace::run_hello_interactive();
                println!("[ok] Back to splash screen.");
            }
            Key::T => {
                let ms = arch::Arch::elapsed_ms();
                println!("[time] {}.{:03} seconds", ms / 1000, ms % 1000);
            }
            _ => {}
        }
    }
}

/// Async display task — watches the app store's `bg_color` and repaints
/// the splash screen when it changes.
async fn display_task(app_store: StoreId) {
    loop {
        if store::watch(app_store, &["bg_color"]).await.is_err() {
            return;
        }

        let color_val = match store::get(app_store, "bg_color") {
            Ok(Value::U32(v)) => v,
            _ => return,
        };

        let fb = framebuffer::FRAMEBUFFER_INFO.get().unwrap();
        let bg = framebuffer::Color(color_val);
        draw_splash(fb, bg);
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

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

/// Paint the splash screen: solid background with centered title text.
///
/// Clears only the area above the status bar so the bottom strip is preserved.
fn draw_splash(fb: &FramebufferInfo, bg: framebuffer::Color) {
    let splash_height = fb.height.saturating_sub(STATUS_BAR_HEIGHT);
    framebuffer::clear_rect(fb.ptr, fb.pitch, 0, 0, fb.width, splash_height, bg);

    let title = "OBOOS v0.0";
    let subtitle = "Off By One Operating System";
    let hint = "Enter=colors / F=fault / H=hello / T=uptime";
    let title_x = (fb.width - title.len() * 8) / 2;
    let subtitle_x = (fb.width - subtitle.len() * 8) / 2;
    let hint_x = (fb.width - hint.len() * 8) / 2;
    let center_y = fb.height / 2 - 16;

    framebuffer::draw_str(fb.ptr, fb.pitch, title_x, center_y, title, framebuffer::Color::WHITE);
    framebuffer::draw_str(fb.ptr, fb.pitch, subtitle_x, center_y + 16, subtitle, framebuffer::Color::LIGHT_GRAY);
    framebuffer::draw_str(fb.ptr, fb.pitch, hint_x, center_y + 40, hint, framebuffer::Color::LIGHT_GRAY);
}

/// Draw the status bar at the bottom of the screen.
///
/// Layout: darkened background strip, left-aligned text showing uptime (with
/// millisecond precision) and free memory. The bar occupies the bottom [`STATUS_BAR_HEIGHT`]
/// pixels and uses a darkened version of the current background color
/// so it visually recedes behind the main content.
fn draw_status_bar(fb: &FramebufferInfo, bg: framebuffer::Color, uptime_ms: u64, free_kb: u64) {
    let bar_bg = bg.darken();
    let bar_y = fb.height - STATUS_BAR_HEIGHT;
    framebuffer::clear_rect(fb.ptr, fb.pitch, 0, bar_y, fb.width, STATUS_BAR_HEIGHT, bar_bg);

    // Format "Uptime: XX.XXXs | Free: XXX KiB" into a stack buffer.
    let mut buf = [0u8; 80];
    let mut pos = 0;

    let prefix = b"Uptime: ";
    buf[pos..pos + prefix.len()].copy_from_slice(prefix);
    pos += prefix.len();

    // Whole seconds part.
    pos += write_u64(&mut buf[pos..], uptime_ms / 1000);

    // Decimal point + zero-padded milliseconds (always 3 digits).
    buf[pos] = b'.';
    pos += 1;
    let frac = (uptime_ms % 1000) as u16;
    buf[pos]     = b'0' + (frac / 100) as u8;
    buf[pos + 1] = b'0' + ((frac / 10) % 10) as u8;
    buf[pos + 2] = b'0' + (frac % 10) as u8;
    pos += 3;

    let mid = b"s | Free: ";
    buf[pos..pos + mid.len()].copy_from_slice(mid);
    pos += mid.len();

    pos += write_u64(&mut buf[pos..], free_kb);

    let suffix = b" KiB";
    buf[pos..pos + suffix.len()].copy_from_slice(suffix);
    pos += suffix.len();

    let text = core::str::from_utf8(&buf[..pos]).unwrap();
    let text_y = bar_y + 12; // vertically centered: (32 - 8) / 2 = 12
    framebuffer::draw_str(fb.ptr, fb.pitch, 16, text_y, text, framebuffer::Color::LIGHT_GRAY);
}

/// Write a `u64` as decimal digits into a byte buffer.
///
/// Returns the number of bytes written. Used by [`draw_status_bar()`] and
/// [`fmt_uptime()`] to format numbers without `format!` (which needs an
/// allocator we want to avoid on the drawing path).
fn write_u64(buf: &mut [u8], val: u64) -> usize {
    if val == 0 {
        buf[0] = b'0';
        return 1;
    }

    let mut digits = [0u8; 20];
    let mut n = val;
    let mut len = 0;
    while n > 0 {
        digits[len] = b'0' + (n % 10) as u8;
        n /= 10;
        len += 1;
    }

    for i in (0..len).rev() {
        buf[len - 1 - i] = digits[i];
    }
    len
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
