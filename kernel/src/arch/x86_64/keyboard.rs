//! PS/2 keyboard driver (IRQ-driven with async support).
//!
//! Translates raw PS/2 scan code set 1 into architecture-independent
//! [`Key`] values. The keyboard IRQ (IRQ 1) pushes scancodes into a
//! ring buffer; the async [`next_key()`] future drains it.
//!
//! ## Console mode
//!
//! When console mode is active (during Ring 3 execution), keyboard
//! scancodes are routed through [`next_console_byte()`] instead of
//! [`next_key()`]. The IRQ handler wakes the [`CONSOLE_WAKER`] so the
//! async [`keyboard_input_driver`](super::syscall::keyboard_input_driver)
//! can translate scancodes to ASCII and push them into the console store.
//!
//! ## Interrupt safety
//!
//! A single [`spin::Mutex`] protects both the buffer and the waker.
//! Code that locks `STATE` from non-interrupt context must disable
//! interrupts first (`cli`) to prevent deadlock — the IRQ handler
//! also locks `STATE`, and interrupt gates already run with IF=0.

extern crate alloc;

use alloc::collections::VecDeque;
use core::future::Future;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::Waker;

use crate::platform::{Key, Platform};
use super::port::inb;

// PS/2 keyboard controller ports. These have been the same since the IBM PC AT (1984).
const DATA_PORT: u16 = 0x60;

// Scan code set 1 — the default that the keyboard controller gives us.
// These are "make codes" (key press). Release codes are make + 0x80.
const SC_ENTER: u8 = 0x1C;
const SC_F: u8 = 0x21;
const SC_H: u8 = 0x23;
const SC_T: u8 = 0x14;

/// Shared keyboard state: scancode buffer + async waker.
///
/// A single struct behind one lock avoids lock-ordering issues that
/// would arise from separate buffer/waker locks.
struct KeyboardState {
    buffer: VecDeque<u8>,
    waker: Option<Waker>,
}

/// Global keyboard state. The IRQ handler and `next_key()` both access
/// this — always with IF=0 to prevent deadlock.
static STATE: spin::Mutex<KeyboardState> = spin::Mutex::new(KeyboardState {
    buffer: VecDeque::new(),
    waker: None,
});

/// Pre-allocate the scancode buffer so the IRQ handler never hits the
/// heap allocator.
///
/// Must be called after [`crate::heap::init()`] and before unmasking
/// IRQ 1. 32 slots is plenty — a fast typist can't fill that between
/// polls.
pub fn init() {
    // cli/sti around the lock to match the interrupt-safety pattern,
    // even though interrupts are typically still disabled at this point
    // in the boot sequence.
    crate::arch::Arch::disable_interrupts();
    STATE.lock().buffer.reserve(32);
    crate::arch::Arch::enable_interrupts();
    crate::println!("[ok] Keyboard buffer pre-allocated (32 slots)");
}

/// When true, keyboard scancodes wake [`CONSOLE_WAKER`] instead of
/// `STATE.waker`. This routes keyboard input through the async
/// [`keyboard_input_driver`](super::syscall::keyboard_input_driver)
/// during Ring 3 execution.
static CONSOLE_MODE: AtomicBool = AtomicBool::new(false);

/// Enable or disable console mode.
///
/// When enabled, scancodes wake the console waker instead of the
/// normal async keyboard future. Call with `true` before entering
/// Ring 3 and `false` after returning.
pub fn set_console_mode(enabled: bool) {
    CONSOLE_MODE.store(enabled, Ordering::SeqCst);
}

/// Waker for the console input driver. Set by [`next_console_byte()`],
/// woken by the IRQ handler when console mode is active.
static CONSOLE_WAKER: spin::Mutex<Option<Waker>> = spin::Mutex::new(None);

/// IRQ 1 handler — reads a scancode from port 0x60 and buffers it.
///
/// Runs with IF=0 (interrupt gate), so locking `STATE` is safe.
/// In normal mode, wakes the async [`next_key()`] future. In console
/// mode, wakes the [`CONSOLE_WAKER`] for the keyboard input driver.
pub fn on_key() {
    let scancode = unsafe { inb(DATA_PORT) };
    let mut state = STATE.lock();
    state.buffer.push_back(scancode);
    if CONSOLE_MODE.load(Ordering::SeqCst) {
        if let Some(waker) = CONSOLE_WAKER.lock().take() {
            waker.wake();
        }
    } else if let Some(waker) = state.waker.take() {
        waker.wake();
    }
}

/// Async future that returns the next key press.
///
/// Drains break codes (bit 7 set) from the buffer and returns on the
/// first make code. If the buffer is empty, stores the waker and
/// returns `Pending` — the IRQ handler will wake us when a scancode
/// arrives.
///
/// Uses [`core::future::poll_fn`] to avoid a separate `Future` struct.
/// The `cli`/`sti` around the lock prevents deadlock with the IRQ
/// handler.
pub fn next_key() -> impl Future<Output = Key> + Send {
    core::future::poll_fn(|cx| {
        crate::arch::Arch::disable_interrupts();
        let mut state = STATE.lock();

        // Drain break codes, return on first make code.
        while let Some(scancode) = state.buffer.pop_front() {
            if scancode & 0x80 != 0 {
                continue; // release code — skip
            }
            drop(state);
            crate::arch::Arch::enable_interrupts();
            return core::task::Poll::Ready(translate_scancode(scancode));
        }

        // Buffer empty — store waker and return Pending.
        state.waker = Some(cx.waker().clone());
        drop(state);
        crate::arch::Arch::enable_interrupts();
        core::task::Poll::Pending
    })
}

/// Async future that returns the next ASCII byte from the keyboard.
///
/// Used by the keyboard input driver during console mode. Drains
/// scancodes from the buffer, skips break codes, translates via
/// [`scancode_to_ascii`], and returns the first ASCII byte. If the
/// buffer is empty, stores the waker in [`CONSOLE_WAKER`] and returns
/// Pending.
pub fn next_console_byte() -> impl Future<Output = u8> + Send {
    core::future::poll_fn(|cx| {
        crate::arch::Arch::disable_interrupts();
        let mut state = STATE.lock();

        // Drain the buffer looking for a translatable make code.
        while let Some(scancode) = state.buffer.pop_front() {
            if scancode & 0x80 != 0 {
                continue; // release code — skip
            }
            if let Some(ascii) = scancode_to_ascii(scancode) {
                drop(state);
                crate::arch::Arch::enable_interrupts();
                return core::task::Poll::Ready(ascii);
            }
            // Untranslatable make code (modifier, function key) — skip.
        }

        // Buffer empty — store waker and return Pending.
        drop(state);
        *CONSOLE_WAKER.lock() = Some(cx.waker().clone());
        crate::arch::Arch::enable_interrupts();
        core::task::Poll::Pending
    })
}

/// Translate a scan code set 1 make code to a [`Key`].
fn translate_scancode(scancode: u8) -> Key {
    match scancode {
        SC_ENTER => Key::Enter,
        SC_F => Key::F,
        SC_H => Key::H,
        SC_T => Key::T,
        other => Key::Other(other),
    }
}

/// Test helper: push a scancode into the buffer and wake any waiting
/// future, simulating what the IRQ handler does without reading the
/// hardware port.
///
/// Must be called with interrupts disabled (IF=0) — same requirement
/// as [`crate::executor::wake()`].
#[cfg(feature = "smoke-test")]
pub fn push_scancode(scancode: u8) {
    let mut state = STATE.lock();
    state.buffer.push_back(scancode);
    // Wake whichever waker is active — console mode or normal mode.
    if CONSOLE_MODE.load(Ordering::SeqCst) {
        if let Some(waker) = CONSOLE_WAKER.lock().take() {
            waker.wake();
        }
    } else if let Some(waker) = state.waker.take() {
        waker.wake();
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Scancode → ASCII translation
// ————————————————————————————————————————————————————————————————————————————

/// Scan code set 1 → ASCII lookup table (128 entries, make codes only).
///
/// Covers printable ASCII: lowercase letters, digits, basic punctuation,
/// plus Enter (→ `\n`), Backspace (→ 0x08), and Space (→ 0x20).
/// Unmapped scancodes have value 0.
///
/// This table represents the US QWERTY layout without shift — good enough
/// for a first interactive console. Shift/caps support comes later.
static SCANCODE_TO_ASCII: [u8; 128] = {
    let mut table = [0u8; 128];

    // Row 1: digits and symbols
    table[0x02] = b'1';
    table[0x03] = b'2';
    table[0x04] = b'3';
    table[0x05] = b'4';
    table[0x06] = b'5';
    table[0x07] = b'6';
    table[0x08] = b'7';
    table[0x09] = b'8';
    table[0x0A] = b'9';
    table[0x0B] = b'0';
    table[0x0C] = b'-';
    table[0x0D] = b'=';
    table[0x0E] = 0x08; // Backspace

    // Row 2: QWERTYUIOP
    table[0x10] = b'q';
    table[0x11] = b'w';
    table[0x12] = b'e';
    table[0x13] = b'r';
    table[0x14] = b't';
    table[0x15] = b'y';
    table[0x16] = b'u';
    table[0x17] = b'i';
    table[0x18] = b'o';
    table[0x19] = b'p';
    table[0x1A] = b'[';
    table[0x1B] = b']';
    table[0x1C] = b'\n'; // Enter

    // Row 3: ASDFGHJKL
    table[0x1E] = b'a';
    table[0x1F] = b's';
    table[0x20] = b'd';
    table[0x21] = b'f';
    table[0x22] = b'g';
    table[0x23] = b'h';
    table[0x24] = b'j';
    table[0x25] = b'k';
    table[0x26] = b'l';
    table[0x27] = b';';
    table[0x28] = b'\'';
    table[0x29] = b'`';

    // Row 4: ZXCVBNM
    table[0x2B] = b'\\';
    table[0x2C] = b'z';
    table[0x2D] = b'x';
    table[0x2E] = b'c';
    table[0x2F] = b'v';
    table[0x30] = b'b';
    table[0x31] = b'n';
    table[0x32] = b'm';
    table[0x33] = b',';
    table[0x34] = b'.';
    table[0x35] = b'/';

    // Space bar
    table[0x39] = b' ';

    table
};

/// Convert a scan code set 1 make code to an ASCII byte.
///
/// Returns `Some(ascii)` for printable characters, Enter, Backspace,
/// and Space. Returns `None` for unmapped scancodes (function keys,
/// modifiers, etc.).
pub fn scancode_to_ascii(scancode: u8) -> Option<u8> {
    let idx = scancode as usize;
    if idx < SCANCODE_TO_ASCII.len() {
        let ch = SCANCODE_TO_ASCII[idx];
        if ch != 0 { Some(ch) } else { None }
    } else {
        None
    }
}
