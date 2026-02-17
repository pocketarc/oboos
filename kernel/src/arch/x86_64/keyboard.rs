//! PS/2 keyboard driver (IRQ-driven with async support).
//!
//! Translates raw PS/2 scan code set 1 into architecture-independent
//! [`Key`] values. The keyboard IRQ (IRQ 1) pushes scancodes into a
//! ring buffer; the async [`next_key()`] future drains it. The sync
//! [`Keyboard::poll()`] also reads from the buffer, so both paths are
//! consistent after IRQ registration.
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
use core::task::Waker;

use crate::platform::{Key, Keyboard, Platform};
use super::port::inb;

// PS/2 keyboard controller ports. These have been the same since the IBM PC AT (1984).
const DATA_PORT: u16 = 0x60;

// Scan code set 1 — the default that the keyboard controller gives us.
// These are "make codes" (key press). Release codes are make + 0x80.
const SC_ENTER: u8 = 0x1C;
const SC_F: u8 = 0x21;
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

/// IRQ 1 handler — reads a scancode from port 0x60 and buffers it.
///
/// Runs with IF=0 (interrupt gate), so locking `STATE` is safe.
/// If an async future is waiting via [`next_key()`], wakes it so
/// the executor will re-poll on the next [`crate::executor::poll_once()`].
pub fn on_key() {
    let scancode = unsafe { inb(DATA_PORT) };
    let mut state = STATE.lock();
    state.buffer.push_back(scancode);
    if let Some(waker) = state.waker.take() {
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

/// Translate a scan code set 1 make code to a [`Key`].
fn translate_scancode(scancode: u8) -> Key {
    match scancode {
        SC_ENTER => Key::Enter,
        SC_F => Key::F,
        SC_T => Key::T,
        other => Key::Other(other),
    }
}

/// PS/2 keyboard driver — translates hardware scancodes into [`Key`] values.
pub struct Ps2Keyboard;

impl Keyboard for Ps2Keyboard {
    /// Read the next key press from the scancode buffer.
    ///
    /// After IRQ registration, scancodes arrive via the interrupt
    /// handler and are buffered. This method drains break codes and
    /// returns the first make code, or `None` if the buffer is empty.
    ///
    /// Disables interrupts while holding the lock to prevent deadlock
    /// with the IRQ handler.
    fn poll() -> Option<Key> {
        crate::arch::Arch::disable_interrupts();
        let mut state = STATE.lock();

        while let Some(scancode) = state.buffer.pop_front() {
            if scancode & 0x80 != 0 {
                continue;
            }
            drop(state);
            crate::arch::Arch::enable_interrupts();
            return Some(translate_scancode(scancode));
        }

        drop(state);
        crate::arch::Arch::enable_interrupts();
        None
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
    if let Some(waker) = state.waker.take() {
        waker.wake();
    }
}
