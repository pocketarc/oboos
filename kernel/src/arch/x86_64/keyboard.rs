//! PS/2 keyboard driver (polling mode).
//!
//! Translates raw PS/2 scan code set 1 into architecture-independent
//! [`Key`] values. The scan code constants are private; they're an
//! implementation detail of this particular keyboard controller. The
//! rest of the kernel works with [`Key`] and never sees a raw scancode.

use crate::platform::{Key, Keyboard};
use super::port::inb;

// PS/2 keyboard controller ports. These have been the same since the IBM PC AT (1984).
const DATA_PORT: u16 = 0x60;
const STATUS_PORT: u16 = 0x64;

// Scan code set 1 — the default that the keyboard controller gives us.
// These are "make codes" (key press). Release codes are make + 0x80.
const SC_ENTER: u8 = 0x1C;
const SC_F: u8 = 0x21;
const SC_T: u8 = 0x14;

/// PS/2 keyboard driver — translates hardware scancodes into [`Key`] values.
pub struct Ps2Keyboard;

impl Keyboard for Ps2Keyboard {
    /// Poll the PS/2 controller and translate the raw scancode to a [`Key`].
    ///
    /// Returns `None` if no key event is pending, or if the event was a
    /// key release (break code). Only key presses are reported.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::platform::{Key, Keyboard};
    ///
    /// if let Some(key) = Ps2Keyboard::poll() {
    ///     match key {
    ///         Key::Enter => println!("Enter pressed!"),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    fn poll() -> Option<Key> {
        let scancode = raw_poll()?;

        // Bit 7 set means this is a break (release) code. We only care
        // about make (press) codes — the PS/2 convention is that the
        // release code = make code | 0x80.
        if scancode & 0x80 != 0 {
            return None;
        }

        let key = match scancode {
            SC_ENTER => Key::Enter,
            SC_F => Key::F,
            SC_T => Key::T,
            other => Key::Other(other),
        };
        Some(key)
    }
}

/// Read a raw scancode from the PS/2 controller, if one is pending.
fn raw_poll() -> Option<u8> {
    unsafe {
        // Bit 0 of the status register = output buffer full (data ready).
        if inb(STATUS_PORT) & 0x01 != 0 {
            Some(inb(DATA_PORT))
        } else {
            None
        }
    }
}
