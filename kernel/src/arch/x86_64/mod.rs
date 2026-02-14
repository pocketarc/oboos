//! x86_64 platform implementation.

pub mod keyboard;
pub mod port;
pub mod serial;

use crate::platform::{Platform, SerialConsole};

/// x86_64 platform — implements [`Platform`] for 64-bit Intel/AMD.
pub struct X86_64;

impl Platform for X86_64 {
    fn init() -> Self {
        serial::Serial::init();
        X86_64
    }

    /// `sti` — Set Interrupt Flag. The CPU will start responding to
    /// hardware interrupts after the *next* instruction completes.
    fn enable_interrupts() {
        unsafe { core::arch::asm!("sti") };
    }

    /// `cli` — Clear Interrupt Flag. Masks all maskable interrupts.
    /// Used to protect critical sections where we can't be preempted.
    fn disable_interrupts() {
        unsafe { core::arch::asm!("cli") };
    }

    /// `hlt` — Halt the CPU until the next interrupt fires.
    fn halt_until_interrupt() {
        unsafe { core::arch::asm!("hlt") };
    }
}

/// Read the CPU's Time Stamp Counter — a 64-bit counter that increments
/// every clock cycle. Useful as a source of entropy since we can't predict
/// exactly when a human presses a key.
pub fn read_tsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        core::arch::asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack));
    }
    ((hi as u64) << 32) | (lo as u64)
}
