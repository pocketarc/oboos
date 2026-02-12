pub mod keyboard;
pub mod port;
pub mod serial;

use crate::platform::{Platform, SerialConsole};

pub struct X86_64;

impl Platform for X86_64 {
    fn init() -> Self {
        serial::Serial::init();
        X86_64
    }

    fn enable_interrupts() {
        unsafe { core::arch::asm!("sti") };
    }

    fn disable_interrupts() {
        unsafe { core::arch::asm!("cli") };
    }

    fn halt_until_interrupt() {
        unsafe { core::arch::asm!("hlt") };
    }
}

/* Read the CPU's Time Stamp Counter — a 64-bit counter that increments
 * every clock cycle. Useful as a source of entropy since we can't predict
 * exactly when a human presses a key.
 */
pub fn read_tsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        core::arch::asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack));
    }
    ((hi as u64) << 32) | (lo as u64)
}
