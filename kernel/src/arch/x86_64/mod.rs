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
