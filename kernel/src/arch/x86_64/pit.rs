//! 8254 PIT (Programmable Interval Timer) driver.
//!
//! The PIT is a timer chip with 3 independent counters. Channel 0 is wired to
//! IRQ 0 on the PIC and provides the system heartbeat — a periodic interrupt
//! at a configurable frequency.
//!
//! The PIT's oscillator runs at **1,193,182 Hz**, a frequency inherited from the
//! original IBM PC. IBM needed a single cheap crystal that could drive both the
//! CGA video controller (which needed a frequency related to the NTSC color burst)
//! and the timer. A 14.31818 MHz crystal (4× the NTSC color burst frequency of
//! 3.579545 MHz) divided by 12 gave 1.193182 MHz for the PIT. Every x86 PC since
//! has kept this frequency for backwards compatibility.
//!
//! To get periodic interrupts at a desired frequency, we load a **divisor** into
//! channel 0. The PIT counts down from the divisor at 1.193182 MHz and fires
//! IRQ 0 each time it hits zero, then reloads. So:
//! frequency = 1,193,182 / divisor. For ~1000 Hz: divisor = 1193.

use super::port::outb;

/// Channel 0 data port — write the divisor here (low byte, then high byte).
const CHANNEL_0: u16 = 0x40;

/// Mode/command register (write-only). Selects which channel to program
/// and what counting mode to use.
const COMMAND: u16 = 0x43;

/// The PIT's base oscillator frequency in Hz.
const PIT_FREQUENCY: u32 = 1_193_182;

/// Total number of IRQ 0 ticks since the PIT was initialized.
///
/// No locking needed: we're single-core, interrupt gates clear IF so IRQ
/// handlers can't re-enter, and u64 reads/writes are atomic on x86_64.
static mut TICKS: u64 = 0;

/// The frequency we programmed, stored for [`elapsed_ms`] conversion.
static mut FREQUENCY_HZ: u32 = 0;

/// Program PIT channel 0 for periodic interrupts at `frequency_hz` Hz.
///
/// This configures the hardware but does NOT unmask the IRQ or enable
/// interrupts — that's handled separately by [`super::interrupts::set_irq_handler`]
/// and [`super::X86_64::enable_interrupts`].
pub fn init(frequency_hz: u32) {
    let divisor = PIT_FREQUENCY / frequency_hz;

    // Command byte 0x36:
    //   bits 6–7 = 00  → select channel 0
    //   bits 4–5 = 11  → access mode: lobyte/hibyte (send both bytes)
    //   bits 1–3 = 011 → mode 3: square wave generator
    //   bit 0    = 0   → binary counting (not BCD)
    unsafe {
        outb(COMMAND, 0x36);
        outb(CHANNEL_0, divisor as u8);         // low byte
        outb(CHANNEL_0, (divisor >> 8) as u8);  // high byte

        FREQUENCY_HZ = frequency_hz;
    }

    let period_us = 1_000_000 / frequency_hz;
    crate::println!("[ok] PIT configured: {} Hz ({} \u{00B5}s per tick)", frequency_hz, period_us);
}

/// Return milliseconds elapsed since the PIT was initialized.
///
/// At 1000 Hz this simplifies to just the tick count. No overflow concern:
/// u64 at 1000 Hz overflows after ~584 million years.
pub fn elapsed_ms() -> u64 {
    unsafe { (TICKS * 1000) / FREQUENCY_HZ as u64 }
}

/// IRQ 0 handler — called by the interrupt stub every time the PIT fires.
///
/// Increments the tick counter, then notifies the scheduler so it can
/// track time slices and preempt if needed. EOI has already been sent
/// by the IRQ stub before we're called (required because we may
/// context-switch away and not return for a while).
pub fn tick() {
    unsafe {
        TICKS += 1;
    }
    crate::scheduler::on_tick();
    crate::timer::check_deadlines();
}
