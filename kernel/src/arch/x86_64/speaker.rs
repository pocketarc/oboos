//! PC speaker driver via PIT channel 2 and port 0x61.
//!
//! The PC speaker is the simplest audio output on x86 — a single square wave
//! at a programmable frequency. The PIT has three independent channels that
//! share a command register (port 0x43) but have separate data ports and
//! counters. Channel 0 drives the system timer (configured in [`super::pit`]);
//! channel 2 drives the PC speaker. Programming one doesn't disturb the other.
//!
//! Port 0x61 is the system control port. Bits 0 and 1 gate and enable the
//! speaker:
//! - Bit 0: PIT channel 2 gate (1 = counter runs, 0 = counter frozen)
//! - Bit 1: Speaker output enable (1 = speaker connected to channel 2 output)
//! - Bits 2–7: NMI status, parity check, etc. — must be preserved via
//!   read-modify-write.
//!
//! To produce a tone: program channel 2 with a divisor for the desired
//! frequency, then set bits 0+1 of port 0x61. To silence: clear those bits.

use super::port::{inb, outb};

/// PIT channel 2 data port — write the frequency divisor here (lobyte then hibyte).
const CHANNEL_2: u16 = 0x42;

/// PIT mode/command register (shared with channel 0, write-only).
const COMMAND: u16 = 0x43;

/// System control port B — bits 0–1 control the PC speaker gate and enable.
const SPEAKER_PORT: u16 = 0x61;

/// The PIT oscillator frequency in Hz. Same crystal for all three channels —
/// inherited from the original IBM PC's 14.31818 MHz crystal divided by 12.
const PIT_FREQUENCY: u32 = 1_193_182;

/// Start a continuous tone at the given frequency.
///
/// Programs PIT channel 2 as a square wave generator at `frequency_hz`,
/// then enables the speaker gate and output. The tone plays until
/// [`stop()`] is called.
///
/// Frequencies below ~19 Hz or above ~596 kHz will produce incorrect
/// results due to divisor overflow/underflow, but anything in the audible
/// range (20 Hz – 20 kHz) works fine.
pub fn beep(frequency_hz: u32) {
    let divisor = PIT_FREQUENCY / frequency_hz;

    unsafe {
        // Command byte 0xB6:
        //   bits 6–7 = 10  → select channel 2
        //   bits 4–5 = 11  → access mode: lobyte/hibyte
        //   bits 1–3 = 011 → mode 3: square wave generator
        //   bit 0    = 0   → binary counting (not BCD)
        outb(COMMAND, 0xB6);
        outb(CHANNEL_2, divisor as u8);        // low byte
        outb(CHANNEL_2, (divisor >> 8) as u8); // high byte

        // Enable the speaker: set bit 0 (gate) and bit 1 (output),
        // preserving all other bits in the system control port.
        let prev = inb(SPEAKER_PORT);
        outb(SPEAKER_PORT, prev | 0x03);
    }
}

/// Silence the PC speaker by clearing the gate and output enable bits.
///
/// This freezes the PIT channel 2 counter and disconnects it from the
/// speaker — the channel's programming is preserved, so a subsequent
/// [`beep()`] call at the same frequency would only need to re-enable
/// the bits.
pub fn stop() {
    unsafe {
        let prev = inb(SPEAKER_PORT);
        outb(SPEAKER_PORT, prev & 0xFC);
    }
}

/// Play a tone for a fixed duration, then silence the speaker.
///
/// This is an async function that uses [`crate::timer::sleep()`] for the
/// delay — our first arch-specific code that calls into arch-independent
/// async infrastructure. It only works inside the executor's run loop.
pub async fn play_tone(frequency_hz: u32, duration_ms: u64) {
    beep(frequency_hz);
    crate::timer::sleep(duration_ms).await;
    stop();
}

/// Check whether the speaker output is currently enabled.
///
/// Returns `true` if bit 1 (speaker output enable) of port 0x61 is set.
/// Used by the smoke test to verify [`beep()`] and [`stop()`] toggle the
/// hardware correctly.
pub(crate) fn is_enabled() -> bool {
    unsafe { inb(SPEAKER_PORT) & 0x02 != 0 }
}
