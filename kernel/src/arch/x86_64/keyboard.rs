use super::port::inb;

/* PS/2 keyboard controller ports.
 * These have been the same since the IBM PC AT (1984).
 */
const DATA_PORT: u16 = 0x60;
const STATUS_PORT: u16 = 0x64;

/* Scan code set 1 — the default that the keyboard controller gives us.
 * These are "make codes" (key press). Release codes are make + 0x80.
 */
pub const SC_ENTER: u8 = 0x1C;

/* Poll the keyboard controller. Returns Some(scancode) if a key event
 * is waiting, None otherwise. Non-blocking.
 *
 * @todo Replace this with interrupt-driven input, which is much cleaner.
 */
pub fn poll_scancode() -> Option<u8> {
    unsafe {
        // Bit 0 of the status register = output buffer full (data ready)
        if inb(STATUS_PORT) & 0x01 != 0 {
            Some(inb(DATA_PORT))
        } else {
            None
        }
    }
}
