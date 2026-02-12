use crate::platform::SerialConsole;
use super::port::{inb, outb};

// COM1 base I/O port address. This has been 0x3F8 since the original IBM PC.
const COM1: u16 = 0x3F8;

pub struct Serial;

impl SerialConsole for Serial {
    /* Initialize the UART. This follows the standard 16550 UART programming
     * sequence that's been the same since the 1980s.
     */
    fn init() {
        unsafe {
            outb(COM1 + 1, 0x00); // Disable all interrupts
            outb(COM1 + 3, 0x80); // Enable DLAB (Divisor Latch Access Bit)
            outb(COM1 + 0, 0x03); // Divisor low byte: 38400 baud
            outb(COM1 + 1, 0x00); // Divisor high byte
            outb(COM1 + 3, 0x03); // 8 data bits, no parity, 1 stop bit (8N1)
            outb(COM1 + 2, 0xC7); // Enable FIFO, clear buffers, 14-byte threshold
            outb(COM1 + 4, 0x0B); // RTS/DSR set, IRQs enabled
        }
    }

    fn write_byte(b: u8) {
        unsafe {
            // Spin until the transmit holding register is empty.
            // Bit 5 of the Line Status Register (COM1+5) = THR empty.
            while inb(COM1 + 5) & 0x20 == 0 {}
            outb(COM1, b);
        }
    }

    fn read_byte() -> Option<u8> {
        unsafe {
            // Bit 0 of LSR = data ready
            if inb(COM1 + 5) & 0x01 != 0 {
                Some(inb(COM1))
            } else {
                None
            }
        }
    }
}

// Allows using `write!` / `writeln!` with the serial port.
impl core::fmt::Write for Serial {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            Serial::write_byte(b);
        }
        Ok(())
    }
}
