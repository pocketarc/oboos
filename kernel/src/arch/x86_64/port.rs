//! Raw x86 port I/O.
//!
//! These are privileged instructions that only work in ring 0 (kernel mode).
//! That's why userspace programs can't talk to hardware directly — they're
//! not running at the same privilege level as the kernel.
//!
//! x86 has a separate 16-bit I/O address space (ports 0x0000–0xFFFF) that's
//! distinct from memory. Legacy hardware like the UART, keyboard controller,
//! and PIC all live here. Modern hardware uses memory-mapped I/O instead,
//! but we need port I/O for the classics.

/// Write a byte to an I/O port (`out dx, al`).
///
/// # Safety
///
/// Writing to an arbitrary I/O port can trigger any hardware side effect —
/// reconfiguring a device, firing a reset, corrupting state. The caller must
/// ensure `port` and `value` are correct for the device being addressed.
pub unsafe fn outb(port: u16, value: u8) {
    unsafe {
        core::arch::asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack));
    }
}

/// Read a byte from an I/O port (`in al, dx`).
///
/// # Safety
///
/// Reading from an I/O port can have side effects — some device registers
/// change state when read (e.g., the keyboard data port clears the buffer).
/// The caller must ensure `port` is valid and that the read is intentional.
pub unsafe fn inb(port: u16) -> u8 {
    let value: u8;
    unsafe {
        core::arch::asm!("in al, dx", in("dx") port, out("al") value, options(nomem, nostack));
    }
    value
}
