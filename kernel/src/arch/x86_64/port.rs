/* Raw x86 port I/O. These are privileged instructions — they only work in
 * ring 0 (kernel mode). That's why userspace programs can't just talk to
 * hardware directly, they are not running in the same privilege level as the kernel.
 *
 * x86 has a separate 16-bit I/O address space (ports 0x0000–0xFFFF) that's
 * distinct from memory. Legacy hardware like the UART, keyboard controller,
 * and PIC all live here. Modern hardware uses memory-mapped I/O instead,
 * but we need port I/O for the classics.
 */
pub unsafe fn outb(port: u16, value: u8) {
    unsafe {
        core::arch::asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack));
    }
}

pub unsafe fn inb(port: u16) -> u8 {
    let value: u8;
    unsafe {
        core::arch::asm!("in al, dx", in("dx") port, out("al") value, options(nomem, nostack));
    }
    value
}
