//! 8259 PIC (Programmable Interrupt Controller) driver.
//!
//! The 8259 is a pair of chips — master and slave — that multiplex 15 hardware
//! interrupt lines (IRQs) into a single INTR pin on the CPU. The design dates
//! back to the IBM PC/AT (1984), but every x86 PC still has one (or emulates
//! one) for backwards compatibility, even machines that also have an APIC.
//!
//! ## Why we need to remap
//!
//! Out of reset the master PIC maps IRQs 0–7 to interrupt vectors 0–7, and
//! the slave maps IRQs 8–15 to vectors 8–15. But the CPU reserves vectors
//! 0–31 for exceptions (divide-by-zero is vector 0, double fault is vector 8,
//! page fault is vector 14...). So a timer tick (IRQ 0) is indistinguishable
//! from a divide-by-zero — the CPU literally can't tell them apart.
//!
//! The fix is to **remap** the PIC so that IRQs start at vector 32, the first
//! vector not reserved by the CPU. After remapping:
//!
//! | IRQ | Vector | Typical device          |
//! |-----|--------|-------------------------|
//! | 0   | 32     | PIT timer               |
//! | 1   | 33     | PS/2 keyboard           |
//! | 2   | 34     | Cascade (slave PIC)     |
//! | 3   | 35     | COM2 / COM4             |
//! | 4   | 36     | COM1 / COM3             |
//! | 5   | 37     | LPT2 / sound card       |
//! | 6   | 38     | Floppy disk             |
//! | 7   | 39     | LPT1 / spurious         |
//! | 8   | 40     | RTC (real-time clock)    |
//! | 9   | 41     | ACPI / redirected IRQ2   |
//! | 10  | 42     | Open                    |
//! | 11  | 43     | Open                    |
//! | 12  | 44     | PS/2 mouse              |
//! | 13  | 45     | FPU / coprocessor       |
//! | 14  | 46     | Primary ATA             |
//! | 15  | 47     | Secondary ATA           |
//!
//! ## Initialization protocol (ICW1–ICW4)
//!
//! The 8259 expects a rigid sequence of 4 "Initialization Command Words"
//! written in order. This hasn't changed since the original Intel datasheet.
//! Each ICW tells the PIC something about its configuration:
//!
//! - **ICW1** (→ command port): "start initialization, I'm going to send ICW4"
//! - **ICW2** (→ data port): the vector offset (where to map IRQ 0)
//! - **ICW3** (→ data port): how master and slave are wired together
//! - **ICW4** (→ data port): "use 8086 mode" (vs. the ancient MCS-80 mode)
//!
//! After initialization, all IRQs are masked (disabled). Individual IRQs get
//! unmasked one at a time as device drivers register their handlers.

use super::port::{inb, outb};

// Master PIC lives at I/O ports 0x20 (command) and 0x21 (data).
const MASTER_CMD: u16 = 0x20;
const MASTER_DATA: u16 = 0x21;

// Slave PIC lives at I/O ports 0xA0 (command) and 0xA1 (data).
const SLAVE_CMD: u16 = 0xA0;
const SLAVE_DATA: u16 = 0xA1;

/// First interrupt vector used for hardware IRQs. IRQ N maps to vector
/// `IRQ_BASE + N`. We pick 32 because vectors 0–31 are reserved by the CPU
/// for exceptions.
pub const IRQ_BASE: u8 = 32;

/// End of Interrupt command byte. After handling an IRQ, we must send this
/// to the PIC (and both PICs for slave IRQs) so it knows it can deliver the
/// next interrupt on that line.
const EOI: u8 = 0x20;

/// Initialize both PICs: remap IRQs to vectors 32–47 and mask everything.
///
/// After this function returns, no hardware interrupts will fire until a
/// driver explicitly calls [`unmask`] for the IRQ it wants to handle.
///
/// Must be called after the IDT is loaded, since the whole point of
/// remapping is to route IRQs to our IDT entries.
pub fn init() {
    unsafe {
        // Save the current masks so we can inspect them in debugging if needed,
        // though we'll overwrite them with 0xFF (all masked) at the end.
        let _master_mask = inb(MASTER_DATA);
        let _slave_mask = inb(SLAVE_DATA);

        // ICW1: bit 0 = ICW4 needed, bit 4 = initialization flag.
        // Writing to the command port with bit 4 set tells the PIC
        // "I'm starting the init sequence, expect ICW2–ICW4 on the data port."
        outb(MASTER_CMD, 0x11);
        io_wait();
        outb(SLAVE_CMD, 0x11);
        io_wait();

        // ICW2: vector offset. This is the whole reason we're here — remap
        // IRQs away from the CPU exception range.
        outb(MASTER_DATA, IRQ_BASE);       // Master: IRQ 0–7  → vectors 32–39
        io_wait();
        outb(SLAVE_DATA, IRQ_BASE + 8);    // Slave:  IRQ 8–15 → vectors 40–47
        io_wait();

        // ICW3: wiring between master and slave.
        // Master: bit 2 set means "there's a slave on IRQ 2" (bit mask, not number).
        // Slave: value 2 means "my cascade identity is IRQ 2" (binary number, not mask).
        // This asymmetry is one of the PIC's many charming quirks.
        outb(MASTER_DATA, 0x04);
        io_wait();
        outb(SLAVE_DATA, 0x02);
        io_wait();

        // ICW4: bit 0 = 8086 mode (vs ancient 8080 mode). That's all we need.
        outb(MASTER_DATA, 0x01);
        io_wait();
        outb(SLAVE_DATA, 0x01);
        io_wait();

        // Mask all IRQs on both PICs. Nothing fires until a driver explicitly
        // unmaskes its IRQ line. This prevents interrupts from devices we
        // don't have handlers for yet.
        outb(MASTER_DATA, 0xFF);
        outb(SLAVE_DATA, 0xFF);
    }

    crate::println!("[ok] PIC remapped (IRQs 32-47), all masked");
}

/// Unmask (enable) a specific IRQ line so the PIC will deliver it to the CPU.
///
/// For slave IRQs (8–15), this also unmasks IRQ 2 on the master — that's the
/// cascade line. If IRQ 2 is masked, slave interrupts can never reach the CPU
/// because they have to pass through the master first.
pub fn unmask(irq: u8) {
    assert!(irq < 16, "IRQ number must be 0-15, got {}", irq);

    unsafe {
        if irq < 8 {
            let mask = inb(MASTER_DATA) & !(1 << irq);
            outb(MASTER_DATA, mask);
        } else {
            let mask = inb(SLAVE_DATA) & !(1 << (irq - 8));
            outb(SLAVE_DATA, mask);

            // Also unmask the cascade line (IRQ 2) on the master, otherwise
            // slave interrupts are blocked at the master and never arrive.
            let master_mask = inb(MASTER_DATA) & !(1 << 2);
            outb(MASTER_DATA, master_mask);
        }
    }
}

/// Send End of Interrupt to acknowledge an IRQ.
///
/// The PIC won't deliver another interrupt on the same line until it gets
/// an EOI. For slave IRQs (8–15) we must send EOI to **both** PICs — first
/// the slave, then the master — because the slave's interrupt went through
/// the master's cascade line.
///
/// This must be called at the end of every IRQ handler, or that IRQ line
/// will be permanently stuck.
pub fn acknowledge(irq: u8) {
    unsafe {
        if irq >= 8 {
            outb(SLAVE_CMD, EOI);
        }
        outb(MASTER_CMD, EOI);
    }
}

/// Disable the 8259 PIC entirely by masking all IRQs.
///
/// Called during the transition to APIC mode. After this, no interrupts
/// will be delivered through the PIC. The PIC hardware remains initialized
/// (it was needed for APIC timer calibration via the PIT), but all 16
/// IRQ lines are masked.
pub fn disable() {
    unsafe {
        outb(MASTER_DATA, 0xFF);
        outb(SLAVE_DATA, 0xFF);
    }
}

/// Brief I/O delay to give the PIC time to process a command.
///
/// Port 0x80 is the "POST diagnostic" port — writing to it does nothing
/// meaningful but takes just long enough for the PIC to digest the
/// previous write. This is a standard technique that goes back to the
/// original IBM PC. Some BIOSes display the value on a diagnostic LED,
/// but we don't care about that.
unsafe fn io_wait() {
    unsafe {
        outb(0x80, 0);
    }
}
