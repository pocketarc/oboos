//! I/O APIC driver — routes external hardware interrupts to CPU cores.
//!
//! The I/O APIC replaces the 8259 PIC as the external interrupt controller.
//! While the PIC can only deliver interrupts to one CPU, the I/O APIC can
//! route each IRQ to any core (or set of cores) via a configurable
//! redirection table.
//!
//! ## Standard base address
//!
//! The I/O APIC is memory-mapped at `0xFEC00000` on virtually all x86
//! systems. This address is standardized and works on QEMU and real
//! hardware. Proper ACPI MADT parsing to discover the actual address is
//! deferred — it's not worth the complexity yet.
//!
//! ## Register access
//!
//! The I/O APIC uses an indirect register access scheme: write the
//! register index to the IOREGSEL port (offset 0x00), then read/write
//! the value from/to IOWIN (offset 0x10). This was common in older
//! hardware to save address space.

use super::memory::phys_to_virt;

/// Standard I/O APIC physical base address.
const IOAPIC_BASE_PHYS: u64 = 0xFEC0_0000;

// Register indices (written to IOREGSEL to select the register).
const IOAPIC_ID: u32 = 0x00;
const IOAPIC_VER: u32 = 0x01;
const IOAPIC_REDTBL_BASE: u32 = 0x10;

// ————————————————————————————————————————————————————————————————————————————
// Register access
// ————————————————————————————————————————————————————————————————————————————

/// Read a 32-bit I/O APIC register.
unsafe fn read(reg: u32) -> u32 {
    let base = phys_to_virt(IOAPIC_BASE_PHYS);
    unsafe {
        // Write register index to IOREGSEL (offset 0x00).
        core::ptr::write_volatile(base as *mut u32, reg);
        // Read value from IOWIN (offset 0x10).
        core::ptr::read_volatile(base.add(0x10) as *const u32)
    }
}

/// Write a 32-bit I/O APIC register.
unsafe fn write(reg: u32, value: u32) {
    let base = phys_to_virt(IOAPIC_BASE_PHYS);
    unsafe {
        core::ptr::write_volatile(base as *mut u32, reg);
        core::ptr::write_volatile(base.add(0x10) as *mut u32, value);
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Public API
// ————————————————————————————————————————————————————————————————————————————

/// Initialize the I/O APIC: disable the 8259 PIC and set up IRQ routing.
///
/// After this function:
/// - The 8259 PIC is fully disabled (all IRQs masked, remapped to unused vectors)
/// - Keyboard IRQ 1 is routed through the I/O APIC to the BSP (vector 33)
///
/// Must be called after [`super::lapic::init()`].
pub fn init() {
    // The I/O APIC is memory-mapped MMIO at a fixed physical address.
    // Limine's HHDM only covers RAM, so we must map this page explicitly.
    super::paging::ensure_mmio_mapped(IOAPIC_BASE_PHYS as usize);

    // Disable the 8259 PIC by masking all IRQs. We remap it to vectors
    // 0x20-0x2F first (in case any stray interrupts fire during the
    // transition), then mask everything.
    super::pic::disable();

    unsafe {
        let ver = read(IOAPIC_VER);
        let max_entries = ((ver >> 16) & 0xFF) + 1;
        crate::println!("[ioapic] {} redirection entries", max_entries);

        // Mask all redirection entries by default. Each entry is 64 bits
        // (two 32-bit registers). Bit 16 of the low word = masked.
        for i in 0..max_entries {
            let reg_low = IOAPIC_REDTBL_BASE + i * 2;
            let reg_high = IOAPIC_REDTBL_BASE + i * 2 + 1;
            write(reg_low, 0x10000); // masked, vector 0
            write(reg_high, 0);
        }

        // Route keyboard (IRQ 1) → vector 33 → BSP (LAPIC ID 0).
        // Low 32 bits: vector 33 (0x21), delivery mode fixed (000),
        // active high, edge-triggered, not masked.
        route_irq(1, super::pic::IRQ_BASE + 1, 0);
    }

    crate::println!("[ok] I/O APIC initialized, PIC disabled");
}

/// Route an IRQ through the I/O APIC to a specific core.
///
/// Sets the redirection table entry for `irq` to deliver `vector` to
/// the core with LAPIC ID `dest_lapic_id`. The interrupt is configured
/// as fixed delivery, edge-triggered, active-high (standard for ISA IRQs).
pub fn route_irq(irq: u8, vector: u8, dest_lapic_id: u32) {
    let reg_low = IOAPIC_REDTBL_BASE + (irq as u32) * 2;
    let reg_high = IOAPIC_REDTBL_BASE + (irq as u32) * 2 + 1;

    unsafe {
        // High 32 bits: destination LAPIC ID in bits 24-31.
        write(reg_high, dest_lapic_id << 24);
        // Low 32 bits: vector, fixed delivery, edge, active high, not masked.
        write(reg_low, vector as u32);
    }
}
