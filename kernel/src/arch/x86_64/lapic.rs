//! Local APIC (Advanced Programmable Interrupt Controller) driver.
//!
//! Every x86_64 CPU core has its own Local APIC — a per-core interrupt
//! controller that handles:
//! - **Timer interrupts**: each core gets an independent hardware timer
//! - **Inter-Processor Interrupts (IPIs)**: cores can signal each other
//! - **External interrupts**: routed from the I/O APIC
//!
//! The Local APIC replaces the 8259 PIC for interrupt handling. While the
//! PIC is a shared chip that can only deliver interrupts to one CPU, the
//! APIC architecture gives each core its own interrupt controller with a
//! private timer — essential for SMP.
//!
//! ## MMIO access
//!
//! The Local APIC's registers are memory-mapped at a physical address
//! read from the `IA32_APIC_BASE` MSR (typically `0xFEE00000`). We access
//! them through the HHDM (Higher Half Direct Map) that Limine set up.
//! All register accesses are 32-bit aligned reads/writes.
//!
//! ## Timer calibration
//!
//! The APIC timer runs at an unknown frequency (it's derived from the
//! CPU's bus clock, which varies by hardware). To program it for 1 kHz,
//! we calibrate against the PIT: run both timers simultaneously for 10ms
//! (measured by the PIT's known 1.193182 MHz oscillator) and count how
//! many APIC ticks elapsed. This gives us the conversion factor.

use core::sync::atomic::{AtomicU64, Ordering};

use super::memory::phys_to_virt;

// ————————————————————————————————————————————————————————————————————————————
// LAPIC register offsets (from the APIC base address)
// ————————————————————————————————————————————————————————————————————————————

const LAPIC_ID: u32 = 0x020;
const LAPIC_EOI: u32 = 0x0B0;
const LAPIC_SPURIOUS: u32 = 0x0F0;
const LAPIC_ICR_LOW: u32 = 0x300;
const LAPIC_ICR_HIGH: u32 = 0x310;
const LAPIC_TIMER_LVT: u32 = 0x320;
const LAPIC_TIMER_INIT: u32 = 0x380;
const LAPIC_TIMER_CURRENT: u32 = 0x390;
const LAPIC_TIMER_DIVIDE: u32 = 0x3E0;

/// MSR that holds the LAPIC base physical address and enable bit.
const IA32_APIC_BASE_MSR: u32 = 0x1B;

/// The APIC timer interrupt vector. Chosen to be above the PIC range (32-47)
/// and below the spurious vector (0xFF).
pub const TIMER_VECTOR: u8 = 48;

/// IPI wake vector — used to break APs out of `hlt` (Phase 4).
pub const IPI_WAKE_VECTOR: u8 = 49;

/// TLB shootdown vector — used to invalidate TLB entries on remote cores (Phase 5).
pub const TLB_SHOOTDOWN_VECTOR: u8 = 50;

/// Spurious interrupt vector. Must be 0xXF (low nibble = 0xF) per Intel spec.
/// The handler is a no-op that does NOT send EOI.
const SPURIOUS_VECTOR: u8 = 0xFF;

// ————————————————————————————————————————————————————————————————————————————
// Global state
// ————————————————————————————————————————————————————————————————————————————

/// LAPIC base virtual address, computed once from the MSR during init.
static mut LAPIC_BASE: u64 = 0;

/// Calibrated APIC timer ticks per millisecond. Set during BSP init,
/// used by all cores (the APIC timer frequency is the same across cores
/// on the same die).
static mut APIC_TICKS_PER_MS: u32 = 0;

/// Global tick counter incremented by the BSP's APIC timer handler only.
/// Other cores drive their own scheduling ticks but don't contribute to
/// `elapsed_ms()` — otherwise we'd multiply the count by the number of cores.
pub static GLOBAL_TICKS: AtomicU64 = AtomicU64::new(0);

/// Whether APIC mode is active. Once set, IRQ handlers use `lapic::eoi()`
/// instead of `pic::acknowledge()`.
static mut APIC_MODE: bool = false;

// ————————————————————————————————————————————————————————————————————————————
// Register access
// ————————————————————————————————————————————————————————————————————————————

/// Read a 32-bit LAPIC register.
unsafe fn read(offset: u32) -> u32 {
    let addr = unsafe { LAPIC_BASE + offset as u64 };
    unsafe { core::ptr::read_volatile(addr as *const u32) }
}

/// Write a 32-bit LAPIC register.
unsafe fn write(offset: u32, value: u32) {
    let addr = unsafe { LAPIC_BASE + offset as u64 };
    unsafe { core::ptr::write_volatile(addr as *mut u32, value) }
}

// ————————————————————————————————————————————————————————————————————————————
// MSR helpers
// ————————————————————————————————————————————————————————————————————————————

unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") msr,
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack, preserves_flags),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

// ————————————————————————————————————————————————————————————————————————————
// Public API
// ————————————————————————————————————————————————————————————————————————————

/// Initialize the BSP's Local APIC.
///
/// Reads the LAPIC base address from the MSR, enables the APIC via the
/// spurious interrupt register, and calibrates the APIC timer against
/// the PIT. After this, the PIT IRQ 0 is no longer needed for timekeeping.
///
/// Must be called after `pit::init()` (needs PIT for calibration) and
/// before `ioapic::init()` (which disables the PIC).
pub fn init() {
    unsafe {
        // Read LAPIC base physical address from the MSR.
        // Bits 12..35 hold the base address (page-aligned).
        let msr_val = rdmsr(IA32_APIC_BASE_MSR);
        let phys_base = msr_val & 0xFFFF_F000;

        // The LAPIC is memory-mapped MMIO, not RAM. Limine's HHDM only
        // covers physical RAM regions, so this page may not be mapped yet.
        super::paging::ensure_mmio_mapped(phys_base as usize);

        LAPIC_BASE = phys_to_virt(phys_base) as u64;

        // Enable the APIC by setting bit 8 (APIC Software Enable) in the
        // spurious interrupt vector register. Also set the spurious vector.
        write(LAPIC_SPURIOUS, 0x100 | SPURIOUS_VECTOR as u32);

        // Set timer divide to 16. The divide register controls how the bus
        // clock is divided before reaching the timer counter. Lower divisors
        // give finer granularity but overflow faster. 16 is a good balance.
        // Encoding: 0b0011 = divide by 16.
        write(LAPIC_TIMER_DIVIDE, 0x03);

        // Calibrate: measure APIC timer ticks in a known PIT interval.
        let ticks_per_ms = calibrate_timer();
        APIC_TICKS_PER_MS = ticks_per_ms;

        crate::println!("[lapic] Calibrated: {} ticks/ms (base {:#X})",
            ticks_per_ms, phys_base);

        // Start the BSP's periodic timer.
        start_timer(ticks_per_ms);

        APIC_MODE = true;
    }
}

/// Initialize an AP's Local APIC and start its timer.
///
/// The LAPIC base address and ticks_per_ms are already known from BSP init.
/// Each AP just needs to enable its own APIC and start its timer.
pub fn init_ap() {
    unsafe {
        // Read this AP's LAPIC base from its own MSR (same physical address
        // on all cores, but each core has its own MSR copy).
        let msr_val = rdmsr(IA32_APIC_BASE_MSR);
        let phys_base = msr_val & 0xFFFF_F000;
        let base = phys_to_virt(phys_base) as u64;

        // Enable APIC + set spurious vector.
        let addr = base + LAPIC_SPURIOUS as u64;
        core::ptr::write_volatile(addr as *mut u32, 0x100 | SPURIOUS_VECTOR as u32);

        // Set timer divide to 16 (same as BSP).
        let addr = base + LAPIC_TIMER_DIVIDE as u64;
        core::ptr::write_volatile(addr as *mut u32, 0x03);

        // Start periodic timer with the calibrated value.
        let ticks_per_ms = APIC_TICKS_PER_MS;

        // Program timer LVT: periodic mode (bit 17), our timer vector.
        let addr = base + LAPIC_TIMER_LVT as u64;
        core::ptr::write_volatile(addr as *mut u32, (1 << 17) | TIMER_VECTOR as u32);

        // Set initial count — timer fires every 1ms.
        let addr = base + LAPIC_TIMER_INIT as u64;
        core::ptr::write_volatile(addr as *mut u32, ticks_per_ms);
    }
}

/// Send End of Interrupt to the Local APIC.
///
/// Must be called at the end of every APIC-delivered interrupt handler.
/// Writing any value to the EOI register signals completion.
pub fn eoi() {
    unsafe { write(LAPIC_EOI, 0) }
}

/// Return `true` if we've switched from PIC to APIC mode.
pub fn is_apic_mode() -> bool {
    unsafe { APIC_MODE }
}

/// Return elapsed milliseconds since APIC timer started (BSP only).
pub fn elapsed_ms() -> u64 {
    GLOBAL_TICKS.load(Ordering::Relaxed)
}

/// Send an IPI (Inter-Processor Interrupt) to a specific core.
///
/// Used for wake IPIs (Phase 4) and TLB shootdown (Phase 5).
pub fn send_ipi(target_lapic_id: u32, vector: u8) {
    unsafe {
        // ICR high: destination LAPIC ID in bits 24-31.
        write(LAPIC_ICR_HIGH, target_lapic_id << 24);
        // ICR low: vector in bits 0-7, delivery mode fixed (000),
        // level assert (bit 14), edge trigger (bit 15 = 0).
        write(LAPIC_ICR_LOW, vector as u32);
    }
}

/// Read this core's LAPIC ID from the LAPIC register.
pub fn id() -> u32 {
    unsafe { read(LAPIC_ID) >> 24 }
}

// ————————————————————————————————————————————————————————————————————————————
// Timer internals
// ————————————————————————————————————————————————————————————————————————————

/// Start the APIC timer in periodic mode at 1 kHz.
fn start_timer(ticks_per_ms: u32) {
    unsafe {
        // Program timer LVT: periodic mode (bit 17), our timer vector.
        write(LAPIC_TIMER_LVT, (1 << 17) | TIMER_VECTOR as u32);
        // Set initial count — timer reloads this value each period.
        write(LAPIC_TIMER_INIT, ticks_per_ms);
    }
}

/// Calibrate the APIC timer against the PIT.
///
/// Programs the PIT for a 10ms one-shot, starts the APIC timer counting
/// down from max, waits for the PIT to fire, then measures how many APIC
/// ticks elapsed. Returns ticks per millisecond.
fn calibrate_timer() -> u32 {
    unsafe {
        // Start APIC timer counting down from max (one-shot, masked so
        // it doesn't fire an interrupt — we just read the current count).
        write(LAPIC_TIMER_LVT, 0x10000); // masked (bit 16)
        write(LAPIC_TIMER_INIT, 0xFFFF_FFFF);

        // Program PIT channel 0 for a 10ms one-shot.
        // Mode 0 (interrupt on terminal count), lobyte/hibyte access.
        // Divisor = 1_193_182 / 100 = 11932 (~10ms).
        let divisor: u16 = 11932;
        super::port::outb(0x43, 0x30); // channel 0, mode 0, lobyte/hibyte
        super::port::outb(0x40, divisor as u8);
        super::port::outb(0x40, (divisor >> 8) as u8);

        // Spin-wait for the PIT one-shot to complete. In mode 0, the OUT
        // pin goes high when the count reaches zero. We poll the PIT's
        // read-back status: bit 7 of the status byte = OUT pin state.
        loop {
            // Latch the status of channel 0 (read-back command).
            super::port::outb(0x43, 0xE2); // read-back, channel 0, status only
            let status = super::port::inb(0x40);
            if status & 0x80 != 0 {
                break; // OUT pin high = countdown complete
            }
        }

        // Read how many APIC ticks elapsed in ~10ms.
        let current = read(LAPIC_TIMER_CURRENT);
        let elapsed = 0xFFFF_FFFFu32 - current;

        // Stop the APIC timer (mask it).
        write(LAPIC_TIMER_LVT, 0x10000);
        write(LAPIC_TIMER_INIT, 0);

        // ticks_per_ms = elapsed_in_10ms / 10
        elapsed / 10
    }
}
