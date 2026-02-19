//! Hardware Abstraction Layer (HAL) traits.
//!
//! Every architecture implements these. The kernel only calls these traits,
//! never arch-specific code directly. This is what makes adding a second
//! architecture (aarch64) tractable later — you implement the same traits
//! for different hardware.

/// Core platform operations: initialization, interrupt control, and halting.
pub trait Platform {
    fn init() -> Self;
    fn enable_interrupts();
    fn disable_interrupts();
    /// Put the CPU to sleep until the next interrupt arrives. This saves
    /// power compared to a busy spin loop — the CPU literally stops
    /// executing until hardware wakes it up.
    fn halt_until_interrupt();

    /// Return a pseudo-random 64-bit value.
    ///
    /// Not cryptographically secure — uses whatever fast entropy source
    /// the hardware provides (TSC on x86_64, `CNTPCT_EL0` on aarch64,
    /// hardware RNG where available, etc.).
    fn entropy() -> u64;

    /// Return milliseconds elapsed since the system timer was initialized.
    fn elapsed_ms() -> u64;

    /// Deliberately trigger a CPU fault for testing exception handling.
    ///
    /// Without an IDT / vector table this will crash the machine (triple
    /// fault on x86_64). Once exception handlers are installed, it
    /// exercises them — useful for verifying fault handlers actually work.
    fn trigger_test_fault() -> !;
}

/// Architecture-independent key identifiers.
///
/// The keyboard driver translates hardware-specific scancodes (PS/2 scan
/// code set 1 on x86, USB HID codes on other platforms) into these
/// variants. The rest of the kernel never sees raw scancodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Enter,
    F,
    H,
    T,
    /// A key we recognize but don't have a named variant for yet.
    /// Carries the raw scancode for callers that want to inspect it.
    Other(u8),
}

/// Serial port access for debug output.
pub trait SerialConsole {
    fn init();
    fn write_byte(b: u8);
    fn read_byte() -> Option<u8>;
}

/// Programmable interrupt controller interface.
pub trait InterruptController {
    fn init(&mut self);
    fn register_handler(vector: u8, handler: fn());
    /// Tell the controller we're done handling this interrupt. If you
    /// forget to acknowledge, the controller won't deliver the next one.
    fn acknowledge(vector: u8);
}

/// Hardware timer interface.
pub trait Timer {
    fn init(frequency_hz: u32);
    fn elapsed_ms() -> u64;
}

/// Virtual memory management.
pub trait MemoryManager {
    fn map_page(virt: usize, phys: usize, flags: PageFlags);
    fn unmap_page(virt: usize);
    /// Allocate a physical frame — a 4 KiB-aligned page of physical RAM.
    fn alloc_physical_frame() -> Option<usize>;
    /// Return a physical frame to the free pool.
    fn free_physical_frame(addr: usize);
}

/// CPU context switching for multitasking.
pub trait ContextSwitch {
    /// Save current register state, restore `next`, jump.
    ///
    /// # Safety
    ///
    /// Both `current` and `next` must point to valid, properly initialized
    /// [`TaskContext`] values. `next` must represent a task that is safe to
    /// resume — its stack and instruction pointer must be valid. Getting
    /// this wrong will corrupt the stack or jump to garbage.
    unsafe fn switch(current: &mut TaskContext, next: &TaskContext);
}

/// Page table entry flags.
///
/// These are architecture-independent constants that map to the permission
/// bits in page table entries. Defined in the HAL so callers write
/// `PageFlags::PRESENT | PageFlags::WRITABLE` without reaching into
/// arch-specific code. The arch layer translates these to hardware bits.
#[derive(Clone, Copy, Debug)]
pub struct PageFlags(pub u64);

impl PageFlags {
    /// Page is present in physical memory (bit 0). Without this bit, any
    /// access to the page triggers a page fault.
    pub const PRESENT: Self = Self(1 << 0);
    /// Page is writable (bit 1). Without this, writes trigger a page fault.
    /// Reads are always allowed when PRESENT is set.
    pub const WRITABLE: Self = Self(1 << 1);
    /// Page is accessible from user mode / ring 3 (bit 2). Without this,
    /// only supervisor code can touch the page.
    pub const USER: Self = Self(1 << 2);
    /// Disable instruction fetch from this page (bit 63). Requires the NXE
    /// bit in the EFER MSR, which Limine enables for us. Prevents code
    /// execution from data pages — a basic exploit mitigation.
    pub const NO_EXECUTE: Self = Self(1 << 63);
}

impl core::ops::BitOr for PageFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

pub use crate::arch::TaskContext;
