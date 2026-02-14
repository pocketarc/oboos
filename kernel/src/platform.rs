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
pub struct PageFlags(pub u64);

/// Saved register state for a suspended task.
pub struct TaskContext {
    _placeholder: u64,
}
