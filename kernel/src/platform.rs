/* Hardware Abstraction Layer (HAL) traits.
 *
 * Every architecture implements these. The kernel only calls these traits,
 * never arch-specific code directly. This is what makes adding a second
 * architecture (aarch64) tractable later — you implement the same traits
 * for different hardware.
*/

pub trait Platform {
    fn init() -> Self;
    fn enable_interrupts();
    fn disable_interrupts();
    fn halt_until_interrupt();
}

pub trait SerialConsole {
    fn init();
    fn write_byte(b: u8);
    fn read_byte() -> Option<u8>;
}

pub trait InterruptController {
    fn init(&mut self);
    fn register_handler(vector: u8, handler: fn());
    fn acknowledge(vector: u8);
}

pub trait Timer {
    fn init(frequency_hz: u32);
    fn elapsed_ms() -> u64;
}

pub trait MemoryManager {
    fn map_page(virt: usize, phys: usize, flags: PageFlags);
    fn unmap_page(virt: usize);
    fn alloc_physical_frame() -> Option<usize>;
    fn free_physical_frame(addr: usize);
}

pub trait ContextSwitch {
    // Save current register state, restore `next`, jump.
    unsafe fn switch(current: &mut TaskContext, next: &TaskContext);
}

// Placeholder types — we'll flesh these out in later phases.
pub struct PageFlags(pub u64);
pub struct TaskContext {
    _placeholder: u64,
}
