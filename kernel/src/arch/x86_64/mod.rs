//! x86_64 platform implementation.

pub mod gdt;
pub mod interrupts;
pub mod keyboard;
pub mod memory;
pub mod paging;
pub mod pic;
pub mod pit;
pub mod port;
pub mod serial;

// Re-export arch types under neutral names so the kernel can use
// `arch::Serial` and `arch::KeyboardDriver` without reaching into
// submodules. When aarch64 is added, it re-exports its own types
// under the same names.
pub type Arch = X86_64;
pub use serial::Serial;
pub use keyboard::Ps2Keyboard as KeyboardDriver;

use crate::platform::{MemoryManager, PageFlags, Platform, SerialConsole};

/// x86_64 platform — implements [`Platform`] for 64-bit Intel/AMD.
pub struct X86_64;

impl Platform for X86_64 {
    fn init() -> Self {
        serial::Serial::init();
        gdt::init();
        interrupts::init();

        // Discover physical memory and initialize the frame allocator.
        // This must happen before anything that needs to allocate frames,
        // but doesn't depend on timers or keyboard.
        memory::init();

        // Program the PIT for 1 kHz, then register its tick handler.
        // This order matters: configure the hardware before unmasking
        // its IRQ, so we don't get an interrupt before we're ready.
        pit::init(1000);
        interrupts::set_irq_handler(0, pit::tick);

        X86_64
    }

    /// `sti` — Set Interrupt Flag. The CPU will start responding to
    /// hardware interrupts after the *next* instruction completes.
    fn enable_interrupts() {
        unsafe { core::arch::asm!("sti") };
    }

    /// `cli` — Clear Interrupt Flag. Masks all maskable interrupts.
    /// Used to protect critical sections where we can't be preempted.
    fn disable_interrupts() {
        unsafe { core::arch::asm!("cli") };
    }

    /// `hlt` — Halt the CPU until the next interrupt fires.
    fn halt_until_interrupt() {
        unsafe { core::arch::asm!("hlt") };
    }

    /// Read the CPU's Time Stamp Counter as an entropy source.
    ///
    /// The TSC increments every clock cycle, so its exact value at the
    /// moment a human presses a key is unpredictable — good enough for
    /// non-cryptographic randomness like picking a color.
    fn entropy() -> u64 {
        let lo: u32;
        let hi: u32;
        unsafe {
            core::arch::asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack));
        }
        ((hi as u64) << 32) | (lo as u64)
    }

    /// Trigger a divide-by-zero exception (#DE, vector 0).
    ///
    /// Without an IDT, this will triple-fault the CPU — the processor tries
    /// to call the #DE handler, finds no IDT, faults again (#DF), fails to
    /// find that handler too, and gives up with a triple fault (machine reset).
    /// With `-no-reboot` in QEMU, you'll see it freeze instead.
    ///
    /// Once an IDT is set up with a #DE handler, this same call will invoke
    /// that handler instead.
    fn trigger_test_fault() -> ! {
        unsafe {
            core::arch::asm!(
                "xor rdx, rdx", // clear upper half of dividend
                "xor rax, rax", // dividend = 0
                "xor rcx, rcx", // divisor = 0
                "div rcx",      // RDX:RAX / RCX → #DE
                options(noreturn, nomem, nostack),
            );
        }
    }
}

impl MemoryManager for X86_64 {
    fn alloc_physical_frame() -> Option<usize> {
        crate::memory::alloc_frame()
    }

    fn free_physical_frame(addr: usize) {
        crate::memory::free_frame(addr)
    }

    fn map_page(virt: usize, phys: usize, flags: PageFlags) {
        paging::map_page(virt, phys, flags);
    }

    fn unmap_page(virt: usize) {
        paging::unmap_page(virt);
    }
}
