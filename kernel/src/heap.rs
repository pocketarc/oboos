//! Kernel heap allocator.
//!
//! Provides a [`#[global_allocator]`] so the kernel can use `Vec`, `Box`,
//! `String`, and other types from `extern crate alloc`.
//!
//! The backing memory is a static array in `.bss` — Limine zeros BSS before
//! jumping to `kmain`, so the memory is ready to use after a single [`init`]
//! call. We use the [`linked_list_allocator`] crate, which maintains a free
//! list *inside* the heap memory itself (no external metadata). Its
//! [`LockedHeap`] wrapper provides a spinlock-protected [`GlobalAlloc`]
//! implementation.
//!
//! The heap starts at 256 KiB. Once we have virtual memory management
//! (page table manipulation), we can grow it dynamically by mapping
//! additional frames and calling `Heap::extend()`.
//!
//! **Important:** Interrupt handlers must never allocate from the heap.
//! The allocator's spinlock is not reentrant — if an interrupt fires while
//! the lock is held and the handler tries to allocate, it will deadlock.

use linked_list_allocator::LockedHeap;

/// Heap size in bytes. 256 KiB is plenty for a kernel that doesn't yet
/// have userspace or complex subsystems. Bump this constant if you hit
/// OOM panics during development.
const HEAP_SIZE: usize = 256 * 1024;

/// The backing storage for the kernel heap. Lives in `.bss`, so it costs
/// zero bytes in the kernel binary on disk — the bootloader allocates and
/// zeros the memory at load time.
///
/// The `repr(align(16))` wrapper ensures the array starts at a 16-byte
/// boundary, which is the maximum fundamental alignment on x86_64. This
/// avoids wasting the first few bytes to internal alignment padding.
#[repr(align(16))]
struct Aligned([u8; HEAP_SIZE]);

static mut HEAP_SPACE: Aligned = Aligned([0u8; HEAP_SIZE]);

/// Global allocator instance. Created empty (no backing memory) at compile
/// time via `const fn`, then initialized in [`init`] with the BSS array.
#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

/// Initialize the kernel heap.
///
/// Must be called exactly once during boot, after BSS is zeroed (which
/// Limine guarantees before `kmain`). After this returns, `alloc` types
/// (`Box`, `Vec`, `String`, etc.) are usable.
pub fn init() {
    let (heap_start, heap_size) = unsafe {
        let ptr = core::ptr::addr_of_mut!(HEAP_SPACE) as *mut u8;
        ALLOCATOR.lock().init(ptr, HEAP_SIZE);
        (ptr as usize, HEAP_SIZE)
    };

    crate::println!(
        "[heap] Initialized: {} KiB at {:#X}",
        heap_size / 1024,
        heap_start,
    );
}
