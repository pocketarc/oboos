//! Task management for kernel multitasking.
//!
//! Each task represents an independent thread of execution in the kernel,
//! with its own stack and saved CPU context. This module handles:
//!
//! - **Task creation**: Allocating a kernel stack, building an initial
//!   stack frame so the context switch can "return" into the entry point.
//! - **Bootstrap task**: Wrapping `kmain`'s existing execution context
//!   as task 0 — no stack allocation needed since it's already running.
//! - **RAII cleanup**: The [`Drop`] impl unmaps and frees stack frames
//!   when a task is destroyed.
//!
//! This module does NOT include scheduling or context switching — those
//! come in Phase 2b and 2c.

use alloc::boxed::Box;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::arch::context::FpuState;
use crate::arch::{self, TaskContext};
use crate::memory::{self, FRAME_SIZE};
use crate::platform::{MemoryManager, PageFlags};
use crate::println;

// ————————————————————————————————————————————————————————————————————————————
// Constants
// ————————————————————————————————————————————————————————————————————————————

/// Base virtual address for kernel task stacks. Sits between the HHDM
/// (`0xFFFF_8000_0000_0000`) and the kernel image (`0xFFFF_FFFF_8000_0000`),
/// giving us a large, unused region of virtual address space.
const STACK_REGION_BASE: usize = 0xFFFF_FE00_0000_0000;

/// Number of 4 KiB pages per task stack (16 KiB total).
/// Linux kernel stacks are typically 8–16 KiB; 16 KiB gives us comfortable
/// headroom for nested function calls and interrupt frames.
const STACK_PAGES: usize = 4;

/// Pages per task slot: 1 guard page + [`STACK_PAGES`].
/// The guard page sits below the stack and is intentionally left unmapped.
/// If the stack overflows into it, the CPU triggers a page fault instead
/// of silently corrupting whatever is below — an essential safety net.
const SLOT_PAGES: usize = STACK_PAGES + 1;

/// Number of u64 values in the initial stack frame: 6 callee-saved
/// registers (rbx, rbp, r12–r15), the [`task_trampoline`](arch::task_trampoline)
/// (which enables interrupts before falling through), and the entry point.
const INITIAL_FRAME_SLOTS: usize = 8;

// ————————————————————————————————————————————————————————————————————————————
// Statics
// ————————————————————————————————————————————————————————————————————————————

/// Next task ID to assign. Starts at 1 because 0 is reserved for the
/// bootstrap task (`kmain`'s existing execution context).
static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

/// Next stack slot to allocate. Each slot is a contiguous 5-page region
/// in virtual address space. Simple bump allocator — no reuse on task death.
static NEXT_SLOT: AtomicU64 = AtomicU64::new(0);

// ————————————————————————————————————————————————————————————————————————————
// Types
// ————————————————————————————————————————————————————————————————————————————

/// A unique identifier for a task. Monotonically increasing, never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskId(pub u64);

/// Lifecycle state of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Runnable — waiting in the ready queue for CPU time.
    Ready,
    /// Currently executing on the CPU.
    Running,
    /// Waiting for an event (I/O, timer, etc.). Not in the ready queue.
    Blocked,
    /// Finished execution. Resources can be reclaimed.
    Dead,
}

/// Tracks the physical frames and virtual address range of a task's kernel
/// stack. Kept private — callers interact with stacks through [`Task`] methods.
struct StackAllocation {
    /// Virtual address of the first mapped page (just above the guard page).
    stack_bottom: usize,
    /// Virtual address one past the last mapped byte — the initial stack
    /// pointer for an empty stack (x86 stacks grow downward).
    stack_top: usize,
    /// Physical frames backing the stack pages. We store these so [`Drop`]
    /// can return them to the frame allocator.
    frames: [usize; STACK_PAGES],
}

/// A kernel task — an independent thread of execution with its own stack
/// and saved CPU context.
pub struct Task {
    pub id: TaskId,
    pub state: TaskState,
    pub context: TaskContext,
    stack: Option<StackAllocation>,
}

// ————————————————————————————————————————————————————————————————————————————
// Task implementation
// ————————————————————————————————————————————————————————————————————————————

impl Task {
    /// Create the bootstrap task representing `kmain`'s current execution.
    ///
    /// This doesn't allocate a stack — `kmain` is already running on the
    /// boot stack provided by the bootloader. The zeroed context will be
    /// filled with real register values on the first context switch.
    pub fn bootstrap() -> Self {
        let mut context = TaskContext::zero();
        context.fpu_state = Box::into_raw(Box::new(FpuState::new()));
        Self {
            id: TaskId(0),
            state: TaskState::Running,
            context,
            stack: None,
        }
    }

    /// Create a new task that will begin executing at `entry_point`.
    ///
    /// Allocates a 16 KiB kernel stack (with guard page) and builds an
    /// initial stack frame so that the context switch assembly can
    /// "return" into the entry point. The entry function must never return
    /// (`-> !`) — there's nothing valid to return to.
    ///
    /// The initial stack frame looks like this (stack grows downward):
    ///
    /// ```text
    /// stack_top:
    ///   entry_point       <- trampoline's ret target
    ///   task_trampoline   <- switch_context's ret target (enables IF)
    ///   0 (rbx)           <- popped by switch assembly
    ///   0 (rbp)
    ///   0 (r12)
    ///   0 (r13)
    ///   0 (r14)
    ///   0 (r15)           <- initial RSP points here
    /// ```
    ///
    /// The trampoline executes `sti; ret` — enabling interrupts so the PIT
    /// can preempt this task, then falling through into the real entry point.
    pub fn new(entry_point: fn() -> !) -> Self {
        let id = TaskId(NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed));
        let slot = NEXT_SLOT.fetch_add(1, Ordering::Relaxed) as usize;
        let alloc = allocate_stack(slot);

        // Place the entry point and trampoline at the top of the stack.
        // switch_context's `ret` pops the trampoline, which does `sti; ret`
        // to enable interrupts and pop the entry point into RIP.
        // The 6 register slots below are already zero from page zeroing.
        unsafe {
            let entry_ptr = (alloc.stack_top as *mut u64).sub(1);
            core::ptr::write(entry_ptr, entry_point as *const () as u64);
            let tramp_ptr = (alloc.stack_top as *mut u64).sub(2);
            core::ptr::write(tramp_ptr, arch::task_trampoline as *const () as u64);
        }

        let mut context = TaskContext {
            rsp: (alloc.stack_top - INITIAL_FRAME_SLOTS * 8) as u64,
            ..TaskContext::zero()
        };
        context.fpu_state = Box::into_raw(Box::new(FpuState::new()));

        Self {
            id,
            state: TaskState::Ready,
            context,
            stack: Some(alloc),
        }
    }

    /// Returns the top of this task's kernel stack, or `None` for the
    /// bootstrap task (which uses the bootloader-provided stack).
    pub fn stack_top(&self) -> Option<usize> {
        self.stack.as_ref().map(|a| a.stack_top)
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        if !self.context.fpu_state.is_null() {
            unsafe { drop(Box::from_raw(self.context.fpu_state)); }
        }
        if let Some(ref alloc) = self.stack {
            for i in 0..STACK_PAGES {
                let virt = alloc.stack_bottom + i * FRAME_SIZE;
                arch::Arch::unmap_page(virt);
                memory::free_frame(alloc.frames[i]);
            }
            println!(
                "[task] Freed stack for task {} ({:#018X}..{:#018X})",
                self.id.0, alloc.stack_bottom, alloc.stack_top,
            );
        }
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Stack allocation
// ————————————————————————————————————————————————————————————————————————————

/// Allocate and map a kernel stack for a new task.
///
/// Each task gets a contiguous 5-page virtual address slot:
/// - Page 0: unmapped guard page (stack overflow -> page fault)
/// - Pages 1–4: mapped stack pages (PRESENT | WRITABLE | NO_EXECUTE)
///
/// The pages are zeroed after mapping to avoid leaking stale data from
/// recycled frames.
fn allocate_stack(slot: usize) -> StackAllocation {
    let slot_base = STACK_REGION_BASE + slot * SLOT_PAGES * FRAME_SIZE;
    let stack_bottom = slot_base + FRAME_SIZE;
    let stack_top = slot_base + SLOT_PAGES * FRAME_SIZE;

    let flags = PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::NO_EXECUTE;
    let mut frames = [0usize; STACK_PAGES];

    for i in 0..STACK_PAGES {
        let frame = memory::alloc_frame().expect("out of memory allocating task stack");
        let virt = stack_bottom + i * FRAME_SIZE;
        arch::Arch::map_page(virt, frame, flags);
        frames[i] = frame;
    }

    // Zero all stack pages through the newly established virtual mapping.
    unsafe {
        core::ptr::write_bytes(stack_bottom as *mut u8, 0, STACK_PAGES * FRAME_SIZE);
    }

    println!(
        "[task] Allocated stack: {:#018X}..{:#018X} (guard at {:#018X})",
        stack_bottom, stack_top, slot_base,
    );

    StackAllocation {
        stack_bottom,
        stack_top,
        frames,
    }
}
