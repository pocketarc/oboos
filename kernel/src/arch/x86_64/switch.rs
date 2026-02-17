//! Context switch assembly for x86_64.
//!
//! This is the beating heart of multitasking: 12 instructions that save one
//! task's execution state and restore another's. The trick exploits how
//! `call`/`ret` work — every `call` pushes a return address onto the stack,
//! every `ret` pops one. By swapping which stack RSP points to between the
//! save and restore, `ret` jumps into a completely different task's code.
//!
//! We only save **callee-saved** registers (rbx, rbp, r12–r15). The System V
//! AMD64 ABI guarantees the compiler already preserves caller-saved registers
//! (rax, rcx, rdx, rsi, rdi, r8–r11) before calling us. RIP is implicit —
//! it's the return address sitting on the stack.
//!
//! For a **new task**, [`Task::new()`] pre-builds a fake stack frame matching
//! this layout: zeroed registers with the entry point as the return address.
//! Our `ret` pops the entry point into RIP and the task starts running.
//!
//! For a **resumed task**, the frame was built by a previous `switch_context`
//! call. The `ret` returns to wherever that task last called into the switch.

use core::arch::naked_asm;

/// Swap execution from one task to another.
///
/// Saves callee-saved registers onto `current`'s stack, stores its RSP,
/// loads `next`'s RSP, restores callee-saved registers from `next`'s stack,
/// and `ret`s into `next`'s code.
///
/// # ABI
///
/// `extern "C"` ensures System V AMD64 calling convention:
/// - `rdi` = pointer to current task's [`TaskContext`] (first argument)
/// - `rsi` = pointer to next task's [`TaskContext`] (second argument)
///
/// Only [`TaskContext::rsp`] (offset 0) is accessed by the assembly.
/// The other fields in `TaskContext` exist for debuggability — the actual
/// register values live on the stack.
///
/// # Safety
///
/// Both pointers must point to valid, initialized [`TaskContext`] values.
/// `next`'s RSP must point to a valid stack with the expected frame layout
/// (6 callee-saved registers + return address). Getting this wrong will
/// corrupt the stack or jump to garbage.
///
/// [`TaskContext`]: super::context::TaskContext
#[unsafe(naked)]
pub(super) unsafe extern "C" fn switch_context(
    _current: *mut u64,
    _next: *const u64,
) {
    naked_asm!(
        // Save callee-saved registers onto the current task's stack.
        // Push order matches Task::new()'s initial frame layout:
        // rbx deepest, r15 at the top (lowest address).
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",

        // Save the current stack pointer into current.rsp (offset 0).
        "mov [rdi], rsp",

        // THE SWITCH: load the next task's stack pointer.
        // From this point on, every stack operation (push/pop/ret)
        // operates on the *next* task's stack.
        "mov rsp, [rsi]",

        // Restore callee-saved registers from the next task's stack.
        // Pop order is the reverse of push — r15 first (it's on top).
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",

        // Pop the return address into RIP. For a new task, this is the
        // entry point placed by Task::new(). For a resumed task, it's
        // wherever it last called switch_context from.
        "ret",
    );
}
