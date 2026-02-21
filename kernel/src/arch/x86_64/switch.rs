//! Context switch assembly for x86_64.
//!
//! This is the beating heart of multitasking: save one task's execution state
//! and restore another's. The trick exploits how `call`/`ret` work — every
//! `call` pushes a return address onto the stack, every `ret` pops one. By
//! swapping which stack RSP points to between the save and restore, `ret`
//! jumps into a completely different task's code.
//!
//! We save **callee-saved** registers (rbx, rbp, r12–r15) onto the stack,
//! plus the FPU/SSE state via `fxsave64`/`fxrstor64` into a per-task buffer.
//! The System V AMD64 ABI guarantees the compiler already preserves
//! caller-saved registers (rax, rcx, rdx, rsi, rdi, r8–r11) before calling
//! us. RIP is implicit — it's the return address sitting on the stack.
//!
//! For a **new task**, [`Task::new()`] pre-builds a fake stack frame matching
//! this layout: zeroed registers with the entry point as the return address.
//! Our `ret` pops the entry point into RIP and the task starts running.
//!
//! For a **resumed task**, the frame was built by a previous `switch_context`
//! call. The `ret` returns to wherever that task last called into the switch.

use core::arch::naked_asm;

/// Entry trampoline for newly spawned tasks.
///
/// When [`switch_context`] first switches to a new task, its `ret` pops
/// this function's address. We enable interrupts (so the PIT can
/// preempt this task) and `ret` again into the real entry point.
///
/// Without this trampoline, new tasks would start with IF=0 (because
/// the scheduler disables interrupts before switching) and could never
/// be preempted by the timer.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn task_trampoline() {
    naked_asm!("sti", "ret");
}

/// Swap execution from one task to another.
///
/// Saves FPU/SSE state (via `fxsave64`) and callee-saved registers onto
/// `current`'s stack, stores its RSP, loads `next`'s RSP, restores
/// callee-saved registers and FPU/SSE state (via `fxrstor64`) from
/// `next`'s stack, and `ret`s into `next`'s code.
///
/// # ABI
///
/// `extern "C"` ensures System V AMD64 calling convention:
/// - `rdi` = pointer to current task's [`TaskContext`] (first argument)
/// - `rsi` = pointer to next task's [`TaskContext`] (second argument)
///
/// The assembly accesses `TaskContext` fields at known offsets:
/// - offset 0: `rsp`
/// - offset 56: `fpu_state` pointer
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
        // Save current task's FPU/SSE state if fpu_state pointer is non-null.
        // The pointer lives at [rdi+56] (TaskContext::fpu_state).
        "mov rax, [rdi + 56]",
        "test rax, rax",
        "jz 2f",
        "fxsave64 [rax]",
        "2:",

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

        // Restore next task's FPU/SSE state if fpu_state pointer is non-null.
        // RSI still holds the next task's TaskContext pointer (caller-saved,
        // not modified by our push/pop sequence).
        "mov rax, [rsi + 56]",
        "test rax, rax",
        "jz 3f",
        "fxrstor64 [rax]",
        "3:",

        // Pop the return address into RIP. For a new task, this is the
        // entry point placed by Task::new(). For a resumed task, it's
        // wherever it last called switch_context from.
        "ret",
    );
}
