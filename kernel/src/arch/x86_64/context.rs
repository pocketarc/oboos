//! Saved CPU context for task switching on x86_64.
//!
//! When we switch between tasks, we need to save the current CPU register
//! state and restore the next task's state. On x86_64, the System V ABI
//! divides registers into two groups:
//!
//! - **Caller-saved** (rax, rcx, rdx, rsi, rdi, r8–r11): The compiler
//!   already saves these before any function call. By the time our
//!   `switch()` function runs, they're safely on the stack.
//!
//! - **Callee-saved** (rbx, rbp, r12–r15): These must be preserved across
//!   function calls. Our switch routine is responsible for saving and
//!   restoring them.
//!
//! The instruction pointer (RIP) is implicit — it's the return address
//! sitting on the stack. When we `ret` from the switch routine, the CPU
//! pops it automatically.
//!
//! The `#[repr(C)]` layout guarantees field order matches the struct
//! declaration, which our assembly will depend on for known offsets:
//! `rsp` at offset 0, `rbx` at 8, `rbp` at 16, etc.

/// Saved register state for a suspended task.
///
/// Contains only the callee-saved registers plus the stack pointer.
/// The System V AMD64 ABI requires callees to preserve rbx, rbp, and
/// r12–r15; everything else is either caller-saved (handled by the
/// compiler) or implicit (RIP is the return address on the stack).
///
/// # Layout
///
/// `#[repr(C)]` ensures predictable field offsets for the assembly
/// context switch routine:
///
/// | Offset | Field |
/// |--------|-------|
/// | 0      | rsp   |
/// | 8      | rbx   |
/// | 16     | rbp   |
/// | 24     | r12   |
/// | 32     | r13   |
/// | 40     | r14   |
/// | 48     | r15   |
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TaskContext {
    /// Stack pointer — the single most important register for switching.
    /// Loading a different RSP effectively switches the entire execution
    /// context, since all the other saved state lives on the stack.
    pub rsp: u64,
    pub rbx: u64,
    /// Base pointer. Zeroed for new tasks so debuggers see a clean end
    /// to the call chain instead of chasing garbage frame pointers.
    pub rbp: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
}

impl TaskContext {
    /// A zeroed context — all registers set to 0.
    ///
    /// Used for the bootstrap task (whose real register values will be
    /// saved on the first context switch) and as the base for new tasks
    /// (whose `rsp` is then set to their prepared stack frame).
    pub const fn zero() -> Self {
        Self {
            rsp: 0,
            rbx: 0,
            rbp: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
        }
    }
}
