//! SYSCALL/SYSRET support for x86_64.
//!
//! SYSCALL is the fast user→kernel transition on x86_64. Unlike `int 0x80`
//! (which goes through the IDT, pushes an interrupt frame, and does privilege
//! checks), SYSCALL is a single instruction that swaps CS/SS, saves
//! RIP→RCX and RFLAGS→R11, and jumps to a fixed address (LSTAR). It's
//! roughly 4x faster than a software interrupt because it skips all the
//! descriptor-table lookups.
//!
//! The return path is SYSRET, which reverses the process: loads CS/SS from
//! STAR, restores RIP from RCX and RFLAGS from R11, and drops back to Ring 3.
//!
//! ## MSR configuration
//!
//! Four Model-Specific Registers control the SYSCALL mechanism:
//! - **IA32_EFER** (0xC0000080): the SCE (System Call Enable) bit must be set
//! - **IA32_STAR** (0xC0000081): segment selectors for SYSCALL and SYSRET
//! - **IA32_LSTAR** (0xC0000082): the kernel entry point address
//! - **IA32_FMASK** (0xC0000084): RFLAGS bits to clear on entry (we clear IF)
//!
//! ## Register convention
//!
//! User programs pass syscall arguments in registers following the Linux
//! convention (RAX=number, RDI/RSI/RDX/R10=args). The entry stub shuffles
//! these into the System V AMD64 calling convention before calling the
//! Rust handler:
//!
//! ```text
//! User register → Handler parameter
//! RAX (number)  → RDI
//! RDI (arg1)    → RSI
//! RSI (arg2)    → RDX
//! RDX (arg3)    → RCX
//! R10 (arg4)    → R8
//! ```
//!
//! The handler returns a `u64` in RAX which flows back to the user
//! untouched via `sysretq`.
//!
//! ## Critical SYSRET pitfall
//!
//! On AMD CPUs, SYSRET checks whether RCX (the return RIP) is canonical
//! *before* switching back to Ring 3. If RCX is non-canonical, the CPU
//! raises #GP **at Ring 0** — a kernel-mode fault triggered by a user-
//! controlled register. Linux mitigates this by checking RCX before SYSRET.
//! Our user code is at 0x40_0000 (canonical), so we're safe for now.

use core::arch::naked_asm;

use crate::platform::SerialConsole;
use crate::store::{self, StoreId};
use oboos_api::Value;

// ————————————————————————————————————————————————————————————————————————————
// MSR addresses
// ————————————————————————————————————————————————————————————————————————————

const IA32_EFER: u32 = 0xC000_0080;
const IA32_STAR: u32 = 0xC000_0081;
const IA32_LSTAR: u32 = 0xC000_0082;
const IA32_FMASK: u32 = 0xC000_0084;

/// SCE — System Call Enable bit in IA32_EFER.
const EFER_SCE: u64 = 1 << 0;

// ————————————————————————————————————————————————————————————————————————————
// Syscall numbers
// ————————————————————————————————————————————————————————————————————————————

const SYS_EXIT: u64 = 0;
const SYS_WRITE: u64 = 1;
const SYS_STORE_GET: u64 = 2;
const SYS_STORE_SET: u64 = 3;
const SYS_GETPID: u64 = 4;

// ————————————————————————————————————————————————————————————————————————————
// MSR helpers
// ————————————————————————————————————————————————————————————————————————————

/// Read a Model-Specific Register.
///
/// MSRs are CPU configuration registers addressed by a 32-bit index.
/// `rdmsr` loads ECX with the index and returns the 64-bit value split
/// across EDX:EAX (high:low) — a convention from the 32-bit era that
/// persists in 64-bit mode.
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

/// Write a Model-Specific Register.
///
/// Same EDX:EAX split as `rdmsr`, just in the write direction.
unsafe fn wrmsr(msr: u32, value: u64) {
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    unsafe {
        core::arch::asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") lo,
            in("edx") hi,
            options(nomem, nostack, preserves_flags),
        );
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Syscall entry point state
// ————————————————————————————————————————————————————————————————————————————

/// Saved user RSP. SYSCALL doesn't touch RSP — it's our job to swap stacks.
/// Written by `syscall_entry` before loading the kernel stack. Single-CPU
/// with IF=0 on entry, so no races.
static mut SAVED_USER_RSP: u64 = 0;

/// Kernel stack pointer loaded by `syscall_entry`. Set by [`set_kernel_rsp`]
/// before entering Ring 3.
static mut KERNEL_RSP: u64 = 0;

/// Saved kernel context for the SYS_EXIT return path. When the user program
/// calls SYS_EXIT, the handler restores these registers and RSP to return
/// control to whoever called [`jump_to_ring3`].
///
/// Layout: `[rsp, rbx, rbp, r12, r13, r14, r15]` — same callee-saved
/// register set as our context switch, stored via `mov` (not `push`) so
/// we don't disturb RSP during the save.
static mut RETURN_CONTEXT: [u64; 7] = [0; 7];

// ————————————————————————————————————————————————————————————————————————————
// Initialization
// ————————————————————————————————————————————————————————————————————————————

/// Configure the SYSCALL/SYSRET mechanism.
///
/// Sets up the four MSRs that control fast system calls. After this,
/// executing `syscall` in Ring 3 will jump to [`syscall_entry`] with
/// kernel CS/SS loaded and interrupts disabled (IF cleared by FMASK).
pub fn init() {
    unsafe {
        // Enable SYSCALL/SYSRET by setting the SCE bit in EFER.
        // EFER also controls NXE (no-execute) and LME (long mode enable),
        // both already set by Limine — we preserve them with read-modify-write.
        let efer = rdmsr(IA32_EFER);
        wrmsr(IA32_EFER, efer | EFER_SCE);

        // STAR register layout:
        //
        //   Bits 47:32 — SYSCALL CS/SS selectors
        //     SYSCALL loads: CS = this value       (0x08 = kernel code)
        //                    SS = this value + 8   (0x10 = kernel data)
        //
        //   Bits 63:48 — SYSRET CS/SS base selector
        //     For 64-bit SYSRET the CPU loads:
        //       SS = base + 8,  with RPL forced to 3  → 0x18 | 3 = 0x1B (user data)
        //       CS = base + 16, with RPL forced to 3  → 0x20 | 3 = 0x23 (user code)
        //     So the base must be 0x10 (GDT index 2, kernel data) for the
        //     arithmetic to land on our user data (index 3) and user code (index 4).
        let star = (0x0010u64 << 48) | (0x0008u64 << 32);
        wrmsr(IA32_STAR, star);

        // LSTAR — the 64-bit entry point for SYSCALL.
        wrmsr(IA32_LSTAR, syscall_entry as *const () as u64);

        // FMASK — RFLAGS bits to clear on SYSCALL entry. We clear IF (bit 9)
        // so interrupts are disabled when we enter the kernel. This gives us
        // a safe window to swap from user RSP to kernel RSP before an interrupt
        // could fire and push a frame onto the wrong stack.
        wrmsr(IA32_FMASK, 0x200);
    }

    crate::println!("[ok] SYSCALL/SYSRET configured");
}

// ————————————————————————————————————————————————————————————————————————————
// Pointer validation
// ————————————————————————————————————————————————————————————————————————————

/// Check that a user-provided pointer + length is within user-space canonical
/// addresses. This prevents the user from tricking the kernel into reading or
/// writing kernel memory through a syscall.
///
/// The x86_64 canonical address split: user addresses are below
/// 0x0000_8000_0000_0000, kernel addresses are above 0xFFFF_8000_0000_0000.
/// We reject null pointers, empty ranges, and any range that extends past
/// the user-space boundary.
fn validate_user_ptr(ptr: u64, len: u64) -> bool {
    const USER_SPACE_END: u64 = 0x0000_8000_0000_0000;

    if ptr == 0 || len == 0 {
        return false;
    }

    // Check for overflow and that the entire range stays in user space.
    match ptr.checked_add(len) {
        Some(end) => end <= USER_SPACE_END,
        None => false,
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Syscall entry (assembly)
// ————————————————————————————————————————————————————————————————————————————

/// Naked assembly entry point for SYSCALL.
///
/// On entry from SYSCALL:
/// - RCX = user RIP (return address)
/// - R11 = user RFLAGS
/// - IF = 0 (cleared by FMASK)
/// - CS/SS = kernel selectors (loaded by SYSCALL hardware)
/// - RSP = **unchanged** (still the user stack!)
/// - RAX = syscall number
/// - RDI = arg1, RSI = arg2, RDX = arg3, R10 = arg4
///
/// We must immediately swap to a kernel stack before doing anything that
/// touches the stack (like calling a Rust function).
///
/// The register shuffle maps the Linux syscall convention to System V
/// AMD64 calling convention for the Rust handler. R9 is used as scratch
/// because the 4-register rotation (RAX→RDI→RSI→RDX) has a cycle that
/// needs breaking:
///
/// ```text
/// mov r9, rdi      ; save arg1 (RDI is about to be overwritten)
/// mov rdi, rax     ; number → 1st param
/// mov r8, r10      ; arg4 → 5th param (no conflict)
/// mov rcx, rdx     ; arg3 → 4th param (must happen before RDX overwritten)
/// mov rdx, rsi     ; arg2 → 3rd param
/// mov rsi, r9      ; arg1 → 2nd param
/// ```
#[unsafe(naked)]
unsafe extern "C" fn syscall_entry() {
    naked_asm!(
        // Save user RSP and load kernel RSP. These two instructions run
        // with IF=0, so no interrupt can fire between them.
        "mov [{saved_user_rsp}], rsp",
        "mov rsp, [{kernel_rsp}]",

        // Save user return state on the kernel stack.
        "push rcx",          // user RIP (SYSCALL saved it in RCX)
        "push r11",          // user RFLAGS (SYSCALL saved it in R11)

        // Set up a stack frame for the Rust handler.
        "push rbp",
        "mov rbp, rsp",

        // Shuffle registers: Linux syscall convention → System V ABI.
        "mov r9, rdi",       // save user arg1
        "mov rdi, rax",      // syscall number → param 1
        "mov r8, r10",       // arg4 → param 5
        "mov rcx, rdx",      // arg3 → param 4 (before rdx overwritten)
        "mov rdx, rsi",      // arg2 → param 3
        "mov rsi, r9",       // arg1 → param 2

        "call {handler}",

        // RAX now holds the return value from the handler. It flows
        // back to the user program untouched via sysretq.

        // Restore frame and user return state.
        "pop rbp",
        "pop r11",           // user RFLAGS → R11 for SYSRET
        "pop rcx",           // user RIP → RCX for SYSRET

        // Restore user RSP and return to Ring 3.
        "mov rsp, [{saved_user_rsp}]",
        "sysretq",

        saved_user_rsp = sym SAVED_USER_RSP,
        kernel_rsp = sym KERNEL_RSP,
        handler = sym syscall_handler,
    );
}

// ————————————————————————————————————————————————————————————————————————————
// Syscall handler (Rust)
// ————————————————————————————————————————————————————————————————————————————

/// Dispatch syscalls by number.
///
/// Called from [`syscall_entry`] with interrupts disabled (FMASK clears IF).
/// Arguments arrive in System V order after the entry stub's register shuffle.
///
/// ## Syscall table
///
/// | Number | Name          | Args                          | Returns          |
/// |--------|---------------|-------------------------------|------------------|
/// | 0      | SYS_EXIT      | exit_code                     | never returns    |
/// | 1      | SYS_WRITE     | buf_ptr, buf_len              | bytes written    |
/// | 2      | SYS_STORE_GET | store_id, field_ptr, field_len | u64 value        |
/// | 3      | SYS_STORE_SET | store_id, field_ptr, field_len, value | 0 or error |
/// | 4      | SYS_GETPID    | —                             | current PID      |
extern "C" fn syscall_handler(number: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64) -> u64 {
    match number {
        SYS_EXIT => {
            let exit_code = arg1;
            crate::process::exit(crate::process::current_pid(), exit_code);
            unsafe { restore_return_context(); }
        },

        SYS_GETPID => {
            crate::process::current_pid().as_raw()
        },

        SYS_WRITE => {
            let buf_ptr = arg1;
            let buf_len = arg2;

            if !validate_user_ptr(buf_ptr, buf_len) {
                return u64::MAX; // error: bad pointer
            }

            let slice = unsafe {
                core::slice::from_raw_parts(buf_ptr as *const u8, buf_len as usize)
            };

            for &b in slice {
                crate::arch::Serial::write_byte(b);
            }
            buf_len
        }

        SYS_STORE_GET => {
            let store_id = arg1;
            let field_ptr = arg2;
            let field_len = arg3;

            if !validate_user_ptr(field_ptr, field_len) {
                return u64::MAX;
            }

            let field_bytes = unsafe {
                core::slice::from_raw_parts(field_ptr as *const u8, field_len as usize)
            };
            let field_name = match core::str::from_utf8(field_bytes) {
                Ok(s) => s,
                Err(_) => return u64::MAX,
            };

            let id = StoreId::from_raw(store_id);
            match store::get_no_cli(id, field_name) {
                Ok(Value::U64(v)) => v,
                _ => u64::MAX,
            }
        }

        SYS_STORE_SET => {
            let store_id = arg1;
            let field_ptr = arg2;
            let field_len = arg3;
            let value = arg4;

            if !validate_user_ptr(field_ptr, field_len) {
                return u64::MAX;
            }

            let field_bytes = unsafe {
                core::slice::from_raw_parts(field_ptr as *const u8, field_len as usize)
            };
            let field_name = match core::str::from_utf8(field_bytes) {
                Ok(s) => s,
                Err(_) => return u64::MAX,
            };

            let id = StoreId::from_raw(store_id);
            match store::set_no_cli(id, &[(field_name, Value::U64(value))]) {
                Ok(()) => 0,
                Err(_) => u64::MAX,
            }
        }

        _ => {
            crate::println!("[syscall] unknown: {}", number);
            u64::MAX
        }
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Return context restore
// ————————————————————————————————————————————————————————————————————————————

/// Restore callee-saved registers and RSP from [`RETURN_CONTEXT`], then `ret`.
///
/// This is the "longjmp" half of the SYS_EXIT mechanism. It restores the
/// register state saved by [`jump_to_ring3`] and executes `ret`, which pops
/// the return address that was on the stack when `jump_to_ring3` was called
/// (pushed by the `call` instruction). Execution resumes in the caller of
/// `jump_to_ring3` as if it returned normally.
///
/// Never returns to its own caller.
unsafe fn restore_return_context() -> ! {
    unsafe {
        core::arch::asm!(
            "mov r15, [{ctx} + 48]",
            "mov r14, [{ctx} + 40]",
            "mov r13, [{ctx} + 32]",
            "mov r12, [{ctx} + 24]",
            "mov rbp, [{ctx} + 16]",
            "mov rbx, [{ctx} + 8]",
            "mov rsp, [{ctx} + 0]",
            "ret",
            ctx = sym RETURN_CONTEXT,
            options(noreturn),
        );
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Ring 3 entry
// ————————————————————————————————————————————————————————————————————————————

/// Set the kernel RSP used by [`syscall_entry`].
///
/// Must be called before entering Ring 3. The syscall entry stub loads
/// this value into RSP immediately after saving the user RSP.
pub fn set_kernel_rsp(rsp: u64) {
    unsafe {
        KERNEL_RSP = rsp;
    }
}

/// Drop to Ring 3 by constructing a fake `iretq` frame.
///
/// This is a naked function because we need precise control over the stack:
/// 1. Save callee-saved registers + RSP to [`RETURN_CONTEXT`] (the "setjmp")
/// 2. Build the iretq frame and execute `iretq` to enter Ring 3
///
/// When the user program calls SYS_EXIT, [`restore_return_context`] loads
/// the saved context and executes `ret`, which pops the return address
/// that the `call jump_to_ring3` instruction pushed. The caller sees
/// `jump_to_ring3` "return" normally.
///
/// ## iretq frame layout (bottom to top on stack)
///
/// ```text
///   SS       — 0x1B (user data selector, RPL=3)
///   RSP      — user stack pointer
///   RFLAGS   — 0x202 (IF=1, reserved bit 1 set)
///   CS       — 0x23 (user code selector, RPL=3)
///   RIP      — user entry point
/// ```
///
/// The third argument (`arg0`) is placed into RDI before `iretq` so the
/// user's `_start(arg0: u64)` receives it per the C calling convention.
/// `iretq` pops SS/RSP/RFLAGS/CS/RIP from the stack but doesn't touch
/// general-purpose registers, so RDI survives the transition.
///
/// # Safety
///
/// - `rip` must point to valid, executable user-mode code mapped with USER.
/// - `rsp` must point to the top of a valid, mapped user stack.
/// - The syscall kernel stack and TSS.RSP0 must already be configured.
/// - SYSCALL MSRs must be initialized ([`init`] must have been called).
///
/// # ABI
///
/// `extern "C"`: RDI = user RIP, RSI = user RSP, RDX = arg0.
#[unsafe(naked)]
pub unsafe extern "C" fn jump_to_ring3(_rip: u64, _rsp: u64, _arg0: u64) {
    naked_asm!(
        // Save callee-saved registers and RSP into RETURN_CONTEXT.
        // Using `mov` (not `push`) so we don't modify RSP during the save.
        // RSP currently points to the return address pushed by `call`.
        "mov [{ctx} + 0], rsp",
        "mov [{ctx} + 8], rbx",
        "mov [{ctx} + 16], rbp",
        "mov [{ctx} + 24], r12",
        "mov [{ctx} + 32], r13",
        "mov [{ctx} + 40], r14",
        "mov [{ctx} + 48], r15",

        // Build iretq frame. RDI = user RIP, RSI = user RSP, RDX = arg0.
        "push 0x1B",         // SS: user data selector (0x18 | RPL=3)
        "push rsi",          // RSP: user stack pointer
        "push 0x202",        // RFLAGS: IF=1 (interrupts enabled in user mode)
        "push 0x23",         // CS: user code selector (0x20 | RPL=3)
        "push rdi",          // RIP: user entry point

        // Pass arg0 to the user's _start function via RDI.
        // iretq doesn't touch general-purpose registers, so this value
        // will be in RDI when user code starts executing.
        "mov rdi, rdx",

        "iretq",

        ctx = sym RETURN_CONTEXT,
    );
}
