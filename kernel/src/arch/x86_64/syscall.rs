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
//! ## Syscall interface
//!
//! Only two syscalls exist — everything is expressed through the store:
//!
//! | # | Name          | Args                                                  | Returns            |
//! |---|---------------|-------------------------------------------------------|--------------------|
//! | 0 | SYS_STORE_GET | store_id, field_ptr, field_len, out_ptr, out_len      | bytes written      |
//! | 1 | SYS_STORE_SET | store_id, field_ptr, field_len, value_ptr, value_len  | 0 on success       |
//!
//! Both return `u64::MAX` on error. The kernel uses the field's [`FieldKind`]
//! from the schema to interpret raw bytes — no type information crosses the
//! syscall boundary.
//!
//! ## Well-known store IDs
//!
//! Bit 63 flags well-known stores resolved per-process by the kernel:
//! - `PROCESS` (1 << 63) — current process's lifecycle store
//! - `CONSOLE` ((1 << 63) | 1) — serial console device store
//!
//! ## Side-effects
//!
//! SET on certain well-known store fields triggers kernel actions:
//! - `CONSOLE`/`"output"` → bytes written to serial inline
//! - `PROCESS`/`"status"` = `"exiting"` → reads `exit_code`, triggers process
//!   exit and longjmp back to [`jump_to_ring3`]'s caller
//!
//! ## Register convention
//!
//! User programs pass 5 syscall arguments in registers following the Linux
//! convention. The entry stub shuffles these into System V AMD64 for the
//! Rust handler:
//!
//! ```text
//! User register → Handler parameter
//! RAX (number)  → RDI (param 1)
//! RDI (arg1)    → RSI (param 2)
//! RSI (arg2)    → RDX (param 3)
//! RDX (arg3)    → RCX (param 4)
//! R10 (arg4)    → R8  (param 5)
//! R8  (arg5)    → R9  (param 6)
//! ```
//!
//! ## Critical SYSRET pitfall
//!
//! On AMD CPUs, SYSRET checks whether RCX (the return RIP) is canonical
//! *before* switching back to Ring 3. If RCX is non-canonical, the CPU
//! raises #GP **at Ring 0** — a kernel-mode fault triggered by a user-
//! controlled register. Linux mitigates this by checking RCX before SYSRET.
//! Our user code is at 0x40_0000 (canonical), so we're safe for now.

extern crate alloc;

use alloc::string::String;
use core::arch::naked_asm;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::platform::SerialConsole;
use crate::process;
use crate::store::{self, StoreId};
use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

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

const SYS_STORE_GET: u64 = 0;
const SYS_STORE_SET: u64 = 1;

// ————————————————————————————————————————————————————————————————————————————
// Well-known store IDs
// ————————————————————————————————————————————————————————————————————————————

/// Bit 63 set means the store ID is a well-known kernel-resolved alias
/// rather than a direct store registry index.
const WELL_KNOWN_BIT: u64 = 1 << 63;

// ————————————————————————————————————————————————————————————————————————————
// Console store
// ————————————————————————————————————————————————————————————————————————————

/// Schema for the console device store — a single string field that triggers
/// serial output as a side-effect when written.
struct ConsoleSchema;

impl StoreSchema for ConsoleSchema {
    fn name() -> &'static str { "Console" }
    fn fields() -> &'static [FieldDef] {
        &[FieldDef { name: "output", kind: FieldKind::Str }]
    }
}

/// The real [`StoreId`] of the console store. Set by [`create_console_store`],
/// cleared by [`destroy_console_store`]. `u64::MAX` means no console store
/// exists yet.
static CONSOLE_STORE_ID: AtomicU64 = AtomicU64::new(u64::MAX);

/// Create the console store and register its ID in the global.
///
/// Must be called before entering Ring 3 so that userspace `write()` calls
/// (which SET on `CONSOLE`/`"output"`) have a store to resolve to.
pub fn create_console_store() -> StoreId {
    let id = store::create::<ConsoleSchema>(&[
        ("output", Value::Str(String::from(""))),
    ]).expect("create console store");
    CONSOLE_STORE_ID.store(id.as_raw(), Ordering::Relaxed);
    id
}

/// Unregister and destroy the console store.
pub fn destroy_console_store(id: StoreId) {
    CONSOLE_STORE_ID.store(u64::MAX, Ordering::Relaxed);
    store::destroy(id).expect("destroy console store");
}

/// Read the current console store ID, if one exists.
fn console_store_id() -> Option<StoreId> {
    let raw = CONSOLE_STORE_ID.load(Ordering::Relaxed);
    if raw == u64::MAX { None } else { Some(StoreId::from_raw(raw)) }
}

// ————————————————————————————————————————————————————————————————————————————
// Well-known store ID resolution
// ————————————————————————————————————————————————————————————————————————————

/// Resolve a user-provided store ID to an internal [`StoreId`].
///
/// If bit 63 is set, the lower bits select a well-known store:
/// - 0 → current process's lifecycle store (via [`process::store_id`])
/// - 1 → console device store (via [`console_store_id`])
///
/// If bit 63 is clear, the raw value is used directly as a store registry ID.
fn resolve_store_id(raw: u64) -> Option<StoreId> {
    if raw & WELL_KNOWN_BIT != 0 {
        match raw & !WELL_KNOWN_BIT {
            0 => process::store_id(process::current_pid()),
            1 => console_store_id(),
            _ => None,
        }
    } else {
        Some(StoreId::from_raw(raw))
    }
}

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

/// Saved kernel context for the process exit return path. When the user
/// program triggers exit (by setting PROCESS/"status" to "exiting"), the
/// handler restores these registers and RSP to return control to whoever
/// called [`jump_to_ring3`].
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
/// - RDI = arg1, RSI = arg2, RDX = arg3, R10 = arg4, R8 = arg5
///
/// We must immediately swap to a kernel stack before doing anything that
/// touches the stack (like calling a Rust function).
///
/// The register shuffle maps the Linux syscall convention (5 args) to
/// System V AMD64 calling convention for the 6-param Rust handler.
/// R9 is now needed for arg5 (param 6), so we use push/pop to save
/// arg1 through the RDI→param1 overwrite:
///
/// ```text
/// push rdi          ; save arg1 (RDI about to become syscall number)
/// mov r9, r8        ; arg5 → param 6 (before R8 overwritten)
/// mov rdi, rax      ; number → param 1
/// mov r8, r10       ; arg4 → param 5
/// mov rcx, rdx      ; arg3 → param 4 (before RDX overwritten)
/// mov rdx, rsi      ; arg2 → param 3
/// pop rsi           ; arg1 → param 2
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

        // Shuffle registers: Linux syscall convention (5 args) → System V ABI (6 params).
        "push rdi",          // save user arg1 (RDI about to become syscall number)
        "mov r9, r8",        // arg5 → param 6 (must happen before R8 overwritten)
        "mov rdi, rax",      // syscall number → param 1
        "mov r8, r10",       // arg4 → param 5
        "mov rcx, rdx",      // arg3 → param 4 (before rdx overwritten)
        "mov rdx, rsi",      // arg2 → param 3
        "pop rsi",           // arg1 → param 2

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
/// Only two syscalls exist — GET and SET. Everything (process identity,
/// console output, process exit) is expressed through well-known store IDs
/// and side-effects triggered by specific field writes.
extern "C" fn syscall_handler(
    number: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64,
) -> u64 {
    match number {
        SYS_STORE_GET => handle_store_get(arg1, arg2, arg3, arg4, arg5),
        SYS_STORE_SET => handle_store_set(arg1, arg2, arg3, arg4, arg5),
        _ => {
            crate::println!("[syscall] unknown: {}", number);
            u64::MAX
        }
    }
}

/// Handle SYS_STORE_GET (syscall 0).
///
/// Reads a field from a store and serializes the value into the caller's
/// output buffer. The kernel uses the field's [`FieldKind`] from the schema
/// to determine the byte representation:
/// - `U64` → 8 bytes, native endian
/// - `Str` → UTF-8 bytes (truncated if buffer too small)
/// - `Bool` → 1 byte (0 or 1)
///
/// Returns the number of bytes written, or `u64::MAX` on error.
fn handle_store_get(
    store_id_raw: u64, field_ptr: u64, field_len: u64, out_ptr: u64, out_len: u64,
) -> u64 {
    if !validate_user_ptr(field_ptr, field_len) || !validate_user_ptr(out_ptr, out_len) {
        return u64::MAX;
    }

    let field_bytes = unsafe {
        core::slice::from_raw_parts(field_ptr as *const u8, field_len as usize)
    };
    let field_name = match core::str::from_utf8(field_bytes) {
        Ok(s) => s,
        Err(_) => return u64::MAX,
    };

    let id = match resolve_store_id(store_id_raw) {
        Some(id) => id,
        None => return u64::MAX,
    };

    let value = match store::get_no_cli(id, field_name) {
        Ok(v) => v,
        Err(_) => return u64::MAX,
    };

    // Serialize the value into the caller's output buffer.
    let out_buf = unsafe {
        core::slice::from_raw_parts_mut(out_ptr as *mut u8, out_len as usize)
    };

    match value {
        Value::U64(v) => {
            if out_buf.len() < 8 { return u64::MAX; }
            out_buf[..8].copy_from_slice(&v.to_ne_bytes());
            8
        }
        Value::I64(v) => {
            if out_buf.len() < 8 { return u64::MAX; }
            out_buf[..8].copy_from_slice(&v.to_ne_bytes());
            8
        }
        Value::U32(v) => {
            if out_buf.len() < 4 { return u64::MAX; }
            out_buf[..4].copy_from_slice(&v.to_ne_bytes());
            4
        }
        Value::Str(ref s) => {
            let src = s.as_bytes();
            let copy_len = src.len().min(out_buf.len());
            out_buf[..copy_len].copy_from_slice(&src[..copy_len]);
            copy_len as u64
        }
        Value::Bool(b) => {
            if out_buf.is_empty() { return u64::MAX; }
            out_buf[0] = b as u8;
            1
        }
        Value::U8(v) => {
            if out_buf.is_empty() { return u64::MAX; }
            out_buf[0] = v;
            1
        }
    }
}

/// Handle SYS_STORE_SET (syscall 1).
///
/// Interprets raw bytes from the caller according to the field's [`FieldKind`]
/// and writes the resulting [`Value`] to the store. After a successful write,
/// checks for side-effects:
///
/// - **Console output:** if the store is [`CONSOLE`] and the field is `"output"`,
///   writes the raw value bytes to the serial port.
/// - **Process exit:** if the store is the current process's lifecycle store,
///   the field is `"status"`, and the value is `"exiting"`, reads `exit_code`
///   from the process store, calls [`process::exit`], and longjmps back to
///   the kernel via [`restore_return_context`] (never returns to userspace).
///
/// Returns 0 on success, `u64::MAX` on error.
fn handle_store_set(
    store_id_raw: u64, field_ptr: u64, field_len: u64, value_ptr: u64, value_len: u64,
) -> u64 {
    if !validate_user_ptr(field_ptr, field_len) || !validate_user_ptr(value_ptr, value_len) {
        return u64::MAX;
    }

    let field_bytes = unsafe {
        core::slice::from_raw_parts(field_ptr as *const u8, field_len as usize)
    };
    let field_name = match core::str::from_utf8(field_bytes) {
        Ok(s) => s,
        Err(_) => return u64::MAX,
    };

    let id = match resolve_store_id(store_id_raw) {
        Some(id) => id,
        None => return u64::MAX,
    };

    // Look up the field's type from the schema so we know how to
    // interpret the raw bytes.
    let kind = match store::field_kind_no_cli(id, field_name) {
        Ok(k) => k,
        Err(_) => return u64::MAX,
    };

    let value_bytes = unsafe {
        core::slice::from_raw_parts(value_ptr as *const u8, value_len as usize)
    };

    // Interpret the raw bytes according to the schema's declared type.
    let value = match kind {
        FieldKind::U64 => {
            if value_bytes.len() != 8 { return u64::MAX; }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(value_bytes);
            Value::U64(u64::from_ne_bytes(buf))
        }
        FieldKind::I64 => {
            if value_bytes.len() != 8 { return u64::MAX; }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(value_bytes);
            Value::I64(i64::from_ne_bytes(buf))
        }
        FieldKind::U32 => {
            if value_bytes.len() != 4 { return u64::MAX; }
            let mut buf = [0u8; 4];
            buf.copy_from_slice(value_bytes);
            Value::U32(u32::from_ne_bytes(buf))
        }
        FieldKind::Str => {
            match core::str::from_utf8(value_bytes) {
                Ok(s) => Value::Str(String::from(s)),
                Err(_) => return u64::MAX,
            }
        }
        FieldKind::Bool => {
            if value_bytes.len() != 1 { return u64::MAX; }
            Value::Bool(value_bytes[0] != 0)
        }
        FieldKind::U8 => {
            if value_bytes.len() != 1 { return u64::MAX; }
            Value::U8(value_bytes[0])
        }
    };

    if store::set_no_cli(id, &[(field_name, value)]).is_err() {
        return u64::MAX;
    }

    // ── Side-effects ──────────────────────────────────────────────────

    // Console output: write the raw value bytes to serial.
    if console_store_id() == Some(id) && field_name == "output" {
        for &b in value_bytes {
            crate::arch::Serial::write_byte(b);
        }
    }

    // Process exit: if userspace set status to "exiting" on its own
    // process store, read the exit_code and trigger the exit path.
    let pid = process::current_pid();
    if let Some(proc_store) = process::store_id(pid) {
        if proc_store == id && field_name == "status" && value_bytes == b"exiting" {
            let exit_code = match store::get_no_cli(id, "exit_code") {
                Ok(Value::U64(v)) => v,
                _ => 0,
            };
            process::exit(pid, exit_code);
            unsafe { restore_return_context(); }
        }
    }

    0
}

// ————————————————————————————————————————————————————————————————————————————
// Return context restore
// ————————————————————————————————————————————————————————————————————————————

/// Restore callee-saved registers and RSP from [`RETURN_CONTEXT`], then `ret`.
///
/// This is the "longjmp" half of the process exit mechanism. It restores the
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
/// When the user program triggers process exit (by setting PROCESS/"status"
/// to "exiting"), [`restore_return_context`] loads the saved context and
/// executes `ret`, which pops the return address that the `call jump_to_ring3`
/// instruction pushed. The caller sees `jump_to_ring3` "return" normally.
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
