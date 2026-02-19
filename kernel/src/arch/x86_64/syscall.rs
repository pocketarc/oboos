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
//! Five syscalls express all kernel interaction through the store:
//!
//! | # | Name            | Args                                               | Returns            |
//! |---|-----------------|----------------------------------------------------|--------------------|
//! | 0 | SYS_STORE_GET   | store_id, fields_ptr, fields_len, out_ptr, out_len | bytes written      |
//! | 1 | SYS_STORE_SET   | store_id, buf_ptr, buf_len, 0, 0                   | 0 on success       |
//! | 2 | SYS_SUBSCRIBE   | store_id, field_ptr, field_len, 0, 0               | sub_id (0-63)      |
//! | 3 | SYS_UNSUBSCRIBE | sub_id, 0, 0, 0, 0                                | 0 on success       |
//! | 4 | SYS_YIELD       | 0, 0, 0, 0, 0                                     | fired bitmask      |
//!
//! GET and SET use packed buffers for multi-field operations. Each field
//! name or value is prefixed with a `u16` length. SUBSCRIBE registers a
//! persistent watcher; YIELD sleeps until a subscription fires. Errors
//! return structured codes from [`oboos_api::error`].
//!
//! ## Well-known store IDs
//!
//! Bit 63 flags well-known stores resolved per-process by the kernel:
//! - `PROCESS` (1 << 63) — current process's lifecycle store
//! - `CONSOLE` ((1 << 63) | 1) — serial console device store
//!
//! ## Side-effects and async processing
//!
//! SET triggers a [`poll_once()`] at the end of every successful write,
//! allowing async tasks to react before the syscall returns. The console
//! driver is one such task — it watches `CONSOLE`/`"output"` (a Queue(Str)
//! field) and drains messages to serial.
//!
//! The only synchronous side-effect left is process exit: when SET writes
//! `"status"` = `"exiting"` on the process store, the kernel reads `exit_code`,
//! triggers process exit, and longjmps back to [`jump_to_ring3`]'s caller.
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

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::arch::naked_asm;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{RawWaker, RawWakerVTable, Waker};

use crate::platform::{Platform, SerialConsole};
use crate::process;
use crate::store::{self, StoreError, StoreId};
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
const SYS_SUBSCRIBE: u64 = 2;
const SYS_UNSUBSCRIBE: u64 = 3;
const SYS_YIELD: u64 = 4;

// ————————————————————————————————————————————————————————————————————————————
// Well-known store IDs
// ————————————————————————————————————————————————————————————————————————————

/// Bit 63 set means the store ID is a well-known kernel-resolved alias
/// rather than a direct store registry index.
const WELL_KNOWN_BIT: u64 = 1 << 63;

// ————————————————————————————————————————————————————————————————————————————
// Console store
// ————————————————————————————————————————————————————————————————————————————

/// Schema for the console device store — bidirectional I/O through queues.
///
/// - `"output"` (Queue(Str)) — userspace pushes strings; the async console
///   driver drains them to serial.
/// - `"input"` (Queue(U8)) — the keyboard IRQ handler pushes ASCII bytes
///   during Ring 3; userspace pops them via WATCH + GET.
struct ConsoleSchema;

impl StoreSchema for ConsoleSchema {
    fn name() -> &'static str { "Console" }
    fn fields() -> &'static [FieldDef] {
        &[
            FieldDef { name: "output", kind: FieldKind::Queue(&FieldKind::Str) },
            FieldDef { name: "input",  kind: FieldKind::Queue(&FieldKind::U8) },
        ]
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
        ("output", Value::Queue(VecDeque::new())),
        ("input",  Value::Queue(VecDeque::new())),
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
pub(crate) fn console_store_id() -> Option<StoreId> {
    let raw = CONSOLE_STORE_ID.load(Ordering::Relaxed);
    if raw == u64::MAX { None } else { Some(StoreId::from_raw(raw)) }
}

/// Async console driver — watches the console store's output queue and
/// drains pending messages to serial.
///
/// This replaces the synchronous console side-effect that was previously
/// hard-coded in `handle_store_set`. The driver uses the same `watch()`
/// mechanism as other reactive tasks (sysmon_renderer, display_task).
/// `poll_once()` at the end of every SYS_STORE_SET ensures the driver
/// runs before the syscall returns to userspace.
pub async fn console_driver() {
    let id = match console_store_id() {
        Some(id) => id,
        None => return,
    };
    loop {
        if store::watch(id, &["output"]).await.is_err() {
            return;
        }
        for item in store::drain(id, "output").unwrap_or_default() {
            if let Value::Str(s) = item {
                for &b in s.as_bytes() {
                    crate::arch::Serial::write_byte(b);
                }
            }
        }
    }
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
/// Five syscalls: GET, SET, SUBSCRIBE, UNSUBSCRIBE, YIELD. Everything
/// (process identity, console output, console input, process exit) is
/// expressed through well-known store IDs and side-effects triggered by
/// specific field writes.
extern "C" fn syscall_handler(
    number: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64,
) -> u64 {
    match number {
        SYS_STORE_GET => handle_store_get(arg1, arg2, arg3, arg4, arg5),
        SYS_STORE_SET => handle_store_set(arg1, arg2, arg3, arg4, arg5),
        SYS_SUBSCRIBE => handle_subscribe(arg1, arg2, arg3),
        SYS_UNSUBSCRIBE => handle_unsubscribe(arg1),
        SYS_YIELD => handle_yield(),
        _ => {
            crate::println!("[syscall] unknown: {}", number);
            oboos_api::ERR_INVALID_ARG
        }
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Error mapping
// ————————————————————————————————————————————————————————————————————————————

/// Map a [`StoreError`] to the corresponding structured error code.
fn map_store_error(e: &StoreError) -> u64 {
    e.to_raw()
}

// ————————————————————————————————————————————————————————————————————————————
// Value serialization helpers
// ————————————————————————————————————————————————————————————————————————————

/// Interpret raw bytes from userspace as a scalar [`Value`] according to
/// the field's declared type.
///
/// Returns `None` for `Queue` kinds — queue pushes deserialize by the
/// inner kind, not the queue kind itself.
fn deserialize_value(kind: &FieldKind, bytes: &[u8]) -> Option<Value> {
    match kind {
        FieldKind::U64 => {
            if bytes.len() != 8 { return None; }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(bytes);
            Some(Value::U64(u64::from_ne_bytes(buf)))
        }
        FieldKind::I64 => {
            if bytes.len() != 8 { return None; }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(bytes);
            Some(Value::I64(i64::from_ne_bytes(buf)))
        }
        FieldKind::U32 => {
            if bytes.len() != 4 { return None; }
            let mut buf = [0u8; 4];
            buf.copy_from_slice(bytes);
            Some(Value::U32(u32::from_ne_bytes(buf)))
        }
        FieldKind::Str => {
            core::str::from_utf8(bytes)
                .ok()
                .map(|s| Value::Str(String::from(s)))
        }
        FieldKind::Bool => {
            if bytes.len() != 1 { return None; }
            Some(Value::Bool(bytes[0] != 0))
        }
        FieldKind::U8 => {
            if bytes.len() != 1 { return None; }
            Some(Value::U8(bytes[0]))
        }
        FieldKind::Queue(_) => None,
    }
}

/// Serialize a scalar [`Value`] into `out_buf`, returning the byte count.
///
/// Returns `None` for `Queue` values (can't be serialized as a single blob)
/// or if the output buffer is too small.
fn serialize_value(value: &Value, out_buf: &mut [u8]) -> Option<usize> {
    match value {
        Value::U64(v) => {
            if out_buf.len() < 8 { return None; }
            out_buf[..8].copy_from_slice(&v.to_ne_bytes());
            Some(8)
        }
        Value::I64(v) => {
            if out_buf.len() < 8 { return None; }
            out_buf[..8].copy_from_slice(&v.to_ne_bytes());
            Some(8)
        }
        Value::U32(v) => {
            if out_buf.len() < 4 { return None; }
            out_buf[..4].copy_from_slice(&v.to_ne_bytes());
            Some(4)
        }
        Value::Str(s) => {
            let src = s.as_bytes();
            if out_buf.len() < src.len() { return None; }
            out_buf[..src.len()].copy_from_slice(src);
            Some(src.len())
        }
        Value::Bool(b) => {
            if out_buf.is_empty() { return None; }
            out_buf[0] = *b as u8;
            Some(1)
        }
        Value::U8(v) => {
            if out_buf.is_empty() { return None; }
            out_buf[0] = *v;
            Some(1)
        }
        Value::Queue(_) => None,
    }
}

// ————————————————————————————————————————————————————————————————————————————
// SYS_STORE_GET — multi-field packed buffer read
// ————————————————————————————————————————————————————————————————————————————

/// Handle SYS_STORE_GET (syscall 0).
///
/// Input buffer contains packed field names: `[u16 name_len, name bytes, ...]`.
/// Output buffer receives packed values: `[u16 val_len, val bytes, ...]`.
/// For Queue fields, pops the front element (val_len=0 if empty).
/// For scalar fields, reads the current value.
///
/// Returns total bytes written to the output buffer, or a structured
/// error code on failure.
fn handle_store_get(
    store_id_raw: u64, fields_ptr: u64, fields_len: u64, out_ptr: u64, out_len: u64,
) -> u64 {
    if !validate_user_ptr(fields_ptr, fields_len) || !validate_user_ptr(out_ptr, out_len) {
        return oboos_api::ERR_INVALID_ARG;
    }

    let fields_buf = unsafe {
        core::slice::from_raw_parts(fields_ptr as *const u8, fields_len as usize)
    };
    let out_buf = unsafe {
        core::slice::from_raw_parts_mut(out_ptr as *mut u8, out_len as usize)
    };

    let id = match resolve_store_id(store_id_raw) {
        Some(id) => id,
        None => return oboos_api::ERR_NOT_FOUND,
    };

    // Parse field names from the input buffer.
    let mut field_names: Vec<&str> = Vec::new();
    let mut off = 0usize;
    while off < fields_buf.len() {
        if off + 2 > fields_buf.len() {
            return oboos_api::ERR_INVALID_ARG;
        }
        let name_len = u16::from_le_bytes([fields_buf[off], fields_buf[off + 1]]) as usize;
        off += 2;
        if off + name_len > fields_buf.len() {
            return oboos_api::ERR_INVALID_ARG;
        }
        match core::str::from_utf8(&fields_buf[off..off + name_len]) {
            Ok(s) => field_names.push(s),
            Err(_) => return oboos_api::ERR_INVALID_ARG,
        }
        off += name_len;
    }

    // For each field, read/pop the value and write it into the output buffer.
    let mut out_off = 0usize;
    for field_name in &field_names {
        let kind = match store::field_kind_no_cli(id, field_name) {
            Ok(k) => k,
            Err(e) => return map_store_error(&e),
        };

        match kind {
            FieldKind::Queue(_) => {
                match store::pop_no_cli(id, field_name) {
                    Ok(Some(value)) => {
                        // Serialize value into a temp buffer to get its length.
                        let mut tmp = [0u8; 1024];
                        let val_len = match serialize_value(&value, &mut tmp) {
                            Some(n) => n,
                            None => return oboos_api::ERR_INVALID_ARG,
                        };
                        // Write u16 length prefix + value bytes.
                        if out_off + 2 + val_len > out_buf.len() {
                            return oboos_api::ERR_INVALID_ARG;
                        }
                        out_buf[out_off..out_off + 2].copy_from_slice(&(val_len as u16).to_le_bytes());
                        out_off += 2;
                        out_buf[out_off..out_off + val_len].copy_from_slice(&tmp[..val_len]);
                        out_off += val_len;
                    }
                    Ok(None) => {
                        // Empty queue → val_len = 0.
                        if out_off + 2 > out_buf.len() {
                            return oboos_api::ERR_INVALID_ARG;
                        }
                        out_buf[out_off..out_off + 2].copy_from_slice(&0u16.to_le_bytes());
                        out_off += 2;
                    }
                    Err(e) => return map_store_error(&e),
                }
            }
            _ => {
                let value = match store::get_no_cli(id, field_name) {
                    Ok(v) => v,
                    Err(e) => return map_store_error(&e),
                };
                let mut tmp = [0u8; 1024];
                let val_len = match serialize_value(&value, &mut tmp) {
                    Some(n) => n,
                    None => return oboos_api::ERR_INVALID_ARG,
                };
                if out_off + 2 + val_len > out_buf.len() {
                    return oboos_api::ERR_INVALID_ARG;
                }
                out_buf[out_off..out_off + 2].copy_from_slice(&(val_len as u16).to_le_bytes());
                out_off += 2;
                out_buf[out_off..out_off + val_len].copy_from_slice(&tmp[..val_len]);
                out_off += val_len;
            }
        }
    }

    out_off as u64
}

// ————————————————————————————————————————————————————————————————————————————
// SYS_STORE_SET — multi-field packed buffer write
// ————————————————————————————————————————————————————————————————————————————

/// Handle SYS_STORE_SET (syscall 1).
///
/// The input buffer contains packed field/value pairs:
/// `[u16 name_len, name, u16 value_len, value, ...]`.
///
/// Scalar fields are written atomically (all-or-nothing). Queue fields
/// are pushed individually. After a successful write, checks for the
/// process-exit side-effect and calls [`poll_once()`].
///
/// Returns 0 on success, or a structured error code on failure.
fn handle_store_set(
    store_id_raw: u64, buf_ptr: u64, buf_len: u64, _arg4: u64, _arg5: u64,
) -> u64 {
    if !validate_user_ptr(buf_ptr, buf_len) {
        return oboos_api::ERR_INVALID_ARG;
    }

    let buf = unsafe {
        core::slice::from_raw_parts(buf_ptr as *const u8, buf_len as usize)
    };

    let id = match resolve_store_id(store_id_raw) {
        Some(id) => id,
        None => return oboos_api::ERR_NOT_FOUND,
    };

    // Parse all field/value pairs from the buffer.
    struct Entry<'a> {
        name: &'a str,
        value_bytes: &'a [u8],
    }

    let mut entries: Vec<Entry> = Vec::new();
    let mut off = 0usize;

    while off < buf.len() {
        // Parse name_len + name.
        if off + 2 > buf.len() { return oboos_api::ERR_INVALID_ARG; }
        let name_len = u16::from_le_bytes([buf[off], buf[off + 1]]) as usize;
        off += 2;
        if off + name_len > buf.len() { return oboos_api::ERR_INVALID_ARG; }
        let name = match core::str::from_utf8(&buf[off..off + name_len]) {
            Ok(s) => s,
            Err(_) => return oboos_api::ERR_INVALID_ARG,
        };
        off += name_len;

        // Parse value_len + value.
        if off + 2 > buf.len() { return oboos_api::ERR_INVALID_ARG; }
        let value_len = u16::from_le_bytes([buf[off], buf[off + 1]]) as usize;
        off += 2;
        if off + value_len > buf.len() { return oboos_api::ERR_INVALID_ARG; }
        let value_bytes = &buf[off..off + value_len];
        off += value_len;

        entries.push(Entry { name, value_bytes });
    }

    // Separate queue pushes from scalar sets.
    let mut scalar_updates: Vec<(&str, Value)> = Vec::new();
    let mut has_status_exiting = false;

    for entry in &entries {
        let kind = match store::field_kind_no_cli(id, entry.name) {
            Ok(k) => k,
            Err(e) => return map_store_error(&e),
        };

        match kind {
            FieldKind::Queue(inner) => {
                let value = match deserialize_value(inner, entry.value_bytes) {
                    Some(v) => v,
                    None => return oboos_api::ERR_TYPE_MISMATCH,
                };
                if let Err(e) = store::push_no_cli(id, entry.name, value) {
                    return map_store_error(&e);
                }
            }
            _ => {
                let value = match deserialize_value(&kind, entry.value_bytes) {
                    Some(v) => v,
                    None => return oboos_api::ERR_TYPE_MISMATCH,
                };
                if entry.name == "status" && entry.value_bytes == b"exiting" {
                    has_status_exiting = true;
                }
                scalar_updates.push((entry.name, value));
            }
        }
    }

    // Write all scalar fields atomically.
    if !scalar_updates.is_empty() {
        if let Err(e) = store::set_no_cli(id, &scalar_updates) {
            return map_store_error(&e);
        }
    }

    // ── Side-effects ──────────────────────────────────────────────────

    // Process exit: if this batch wrote status="exiting" on the process store.
    if has_status_exiting {
        let pid = process::current_pid();
        if let Some(proc_store) = process::store_id(pid) {
            if proc_store == id {
                let exit_code = match store::get_no_cli(id, "exit_code") {
                    Ok(Value::U64(v)) => v,
                    _ => 0,
                };
                process::exit(pid, exit_code);
                unsafe { restore_return_context(); }
            }
        }
    }

    // Run async tasks so subscribers (like the console driver) can
    // process the new data before we return to userspace. poll_once()
    // manages its own interrupt state and returns with IF=1, so we
    // must re-disable interrupts for the sysretq path.
    crate::executor::poll_once();
    crate::arch::Arch::disable_interrupts();

    0
}

// ————————————————————————————————————————————————————————————————————————————
// Subscription tracking (SYS_SUBSCRIBE / SYS_UNSUBSCRIBE / SYS_YIELD)
// ————————————————————————————————————————————————————————————————————————————

/// Tracks which store each subscription slot belongs to.
struct SubscriptionSlot {
    store: StoreId,
}

/// 64 subscription slots, one per bit in [`FIRED_MASK`].
static SUBSCRIPTIONS: spin::Mutex<[Option<SubscriptionSlot>; 64]> =
    spin::Mutex::new([const { None }; 64]);

/// Bitmask of fired subscriptions. Bit N is set when subscription N's
/// watcher fires. [`handle_yield`] atomically swaps this to 0 and
/// returns the old value to userspace.
static FIRED_MASK: AtomicU64 = AtomicU64::new(0);

/// Custom waker vtable for subscription slots. `wake()` sets bit
/// `sub_id` in [`FIRED_MASK`] via `fetch_or`. Same zero-alloc pattern
/// as the executor's wakers — the sub_id is packed into the data pointer.
static SUB_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |data| RawWaker::new(data, &SUB_WAKER_VTABLE), // clone
    |data| {
        let sub_id = data as u64;
        FIRED_MASK.fetch_or(1 << sub_id, Ordering::SeqCst);
    }, // wake
    |data| {
        let sub_id = data as u64;
        FIRED_MASK.fetch_or(1 << sub_id, Ordering::SeqCst);
    }, // wake_by_ref
    |_| {}, // drop
);

/// Create a [`Waker`] that sets bit `sub_id` in [`FIRED_MASK`] on wake.
fn subscription_waker(sub_id: u64) -> Waker {
    let raw = RawWaker::new(sub_id as *const (), &SUB_WAKER_VTABLE);
    unsafe { Waker::from_raw(raw) }
}

/// Handle SYS_SUBSCRIBE (syscall 2).
///
/// Registers a persistent watcher on a store field. Returns the
/// subscription ID (0-63) on success, or an error code on failure.
fn handle_subscribe(store_id_raw: u64, field_ptr: u64, field_len: u64) -> u64 {
    if !validate_user_ptr(field_ptr, field_len) {
        return oboos_api::ERR_INVALID_ARG;
    }

    let field_bytes = unsafe {
        core::slice::from_raw_parts(field_ptr as *const u8, field_len as usize)
    };
    let field_name = match core::str::from_utf8(field_bytes) {
        Ok(s) => s,
        Err(_) => return oboos_api::ERR_INVALID_ARG,
    };

    let id = match resolve_store_id(store_id_raw) {
        Some(id) => id,
        None => return oboos_api::ERR_NOT_FOUND,
    };

    // Allocate a free subscription slot.
    let mut subs = SUBSCRIPTIONS.lock();
    let sub_id = match subs.iter().position(|s| s.is_none()) {
        Some(i) => i as u64,
        None => return oboos_api::ERR_WOULD_BLOCK,
    };

    let waker = subscription_waker(sub_id);
    match store::add_watcher_no_cli(id, field_name, waker, sub_id) {
        Ok(fire_immediately) => {
            subs[sub_id as usize] = Some(SubscriptionSlot { store: id });
            if fire_immediately {
                FIRED_MASK.fetch_or(1 << sub_id, Ordering::SeqCst);
            }
            sub_id
        }
        Err(e) => map_store_error(&e),
    }
}

/// Handle SYS_UNSUBSCRIBE (syscall 3).
///
/// Removes a subscription by its ID. Returns 0 on success, or an
/// error code if the slot was already empty.
fn handle_unsubscribe(sub_id: u64) -> u64 {
    if sub_id >= 64 {
        return oboos_api::ERR_INVALID_ARG;
    }

    let mut subs = SUBSCRIPTIONS.lock();
    let slot = match subs[sub_id as usize].take() {
        Some(s) => s,
        None => return oboos_api::ERR_NOT_FOUND,
    };
    drop(subs);

    let _ = store::remove_watcher_no_cli(slot.store, sub_id);
    FIRED_MASK.fetch_and(!(1 << sub_id), Ordering::SeqCst);
    0
}

/// Handle SYS_YIELD (syscall 4).
///
/// Sleeps until at least one subscription fires, then returns the
/// bitmask of fired subscription IDs. Drives async tasks via
/// [`poll_once()`] while waiting. Preemption is suppressed so PIT
/// ticks don't context-switch us off the syscall kernel stack.
fn handle_yield() -> u64 {
    crate::scheduler::disable_preemption();
    loop {
        let mask = FIRED_MASK.swap(0, Ordering::SeqCst);
        if mask != 0 {
            crate::scheduler::enable_preemption();
            crate::arch::Arch::disable_interrupts();
            return mask;
        }
        crate::executor::poll_once();
        crate::arch::Arch::halt_until_interrupt();
    }
}

/// Clear all subscriptions. Called on process exit to prevent stale
/// watchers from firing after the process is gone.
pub fn clear_all_subscriptions() {
    let mut subs = SUBSCRIPTIONS.lock();
    for i in 0..64 {
        if let Some(slot) = subs[i].take() {
            let _ = store::remove_watcher_no_cli(slot.store, i as u64);
        }
    }
    FIRED_MASK.store(0, Ordering::SeqCst);
}

/// Async keyboard input driver — reads scancodes from the keyboard
/// buffer, translates them to ASCII, and pushes them into the console
/// store's `"input"` queue.
///
/// Each push fires persistent watchers → sets FIRED_MASK bits →
/// SYS_YIELD returns to userspace.
pub async fn keyboard_input_driver() {
    let id = match console_store_id() {
        Some(id) => id,
        None => return,
    };
    loop {
        let byte = super::keyboard::next_console_byte().await;
        let _ = store::push(id, "input", Value::U8(byte));
    }
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
