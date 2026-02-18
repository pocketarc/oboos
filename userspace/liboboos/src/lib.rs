//! OBOOS userspace runtime library.
//!
//! Provides the syscall ABI, high-level wrappers, and a panic handler for
//! all Ring 3 programs. Any code that a second userspace binary would need
//! to copy-paste belongs here instead of in the application.
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
//! Both use 5 arguments passed in RDI, RSI, RDX, R10, R8 (Linux convention).
//! SYSCALL clobbers RCX (saves RIP) and R11 (saves RFLAGS).
//!
//! ## Well-known store IDs
//!
//! Bit 63 flags well-known stores resolved per-process by the kernel:
//! - [`PROCESS`] — current process's lifecycle store (pid, status, exit_code)
//! - [`CONSOLE`] — serial console device store (write "output" to print)

#![no_std]

// ————————————————————————————————————————————————————————————————————————————
// Syscall numbers — must match kernel/src/arch/x86_64/syscall.rs
// ————————————————————————————————————————————————————————————————————————————

pub const SYS_STORE_GET: u64 = 0;
pub const SYS_STORE_SET: u64 = 1;

// ————————————————————————————————————————————————————————————————————————————
// Well-known store IDs
// ————————————————————————————————————————————————————————————————————————————

/// The current process's lifecycle store. The kernel resolves this to the
/// process store created by `process::spawn()`, containing fields like
/// `pid` (U64), `name` (Str), `status` (Str), and `exit_code` (U64).
pub const PROCESS: u64 = 1 << 63;

/// The serial console device store. Writing to the `"output"` field (Str)
/// triggers a side-effect: the kernel writes the raw bytes to the serial port.
pub const CONSOLE: u64 = (1 << 63) | 1;

// ————————————————————————————————————————————————————————————————————————————
// Raw syscall wrapper
// ————————————————————————————————————————————————————————————————————————————

/// Syscall with 5 arguments — the only raw syscall needed, since both
/// SYS_STORE_GET and SYS_STORE_SET take exactly 5 args.
///
/// Register mapping follows the Linux SYSCALL convention:
/// RAX = number, RDI = arg1, RSI = arg2, RDX = arg3, R10 = arg4, R8 = arg5.
#[inline(always)]
pub fn syscall5(number: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> u64 {
    let ret: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") number,
            in("rdi") arg1,
            in("rsi") arg2,
            in("rdx") arg3,
            in("r10") arg4,
            in("r8") arg5,
            lateout("rax") ret,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        );
    }
    ret
}

// ————————————————————————————————————————————————————————————————————————————
// High-level syscall wrappers
// ————————————————————————————————————————————————————————————————————————————

/// Read a U64 value from a store field via SYS_STORE_GET.
///
/// Provides an 8-byte output buffer and interprets the result as a native-
/// endian `u64`. Works for any U64 field (e.g. pid, exit_code, counters).
pub fn store_get(store_id: u64, field: &str) -> u64 {
    let mut val: u64 = 0;
    syscall5(
        SYS_STORE_GET,
        store_id,
        field.as_ptr() as u64,
        field.len() as u64,
        &mut val as *mut u64 as u64,
        8,
    );
    val
}

/// Write a U64 value to a store field via SYS_STORE_SET.
///
/// Sends the 8 native-endian bytes of `value`. The kernel looks up the
/// field's schema type (must be U64) and constructs the Value internally.
pub fn store_set(store_id: u64, field: &str, value: u64) -> u64 {
    let bytes = value.to_ne_bytes();
    syscall5(
        SYS_STORE_SET,
        store_id,
        field.as_ptr() as u64,
        field.len() as u64,
        bytes.as_ptr() as u64,
        8,
    )
}

/// Write a string value to a store field via SYS_STORE_SET.
///
/// Sends the raw UTF-8 bytes. The kernel looks up the field's schema type
/// (must be Str) and constructs the Value internally.
pub fn store_set_str(store_id: u64, field: &str, value: &str) -> u64 {
    syscall5(
        SYS_STORE_SET,
        store_id,
        field.as_ptr() as u64,
        field.len() as u64,
        value.as_ptr() as u64,
        value.len() as u64,
    )
}

/// Write a string to the serial console.
///
/// Sets the `"output"` field on the [`CONSOLE`] well-known store. The kernel
/// triggers a side-effect that writes the raw bytes to the serial port.
pub fn write(msg: &str) {
    store_set_str(CONSOLE, "output", msg);
}

/// Get the current process's PID.
///
/// Reads the `"pid"` field from the [`PROCESS`] well-known store.
pub fn getpid() -> u64 {
    store_get(PROCESS, "pid")
}

/// Exit the program with an exit code, returning control to the kernel.
///
/// Sets `exit_code` on the process store, then sets `status` to `"exiting"`.
/// The kernel detects the status write as a side-effect trigger: it reads
/// the exit code, transitions the process to Exited, and longjmps back to
/// whoever called `jump_to_ring3`. The second syscall never returns.
pub fn exit(code: u64) -> ! {
    store_set(PROCESS, "exit_code", code);
    store_set_str(PROCESS, "status", "exiting");
    // The kernel's exit side-effect longjmps — this is a safety net.
    loop {}
}

// ————————————————————————————————————————————————————————————————————————————
// Decimal formatting (no alloc, no format!)
// ————————————————————————————————————————————————————————————————————————————

/// Write a u64 as decimal digits into a byte buffer. Returns bytes written.
pub fn write_u64(buf: &mut [u8], val: u64) -> usize {
    if val == 0 {
        buf[0] = b'0';
        return 1;
    }

    let mut digits = [0u8; 20];
    let mut n = val;
    let mut len = 0;
    while n > 0 {
        digits[len] = b'0' + (n % 10) as u8;
        n /= 10;
        len += 1;
    }

    for i in (0..len).rev() {
        buf[len - 1 - i] = digits[i];
    }
    len
}

// ————————————————————————————————————————————————————————————————————————————
// Panic handler
// ————————————————————————————————————————————————————————————————————————————

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    // Best-effort: print what we can to serial.
    write("!!! USERSPACE PANIC !!!\n");

    // We can't use format! (no alloc), so just print the static parts.
    if let Some(location) = info.location() {
        write(location.file());
        write(":");
        let mut buf = [0u8; 20];
        let len = write_u64(&mut buf, location.line() as u64);
        let s = unsafe { core::str::from_utf8_unchecked(&buf[..len]) };
        write(s);
        write("\n");
    }

    exit(1);
}
