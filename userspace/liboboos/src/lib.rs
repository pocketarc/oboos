//! OBOOS userspace runtime library.
//!
//! Provides the syscall ABI, high-level wrappers, and a panic handler for
//! all Ring 3 programs. Any code that a second userspace binary would need
//! to copy-paste belongs here instead of in the application.
//!
//! ## Syscall interface
//!
//! Three syscalls express all kernel interaction through the store:
//!
//! | # | Name            | Args                                               | Returns            |
//! |---|-----------------|----------------------------------------------------|--------------------|
//! | 0 | SYS_STORE_GET   | store_id, fields_ptr, fields_len, out_ptr, out_len | bytes written      |
//! | 1 | SYS_STORE_SET   | store_id, buf_ptr, buf_len, 0, 0                   | 0 on success       |
//! | 2 | SYS_STORE_WATCH | store_id, field_ptr, field_len, 0, 0               | 0 on success       |
//!
//! GET and SET use packed buffers with u16 length prefixes for multi-field
//! operations. WATCH blocks until a field is written.
//!
//! All three use 5 arguments passed in RDI, RSI, RDX, R10, R8 (Linux convention).
//! SYSCALL clobbers RCX (saves RIP) and R11 (saves RFLAGS).
//!
//! ## Error codes
//!
//! Errors are returned as `u64` values at the top of the address space
//! (>= `ERR_THRESHOLD`). Use [`is_error()`] to check.
//!
//! ## Well-known store IDs
//!
//! Bit 63 flags well-known stores resolved per-process by the kernel:
//! - [`PROCESS`] — current process's lifecycle store (pid, status, exit_code)
//! - [`CONSOLE`] — serial console device store (output + input queues)

#![no_std]

// ————————————————————————————————————————————————————————————————————————————
// Syscall numbers — must match kernel/src/arch/x86_64/syscall.rs
// ————————————————————————————————————————————————————————————————————————————

pub const SYS_STORE_GET: u64 = 0;
pub const SYS_STORE_SET: u64 = 1;
pub const SYS_STORE_WATCH: u64 = 2;

// ————————————————————————————————————————————————————————————————————————————
// Error codes — must match api/src/error.rs
// ————————————————————————————————————————————————————————————————————————————

pub const ERR_NOT_FOUND: u64     = u64::MAX;
pub const ERR_UNKNOWN_FIELD: u64 = u64::MAX - 1;
pub const ERR_TYPE_MISMATCH: u64 = u64::MAX - 2;
pub const ERR_INVALID_ARG: u64   = u64::MAX - 3;
pub const ERR_WOULD_BLOCK: u64   = u64::MAX - 4;
pub const ERR_THRESHOLD: u64     = u64::MAX - 15;

/// Check whether a syscall return value is an error code.
pub fn is_error(result: u64) -> bool {
    result > ERR_THRESHOLD
}

// ————————————————————————————————————————————————————————————————————————————
// Well-known store IDs
// ————————————————————————————————————————————————————————————————————————————

/// The current process's lifecycle store. The kernel resolves this to the
/// process store created by `process::spawn()`, containing fields like
/// `pid` (U64), `name` (Str), `status` (Str), and `exit_code` (U64).
pub const PROCESS: u64 = 1 << 63;

/// The serial console device store. Contains `"output"` (Queue(Str)) for
/// writing and `"input"` (Queue(U8)) for reading keyboard input.
pub const CONSOLE: u64 = (1 << 63) | 1;

// ————————————————————————————————————————————————————————————————————————————
// Raw syscall wrapper
// ————————————————————————————————————————————————————————————————————————————

/// Syscall with 5 arguments — the only raw syscall needed, since all three
/// syscalls take exactly 5 args (unused args are 0).
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
// Packed buffer encoding/decoding helpers
// ————————————————————————————————————————————————————————————————————————————

/// Encode a single field name into a GET input buffer.
///
/// Writes `[u16 name_len (LE), name bytes]` starting at `buf[off]`.
/// Advances `off` past the written data.
fn encode_field(buf: &mut [u8], off: &mut usize, name: &str) {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len() as u16;
    buf[*off..*off + 2].copy_from_slice(&name_len.to_le_bytes());
    *off += 2;
    buf[*off..*off + name_bytes.len()].copy_from_slice(name_bytes);
    *off += name_bytes.len();
}

/// Encode a field name + value into a SET buffer.
///
/// Writes `[u16 name_len, name, u16 value_len, value]` starting at `buf[off]`.
/// Advances `off` past the written data.
fn encode_pair(buf: &mut [u8], off: &mut usize, name: &str, value: &[u8]) {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len() as u16;
    buf[*off..*off + 2].copy_from_slice(&name_len.to_le_bytes());
    *off += 2;
    buf[*off..*off + name_bytes.len()].copy_from_slice(name_bytes);
    *off += name_bytes.len();

    let val_len = value.len() as u16;
    buf[*off..*off + 2].copy_from_slice(&val_len.to_le_bytes());
    *off += 2;
    buf[*off..*off + value.len()].copy_from_slice(value);
    *off += value.len();
}

/// Decode one value from a GET output buffer.
///
/// Reads `[u16 val_len, val bytes]` starting at `buf[off]`.
/// Advances `off` past the read data. Returns the value bytes.
fn decode_value<'a>(buf: &'a [u8], off: &mut usize) -> &'a [u8] {
    let val_len = u16::from_le_bytes([buf[*off], buf[*off + 1]]) as usize;
    *off += 2;
    let val = &buf[*off..*off + val_len];
    *off += val_len;
    val
}

// ————————————————————————————————————————————————————————————————————————————
// High-level syscall wrappers
// ————————————————————————————————————————————————————————————————————————————

/// Read a U64 value from a store field via SYS_STORE_GET.
///
/// Encodes the field name in packed format, provides an output buffer,
/// and interprets the result as a native-endian `u64`. Returns 0 on error.
pub fn store_get(store_id: u64, field: &str) -> u64 {
    let mut fields = [0u8; 64];
    let mut off = 0;
    encode_field(&mut fields, &mut off, field);

    let mut out = [0u8; 16]; // u16 prefix + 8 bytes for u64
    let ret = syscall5(
        SYS_STORE_GET,
        store_id,
        fields.as_ptr() as u64, off as u64,
        out.as_mut_ptr() as u64, out.len() as u64,
    );
    if is_error(ret) || ret < 10 { return 0; }

    // Skip u16 length prefix, read u64.
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&out[2..10]);
    u64::from_ne_bytes(arr)
}

/// Read a U8 value from a store field via SYS_STORE_GET.
///
/// For Queue(U8) fields this pops the front element. Returns `None` if
/// the queue is empty or on error.
pub fn store_get_u8(store_id: u64, field: &str) -> Option<u8> {
    let mut fields = [0u8; 64];
    let mut off = 0;
    encode_field(&mut fields, &mut off, field);

    let mut out = [0u8; 8]; // u16 prefix + 1 byte for u8
    let ret = syscall5(
        SYS_STORE_GET,
        store_id,
        fields.as_ptr() as u64, off as u64,
        out.as_mut_ptr() as u64, out.len() as u64,
    );
    if is_error(ret) { return None; }

    let mut decode_off = 0;
    let val = decode_value(&out, &mut decode_off);
    if val.is_empty() { return None; }
    Some(val[0])
}

/// Read a string value from a store field via SYS_STORE_GET.
///
/// Copies the string bytes into the provided buffer and returns the
/// number of bytes written. Returns 0 on error or if the field is empty.
pub fn store_get_str(store_id: u64, field: &str, out: &mut [u8]) -> usize {
    let mut fields = [0u8; 64];
    let mut off = 0;
    encode_field(&mut fields, &mut off, field);

    // We need the u16 prefix + up to out.len() bytes of string data.
    // Use the caller's buffer with 2 extra bytes for the prefix.
    let total_out_len = out.len() + 2;
    let mut tmp = [0u8; 1024];
    let buf_len = total_out_len.min(tmp.len());

    let ret = syscall5(
        SYS_STORE_GET,
        store_id,
        fields.as_ptr() as u64, off as u64,
        tmp.as_mut_ptr() as u64, buf_len as u64,
    );
    if is_error(ret) || ret < 2 { return 0; }

    let mut decode_off = 0;
    let val = decode_value(&tmp[..ret as usize], &mut decode_off);
    let copy_len = val.len().min(out.len());
    out[..copy_len].copy_from_slice(&val[..copy_len]);
    copy_len
}

/// Write a U64 value to a store field via SYS_STORE_SET.
///
/// Encodes the field name + value in packed format.
pub fn store_set(store_id: u64, field: &str, value: u64) -> u64 {
    let mut buf = [0u8; 64];
    let mut off = 0;
    encode_pair(&mut buf, &mut off, field, &value.to_ne_bytes());
    syscall5(SYS_STORE_SET, store_id, buf.as_ptr() as u64, off as u64, 0, 0)
}

/// Write a U8 value to a store field via SYS_STORE_SET.
pub fn store_set_u8(store_id: u64, field: &str, value: u8) -> u64 {
    let mut buf = [0u8; 64];
    let mut off = 0;
    encode_pair(&mut buf, &mut off, field, &[value]);
    syscall5(SYS_STORE_SET, store_id, buf.as_ptr() as u64, off as u64, 0, 0)
}

/// Write a string value to a store field via SYS_STORE_SET.
///
/// Encodes the field name + string bytes in packed format.
pub fn store_set_str(store_id: u64, field: &str, value: &str) -> u64 {
    let mut buf = [0u8; 256];
    let mut off = 0;
    encode_pair(&mut buf, &mut off, field, value.as_bytes());
    syscall5(SYS_STORE_SET, store_id, buf.as_ptr() as u64, off as u64, 0, 0)
}

/// Block until a store field is written.
///
/// Returns 0 on success, or an error code if the store doesn't exist
/// or the field is unknown.
pub fn store_watch(store_id: u64, field: &str) -> u64 {
    syscall5(
        SYS_STORE_WATCH,
        store_id,
        field.as_ptr() as u64, field.len() as u64,
        0, 0,
    )
}

/// Write a string to the serial console.
///
/// Pushes a string onto the `"output"` queue on the [`CONSOLE`] store.
/// The kernel's async console driver drains the queue to serial.
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
/// Sets both `exit_code` and `status` in a single atomic SYS_STORE_SET call.
/// The kernel detects `status="exiting"` as a side-effect trigger: it reads
/// the exit code, transitions the process to Exited, and longjmps back to
/// whoever called `jump_to_ring3`. The syscall never returns.
pub fn exit(code: u64) -> ! {
    let mut buf = [0u8; 64];
    let mut off = 0;
    encode_pair(&mut buf, &mut off, "exit_code", &code.to_ne_bytes());
    encode_pair(&mut buf, &mut off, "status", b"exiting");
    syscall5(SYS_STORE_SET, PROCESS, buf.as_ptr() as u64, off as u64, 0, 0);
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
