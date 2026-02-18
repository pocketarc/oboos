//! OBOOS userspace "hello" program.
//!
//! This is the first Rust program to run in Ring 3 on OBOOS. It proves
//! that the store-centric IPC model works across the privilege boundary:
//! writes a value to a kernel store field via syscall, reads it back, and
//! prints the result to the serial console.
//!
//! ## Syscall convention
//!
//! Follows the Linux register convention for SYSCALL:
//! - RAX = syscall number
//! - RDI = arg1, RSI = arg2, RDX = arg3, R10 = arg4
//! - Return value in RAX
//!
//! SYSCALL clobbers RCX (saves RIP) and R11 (saves RFLAGS), so both
//! must be listed as clobbers in the inline asm.

#![no_std]
#![no_main]

// ————————————————————————————————————————————————————————————————————————————
// Syscall numbers — must match kernel/src/arch/x86_64/syscall.rs
// ————————————————————————————————————————————————————————————————————————————

const SYS_EXIT: u64 = 0;
const SYS_WRITE: u64 = 1;
const SYS_STORE_GET: u64 = 2;
const SYS_STORE_SET: u64 = 3;
const SYS_GETPID: u64 = 4;

// ————————————————————————————————————————————————————————————————————————————
// Raw syscall wrappers
// ————————————————————————————————————————————————————————————————————————————

/// Syscall with no arguments (just the number).
#[inline(always)]
fn syscall0(number: u64) -> u64 {
    let ret: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") number,
            lateout("rax") ret,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        );
    }
    ret
}

/// Syscall with 1 argument.
#[inline(always)]
fn syscall1(number: u64, arg1: u64) -> u64 {
    let ret: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") number,
            in("rdi") arg1,
            lateout("rax") ret,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        );
    }
    ret
}

/// Syscall with 2 arguments.
#[inline(always)]
fn syscall2(number: u64, arg1: u64, arg2: u64) -> u64 {
    let ret: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") number,
            in("rdi") arg1,
            in("rsi") arg2,
            lateout("rax") ret,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        );
    }
    ret
}

/// Syscall with 3 arguments.
#[inline(always)]
fn syscall3(number: u64, arg1: u64, arg2: u64, arg3: u64) -> u64 {
    let ret: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") number,
            in("rdi") arg1,
            in("rsi") arg2,
            in("rdx") arg3,
            lateout("rax") ret,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        );
    }
    ret
}

/// Syscall with 4 arguments. R10 carries arg4 (not RCX, because SYSCALL
/// clobbers RCX with the return address).
#[inline(always)]
fn syscall4(number: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64) -> u64 {
    let ret: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") number,
            in("rdi") arg1,
            in("rsi") arg2,
            in("rdx") arg3,
            in("r10") arg4,
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

/// Write a string to the serial console via SYS_WRITE.
fn write(msg: &str) {
    syscall2(SYS_WRITE, msg.as_ptr() as u64, msg.len() as u64);
}

/// Read a U64 value from a store field via SYS_STORE_GET.
fn store_get(store_id: u64, field: &str) -> u64 {
    syscall3(SYS_STORE_GET, store_id, field.as_ptr() as u64, field.len() as u64)
}

/// Write a U64 value to a store field via SYS_STORE_SET.
fn store_set(store_id: u64, field: &str, value: u64) -> u64 {
    syscall4(SYS_STORE_SET, store_id, field.as_ptr() as u64, field.len() as u64, value)
}

/// Get the current process's PID via SYS_GETPID.
fn getpid() -> u64 {
    syscall0(SYS_GETPID)
}

/// Exit the program with an exit code, returning control to the kernel.
fn exit(code: u64) -> ! {
    syscall1(SYS_EXIT, code);
    // Safety net — SYS_EXIT never returns, but the compiler doesn't know that.
    loop {}
}

// ————————————————————————————————————————————————————————————————————————————
// Decimal formatting (no alloc, no format!)
// ————————————————————————————————————————————————————————————————————————————

/// Write a u64 as decimal digits into a byte buffer. Returns bytes written.
fn write_u64(buf: &mut [u8], val: u64) -> usize {
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
// Entry point
// ————————————————————————————————————————————————————————————————————————————

/// Userspace entry point. Receives the store ID in RDI (passed by the
/// kernel via `jump_to_ring3`'s third argument).
#[unsafe(no_mangle)]
pub extern "C" fn _start(store_id: u64) -> ! {
    write("Hello from userspace Rust!\n");

    // Print our PID to prove SYS_GETPID works.
    let pid = getpid();
    let mut buf = [0u8; 64];
    let prefix = b"My PID: ";
    buf[..prefix.len()].copy_from_slice(prefix);
    let mut pos = prefix.len();
    pos += write_u64(&mut buf[pos..], pid);
    buf[pos] = b'\n';
    pos += 1;
    let msg = unsafe { core::str::from_utf8_unchecked(&buf[..pos]) };
    write(msg);

    // Set counter=42 in the kernel store.
    store_set(store_id, "counter", 42);

    // Read it back to verify the round trip.
    let val = store_get(store_id, "counter");

    // Print the result: "Store round-trip OK: counter=NN\n"
    let prefix2 = b"Store round-trip OK: counter=";
    buf[..prefix2.len()].copy_from_slice(prefix2);
    pos = prefix2.len();
    pos += write_u64(&mut buf[pos..], val);
    buf[pos] = b'\n';
    pos += 1;
    let msg = unsafe { core::str::from_utf8_unchecked(&buf[..pos]) };
    write(msg);

    exit(0);
}

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
