//! OBOOS userspace "hello" program.
//!
//! First Rust program to run in Ring 3 on OBOOS. Proves that the
//! store-centric IPC model works across the privilege boundary: writes
//! a value to a kernel store field via syscall, reads it back, and
//! prints the result to the serial console. Also demos interactive
//! keyboard input via the console store's `"input"` queue and
//! `SYS_STORE_WATCH`.

#![no_std]
#![no_main]

use liboboos::{
    exit, getpid, store_get, store_get_u8, store_set, store_watch, write, write_u64,
    CONSOLE,
};

/// Userspace entry point. Receives the store ID in RDI (passed by the
/// kernel via `jump_to_ring3`'s third argument).
#[unsafe(no_mangle)]
pub extern "C" fn _start(store_id: u64) -> ! {
    write("Hello from userspace Rust!\n");

    // Print our PID to prove SYS_STORE_GET works.
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

    // Interactive input: read keyboard input via the console store.
    write("Type something (Enter to finish): ");

    loop {
        // Block until a byte arrives on the console input queue.
        store_watch(CONSOLE, "input");

        // Pop all available bytes.
        loop {
            match store_get_u8(CONSOLE, "input") {
                Some(b'\n') => {
                    write("\n");
                    // Done with input — exit cleanly.
                    exit(0);
                }
                Some(ch) => {
                    // Echo the character back to the console.
                    let echo = [ch];
                    let s = unsafe { core::str::from_utf8_unchecked(&echo) };
                    write(s);
                }
                None => break, // queue empty, go back to watching
            }
        }
    }
}
