//! OBOOS userspace "hello" program.
//!
//! First Rust program to run in Ring 3 on OBOOS. Proves that the
//! store-centric IPC model works across the privilege boundary: writes
//! a value to a kernel store field via syscall, reads it back, and
//! prints the result to the serial console. Also demos interactive
//! keyboard input via the console store's `"input"` queue and
//! async subscriptions (SYS_SUBSCRIBE/SYS_YIELD).
//!
//! Now also demonstrates the userspace heap allocator: `format!()`,
//! `String`, and `Vec` work via the `#[global_allocator]` in liboboos,
//! which grows the heap through MUTATE/MapHeap syscalls.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use liboboos::{
    block_on, exit, getpid, store_get, store_set, watch, write,
    CONSOLE,
};

/// Userspace entry point. Receives the store ID in RDI (passed by the
/// kernel via `jump_to_ring3`'s third argument).
#[unsafe(no_mangle)]
pub extern "C" fn _start(store_id: u64) -> ! {
    block_on(main(store_id));
    exit(0);
}

async fn main(store_id: u64) {
    write("Hello from userspace Rust!\n");

    // Print our PID using format!() — proves the heap allocator works.
    let pid = getpid();
    let msg = format!("My PID: {}\n", pid);
    write(&msg);

    // Set counter=42 in the kernel store.
    store_set(store_id, "counter", 42u64).expect("set counter");

    // Read it back to verify the round trip.
    let val: u64 = store_get(store_id, "counter").expect("get counter");
    let msg = format!("Store round-trip OK: counter={}\n", val);
    write(&msg);

    // Prove String works: build a greeting dynamically.
    let mut greeting = String::from("Heap works! ");
    greeting.push_str("String + format!() from userspace.\n");
    write(&greeting);

    // Interactive input: read keyboard input via the console store.
    write("Type something (Enter to finish): ");

    let mut input = watch(CONSOLE, "input");
    loop {
        match input.next::<Option<u8>>().await {
            Some(Some(b'\n')) => {
                write("\n");
                return;
            }
            Some(Some(ch)) => {
                let echo = [ch];
                let s = unsafe { core::str::from_utf8_unchecked(&echo) };
                write(s);
            }
            _ => {}
        }
    }
}
