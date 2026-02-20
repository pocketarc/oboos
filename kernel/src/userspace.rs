//! Ring 3 (user mode) test harness with ELF loading, store syscalls, and
//! process lifecycle tracking.
//!
//! This module proves that the full userspace stack works: an ELF binary
//! compiled from Rust is loaded into user-space memory, tracked through
//! the process table, dropped to Ring 3, and communicates with the kernel
//! via store syscalls. The user program writes a value to a kernel store
//! field, reads it back, and exits. The kernel then verifies both the
//! data store (counter=42) and the process store (status=exited, exit_code=0).
//!
//! ## Memory layout
//!
//! ```text
//! 0x0040_0000+  ELF segments  (PRESENT | USER, permissions from ELF flags)
//! 0x0080_0000           User stack    (PRESENT | WRITABLE | USER | NO_EXECUTE)
//! 0xFFFF_FD00_...       Syscall kernel stack — 16 KiB + guard page (PRESENT | WRITABLE | NO_EXECUTE)
//! ```
//!
//! ## Why a separate syscall kernel stack?
//!
//! When the user program SYSCALLs, the entry stub loads KERNEL_RSP and
//! pushes onto it. If KERNEL_RSP pointed into the same stack as the saved
//! return context (from [`save_return_context`]), the syscall handler's
//! pushes would overwrite the saved registers. A separate page avoids this.
//!
//! This same stack also serves as TSS.RSP0 — used by the CPU when an
//! interrupt (like the PIT) fires while in Ring 3.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::arch;
use crate::elf;
use crate::executor;
use crate::memory::{self, FRAME_SIZE};
use crate::platform::{MemoryManager, PageFlags};
use crate::platform::Platform;

use crate::println;
use crate::process;
use crate::store;
use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

// ————————————————————————————————————————————————————————————————————————————
// Syscall kernel stack allocator
// ————————————————————————————————————————————————————————————————————————————

/// Base virtual address for syscall kernel stacks. Sits below task stacks
/// (`0xFFFF_FE00_0000_0000`) with 1 TiB of gap between them.
const KERN_STACK_REGION_BASE: usize = 0xFFFF_FD00_0000_0000;

/// Number of 4 KiB pages per syscall kernel stack (16 KiB total).
/// Matches task stack size, and gives comfortable room for serialization
/// buffers, nested calls, and interrupt frames that can arrive via TSS.RSP0.
const KERN_STACK_PAGES: usize = 4;

/// Pages per slot: 1 guard page + [`KERN_STACK_PAGES`] mapped pages.
const KERN_SLOT_PAGES: usize = KERN_STACK_PAGES + 1;

/// Bump counter for syscall kernel stack slots.
static NEXT_KERN_SLOT: AtomicU64 = AtomicU64::new(0);

/// Tracks the physical frames and virtual address range of a syscall kernel
/// stack, so the caller can free it when the process exits.
struct KernelStackAlloc {
    /// Virtual address of the first mapped page (just above the guard page).
    stack_bottom: usize,
    /// Virtual address one past the last mapped byte — the initial RSP
    /// (x86 stacks grow downward).
    stack_top: usize,
    /// Physical frames backing the stack pages.
    frames: [usize; KERN_STACK_PAGES],
}

/// Allocate a 16 KiB syscall kernel stack with a guard page below.
///
/// Same layout as task stacks in [`task.rs`]: bump-allocate a virtual slot,
/// leave page 0 unmapped (guard), map pages 1–4 with PRESENT | WRITABLE |
/// NO_EXECUTE (no USER — this is a kernel-only stack). All mapped pages are
/// zeroed.
fn alloc_kernel_stack() -> KernelStackAlloc {
    let slot = NEXT_KERN_SLOT.fetch_add(1, Ordering::Relaxed) as usize;
    let slot_base = KERN_STACK_REGION_BASE + slot * KERN_SLOT_PAGES * FRAME_SIZE;
    let stack_bottom = slot_base + FRAME_SIZE; // skip guard page
    let stack_top = slot_base + KERN_SLOT_PAGES * FRAME_SIZE;

    let flags = PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::NO_EXECUTE;
    let mut frames = [0usize; KERN_STACK_PAGES];

    for i in 0..KERN_STACK_PAGES {
        let frame = memory::alloc_frame().expect("out of memory allocating syscall kernel stack");
        let virt = stack_bottom + i * FRAME_SIZE;
        arch::Arch::map_page(virt, frame, flags);
        frames[i] = frame;
    }

    // Zero all stack pages through the virtual mapping.
    unsafe {
        core::ptr::write_bytes(stack_bottom as *mut u8, 0, KERN_STACK_PAGES * FRAME_SIZE);
    }

    KernelStackAlloc { stack_bottom, stack_top, frames }
}

/// Unmap and free a syscall kernel stack previously allocated by
/// [`alloc_kernel_stack`].
fn free_kernel_stack(alloc: KernelStackAlloc) {
    for i in 0..KERN_STACK_PAGES {
        let virt = alloc.stack_bottom + i * FRAME_SIZE;
        arch::Arch::unmap_page(virt);
        memory::free_frame(alloc.frames[i]);
    }
}

/// The userspace ELF binary, compiled separately and embedded at build time.
/// The Makefile ensures this is built before the kernel.
static USER_ELF: &[u8] = include_bytes!("../../userspace/hello/target/x86_64-unknown-none/debug/oboos-hello");

/// Virtual address for the user stack (top of page = initial RSP).
const USER_STACK_VIRT: usize = 0x0080_0000;

/// Store schema for the userspace test — a single U64 counter field.
///
/// This is the application data store, separate from the process lifecycle
/// store that the process table manages automatically.
struct UserTestSchema;

impl StoreSchema for UserTestSchema {
    fn name() -> &'static str { "UserTest" }
    fn fields() -> &'static [FieldDef] {
        &[FieldDef { name: "counter", kind: FieldKind::U64 }]
    }
}

/// Run the Ring 3 store round-trip test with process lifecycle tracking.
///
/// 1. Spawn a process (creates PID + process store)
/// 2. Load the userspace ELF binary
/// 3. Allocate user stack and syscall kernel stack
/// 4. Create the application data store with `counter = 0`
/// 5. Set the current process and start it
/// 6. Drop to Ring 3, passing the data store ID
/// 7. The user program sets counter=42 and exits (SYS_EXIT sets process status)
/// 8. Verify data store (counter=42) and process store (status=exited)
/// 9. Clean up everything
pub fn run_ring3_smoke_test() {
    // ── Step 1: Spawn a process ────────────────────────────────────
    let pid = process::spawn("hello");
    println!("[user] Spawned process \"hello\" (pid={})", pid.as_raw());

    // ── Step 2: Load the ELF binary ────────────────────────────────
    let loaded = elf::load_elf(USER_ELF);

    // ── Step 3: Allocate and map the user stack page ───────────────
    let stack_frame = memory::alloc_frame().expect("alloc user stack frame");
    arch::Arch::map_page(
        USER_STACK_VIRT,
        stack_frame,
        PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::USER | PageFlags::NO_EXECUTE,
    );
    let user_rsp = (USER_STACK_VIRT + memory::FRAME_SIZE) as u64;

    // ── Step 4: Allocate the syscall kernel stack ──────────────────
    let kern_stack = alloc_kernel_stack();
    let kern_stack_rsp = kern_stack.stack_top as u64;

    arch::syscall::set_kernel_rsp(kern_stack_rsp);
    unsafe {
        arch::gdt::set_rsp0(kern_stack_rsp);
    }

    // ── Step 5: Create the console and application data stores ─────
    let console_store_id = arch::syscall::create_console_store();
    println!("[user] Created console store (id={})", console_store_id.as_raw());

    // Spawn the async console driver and keyboard input driver, then
    // do an initial poll so their watch() calls register subscribers
    // before userspace starts writing.
    executor::spawn(arch::syscall::console_driver());
    executor::spawn(arch::syscall::keyboard_input_driver());

    let data_store_id = store::create::<UserTestSchema>(&[
        ("counter", Value::U64(0)),
    ]).expect("create user test store");

    println!("[user] Created test store (id={}, counter=0)", data_store_id.as_raw());

    // ── Step 6: Set current process and start it ───────────────────
    process::set_current(pid);
    process::start(pid);

    // ── Step 7: Drop to Ring 3 ─────────────────────────────────────
    //
    // The hello program uses SYS_SUBSCRIBE/SYS_YIELD for keyboard input.
    // During smoke tests, pre-fill the keyboard buffer with "hi\n"
    // scancodes. The keyboard_input_driver async task converts them to
    // ASCII and pushes them into the console store's input queue.
    #[cfg(feature = "smoke-test")]
    {
        arch::Arch::disable_interrupts();
        arch::keyboard::push_scancode(0x23); // 'h' make
        arch::keyboard::push_scancode(0x17); // 'i' make
        arch::keyboard::push_scancode(0x1C); // Enter make
        arch::Arch::enable_interrupts();
    }

    // Initial poll drains any pre-filled scancodes through the input
    // driver into the console store's input queue.
    executor::poll_once();

    println!("[user] Jumping to Ring 3...");
    arch::keyboard::set_console_mode(true);
    unsafe {
        arch::syscall::jump_to_ring3(loaded.entry, user_rsp, data_store_id.as_raw());
    }
    arch::keyboard::set_console_mode(false);

    // ── Step 8: Back from Ring 3 — clean up subscriptions and verify ─
    arch::syscall::clear_all_subscriptions();
    process::clear_current();

    // Verify the data store was modified by userspace.
    let counter = store::get(data_store_id, "counter").expect("get counter after Ring 3");
    assert_eq!(counter, Value::U64(42), "expected counter=42, got {:?}", counter);
    println!("[ok] Ring 3 store round-trip verified (counter=42)");

    // Verify the process store reflects the exit.
    let proc_store_id = process::store_id(pid).expect("process store should exist");
    let status = store::get(proc_store_id, "status").expect("get process status");
    let exit_code = store::get(proc_store_id, "exit_code").expect("get process exit_code");
    assert_eq!(status, Value::Str(alloc::string::String::from("exited")),
        "expected status=exited, got {:?}", status);
    assert_eq!(exit_code, Value::U64(0),
        "expected exit_code=0, got {:?}", exit_code);
    println!("[ok] Process store verified (status=exited, exit_code=0)");

    // ── Step 9: Clean up ───────────────────────────────────────────
    arch::syscall::destroy_console_store(console_store_id);
    store::destroy(data_store_id).expect("destroy user test store");
    process::destroy(pid);
    loaded.unload();

    arch::Arch::unmap_page(USER_STACK_VIRT);
    memory::free_frame(stack_frame);
    free_kernel_stack(kern_stack);
}

/// Launch the hello program interactively from the splash screen.
///
/// Same setup as [`run_ring3_smoke_test`] but without assertions — just
/// runs the program, prints the exit status, and cleans up. Keyboard
/// input is live (no pre-filled scancodes).
pub fn run_hello_interactive() {
    // Disable interrupts while we set up (same pattern as the smoke test).
    arch::Arch::disable_interrupts();

    let pid = process::spawn("hello");
    println!("[user] Spawned process \"hello\" (pid={})", pid.as_raw());

    let loaded = elf::load_elf(USER_ELF);

    let stack_frame = memory::alloc_frame().expect("alloc user stack frame");
    arch::Arch::map_page(
        USER_STACK_VIRT,
        stack_frame,
        PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::USER | PageFlags::NO_EXECUTE,
    );
    let user_rsp = (USER_STACK_VIRT + memory::FRAME_SIZE) as u64;

    let kern_stack = alloc_kernel_stack();
    let kern_stack_rsp = kern_stack.stack_top as u64;

    arch::syscall::set_kernel_rsp(kern_stack_rsp);
    unsafe { arch::gdt::set_rsp0(kern_stack_rsp); }

    let console_store_id = arch::syscall::create_console_store();
    executor::spawn(arch::syscall::console_driver());
    executor::spawn(arch::syscall::keyboard_input_driver());
    executor::poll_once();

    let data_store_id = store::create::<UserTestSchema>(&[
        ("counter", Value::U64(0)),
    ]).expect("create user data store");

    process::set_current(pid);
    process::start(pid);

    println!("[user] Jumping to Ring 3 (interactive)...");
    arch::keyboard::set_console_mode(true);
    unsafe {
        arch::syscall::jump_to_ring3(loaded.entry, user_rsp, data_store_id.as_raw());
    }
    arch::keyboard::set_console_mode(false);

    arch::syscall::clear_all_subscriptions();
    process::clear_current();

    // Report exit status.
    let proc_store_id = process::store_id(pid).expect("process store should exist");
    let exit_code = match store::get(proc_store_id, "exit_code") {
        Ok(Value::U64(v)) => v,
        _ => u64::MAX,
    };
    println!("[user] Process exited (code={})", exit_code);

    // Clean up.
    arch::syscall::destroy_console_store(console_store_id);
    store::destroy(data_store_id).ok();
    process::destroy(pid);
    loaded.unload();

    arch::Arch::unmap_page(USER_STACK_VIRT);
    memory::free_frame(stack_frame);
    free_kernel_stack(kern_stack);
}
