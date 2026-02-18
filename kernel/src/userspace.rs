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
//! 0x0080_0000   User stack    (PRESENT | WRITABLE | USER | NO_EXECUTE)
//! kernel-space  Syscall kernel stack (PRESENT | WRITABLE, no USER)
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

use crate::arch;
use crate::elf;
use crate::memory;
use crate::platform::{MemoryManager, PageFlags};
use crate::println;
use crate::process;
use crate::store;
use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

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
    let kern_stack_frame = memory::alloc_frame().expect("alloc syscall kernel stack");
    let kern_stack_top = arch::memory::phys_to_virt(kern_stack_frame as u64);
    let kern_stack_rsp = kern_stack_top as u64 + memory::FRAME_SIZE as u64;

    arch::syscall::set_kernel_rsp(kern_stack_rsp);
    unsafe {
        arch::gdt::set_rsp0(kern_stack_rsp);
    }

    // ── Step 5: Create the application data store ──────────────────
    let data_store_id = store::create::<UserTestSchema>(&[
        ("counter", Value::U64(0)),
    ]).expect("create user test store");

    println!("[user] Created test store (id={}, counter=0)", data_store_id.as_raw());

    // ── Step 6: Set current process and start it ───────────────────
    process::set_current(pid);
    process::start(pid);

    // ── Step 7: Drop to Ring 3 ─────────────────────────────────────
    println!("[user] Jumping to Ring 3...");
    unsafe {
        arch::syscall::jump_to_ring3(loaded.entry, user_rsp, data_store_id.as_raw());
    }

    // ── Step 8: Back from Ring 3 — clear current and verify ────────
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
    store::destroy(data_store_id).expect("destroy user test store");
    process::destroy(pid);
    loaded.unload();

    arch::Arch::unmap_page(USER_STACK_VIRT);
    memory::free_frame(stack_frame);
    memory::free_frame(kern_stack_frame);
}
