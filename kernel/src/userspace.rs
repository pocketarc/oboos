//! Ring 3 (user mode) test harness with ELF loading and store syscalls.
//!
//! This module proves that the full userspace stack works: an ELF binary
//! compiled from Rust is loaded into user-space memory, dropped to Ring 3,
//! and communicates with the kernel via store syscalls. The user program
//! writes a value to a kernel store field, reads it back, and exits.
//! The kernel then verifies the store was correctly modified.
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
use crate::store;
use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

/// The userspace ELF binary, compiled separately and embedded at build time.
/// The Makefile ensures this is built before the kernel.
static USER_ELF: &[u8] = include_bytes!("../../userspace/hello/target/x86_64-unknown-none/debug/oboos-hello");

/// Virtual address for the user stack (top of page = initial RSP).
const USER_STACK_VIRT: usize = 0x0080_0000;

/// Store schema for the userspace test — a single U64 counter field.
struct UserTestSchema;

impl StoreSchema for UserTestSchema {
    fn name() -> &'static str { "UserTest" }
    fn fields() -> &'static [FieldDef] {
        &[FieldDef { name: "counter", kind: FieldKind::U64 }]
    }
}

/// Run the Ring 3 store round-trip test.
///
/// 1. Load the userspace ELF binary
/// 2. Allocate user stack and syscall kernel stack
/// 3. Create a test store with `counter = 0`
/// 4. Drop to Ring 3, passing the store ID
/// 5. The user program sets counter=42 and exits
/// 6. Verify counter==42 from the kernel side
/// 7. Clean up everything
pub fn run_ring3_smoke_test() {
    // ── Step 1: Load the ELF binary ─────────────────────────────────
    let loaded = elf::load_elf(USER_ELF);

    // ── Step 2: Allocate and map the user stack page ────────────────
    let stack_frame = memory::alloc_frame().expect("alloc user stack frame");
    arch::Arch::map_page(
        USER_STACK_VIRT,
        stack_frame,
        PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::USER | PageFlags::NO_EXECUTE,
    );
    let user_rsp = (USER_STACK_VIRT + memory::FRAME_SIZE) as u64;

    // ── Step 3: Allocate the syscall kernel stack ───────────────────
    let kern_stack_frame = memory::alloc_frame().expect("alloc syscall kernel stack");
    let kern_stack_top = arch::memory::phys_to_virt(kern_stack_frame as u64);
    let kern_stack_rsp = kern_stack_top as u64 + memory::FRAME_SIZE as u64;

    arch::syscall::set_kernel_rsp(kern_stack_rsp);
    unsafe {
        arch::gdt::set_rsp0(kern_stack_rsp);
    }

    // ── Step 4: Create the test store ───────────────────────────────
    let store_id = store::create::<UserTestSchema>(&[
        ("counter", Value::U64(0)),
    ]).expect("create user test store");

    println!("[user] Created test store (id={}, counter=0)", store_id.as_raw());

    // ── Step 5: Drop to Ring 3 ──────────────────────────────────────
    println!("[user] Jumping to Ring 3...");
    unsafe {
        arch::syscall::jump_to_ring3(loaded.entry, user_rsp, store_id.as_raw());
    }

    // ── Step 6: Verify the store was modified by userspace ──────────
    let counter = store::get(store_id, "counter").expect("get counter after Ring 3");
    assert_eq!(counter, Value::U64(42), "expected counter=42, got {:?}", counter);
    println!("[ok] Ring 3 store round-trip verified (counter=42)");

    // ── Step 7: Clean up ────────────────────────────────────────────
    store::destroy(store_id).expect("destroy user test store");
    loaded.unload();

    arch::Arch::unmap_page(USER_STACK_VIRT);
    memory::free_frame(stack_frame);
    memory::free_frame(kern_stack_frame);
}
