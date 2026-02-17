//! Smoke tests for kernel subsystems.
//!
//! These run during boot when the `smoke-test` feature is enabled. They
//! exercise each subsystem just enough to prove it works — allocate, use,
//! free, check invariants. If any assertion fails, the kernel panics with
//! a clear message on the serial console.
//!
//! Run with `make test`. The normal `make run` skips these entirely.

use crate::{arch, memory, println};
use crate::arch::TaskContext;

/// Run all smoke tests. Called from `kmain` when `smoke-test` is active.
pub fn run_all() {
    test_heap();
    test_frame_allocator();
    test_paging();
    test_task();
    test_context_switch();
    println!();
    println!("[ok] All smoke tests passed");
}

/// Verify the heap allocator handles Vec, Box, and String correctly.
fn test_heap() {
    use alloc::boxed::Box;
    use alloc::string::String;
    use alloc::vec::Vec;

    let mut v = Vec::new();
    v.push(1u32);
    v.push(2);
    v.push(3);
    assert_eq!(v.iter().sum::<u32>(), 6);

    let b = Box::new(42u64);
    assert_eq!(*b, 42);

    let mut s = String::from("OBOOS");
    s.push_str(" heap works!");
    assert_eq!(s, "OBOOS heap works!");

    println!("[ok] Heap allocator verified (Vec, Box, String)");
}

/// Verify frame allocation, freeing, and recycling.
fn test_frame_allocator() {
    let f1 = memory::alloc_frame().expect("alloc frame 1");
    let f2 = memory::alloc_frame().expect("alloc frame 2");
    let f3 = memory::alloc_frame().expect("alloc frame 3");
    println!("[test] Allocated frame: {:#018X}", f1);
    println!("[test] Allocated frame: {:#018X}", f2);
    println!("[test] Allocated frame: {:#018X}", f3);

    memory::free_frame(f2);
    println!("[test] Freed frame:     {:#018X}", f2);

    let f4 = memory::alloc_frame().expect("alloc frame 4");
    println!("[test] Re-allocated:    {:#018X} (recycled: {})", f4, f4 == f2);

    memory::free_frame(f1);
    memory::free_frame(f3);
    memory::free_frame(f4);
    println!("[test] Free frames remaining: {}", memory::free_frame_count());
    println!("[ok] Frame allocator verified");
}

/// Verify page table map/unmap by writing through a new mapping and
/// cross-checking via the HHDM.
fn test_paging() {
    use crate::platform::{MemoryManager, PageFlags};

    let test_virt: usize = 0xFFFF_FFFF_C000_0000;
    let frame = memory::alloc_frame().expect("alloc frame for paging test");
    println!("[test] Paging: mapping {:#018X} -> {:#018X}", test_virt, frame);

    arch::Arch::map_page(test_virt, frame, PageFlags::PRESENT | PageFlags::WRITABLE);

    // Write a magic value through the new mapping.
    let ptr = test_virt as *mut u64;
    unsafe { core::ptr::write_volatile(ptr, 0xDEAD_BEEF_CAFE_BABE) };
    let readback = unsafe { core::ptr::read_volatile(ptr) };
    assert_eq!(readback, 0xDEAD_BEEF_CAFE_BABE);
    println!("[test] Paging: write/read through mapped page succeeded");

    // Cross-check: read the same physical frame through the HHDM.
    // If our mapping points to the right frame, both views see the same data.
    let hhdm_ptr = arch::x86_64::memory::phys_to_virt(frame as u64) as *const u64;
    let hhdm_val = unsafe { core::ptr::read_volatile(hhdm_ptr) };
    assert_eq!(hhdm_val, 0xDEAD_BEEF_CAFE_BABE);
    println!("[test] Paging: HHDM cross-check confirmed");

    arch::Arch::unmap_page(test_virt);
    memory::free_frame(frame);
    println!("[ok] Page table manipulation verified");
}

/// Verify task creation, stack allocation, initial context, and RAII cleanup.
fn test_task() {
    use crate::task::{Task, TaskId, TaskState};

    // Bootstrap task: id=0, Running, no allocated stack.
    let boot = Task::bootstrap();
    assert_eq!(boot.id, TaskId(0));
    assert_eq!(boot.state, TaskState::Running);
    assert!(boot.stack_top().is_none());
    println!("[test] Bootstrap task created (id=0, no allocated stack)");

    // Dummy entry point — tasks must diverge (-> !).
    fn dummy_entry() -> ! {
        loop {
            core::hint::spin_loop();
        }
    }

    // Create a new task and verify its stack, then drop it and check cleanup.
    let task = Task::new(dummy_entry);
    assert_eq!(task.id, TaskId(1));
    assert_eq!(task.state, TaskState::Ready);
    let top = task.stack_top().expect("new task should have a stack");
    println!("[test] Task created (id=1, stack_top={:#018X})", top);

    // Write/read through the stack virtual address to verify mapping works.
    let test_ptr = (top - 4096) as *mut u64;
    unsafe {
        core::ptr::write_volatile(test_ptr, 0xCAFE_BABE_DEAD_BEEF);
        let readback = core::ptr::read_volatile(test_ptr);
        assert_eq!(readback, 0xCAFE_BABE_DEAD_BEEF);
    }
    println!("[test] Stack write/read verified");

    // Verify the entry point was placed at the top of the initial stack frame.
    let ret_addr_ptr = (top - 8) as *const u64;
    let ret_addr = unsafe { core::ptr::read_volatile(ret_addr_ptr) };
    assert_eq!(ret_addr, dummy_entry as *const () as u64);
    println!("[test] Initial context verified (entry_point at top of stack)");

    // Drop the task and verify stack frames are returned to the allocator.
    let free_before_drop = memory::free_frame_count();
    drop(task);
    let free_after_drop = memory::free_frame_count();
    assert_eq!(
        free_after_drop - free_before_drop,
        4,
        "expected 4 stack frames freed, got {}",
        free_after_drop - free_before_drop
    );
    println!("[test] Task drop freed stack frames correctly");

    println!("[ok] Task creation and stack allocation verified");
}

/// Verify context switching by doing a round-trip: switch from the boot
/// context into a new task, which sets a flag and switches back. If we
/// resume and the flag is set, the switch worked in both directions.
fn test_context_switch() {
    use core::sync::atomic::{AtomicBool, Ordering};
    use crate::platform::ContextSwitch;
    use crate::task::Task;

    static FLAG: AtomicBool = AtomicBool::new(false);

    // Two TaskContext values: one for the boot side of this test, one
    // for the spawned task. We use `static mut` because the switch
    // assembly needs stable addresses, and we're single-threaded with
    // interrupts disabled during tests.
    static mut BOOT_CTX: TaskContext = TaskContext::zero();
    static mut TASK_CTX: TaskContext = TaskContext::zero();

    fn task_entry() -> ! {
        // We've landed in the new task. Set the flag to prove we got here,
        // then switch back to the boot context so the test can verify.
        FLAG.store(true, Ordering::SeqCst);

        unsafe {
            arch::Arch::switch(
                &mut *(&raw mut TASK_CTX),
                &*(&raw const BOOT_CTX),
            );
        }

        // Should never reach here — the boot side drops the task after
        // verifying the flag. But tasks must diverge.
        loop {
            core::hint::spin_loop();
        }
    }

    let task = Task::new(task_entry);
    println!("[test] Context switch: created task {}", task.id.0);

    // Copy the task's pre-built context into the static so the assembly
    // can save/restore through stable addresses.
    unsafe {
        *(&raw mut TASK_CTX) = task.context;
    }

    // Switch into the new task. When it switches back, we resume here.
    unsafe {
        arch::Arch::switch(
            &mut *(&raw mut BOOT_CTX),
            &*(&raw const TASK_CTX),
        );
    }

    assert!(FLAG.load(Ordering::SeqCst), "task entry point never ran");
    println!("[test] Context switch: round-trip verified (flag was set)");

    drop(task);
    println!("[ok] Context switch verified");
}
