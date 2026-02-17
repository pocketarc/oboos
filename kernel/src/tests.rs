//! Smoke tests for kernel subsystems.
//!
//! These run during boot when the `smoke-test` feature is enabled. They
//! exercise each subsystem just enough to prove it works — allocate, use,
//! free, check invariants. If any assertion fails, the kernel panics with
//! a clear message on the serial console.
//!
//! Run with `make test`. The normal `make run` skips these entirely.

use crate::{arch, executor, memory, println, scheduler, timer};
use crate::arch::TaskContext;
use crate::platform::Platform;

/// Run all smoke tests. Called from `kmain` when `smoke-test` is active.
pub fn run_all() {
    test_heap();
    test_frame_allocator();
    test_paging();
    test_task();
    test_context_switch();
    test_scheduler();
    test_preemption();
    test_async_executor();
    test_async_keyboard();
    test_async_sleep();
    test_block_glyph();
    test_speaker();
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
    let hhdm_ptr = arch::memory::phys_to_virt(frame as u64) as *const u64;
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

    // The task_trampoline enables interrupts when the new task starts,
    // so IF=1 here after the round-trip. Restore IF=0 for subsequent
    // tests that assume interrupts are disabled.
    arch::Arch::disable_interrupts();

    drop(task);
    println!("[ok] Context switch verified");
}

/// Verify cooperative round-robin scheduling with multiple tasks.
///
/// Spawns two tasks (A and B) that each increment a shared atomic counter
/// 5 times, yielding after each increment. kmain yields 5 times to cycle
/// through all three tasks in round-robin order. After the yields, both
/// counters should have reached 5 — proving that the scheduler correctly
/// rotates through tasks and that yield/resume works across multiple tasks.
fn test_scheduler() {
    use core::sync::atomic::{AtomicU32, Ordering};
    use crate::scheduler;

    static COUNTER_A: AtomicU32 = AtomicU32::new(0);
    static COUNTER_B: AtomicU32 = AtomicU32::new(0);

    fn task_a() -> ! {
        use core::sync::atomic::Ordering;
        for _ in 0..5 {
            COUNTER_A.fetch_add(1, Ordering::SeqCst);
            crate::scheduler::yield_now();
        }
        // Work is done — spin-yield forever. No task exit mechanism yet.
        loop {
            crate::scheduler::yield_now();
        }
    }

    fn task_b() -> ! {
        use core::sync::atomic::Ordering;
        for _ in 0..5 {
            COUNTER_B.fetch_add(1, Ordering::SeqCst);
            crate::scheduler::yield_now();
        }
        loop {
            crate::scheduler::yield_now();
        }
    }

    // Reset counters in case the test ever runs more than once.
    COUNTER_A.store(0, Ordering::SeqCst);
    COUNTER_B.store(0, Ordering::SeqCst);

    scheduler::spawn(task_a);
    scheduler::spawn(task_b);
    println!("[test] Scheduler: spawned 2 tasks, yielding 5 times");

    for _ in 0..5 {
        scheduler::yield_now();
    }

    let a = COUNTER_A.load(Ordering::SeqCst);
    let b = COUNTER_B.load(Ordering::SeqCst);
    println!("[test] Scheduler: task A ran {} times, task B ran {} times", a, b);

    assert_eq!(a, 5, "expected task A counter = 5, got {}", a);
    assert_eq!(b, 5, "expected task B counter = 5, got {}", b);

    println!("[ok] Cooperative round-robin scheduler verified");
}

/// Verify preemptive scheduling by spawning a spinning task that never yields.
///
/// The spinning task increments an atomic counter in a tight loop without
/// ever calling `yield_now()`. We enable interrupts and wait — if preemption
/// works, the PIT will fire, expire the spinner's time slice, and switch
/// back to us. We then check that the counter is non-zero, proving the
/// spinner actually ran and was preempted.
fn test_preemption() {
    use core::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn spinning_task() -> ! {
        loop {
            COUNTER.fetch_add(1, Ordering::Relaxed);
        }
    }

    COUNTER.store(0, Ordering::SeqCst);
    scheduler::spawn(spinning_task);

    // Enable interrupts so the PIT can fire and preempt.
    arch::Arch::enable_interrupts();

    // Wait until the spinning task has had a time slice and incremented.
    // Each hlt wakes on a PIT tick (~1ms). After ~20ms (2 time slices)
    // the spinner should have run and we should have been switched back.
    while COUNTER.load(Ordering::SeqCst) == 0 {
        arch::Arch::halt_until_interrupt();
    }

    arch::Arch::disable_interrupts();

    let count = COUNTER.load(Ordering::SeqCst);
    assert!(count > 0, "spinning task never ran — preemption failed");

    println!("[ok] Preemptive scheduling verified (counter = {})", count);
}

/// Verify the async executor's spawn → poll → wake → poll lifecycle.
///
/// Spawns a custom future that returns `Pending` on its first poll
/// (storing its [`Waker`] in a global) and `Ready(())` on the second.
/// Between the two polls, we manually wake the future — simulating
/// what an IRQ handler would do. This is fully deterministic: no
/// timer dependency, just two polls and one wake.
fn test_async_executor() {
    use core::future::Future;
    use core::pin::Pin;
    use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use core::task::{Context, Poll, Waker};

    static DONE: AtomicBool = AtomicBool::new(false);
    static POLL_COUNT: AtomicU32 = AtomicU32::new(0);
    static TEST_WAKER: spin::Mutex<Option<Waker>> = spin::Mutex::new(None);

    struct TestFuture;

    impl Future for TestFuture {
        type Output = ();

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            let count = POLL_COUNT.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                // First poll: stash the waker so the test can wake us later.
                *TEST_WAKER.lock() = Some(cx.waker().clone());
                Poll::Pending
            } else {
                // Second poll: signal completion.
                DONE.store(true, Ordering::SeqCst);
                Poll::Ready(())
            }
        }
    }

    // Reset statics in case the test runs more than once.
    DONE.store(false, Ordering::SeqCst);
    POLL_COUNT.store(0, Ordering::SeqCst);
    *TEST_WAKER.lock() = None;

    executor::spawn(TestFuture);
    executor::poll_once(); // first poll → Pending, waker stored

    assert_eq!(POLL_COUNT.load(Ordering::SeqCst), 1);
    assert!(!DONE.load(Ordering::SeqCst));

    // Simulate an IRQ waking the future. wake() requires IF=0.
    arch::Arch::disable_interrupts();
    TEST_WAKER.lock().take().unwrap().wake();
    arch::Arch::enable_interrupts();

    let completed = executor::poll_once(); // second poll → Ready

    assert!(DONE.load(Ordering::SeqCst));
    assert_eq!(POLL_COUNT.load(Ordering::SeqCst), 2);
    assert_eq!(completed, 1);

    // poll_once() leaves IF=1. Restore IF=0 for subsequent code.
    arch::Arch::disable_interrupts();

    println!("[ok] Async executor verified (future spawned, woken, completed in 2 polls)");
}

/// Verify async keyboard input: push_scancode → next_key → Key::Enter.
///
/// Spawns a future that awaits [`next_key()`], then simulates an IRQ
/// by pushing a scancode into the buffer. The executor polls the future
/// twice: once to store the waker (Pending), once after the scancode
/// arrives (Ready). Also tests that break codes are filtered out.
fn test_async_keyboard() {
    use core::sync::atomic::{AtomicBool, Ordering};

    static GOT_ENTER: AtomicBool = AtomicBool::new(false);
    GOT_ENTER.store(false, Ordering::SeqCst);

    // Spawn a future that awaits one key press and asserts it's Enter.
    executor::spawn(async {
        let key = arch::keyboard::next_key().await;
        assert_eq!(key, crate::platform::Key::Enter);
        GOT_ENTER.store(true, Ordering::SeqCst);
    });

    // First poll — buffer is empty, future stores its waker and returns Pending.
    executor::poll_once();
    assert!(!GOT_ENTER.load(Ordering::SeqCst));

    // Simulate IRQ: push a release code (should be skipped) then a make code.
    // Must be called with IF=0 — same as a real IRQ handler.
    arch::Arch::disable_interrupts();
    arch::keyboard::push_scancode(0x9C); // Enter release (0x1C | 0x80)
    arch::keyboard::push_scancode(0x1C); // Enter press
    arch::Arch::enable_interrupts();

    // Second poll — future finds the scancode, returns Ready(Key::Enter).
    let completed = executor::poll_once();
    assert!(GOT_ENTER.load(Ordering::SeqCst));
    assert_eq!(completed, 1);

    // poll_once() leaves IF=1. Restore IF=0 for subsequent code.
    arch::Arch::disable_interrupts();

    println!("[ok] Async keyboard input verified (push_scancode \u{2192} next_key \u{2192} Enter)");
}

/// Verify async sleep by spawning a future that sleeps for 50 ms.
///
/// The test enables interrupts so the PIT fires and calls
/// `check_deadlines()`, which wakes the sleeping future once the
/// deadline has passed. After enough ticks, the executor polls the
/// future to completion.
fn test_async_sleep() {
    use core::sync::atomic::{AtomicBool, Ordering};

    static DONE: AtomicBool = AtomicBool::new(false);
    DONE.store(false, Ordering::SeqCst);

    executor::spawn(async {
        timer::sleep(50).await;
        DONE.store(true, Ordering::SeqCst);
    });

    // First poll — computes deadline, registers waker, returns Pending.
    executor::poll_once();
    assert!(!DONE.load(Ordering::SeqCst));

    // Enable interrupts and wait for the PIT to tick past the 50 ms
    // deadline. check_deadlines() runs inside the tick handler and
    // will wake our future.
    arch::Arch::enable_interrupts();
    while !DONE.load(Ordering::SeqCst) {
        // Each hlt wakes on a PIT tick (~1 ms). After ~50 ticks the
        // deadline expires, check_deadlines() wakes the future, and
        // the next poll_once() completes it.
        executor::poll_once();
        arch::Arch::halt_until_interrupt();
    }
    arch::Arch::disable_interrupts();

    assert!(DONE.load(Ordering::SeqCst));
    println!("[ok] Async sleep verified (50 ms sleep completed)");
}

/// Verify that the block element glyph lookup returns correct data.
///
/// U+2588 (█ FULL BLOCK) should map to BLOCK_LEGACY[8] which is all
/// 0xFF — every pixel lit in every row.
fn test_block_glyph() {
    use crate::framebuffer;

    let full_block = framebuffer::glyph_for('\u{2588}');
    println!("[test] U+2588 glyph: {:02X?}", full_block);
    assert_eq!(full_block, [0xFF; 8], "U+2588 should be a solid 8x8 block");

    // Sanity check: 'A' should come from BASIC_LEGACY.
    let a_glyph = framebuffer::glyph_for('A');
    println!("[test] 'A' glyph: {:02X?}", a_glyph);
    assert_ne!(a_glyph, [0x00; 8], "'A' glyph should not be empty");

    // U+2580 (▀ upper half block) — top 4 rows lit, bottom 4 empty.
    let upper_half = framebuffer::glyph_for('\u{2580}');
    println!("[test] U+2580 glyph: {:02X?}", upper_half);
    assert_eq!(upper_half, [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]);

    // Render U+2588 into a memory buffer and verify every pixel is lit.
    // Pitch = 8 pixels * 4 bytes = 32 bytes per row. Buffer = 8 rows.
    let mut buf = [0u8; 32 * 8];
    let pitch = 32;
    framebuffer::draw_str(buf.as_mut_ptr(), pitch, 0, 0, "\u{2588}", framebuffer::Color::WHITE);

    let mut lit = 0;
    let mut unlit = 0;
    for row in 0..8 {
        for col in 0..8 {
            let offset = row * pitch + col * 4;
            let b = buf[offset];
            let g = buf[offset + 1];
            let r = buf[offset + 2];
            if r == 0xFF && g == 0xFF && b == 0xFF {
                lit += 1;
            } else {
                unlit += 1;
                println!("[test] UNLIT pixel at row={}, col={}: r={:02X} g={:02X} b={:02X}", row, col, r, g, b);
            }
        }
    }
    println!("[test] Full block render: {}/64 pixels lit, {}/64 unlit", lit, unlit);
    assert_eq!(lit, 64, "full block should light all 64 pixels");

    println!("[ok] Block element glyph lookup verified");
}

/// Verify PC speaker beep/stop by checking port 0x61 bits.
///
/// Can't test `play_tone()` here because it's async and needs the PIT
/// running with interrupts enabled. But beep/stop are synchronous port
/// I/O — we just verify the speaker enable bit toggles correctly.
fn test_speaker() {
    arch::speaker::beep(440); // A4 — concert pitch
    assert!(arch::speaker::is_enabled(), "speaker should be enabled after beep");
    arch::speaker::stop();
    assert!(!arch::speaker::is_enabled(), "speaker should be disabled after stop");
    println!("[ok] PC speaker beep/stop verified");
}
