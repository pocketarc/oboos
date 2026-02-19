//! Smoke tests for kernel subsystems.
//!
//! These run during boot when the `smoke-test` feature is enabled. They
//! exercise each subsystem just enough to prove it works — allocate, use,
//! free, check invariants. If any assertion fails, the kernel panics with
//! a clear message on the serial console.
//!
//! Run with `make test`. The normal `make run` skips these entirely.

use crate::{arch, executor, memory, println, process, scheduler, store, timer, userspace};
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
    test_store_basic();
    test_store_validation();
    test_store_subscribe();
    test_store_queue();
    test_store_queue_watch();
    test_process_lifecycle();
    test_process_store_watch();
    test_error_codes();
    test_multi_field_set();
    test_persistent_watcher();
    test_ring3();
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

/// Verify store create, get, set, and destroy.
///
/// Creates a 3-field store (count/U32, label/Str, active/Bool), reads
/// defaults back, writes new values, reads them back, then destroys the
/// store and verifies it's gone.
fn test_store_basic() {
    use alloc::string::String;
    use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

    struct CounterSchema;
    impl StoreSchema for CounterSchema {
        fn name() -> &'static str { "Counter" }
        fn fields() -> &'static [FieldDef] {
            &[
                FieldDef { name: "count", kind: FieldKind::U32 },
                FieldDef { name: "label", kind: FieldKind::Str },
                FieldDef { name: "active", kind: FieldKind::Bool },
            ]
        }
    }

    // Create with defaults.
    let id = store::create::<CounterSchema>(&[
        ("count", Value::U32(0)),
        ("label", Value::Str(String::from("hello"))),
        ("active", Value::Bool(true)),
    ])
    .expect("create failed");

    // Read defaults back.
    assert_eq!(store::get(id, "count").unwrap(), Value::U32(0));
    assert_eq!(store::get(id, "label").unwrap(), Value::Str(String::from("hello")));
    assert_eq!(store::get(id, "active").unwrap(), Value::Bool(true));

    // Write new values atomically.
    store::set(id, &[
        ("count", Value::U32(42)),
        ("label", Value::Str(String::from("world"))),
        ("active", Value::Bool(false)),
    ]).expect("set fields");

    // Read them back.
    assert_eq!(store::get(id, "count").unwrap(), Value::U32(42));
    assert_eq!(store::get(id, "label").unwrap(), Value::Str(String::from("world")));
    assert_eq!(store::get(id, "active").unwrap(), Value::Bool(false));

    // Destroy and verify NotFound.
    store::destroy(id).expect("destroy");
    assert!(store::get(id, "count").is_err());
    assert!(store::set(id, &[("count", Value::U32(1))]).is_err());
    assert!(store::destroy(id).is_err());

    println!("[ok] Store basic operations verified (create, get, set, destroy)");
}

/// Verify store schema validation — unknown fields and type mismatches.
///
/// Creates a 2-field store (flag/Bool, score/U32) and verifies that
/// accessing nonexistent fields and writing wrong types both produce
/// the correct errors, while valid operations still work after errors.
fn test_store_validation() {
    use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

    struct TinySchema;
    impl StoreSchema for TinySchema {
        fn name() -> &'static str { "Tiny" }
        fn fields() -> &'static [FieldDef] {
            &[
                FieldDef { name: "flag", kind: FieldKind::Bool },
                FieldDef { name: "score", kind: FieldKind::U32 },
            ]
        }
    }

    let id = store::create::<TinySchema>(&[
        ("flag", Value::Bool(false)),
        ("score", Value::U32(100)),
    ])
    .expect("create failed");

    // Unknown field on get.
    match store::get(id, "nonexistent") {
        Err(_) => {}
        Ok(_) => panic!("expected UnknownField error on get"),
    }

    // Unknown field on set.
    match store::set(id, &[("nonexistent", Value::U32(1))]) {
        Err(_) => {}
        Ok(_) => panic!("expected UnknownField error on set"),
    }

    // Type mismatch: write U32 to a Bool field.
    match store::set(id, &[("flag", Value::U32(1))]) {
        Err(_) => {}
        Ok(_) => panic!("expected TypeMismatch error"),
    }

    // Type mismatch: write Bool to a U32 field.
    match store::set(id, &[("score", Value::Bool(true))]) {
        Err(_) => {}
        Ok(_) => panic!("expected TypeMismatch error"),
    }

    // All-or-nothing: if one field in a batch fails, nothing is written.
    let old_flag = store::get(id, "flag").unwrap();
    match store::set(id, &[("flag", Value::Bool(true)), ("score", Value::Bool(false))]) {
        Err(_) => {}
        Ok(_) => panic!("expected TypeMismatch error on batch"),
    }
    assert_eq!(store::get(id, "flag").unwrap(), old_flag, "flag should be unchanged after failed batch");

    // Valid operations still work after errors.
    store::set(id, &[("flag", Value::Bool(true))]).expect("valid set after error");
    assert_eq!(store::get(id, "flag").unwrap(), Value::Bool(true));
    store::set(id, &[("score", Value::U32(200))]).expect("valid set after error");
    assert_eq!(store::get(id, "score").unwrap(), Value::U32(200));

    store::destroy(id).expect("cleanup");
    println!("[ok] Store schema validation verified (unknown field, type mismatch)");
}

/// Verify store subscriptions: watch() → set() → waker fires → value received.
///
/// Deterministic, no timer or IRQ dependency. Creates a 1-field store,
/// spawns a future that watches the field, manually drives it through
/// subscribe → wake → read via poll_once() and set().
fn test_store_subscribe() {
    use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

    struct WatchSchema;
    impl StoreSchema for WatchSchema {
        fn name() -> &'static str { "Watch" }
        fn fields() -> &'static [FieldDef] {
            &[FieldDef { name: "counter", kind: FieldKind::U32 }]
        }
    }

    static DONE: AtomicBool = AtomicBool::new(false);
    static RECEIVED: AtomicU32 = AtomicU32::new(0);
    DONE.store(false, Ordering::SeqCst);
    RECEIVED.store(0, Ordering::SeqCst);

    let id = store::create::<WatchSchema>(&[
        ("counter", Value::U32(0)),
    ]).expect("create watch store");

    // Spawn a future that watches the counter field.
    executor::spawn(async move {
        if store::watch(id, &["counter"]).await.is_err() {
            panic!("watch failed");
        }
        match store::get(id, "counter") {
            Ok(Value::U32(v)) => {
                RECEIVED.store(v, Ordering::SeqCst);
                DONE.store(true, Ordering::SeqCst);
            }
            other => panic!("unexpected get result: {:?}", other),
        }
    });

    // First poll: the future calls subscribe(), returns Pending.
    executor::poll_once();
    assert!(!DONE.load(Ordering::SeqCst), "future should be pending after first poll");

    // Write a new value — this wakes the subscriber.
    store::set(id, &[("counter", Value::U32(42))]).expect("set counter");

    // Second poll: the future reads the new value and completes.
    let completed = executor::poll_once();
    assert!(DONE.load(Ordering::SeqCst), "future should be done after second poll");
    assert_eq!(RECEIVED.load(Ordering::SeqCst), 42);
    assert_eq!(completed, 1);

    store::destroy(id).expect("cleanup watch store");
    println!("[ok] Store subscription verified (watch wakes on set)");
}

/// Verify Queue field operations: push, pop, drain, empty checks, type validation.
///
/// Creates a store with a Queue(Str) field, exercises the full FIFO lifecycle,
/// and verifies type mismatches are rejected.
fn test_store_queue() {
    use alloc::string::String;
    use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

    struct QueueSchema;
    impl StoreSchema for QueueSchema {
        fn name() -> &'static str { "QueueTest" }
        fn fields() -> &'static [FieldDef] {
            &[FieldDef { name: "msgs", kind: FieldKind::Queue(&FieldKind::Str) }]
        }
    }

    let id = store::create::<QueueSchema>(&[
        ("msgs", Value::Queue(alloc::collections::VecDeque::new())),
    ]).expect("create queue store");

    // Pop from empty queue returns None.
    assert_eq!(store::pop(id, "msgs").unwrap(), None);

    // Push 3 elements.
    store::push(id, "msgs", Value::Str(String::from("a"))).expect("push a");
    store::push(id, "msgs", Value::Str(String::from("b"))).expect("push b");
    store::push(id, "msgs", Value::Str(String::from("c"))).expect("push c");

    // Pop 2 in FIFO order.
    assert_eq!(store::pop(id, "msgs").unwrap(), Some(Value::Str(String::from("a"))));
    assert_eq!(store::pop(id, "msgs").unwrap(), Some(Value::Str(String::from("b"))));

    // Drain remaining 1.
    let remaining = store::drain(id, "msgs").unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0], Value::Str(String::from("c")));

    // Queue is empty again.
    assert_eq!(store::pop(id, "msgs").unwrap(), None);
    assert!(store::drain(id, "msgs").unwrap().is_empty());

    // Type mismatch: push a U32 into a Queue(Str).
    assert!(store::push(id, "msgs", Value::U32(42)).is_err());

    store::destroy(id).expect("cleanup queue store");
    println!("[ok] Store queue operations verified (push, pop, drain, type validation)");
}

/// Verify that push on a Queue field wakes subscribers, just like set() on scalars.
///
/// Creates a Queue(U32) store, spawns a watcher, pushes a value, and verifies
/// the watcher is woken and can pop the value.
fn test_store_queue_watch() {
    use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

    struct QueueWatchSchema;
    impl StoreSchema for QueueWatchSchema {
        fn name() -> &'static str { "QueueWatch" }
        fn fields() -> &'static [FieldDef] {
            &[FieldDef { name: "items", kind: FieldKind::Queue(&FieldKind::U32) }]
        }
    }

    static DONE: AtomicBool = AtomicBool::new(false);
    static RECEIVED: AtomicU32 = AtomicU32::new(0);
    DONE.store(false, Ordering::SeqCst);
    RECEIVED.store(0, Ordering::SeqCst);

    let id = store::create::<QueueWatchSchema>(&[
        ("items", Value::Queue(alloc::collections::VecDeque::new())),
    ]).expect("create queue watch store");

    // Spawn a future that watches the queue field.
    executor::spawn(async move {
        if store::watch(id, &["items"]).await.is_err() {
            panic!("queue watch failed");
        }
        match store::pop(id, "items") {
            Ok(Some(Value::U32(v))) => {
                RECEIVED.store(v, Ordering::SeqCst);
                DONE.store(true, Ordering::SeqCst);
            }
            other => panic!("unexpected pop result: {:?}", other),
        }
    });

    // First poll: the future subscribes, returns Pending.
    executor::poll_once();
    assert!(!DONE.load(Ordering::SeqCst), "future should be pending after first poll");

    // Push a value — this wakes the subscriber.
    store::push(id, "items", Value::U32(99)).expect("push to queue");

    // Second poll: the future pops the value and completes.
    let completed = executor::poll_once();
    assert!(DONE.load(Ordering::SeqCst), "future should be done after second poll");
    assert_eq!(RECEIVED.load(Ordering::SeqCst), 99);
    assert_eq!(completed, 1);

    store::destroy(id).expect("cleanup queue watch store");
    println!("[ok] Store queue watch verified (push wakes subscriber)");
}

/// Verify the process table and process store lifecycle without touching Ring 3.
///
/// Tests the full spawn → start → exit → destroy cycle purely from the
/// kernel side, verifying that each state transition correctly updates the
/// process store fields.
fn test_process_lifecycle() {
    use alloc::string::String;
    use oboos_api::Value;

    // Spawn — creates PID and process store in "created" state.
    let pid = process::spawn("test-proc");
    let sid = process::store_id(pid).expect("process should have a store");

    // Verify initial state.
    assert_eq!(store::get(sid, "status").unwrap(), Value::Str(String::from("created")));
    assert_eq!(store::get(sid, "pid").unwrap(), Value::U64(pid.as_raw()));
    assert_eq!(store::get(sid, "name").unwrap(), Value::Str(String::from("test-proc")));
    assert_eq!(store::get(sid, "exit_code").unwrap(), Value::U64(0));

    // Start — Created → Running.
    process::start(pid);
    assert_eq!(store::get(sid, "status").unwrap(), Value::Str(String::from("running")));

    // Exit — Running → Exited with code 42.
    // exit() uses set_no_cli, so we need IF=0. Tests already run with IF=0.
    process::exit(pid, 42);
    assert_eq!(store::get(sid, "status").unwrap(), Value::Str(String::from("exited")));
    assert_eq!(store::get(sid, "exit_code").unwrap(), Value::U64(42));

    // Destroy — removes from table and destroys the store.
    process::destroy(pid);
    assert!(store::get(sid, "status").is_err(), "store should be gone after destroy");

    println!("[ok] Process lifecycle verified (spawn \u{2192} start \u{2192} exit \u{2192} destroy)");
}

/// Verify that store subscriptions work for observing process lifecycle
/// transitions — the key Phase 3c proof.
///
/// Spawns a process, watches its `status` field, triggers a state transition,
/// and verifies the watcher observes the new value.
fn test_process_store_watch() {
    use core::sync::atomic::{AtomicBool, Ordering};
    use oboos_api::Value;

    static WATCH_DONE: AtomicBool = AtomicBool::new(false);
    static SAW_RUNNING: AtomicBool = AtomicBool::new(false);
    WATCH_DONE.store(false, Ordering::SeqCst);
    SAW_RUNNING.store(false, Ordering::SeqCst);

    let pid = process::spawn("watched-proc");
    let sid = process::store_id(pid).expect("process should have a store");

    // Spawn a watcher that waits for a status change, reads the new value.
    executor::spawn(async move {
        if store::watch(sid, &["status"]).await.is_err() {
            panic!("watch failed");
        }
        match store::get(sid, "status") {
            Ok(Value::Str(s)) if s == "running" => {
                SAW_RUNNING.store(true, Ordering::SeqCst);
            }
            other => panic!("expected status=running, got {:?}", other),
        }
        WATCH_DONE.store(true, Ordering::SeqCst);
    });

    // First poll: watcher subscribes, returns Pending.
    executor::poll_once();
    assert!(!WATCH_DONE.load(Ordering::SeqCst), "watcher should be pending");

    // Transition to Running — wakes the watcher.
    process::start(pid);

    // Second poll: watcher reads "running" and completes.
    let completed = executor::poll_once();
    assert!(WATCH_DONE.load(Ordering::SeqCst), "watcher should be done");
    assert!(SAW_RUNNING.load(Ordering::SeqCst), "watcher should have seen running");
    assert_eq!(completed, 1);

    // Now test the exit transition with a second watcher.
    static WATCH_DONE2: AtomicBool = AtomicBool::new(false);
    static SAW_EXITED: AtomicBool = AtomicBool::new(false);
    WATCH_DONE2.store(false, Ordering::SeqCst);
    SAW_EXITED.store(false, Ordering::SeqCst);

    executor::spawn(async move {
        if store::watch(sid, &["status"]).await.is_err() {
            panic!("watch failed");
        }
        match store::get(sid, "status") {
            Ok(Value::Str(s)) if s == "exited" => {
                SAW_EXITED.store(true, Ordering::SeqCst);
            }
            other => panic!("expected status=exited, got {:?}", other),
        }
        WATCH_DONE2.store(true, Ordering::SeqCst);
    });

    // First poll: second watcher subscribes.
    executor::poll_once();
    assert!(!WATCH_DONE2.load(Ordering::SeqCst));

    // Exit the process — wakes the second watcher.
    process::exit(pid, 0);

    // Second poll: watcher reads "exited" and completes.
    executor::poll_once();
    assert!(WATCH_DONE2.load(Ordering::SeqCst));
    assert!(SAW_EXITED.load(Ordering::SeqCst));

    process::destroy(pid);
    println!("[ok] Process store watch verified (status transitions observable)");
}

/// Verify structured error codes from store operations.
///
/// Tests that specific error variants (NotFound, UnknownField, TypeMismatch)
/// are correctly distinguished — not just "it failed" but "it failed for
/// the right reason."
fn test_error_codes() {
    use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

    struct ErrTestSchema;
    impl StoreSchema for ErrTestSchema {
        fn name() -> &'static str { "ErrTest" }
        fn fields() -> &'static [FieldDef] {
            &[
                FieldDef { name: "count", kind: FieldKind::U32 },
                FieldDef { name: "label", kind: FieldKind::Str },
            ]
        }
    }

    let id = store::create::<ErrTestSchema>(&[
        ("count", Value::U32(0)),
        ("label", Value::Str(alloc::string::String::from("test"))),
    ]).expect("create err test store");

    // NotFound: get/set on a non-existent store ID.
    let bogus = store::StoreId::from_raw(99999);
    match store::get(bogus, "count") {
        Err(store::StoreError::NotFound) => {}
        other => panic!("expected NotFound, got {:?}", other),
    }
    match store::set(bogus, &[("count", Value::U32(1))]) {
        Err(store::StoreError::NotFound) => {}
        other => panic!("expected NotFound on set, got {:?}", other),
    }

    // UnknownField: get/set a field that doesn't exist in the schema.
    match store::get(id, "nonexistent") {
        Err(store::StoreError::UnknownField) => {}
        other => panic!("expected UnknownField, got {:?}", other),
    }
    match store::set(id, &[("nonexistent", Value::U32(1))]) {
        Err(store::StoreError::UnknownField) => {}
        other => panic!("expected UnknownField on set, got {:?}", other),
    }

    // TypeMismatch: write wrong type.
    match store::set(id, &[("count", Value::Bool(true))]) {
        Err(store::StoreError::TypeMismatch) => {}
        other => panic!("expected TypeMismatch, got {:?}", other),
    }

    store::destroy(id).expect("cleanup err test store");
    println!("[ok] Structured error codes verified (NotFound, UnknownField, TypeMismatch)");
}

/// Verify multi-field atomic SET: write two fields at once and verify
/// both are updated atomically.
fn test_multi_field_set() {
    use alloc::string::String;
    use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

    struct MultiSchema;
    impl StoreSchema for MultiSchema {
        fn name() -> &'static str { "MultiSet" }
        fn fields() -> &'static [FieldDef] {
            &[
                FieldDef { name: "x", kind: FieldKind::U32 },
                FieldDef { name: "y", kind: FieldKind::U32 },
                FieldDef { name: "name", kind: FieldKind::Str },
            ]
        }
    }

    let id = store::create::<MultiSchema>(&[
        ("x", Value::U32(0)),
        ("y", Value::U32(0)),
        ("name", Value::Str(String::from("init"))),
    ]).expect("create multi store");

    // Write two fields atomically.
    store::set(id, &[
        ("x", Value::U32(10)),
        ("y", Value::U32(20)),
    ]).expect("multi set x,y");

    // Verify both were written.
    assert_eq!(store::get(id, "x").unwrap(), Value::U32(10));
    assert_eq!(store::get(id, "y").unwrap(), Value::U32(20));
    // Third field should be unchanged.
    assert_eq!(store::get(id, "name").unwrap(), Value::Str(String::from("init")));

    // All-or-nothing: if one field in a batch has wrong type, nothing changes.
    let x_before = store::get(id, "x").unwrap();
    match store::set(id, &[
        ("x", Value::U32(99)),
        ("y", Value::Bool(false)), // wrong type for U32 field
    ]) {
        Err(store::StoreError::TypeMismatch) => {}
        other => panic!("expected TypeMismatch, got {:?}", other),
    }
    assert_eq!(store::get(id, "x").unwrap(), x_before, "x should be unchanged after failed batch");

    store::destroy(id).expect("cleanup multi store");
    println!("[ok] Multi-field atomic SET verified");
}

/// Verify that persistent watchers fire on every write, not just the first.
///
/// Unlike one-shot subscribers (which are drained after firing), persistent
/// watchers stay registered. This test creates a store, adds a persistent
/// watcher, writes the field twice, and verifies the watcher fires both times.
fn test_persistent_watcher() {
    use core::sync::atomic::{AtomicU32, Ordering};
    use core::task::{RawWaker, RawWakerVTable, Waker};
    use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

    struct WatcherSchema;
    impl StoreSchema for WatcherSchema {
        fn name() -> &'static str { "PersistentWatcher" }
        fn fields() -> &'static [FieldDef] {
            &[FieldDef { name: "value", kind: FieldKind::U32 }]
        }
    }

    static WAKE_COUNT: AtomicU32 = AtomicU32::new(0);
    WAKE_COUNT.store(0, Ordering::SeqCst);

    // Custom waker that increments a counter each time it's woken.
    static COUNT_VTABLE: RawWakerVTable = RawWakerVTable::new(
        |data| RawWaker::new(data, &COUNT_VTABLE),
        |_| { WAKE_COUNT.fetch_add(1, Ordering::SeqCst); },
        |_| { WAKE_COUNT.fetch_add(1, Ordering::SeqCst); },
        |_| {},
    );
    let waker = unsafe {
        Waker::from_raw(RawWaker::new(core::ptr::null(), &COUNT_VTABLE))
    };

    let id = store::create::<WatcherSchema>(&[
        ("value", Value::U32(0)),
    ]).expect("create persistent watcher store");

    // Add a persistent watcher (simulating what SYS_SUBSCRIBE does).
    arch::Arch::disable_interrupts();
    store::add_watcher_no_cli(id, "value", waker, 0).expect("add watcher");
    arch::Arch::enable_interrupts();

    // First write — watcher should fire.
    store::set(id, &[("value", Value::U32(1))]).expect("set value 1");
    assert_eq!(WAKE_COUNT.load(Ordering::SeqCst), 1,
        "persistent watcher should fire on first write");

    // Second write — watcher should fire again (not drained).
    store::set(id, &[("value", Value::U32(2))]).expect("set value 2");
    assert_eq!(WAKE_COUNT.load(Ordering::SeqCst), 2,
        "persistent watcher should fire on second write");

    // Remove the watcher and verify it no longer fires.
    arch::Arch::disable_interrupts();
    store::remove_watcher_no_cli(id, 0).expect("remove watcher");
    arch::Arch::enable_interrupts();

    store::set(id, &[("value", Value::U32(3))]).expect("set value 3");
    assert_eq!(WAKE_COUNT.load(Ordering::SeqCst), 2,
        "watcher should not fire after removal");

    store::destroy(id).expect("cleanup persistent watcher store");
    println!("[ok] Persistent watcher verified (fires on every write, removable)");
}

/// Verify the Ring 3 store round trip: load an ELF, drop to user mode,
/// read/write a kernel store via syscalls, and return.
///
/// The userspace Rust program (compiled as ELF, embedded via include_bytes)
/// sets a store field to 42 via SYS_STORE_SET, reads it back via
/// SYS_STORE_GET, and exits via SYS_EXIT. The kernel verifies counter==42.
fn test_ring3() {
    userspace::run_ring3_smoke_test();
}
