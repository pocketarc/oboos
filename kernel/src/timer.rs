//! Async sleep support via PIT tick-driven deadlines.
//!
//! Provides [`sleep()`], an async primitive that suspends a future for a
//! given number of milliseconds. The PIT IRQ handler calls
//! [`check_deadlines()`] on every tick to wake expired sleepers — no
//! polling or busy-waiting needed.
//!
//! ## How it works
//!
//! When a future calls `sleep(500).await`, the first poll computes a
//! deadline (`elapsed_ms() + 500`), registers a `(deadline, waker)` pair
//! in the global registry, and returns `Pending`. On subsequent PIT
//! ticks, `check_deadlines()` scans the registry and wakes any entry
//! whose deadline has passed. The executor then re-polls the future,
//! which sees the deadline has been met and returns `Ready`.
//!
//! ## Interrupt safety
//!
//! Same pattern as the keyboard driver: non-IRQ callers must `cli`
//! before locking the registry to prevent deadlock with the tick
//! handler (which runs with IF=0 inherently).

extern crate alloc;

use alloc::vec::Vec;
use core::future::Future;
use core::task::Waker;

use crate::arch::Arch;
use crate::platform::Platform;

/// A registered sleep: the absolute deadline (in ms since boot) and the
/// waker to call when the deadline expires.
struct SleepEntry {
    deadline_ms: u64,
    waker: Waker,
}

/// The sleep registry — a simple Vec of pending deadlines behind a
/// spinlock. Pre-allocated in [`init()`] so the tick handler never
/// needs to grow the Vec.
struct SleepRegistry {
    entries: Vec<SleepEntry>,
}

/// Global registry instance, initialized once from `kmain` via [`init()`].
static REGISTRY: spin::Mutex<SleepRegistry> = spin::Mutex::new(SleepRegistry {
    entries: Vec::new(),
});

/// Pre-allocate the registry so the tick handler never hits the allocator.
///
/// Must be called after [`crate::heap::init()`]. 16 slots is generous —
/// we only have a handful of concurrent async tasks.
pub fn init() {
    Arch::disable_interrupts();
    REGISTRY.lock().entries.reserve(16);
    Arch::enable_interrupts();
    crate::println!("[ok] Sleep registry pre-allocated (16 slots)");
}

/// Async sleep for `duration_ms` milliseconds.
///
/// Returns a future that resolves after at least `duration_ms` have
/// elapsed. Actual resolution depends on the PIT tick rate (1 ms at
/// 1000 Hz). A duration of 0 completes on the next poll.
///
/// # Examples
///
/// ```
/// // Blink something at 2 Hz:
/// loop {
///     draw_cursor(true);
///     timer::sleep(500).await;
///     draw_cursor(false);
///     timer::sleep(500).await;
/// }
/// ```
pub fn sleep(duration_ms: u64) -> impl Future<Output = ()> + Send {
    let mut deadline: Option<u64> = None;

    core::future::poll_fn(move |cx| {
        let now = Arch::elapsed_ms();

        // Compute the deadline once, on the first poll.
        let target = *deadline.get_or_insert(now + duration_ms);

        if now >= target {
            return core::task::Poll::Ready(());
        }

        // Not yet — register the waker so check_deadlines() can wake us.
        Arch::disable_interrupts();
        REGISTRY.lock().entries.push(SleepEntry {
            deadline_ms: target,
            waker: cx.waker().clone(),
        });
        Arch::enable_interrupts();

        core::task::Poll::Pending
    })
}

/// Wake all futures whose deadlines have expired.
///
/// Called from [`crate::arch::x86_64::pit::tick()`] on every PIT IRQ
/// (IF=0 inherently, so locking is safe). Scans the registry, wakes
/// expired entries, and retains only those still pending.
pub fn check_deadlines() {
    let now = Arch::elapsed_ms();
    let mut reg = REGISTRY.lock();

    // Partition: wake expired entries, keep the rest.
    // We drain into a temp Vec to avoid borrowing issues — the Vec is
    // pre-allocated so this doesn't hit the allocator in practice.
    let mut i = 0;
    while i < reg.entries.len() {
        if reg.entries[i].deadline_ms <= now {
            let entry = reg.entries.swap_remove(i);
            entry.waker.wake();
            // Don't increment i — swap_remove moved the last element here.
        } else {
            i += 1;
        }
    }
}
