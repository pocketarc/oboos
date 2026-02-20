//! Per-core async sleep support via APIC timer-driven deadlines.
//!
//! Provides [`sleep()`], an async primitive that suspends a future for a
//! given number of milliseconds. Each core's APIC timer handler calls
//! [`check_deadlines()`] on every tick to wake expired sleepers — no
//! polling or busy-waiting needed.
//!
//! ## Per-core registries
//!
//! Each core has its own sleep registry. When a future calls
//! `sleep(500).await`, its deadline is registered on whichever core
//! polls it. Since futures stay on the core where they were spawned,
//! this naturally partitions sleep entries by core.
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

use crate::arch::{self, Arch};
use crate::platform::Platform;

/// Maximum CPUs — must match [`smp::MAX_CPUS`].
const MAX_CPUS: usize = 16;

/// A registered sleep: the absolute deadline (in ms since boot) and the
/// waker to call when the deadline expires.
struct SleepEntry {
    deadline_ms: u64,
    waker: Waker,
}

/// Per-core sleep registry — a Vec of pending deadlines behind a spinlock.
struct SleepRegistry {
    entries: Vec<SleepEntry>,
}

/// Per-core registries. Pre-allocated on the BSP in [`init()`]; APs start
/// empty and grow on demand.
static REGISTRIES: [spin::Mutex<SleepRegistry>; MAX_CPUS] =
    [const { spin::Mutex::new(SleepRegistry { entries: Vec::new() }) }; MAX_CPUS];

/// Pre-allocate the BSP's sleep registry so the tick handler never
/// hits the allocator.
///
/// Must be called after [`crate::heap::init()`].
pub fn init() {
    Arch::disable_interrupts();
    REGISTRIES[0].lock().entries.reserve(16);
    Arch::enable_interrupts();
    crate::println!("[ok] Sleep registry pre-allocated (16 slots)");
}

/// Async sleep for `duration_ms` milliseconds.
///
/// Returns a future that resolves after at least `duration_ms` have
/// elapsed. Actual resolution depends on the APIC timer tick rate
/// (1 ms at 1 kHz). A duration of 0 completes on the next poll.
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

        let target = *deadline.get_or_insert(now + duration_ms);

        if now >= target {
            return core::task::Poll::Ready(());
        }

        // Register on the current core's sleep registry.
        let cpu = arch::smp::current_cpu() as usize;
        Arch::disable_interrupts();
        REGISTRIES[cpu].lock().entries.push(SleepEntry {
            deadline_ms: target,
            waker: cx.waker().clone(),
        });
        Arch::enable_interrupts();

        core::task::Poll::Pending
    })
}

/// Wake all futures whose deadlines have expired on the current core.
///
/// Called from the APIC timer handler on every core (IF=0 inherently).
/// Scans this core's registry, wakes expired entries, and retains only
/// those still pending.
pub fn check_deadlines() {
    let cpu = arch::smp::current_cpu() as usize;
    let now = Arch::elapsed_ms();
    let mut reg = REGISTRIES[cpu].lock();

    let mut i = 0;
    while i < reg.entries.len() {
        if reg.entries[i].deadline_ms <= now {
            let entry = reg.entries.swap_remove(i);
            entry.waker.wake();
        } else {
            i += 1;
        }
    }
}
