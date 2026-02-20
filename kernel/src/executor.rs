//! Per-core kernel-space async executor.
//!
//! Each CPU core has its own executor — a task map and wake queue. Futures
//! are spawned on a specific core via [`spawn()`] (local) or [`spawn_on()`]
//! (remote). Cross-core waking (e.g., a store write on core 1 waking a
//! watcher on core 0) pushes the task id into the target core's wake queue
//! and sends an IPI to break that core out of `hlt`.
//!
//! ## Work stealing
//!
//! Idle cores call [`try_steal_and_poll()`] from `run()` after draining
//! their local queue. This lets APs help busy cores instead of sleeping.
//! Stolen futures are polled on the stealing core but returned to the
//! original core's task map if they're still `Pending` — the waker
//! encodes the home core, so future wakes route correctly.
//!
//! ## AsyncTaskId encoding
//!
//! The upper 8 bits of the id encode the owning CPU index, and the lower
//! 56 bits are a monotonically increasing local counter. This lets
//! [`wake()`] determine which core's executor to target without any
//! external lookup.
//!
//! ## Interrupt safety
//!
//! The executor lock must never be held when an IRQ fires that tries
//! to call [`wake()`] on the same core. Rule: **always disable interrupts
//! before taking the executor lock.** [`wake()`] handles this internally
//! by saving and restoring the interrupt flag.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use crate::arch::{self, Arch};
use crate::platform::Platform;
use crate::println;

/// Maximum CPUs — must match [`smp::MAX_CPUS`].
const MAX_CPUS: usize = 16;

/// Unique identifier for an async task.
///
/// Upper 8 bits = owning CPU index, lower 56 bits = local counter.
/// This encoding lets [`wake()`] route wakes to the correct core's
/// executor without an external lookup table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AsyncTaskId(u64);

impl AsyncTaskId {
    fn new(cpu: u32) -> Self {
        let local = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Self(((cpu as u64) << 56) | (local & 0x00FF_FFFF_FFFF_FFFF))
    }

    fn cpu(self) -> u32 {
        (self.0 >> 56) as u32
    }
}

/// Monotonic counter for the local part of [`AsyncTaskId`].
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// A spawned async task: a pinned, boxed, type-erased future.
struct AsyncTask {
    future: Pin<Box<dyn Future<Output = ()> + Send>>,
}

/// Per-core executor state.
struct CoreExecutor {
    /// All live async tasks on this core.
    tasks: BTreeMap<AsyncTaskId, AsyncTask>,
    /// Ids of tasks woken and pending re-poll.
    wake_queue: Vec<AsyncTaskId>,
}

/// Per-core executors. Each slot starts as `None` and is initialized
/// when that core comes online.
static EXECUTORS: [spin::Mutex<Option<CoreExecutor>>; MAX_CPUS] =
    [const { spin::Mutex::new(None) }; MAX_CPUS];

/// Gate for work stealing. Off by default so deterministic tests (which
/// manually drive poll_once) aren't disrupted by APs stealing tasks.
/// Enabled by [`enable_work_stealing()`] once tests are done and the
/// real executor loops are running.
static WORK_STEALING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable cross-core work stealing.
///
/// Call after deterministic tests (which rely on manual poll_once
/// sequencing) are done. APs' `run()` loops will start helping busy
/// cores once this is set.
pub fn enable_work_stealing() {
    WORK_STEALING_ENABLED.store(true, Ordering::Release);
}

/// Initialize the BSP's executor (core 0).
///
/// Creates an empty task map and wake queue. Call after `heap::init()`.
pub fn init() {
    EXECUTORS[0].lock().replace(CoreExecutor {
        tasks: BTreeMap::new(),
        wake_queue: Vec::new(),
    });
    println!("[ok] Async executor initialized");
}

/// Initialize an AP's executor.
///
/// Called from the AP entry function after GDT/TSS/IDT/LAPIC/scheduler setup.
pub fn init_ap(cpu_index: u32) {
    EXECUTORS[cpu_index as usize].lock().replace(CoreExecutor {
        tasks: BTreeMap::new(),
        wake_queue: Vec::new(),
    });
}

/// Spawn an async task on the current core's executor.
///
/// Box-pins the future, assigns a unique [`AsyncTaskId`] encoding
/// this core's index, and enqueues a wake for the initial poll.
pub fn spawn(future: impl Future<Output = ()> + Send + 'static) {
    let cpu = arch::smp::current_cpu();
    let id = AsyncTaskId::new(cpu);
    let task = AsyncTask {
        future: Box::pin(future),
    };

    let was_enabled = interrupts_enabled();
    Arch::disable_interrupts();
    let mut guard = EXECUTORS[cpu as usize].lock();
    let exec = guard.as_mut().expect("executor not initialized");
    exec.tasks.insert(id, task);
    exec.wake_queue.push(id);
    drop(guard);
    if was_enabled {
        Arch::enable_interrupts();
    }
}

/// Spawn an async task on a specific core's executor.
///
/// Like [`spawn()`], but targets `target_cpu` instead of the current core.
/// If the target is a remote core, sends an IPI to break it out of `hlt`
/// so it polls the new task promptly. Falls back to local spawn behavior
/// when `target_cpu == current_cpu()`.
pub fn spawn_on(target_cpu: u32, future: impl Future<Output = ()> + Send + 'static) {
    let id = AsyncTaskId::new(target_cpu);
    let task = AsyncTask {
        future: Box::pin(future),
    };

    let was_enabled = interrupts_enabled();
    Arch::disable_interrupts();
    let mut guard = EXECUTORS[target_cpu as usize].lock();
    let exec = guard.as_mut().expect("executor not initialized");
    exec.tasks.insert(id, task);
    exec.wake_queue.push(id);
    drop(guard);
    if was_enabled {
        Arch::enable_interrupts();
    }

    // If targeting a remote core, send an IPI to wake it from hlt.
    let current = arch::smp::current_cpu();
    if target_cpu != current {
        let lapic_id = arch::smp::lapic_id_for_cpu(target_cpu);
        arch::lapic::send_ipi(lapic_id, arch::lapic::IPI_WAKE_VECTOR);
    }
}

/// Wake an async task by id, scheduling it for re-polling.
///
/// Routes to the correct core's executor based on the id's encoded
/// CPU index. If the target is a different core, sends an IPI to
/// break it out of `hlt`.
///
/// Safe to call from any interrupt or non-interrupt context — this
/// function saves and restores the interrupt flag internally.
pub fn wake(id: AsyncTaskId) {
    let target_cpu = id.cpu() as usize;

    // Save IF state and disable. Prevents deadlock: if a timer fires
    // while we hold the executor lock and calls wake() on the same
    // core, the nested lock acquisition would deadlock.
    let was_enabled = interrupts_enabled();
    Arch::disable_interrupts();

    if let Some(exec) = EXECUTORS[target_cpu].lock().as_mut() {
        exec.wake_queue.push(id);
    }

    if was_enabled {
        Arch::enable_interrupts();
    }

    // If targeting a remote core, send an IPI to wake it from hlt.
    let current = arch::smp::current_cpu() as usize;
    if target_cpu != current {
        let lapic_id = arch::smp::lapic_id_for_cpu(target_cpu as u32);
        arch::lapic::send_ipi(lapic_id, arch::lapic::IPI_WAKE_VECTOR);
    }
}

/// Poll all woken futures on the current core once.
///
/// Drains this core's wake queue and polls each woken future. Futures
/// that return `Ready` are dropped; `Pending` futures are re-inserted.
/// Returns the number of completed futures.
pub fn poll_once() -> usize {
    let cpu = arch::smp::current_cpu() as usize;

    // Step 1: drain the wake queue with IF=0.
    Arch::disable_interrupts();
    let woken = {
        let mut guard = EXECUTORS[cpu].lock();
        let exec = guard.as_mut().expect("executor not initialized");
        core::mem::take(&mut exec.wake_queue)
    };
    Arch::enable_interrupts();

    let mut completed = 0;

    // Step 2: poll each woken future.
    for id in woken {
        Arch::disable_interrupts();
        let task = EXECUTORS[cpu]
            .lock()
            .as_mut()
            .expect("executor not initialized")
            .tasks
            .remove(&id);
        Arch::enable_interrupts();

        let Some(mut task) = task else {
            continue;
        };

        let waker = create_waker(id);
        let mut cx = Context::from_waker(&waker);

        match task.future.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                completed += 1;
            }
            Poll::Pending => {
                Arch::disable_interrupts();
                let mut guard = EXECUTORS[cpu].lock();
                let exec = guard.as_mut().expect("executor not initialized");
                exec.tasks.insert(id, task);
                drop(guard);
                Arch::enable_interrupts();
            }
        }
    }

    completed
}

/// Try to steal one woken task from another core and poll it locally.
///
/// Called from [`run()`] after `poll_once()` drains the local queue.
/// Scans other cores round-robin, steals one task, polls it, and
/// returns the future to its home core if still `Pending`. This lets
/// idle APs help busy cores instead of sleeping.
///
/// Only steals one task per call to avoid starving the victim core.
///
/// // TODO: NUMA-aware steal order — prefer same-socket cores first.
/// QEMU's flat memory model means no performance difference today.
fn try_steal_and_poll() {
    if !WORK_STEALING_ENABLED.load(Ordering::Acquire) {
        return;
    }

    let cpu = arch::smp::current_cpu();
    let count = arch::smp::cpu_count();
    if count <= 1 {
        return;
    }

    for offset in 1..count {
        let victim = ((cpu + offset) % count) as usize;

        // Try to pop one woken task ID from the victim's wake queue.
        Arch::disable_interrupts();
        let stolen_id = {
            let mut guard = EXECUTORS[victim].lock();
            match guard.as_mut() {
                Some(exec) if !exec.wake_queue.is_empty() => {
                    Some(exec.wake_queue.remove(0))
                }
                _ => None,
            }
        };
        Arch::enable_interrupts();

        let Some(id) = stolen_id else { continue };

        // Remove the future from the victim's task map. If the victim
        // already polled it between our stealing the ID and this removal,
        // tasks.remove() returns None — just skip to the next core.
        Arch::disable_interrupts();
        let stolen_task = EXECUTORS[victim]
            .lock()
            .as_mut()
            .and_then(|exec| exec.tasks.remove(&id));
        Arch::enable_interrupts();

        let Some(mut task) = stolen_task else { continue };

        // Poll the stolen future on this core. The waker still encodes
        // the home core's ID, so any future wake() routes back there.
        let waker = create_waker(id);
        let mut cx = Context::from_waker(&waker);

        match task.future.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                // Task completed on this core — nothing to return.
            }
            Poll::Pending => {
                // Return the future to its home core's task map but
                // don't re-enqueue — the task is dormant until a real
                // wake() call pushes its ID. If a wake arrived during
                // the steal window, wake() already enqueued it independently.
                Arch::disable_interrupts();
                let mut guard = EXECUTORS[victim].lock();
                if let Some(exec) = guard.as_mut() {
                    exec.tasks.insert(id, task);
                }
                drop(guard);
                Arch::enable_interrupts();
            }
        }

        // One stolen task per call — don't starve the victim.
        return;
    }
}

/// Run the executor forever, polling woken futures and sleeping when idle.
///
/// Each core calls this from its idle loop. After draining the local
/// wake queue, tries to steal work from other cores before sleeping.
/// The APIC timer breaks `hlt` every 1ms for scheduling, and IPI
/// wakes break it for cross-core future wakeups.
#[allow(dead_code)]
pub fn run() -> ! {
    loop {
        poll_once();
        try_steal_and_poll();
        Arch::halt_until_interrupt();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether the CPU's interrupt flag (IF) is currently set.
fn interrupts_enabled() -> bool {
    let rflags: u64;
    unsafe {
        core::arch::asm!("pushfq; pop {}", out(reg) rflags, options(nomem));
    }
    rflags & 0x200 != 0
}

// ---------------------------------------------------------------------------
// Waker implementation
// ---------------------------------------------------------------------------

/// Create a [`Waker`] for the given async task id.
///
/// The full 64-bit id (including CPU index) is packed into the
/// [`RawWaker`]'s data pointer — zero-allocation on 64-bit platforms.
fn create_waker(id: AsyncTaskId) -> Waker {
    let raw = RawWaker::new(id.0 as *const (), &WAKER_VTABLE);
    unsafe { Waker::from_raw(raw) }
}

fn id_from_data(data: *const ()) -> AsyncTaskId {
    AsyncTaskId(data as u64)
}

static WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |data| RawWaker::new(data, &WAKER_VTABLE), // clone
    |data| wake(id_from_data(data)),            // wake
    |data| wake(id_from_data(data)),            // wake_by_ref
    |_| {},                                     // drop
);
