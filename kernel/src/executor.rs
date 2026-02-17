//! Kernel-space async executor.
//!
//! Provides a simple async runtime that polls [`Future`]s within the
//! bootstrap task's main loop. This sits *on top of* the preemptive
//! scheduler — scheduler tasks (`fn() -> !`) and async tasks
//! (`impl Future<Output = ()>`) coexist peacefully.
//!
//! ## How it integrates with kmain
//!
//! Rather than replacing `kmain`'s main loop, the executor exposes
//! [`poll_once()`] which is called on each loop iteration. This way
//! async futures are polled alongside keyboard polling and framebuffer
//! interaction. The preemptive scheduler still switches to other tasks
//! via PIT ticks — the executor only runs when the bootstrap task
//! (task 0) has the CPU.
//!
//! ## Waker mechanism
//!
//! Rust's [`Future::poll()`] receives a [`Context`] containing a
//! [`Waker`]. When a future returns `Pending`, it must arrange for its
//! waker to be called when progress can be made. Our waker stores the
//! [`AsyncTaskId`] directly in the [`RawWaker`]'s data pointer — a
//! `u64` fits in a pointer on 64-bit, so there's zero heap allocation
//! for wakers. When woken, the id is pushed into the executor's wake
//! queue, and the next [`poll_once()`] call drains the queue and
//! re-polls the woken futures.
//!
//! ## Interrupt safety
//!
//! The executor lock must never be held when an IRQ fires that tries
//! to call [`wake()`]. Rule: **always disable interrupts before taking
//! the executor lock.** This mirrors the scheduler's pattern.
//! [`wake()`] documents that it must be called with IF=0 (IRQ handlers
//! satisfy this inherently via interrupt gates).

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use crate::arch::Arch;
use crate::platform::Platform;
use crate::println;

/// Unique identifier for an async task.
///
/// Separate from the scheduler's [`TaskId`](crate::task::TaskId) —
/// async tasks live in the executor's task map, not the scheduler's
/// ready queue. The id is a monotonically increasing `u64`, which
/// conveniently fits in a pointer on 64-bit platforms (used for
/// zero-allocation wakers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AsyncTaskId(u64);

/// Monotonic counter for assigning unique [`AsyncTaskId`]s.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// A spawned async task: a pinned, boxed, type-erased future.
///
/// [`Pin`]`<`[`Box`]`<...>>` because futures may be self-referential
/// (e.g. across `.await` points), and [`Box`] because `dyn Future` is
/// dynamically sized.
struct AsyncTask {
    future: Pin<Box<dyn Future<Output = ()> + Send>>,
}

/// Executor state protected by a spinlock.
struct ExecutorInner {
    /// All live async tasks, keyed by their id. Tasks are removed when
    /// they return [`Poll::Ready`], or temporarily removed during
    /// polling and re-inserted if they return [`Poll::Pending`].
    tasks: BTreeMap<AsyncTaskId, AsyncTask>,

    /// Ids of tasks that have been woken and need to be polled on the
    /// next [`poll_once()`] call. Populated by [`wake()`].
    wake_queue: Vec<AsyncTaskId>,
}

/// Global executor instance, initialized once from `kmain` via [`init()`].
///
/// Uses [`spin::Once`] for one-time initialization (panics on
/// double-init) and [`spin::Mutex`] for interior mutability. Always
/// disable interrupts before locking — see module-level docs on
/// interrupt safety.
static EXECUTOR: spin::Once<spin::Mutex<ExecutorInner>> = spin::Once::new();

/// Initialize the executor.
///
/// Creates the executor with an empty task map and wake queue. Call
/// this from `kmain` after [`scheduler::init()`](crate::scheduler::init)
/// — the executor needs the heap allocator for [`BTreeMap`] and [`Vec`].
pub fn init() {
    EXECUTOR.call_once(|| {
        spin::Mutex::new(ExecutorInner {
            tasks: BTreeMap::new(),
            wake_queue: Vec::new(),
        })
    });
    println!("[ok] Async executor initialized");
}

/// Spawn an async task.
///
/// Box-pins the future, assigns a unique [`AsyncTaskId`], inserts it
/// into the task map, and enqueues a wake so it gets an initial poll
/// on the next [`poll_once()`] call.
pub fn spawn(future: impl Future<Output = ()> + Send + 'static) {
    let id = AsyncTaskId(NEXT_ID.fetch_add(1, Ordering::Relaxed));
    let task = AsyncTask {
        future: Box::pin(future),
    };

    Arch::disable_interrupts();
    let mut inner = EXECUTOR
        .get()
        .expect("executor not initialized")
        .lock();
    inner.tasks.insert(id, task);
    inner.wake_queue.push(id);
    drop(inner);
    Arch::enable_interrupts();
}

/// Wake an async task by id, scheduling it for re-polling.
///
/// Pushes the id into the executor's wake queue. The next
/// [`poll_once()`] call will drain the queue and poll all woken
/// futures.
///
/// Must be called with interrupts disabled (IF=0). IRQ handlers
/// satisfy this inherently via interrupt gates. If called from
/// non-interrupt context, the caller must `cli` first. This ensures
/// the executor lock can be taken without risking deadlock from a
/// nested IRQ that also calls `wake()`.
pub fn wake(id: AsyncTaskId) {
    EXECUTOR
        .get()
        .expect("executor not initialized")
        .lock()
        .wake_queue
        .push(id);
}

/// Poll all woken futures once.
///
/// Drains the wake queue and polls each woken future. Futures that
/// return [`Poll::Ready`] are dropped; futures that return
/// [`Poll::Pending`] are re-inserted into the task map (they'll be
/// polled again when their waker fires).
///
/// Returns the number of futures that completed (returned `Ready`)
/// during this call. Non-diverging — safe to call from the main loop.
pub fn poll_once() -> usize {
    // Step 1: drain the wake queue with interrupts disabled.
    Arch::disable_interrupts();
    let woken = {
        let mut inner = EXECUTOR
            .get()
            .expect("executor not initialized")
            .lock();
        core::mem::take(&mut inner.wake_queue)
    };
    Arch::enable_interrupts();

    let mut completed = 0;

    // Step 2: poll each woken future.
    for id in woken {
        // Remove the task from the map so we can poll it without
        // holding the lock — polling may call spawn() or wake().
        Arch::disable_interrupts();
        let task = EXECUTOR
            .get()
            .expect("executor not initialized")
            .lock()
            .tasks
            .remove(&id);
        Arch::enable_interrupts();

        // Task may have been removed already (completed in an earlier
        // iteration, or double-woken). Skip it.
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
                // Not done yet — put it back in the task map.
                Arch::disable_interrupts();
                let mut inner = EXECUTOR
                    .get()
                    .expect("executor not initialized")
                    .lock();
                inner.tasks.insert(id, task);
                drop(inner);
                Arch::enable_interrupts();
            }
        }
    }

    completed
}

/// Run the executor forever, polling woken futures and sleeping when
/// idle.
///
/// Convenience loop: [`poll_once()`] + `sti; hlt` when there's
/// nothing to do. Available for future phases that want the executor
/// to own the main loop, but not used in Phase 2e (where
/// [`poll_once()`] is called from `kmain`'s existing loop).
#[allow(dead_code)]
pub fn run() -> ! {
    loop {
        poll_once();
        Arch::halt_until_interrupt();
    }
}

// ---------------------------------------------------------------------------
// Waker implementation
// ---------------------------------------------------------------------------

/// Create a [`Waker`] for the given async task id.
///
/// The id is packed directly into the [`RawWaker`]'s data pointer — a
/// `u64` fits in `*const ()` on 64-bit platforms, so this is a
/// zero-allocation waker. The vtable methods simply unpack the id and
/// push it to the executor's wake queue.
fn create_waker(id: AsyncTaskId) -> Waker {
    let raw = RawWaker::new(id.0 as *const (), &WAKER_VTABLE);
    // SAFETY: Our vtable correctly implements the RawWaker contract:
    // clone returns a valid RawWaker, wake/wake_by_ref push to the
    // wake queue, and drop is a no-op (no resources to free).
    unsafe { Waker::from_raw(raw) }
}

/// Extract the [`AsyncTaskId`] from a waker's data pointer.
fn id_from_data(data: *const ()) -> AsyncTaskId {
    AsyncTaskId(data as u64)
}

/// Waker vtable for async tasks.
///
/// - **clone**: Copy the data pointer (it's just a `u64` id, no heap
///   allocation to refcount).
/// - **wake**: Push the id to the executor's wake queue, consuming the
///   waker. Inherits the IF=0 requirement from [`wake()`].
/// - **wake_by_ref**: Same as wake, but by reference.
/// - **drop**: No-op — no resources to free.
static WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    // clone
    |data| RawWaker::new(data, &WAKER_VTABLE),
    // wake (by value)
    |data| wake(id_from_data(data)),
    // wake_by_ref
    |data| wake(id_from_data(data)),
    // drop
    |_| {},
);
