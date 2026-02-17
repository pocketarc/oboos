//! Cooperative round-robin scheduler.
//!
//! Manages a set of kernel tasks and decides which one runs next. Tasks
//! voluntarily give up the CPU by calling [`yield_now()`] — there's no
//! timer-driven preemption yet.
//!
//! The scheduler uses a simple round-robin policy: each task gets one
//! turn before cycling back. The ready queue is a [`VecDeque`] — pop
//! from the front to run, push to the back when yielding.
//!
//! ## The lock-across-switch problem
//!
//! The scheduler state lives in a [`spin::Mutex`]. But we can't hold
//! the lock during a context switch — the switched-to task would
//! deadlock trying to lock the scheduler on its next yield. The
//! solution: extract raw pointers to the two [`TaskContext`] values,
//! drop the lock, then switch. This is safe because interrupts are
//! disabled (no concurrent access) and the scheduler data is `'static`
//! (pointers remain valid).

use alloc::collections::VecDeque;

use crate::arch::{self, TaskContext};
use crate::platform::{ContextSwitch, Platform};
use crate::println;
use crate::task::{Task, TaskState};

/// Global scheduler instance. Initialized once from `kmain` via [`init()`],
/// then accessed by [`spawn()`] and [`yield_now()`].
///
/// [`spin::Once`] provides one-time initialization with a panic on
/// double-init. The inner [`spin::Mutex`] protects concurrent access
/// (relevant once we have preemption — for now, we disable interrupts
/// around every access anyway).
static SCHEDULER: spin::Once<spin::Mutex<Scheduler>> = spin::Once::new();

/// The scheduler's internal state: which task is running and which are
/// waiting for CPU time.
struct Scheduler {
    /// The task currently executing on the CPU.
    current: Task,
    /// Tasks waiting for CPU time, in round-robin order. New tasks go
    /// to the back; [`yield_now()`] pops from the front and pushes the
    /// old current to the back.
    ready: VecDeque<Task>,
}

/// Initialize the scheduler.
///
/// Creates the scheduler with [`Task::bootstrap()`] as the current task,
/// wrapping `kmain`'s existing execution context. Must be called once
/// from `kmain`, after [`heap::init()`](crate::heap::init) — the ready
/// queue needs the allocator.
pub fn init() {
    SCHEDULER.call_once(|| {
        spin::Mutex::new(Scheduler {
            current: Task::bootstrap(),
            ready: VecDeque::new(),
        })
    });
    println!("[ok] Scheduler initialized");
}

/// Spawn a new task that will begin executing at `entry_point`.
///
/// The task is created in the [`Ready`](TaskState::Ready) state and
/// placed at the back of the ready queue. It won't run until the
/// current task yields.
pub fn spawn(entry_point: fn() -> !) {
    let task = Task::new(entry_point);
    let id = task.id.0;
    SCHEDULER
        .get()
        .expect("scheduler not initialized")
        .lock()
        .ready
        .push_back(task);
    println!("[sched] Spawned task {}", id);
}

/// Yield the CPU to the next ready task.
///
/// Performs a round-robin swap: the current task moves to the back of
/// the ready queue, and the front task becomes the new current. If no
/// other tasks are ready, returns immediately (nothing to switch to).
///
/// This is the cooperative scheduling primitive — tasks call this when
/// they're done with their current work slice. Without preemption, a
/// task that never yields will starve all others.
pub fn yield_now() {
    // Disable interrupts for the entire schedule operation. We re-enable
    // them after we resume (which may be much later, after other tasks
    // have run and yielded back to us).
    arch::Arch::disable_interrupts();

    // Scope the lock so it's dropped before the context switch.
    let (save_to, restore_from) = {
        let mut sched = SCHEDULER
            .get()
            .expect("scheduler not initialized")
            .lock();

        // Nothing to switch to — return early.
        let Some(next) = sched.ready.pop_front() else {
            arch::Arch::enable_interrupts();
            return;
        };

        // Swap current and next: the old current goes to the ready queue,
        // the popped task becomes current.
        let mut prev = core::mem::replace(&mut sched.current, next);
        prev.state = TaskState::Ready;
        sched.current.state = TaskState::Running;
        sched.ready.push_back(prev);

        // Extract raw pointers while we still hold the lock. The save_to
        // pointer targets the task we just pushed to the back of the ready
        // queue — that's where we want our registers saved so we can
        // resume later. The restore_from pointer targets the new current
        // task whose registers we want to load.
        let save_to = &mut sched.ready.back_mut().unwrap().context
            as *mut TaskContext;
        let restore_from = &sched.current.context as *const TaskContext;

        (save_to, restore_from)
    };
    // Lock is dropped here — the switched-to task can safely lock the
    // scheduler on its next yield without deadlocking.

    // Perform the actual context switch. This saves our registers into
    // save_to, loads registers from restore_from, and swaps stacks.
    // We won't execute the next line until some future yield switches
    // back to us.
    unsafe {
        arch::Arch::switch(&mut *save_to, &*restore_from);
    }

    // We've been switched back. Re-enable interrupts now that we're
    // running again.
    arch::Arch::enable_interrupts();
}
