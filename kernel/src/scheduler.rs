//! Preemptive round-robin scheduler.
//!
//! Manages a set of kernel tasks and decides which one runs next. Tasks
//! can voluntarily give up the CPU by calling [`yield_now()`], or be
//! forcibly preempted by the PIT timer via [`on_tick()`].
//!
//! The scheduler uses a simple round-robin policy: each task gets a
//! 10ms time slice before being preempted. The ready queue is a
//! [`VecDeque`] — pop from the front to run, push to the back when
//! switching out.
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
//!
//! ## Two paths into schedule()
//!
//! Both [`yield_now()`] and [`on_tick()`] funnel into the private
//! [`schedule()`] function which does the actual round-robin swap:
//!
//! - **Cooperative** (`yield_now`): `cli` → `schedule()` → `sti`.
//!   The task explicitly gives up its time slice.
//!
//! - **Preemptive** (`on_tick`): PIT fires → interrupt gate clears IF →
//!   `schedule()` → `iretq` restores IF. The task is forcibly switched
//!   out when its time slice expires.

use alloc::collections::VecDeque;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::arch::{self, TaskContext};
use crate::platform::{ContextSwitch, Platform};
use crate::println;
use crate::task::{Task, TaskState};

/// 10ms time slice at 1 kHz PIT rate. Each PIT tick is ~1ms, so 10
/// ticks gives each task 10ms of CPU time before forced preemption.
const PREEMPT_TICKS: u32 = 10;

/// Ticks remaining in the current task's time slice. Decremented by
/// [`on_tick()`] on every PIT interrupt. When it hits zero, the
/// scheduler forces a context switch and resets the counter.
static TICKS_REMAINING: AtomicU32 = AtomicU32::new(PREEMPT_TICKS);

/// Global scheduler instance. Initialized once from `kmain` via [`init()`],
/// then accessed by [`spawn()`], [`yield_now()`], and [`on_tick()`].
///
/// [`spin::Once`] provides one-time initialization with a panic on
/// double-init. The inner [`spin::Mutex`] protects concurrent access
/// from both cooperative yields and preemptive timer interrupts.
static SCHEDULER: spin::Once<spin::Mutex<Scheduler>> = spin::Once::new();

/// The scheduler's internal state: which task is running and which are
/// waiting for CPU time.
struct Scheduler {
    /// The task currently executing on the CPU.
    current: Task,
    /// Tasks waiting for CPU time, in round-robin order. New tasks go
    /// to the back; scheduling pops from the front and pushes the
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
/// current task yields or is preempted.
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

/// The shared scheduling core — performs the round-robin swap and
/// context switch.
///
/// Callers must ensure interrupts are disabled before calling. This
/// function does not manage interrupt state — `yield_now()` does
/// `cli`/`sti`, and `on_tick()` relies on the interrupt gate's IF=0
/// and `iretq`'s IF restore.
fn schedule() {
    // Scope the lock so it's dropped before the context switch.
    let switch_targets = {
        let mut sched = SCHEDULER
            .get()
            .expect("scheduler not initialized")
            .lock();

        // Nothing to switch to — return immediately.
        let Some(next) = sched.ready.pop_front() else {
            return;
        };

        // Swap current and next: the old current goes to the ready queue,
        // the popped task becomes current.
        let mut prev = core::mem::replace(&mut sched.current, next);
        prev.state = TaskState::Ready;
        sched.current.state = TaskState::Running;
        sched.ready.push_back(prev);

        // Reset the time slice for the incoming task.
        TICKS_REMAINING.store(PREEMPT_TICKS, Ordering::Relaxed);

        // Extract raw pointers while we still hold the lock. The save_to
        // pointer targets the task we just pushed to the back of the ready
        // queue — that's where we want our registers saved so we can
        // resume later. The restore_from pointer targets the new current
        // task whose registers we want to load.
        let save_to = &mut sched.ready.back_mut().unwrap().context
            as *mut TaskContext;
        let restore_from = &sched.current.context as *const TaskContext;

        Some((save_to, restore_from))
    };
    // Lock is dropped here — the switched-to task can safely lock the
    // scheduler on its next yield without deadlocking.

    if let Some((save_to, restore_from)) = switch_targets {
        // Perform the actual context switch. This saves our registers into
        // save_to, loads registers from restore_from, and swaps stacks.
        // We won't execute the next line until some future switch brings
        // us back.
        unsafe {
            arch::Arch::switch(&mut *save_to, &*restore_from);
        }
    }
}

/// Yield the CPU to the next ready task.
///
/// Performs a round-robin swap: the current task moves to the back of
/// the ready queue, and the front task becomes the new current. If no
/// other tasks are ready, returns immediately (nothing to switch to).
///
/// This is the cooperative scheduling primitive — tasks call this when
/// they're done with their current work slice.
pub fn yield_now() {
    arch::Arch::disable_interrupts();
    schedule();
    arch::Arch::enable_interrupts();
}

/// Called from [`pit::tick()`](crate::arch::pit::tick) on every
/// PIT interrupt (1 kHz = every ~1ms).
///
/// Decrements the current task's remaining time slice. When it expires
/// (hits zero), forces a context switch via [`schedule()`]. The
/// interrupt gate guarantees IF=0, and `iretq` will restore the
/// preempted task's RFLAGS (with IF=1) when it resumes.
pub fn on_tick() {
    if TICKS_REMAINING.fetch_sub(1, Ordering::Relaxed) == 1 {
        schedule();
    }
}
