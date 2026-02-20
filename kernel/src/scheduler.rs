//! Per-core preemptive round-robin scheduler with work stealing.
//!
//! Each CPU core has its own [`CoreScheduler`] — a current task and a ready
//! queue. Tasks can yield cooperatively via [`yield_now()`] or be preempted
//! by the APIC timer via [`on_tick()`].
//!
//! When a core's local queue is empty, [`schedule()`] attempts to steal a
//! task from another core's queue. This keeps all cores busy without
//! requiring an explicit task migration API.
//!
//! ## Per-core state
//!
//! Each core's scheduler is indexed by `cpu_index` (0 = BSP, 1..N = APs).
//! The current core's index is read from the PerCpu struct via the GS
//! segment base register — set up during SMP init.
//!
//! ## The lock-across-switch problem
//!
//! We can't hold the scheduler lock during a context switch — the
//! switched-to task would deadlock trying to lock the scheduler on its
//! next yield. The solution: extract raw pointers to the two
//! [`TaskContext`] values, drop the lock, then switch. This is safe
//! because interrupts are disabled (no concurrent access on this core)
//! and the scheduler data is `'static` (pointers remain valid).

use alloc::collections::VecDeque;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::arch::{self, TaskContext};
use crate::platform::{ContextSwitch, Platform};
use crate::println;
use crate::task::{Task, TaskState};

/// 10ms time slice at 1 kHz APIC timer rate.
const PREEMPT_TICKS: u32 = 10;

/// Maximum CPUs — must match [`smp::MAX_CPUS`].
const MAX_CPUS: usize = 16;

/// Per-core scheduler state: one current task and a ready queue.
struct CoreScheduler {
    current: Task,
    ready: VecDeque<Task>,
}

/// Per-core schedulers. Each slot starts as `None` and is initialized
/// when that core comes online (`init()` for BSP, `init_ap()` for APs).
/// The mutex protects against concurrent access from timer interrupts
/// (same core) and work stealing (remote cores).
static SCHEDULERS: [spin::Mutex<Option<CoreScheduler>>; MAX_CPUS] =
    [const { spin::Mutex::new(None) }; MAX_CPUS];

/// Per-core tick counters. Each core independently tracks its current
/// task's remaining time slice.
static TICKS_REMAINING: [AtomicU32; MAX_CPUS] =
    [const { AtomicU32::new(PREEMPT_TICKS) }; MAX_CPUS];

/// Per-core preemption disable flags. When true, `on_tick()` skips
/// scheduling even when the time slice expires. Used by the syscall
/// WATCH wait loop to avoid preemption during kernel-stack-bound work.
static PREEMPT_DISABLED: [AtomicBool; MAX_CPUS] =
    [const { AtomicBool::new(false) }; MAX_CPUS];

/// Initialize the BSP's scheduler (core 0).
///
/// Creates a [`CoreScheduler`] with [`Task::bootstrap()`] as the current
/// task, wrapping `kmain`'s existing execution context. Must be called
/// after `heap::init()` and `smp::init_bsp_percpu()`.
pub fn init() {
    SCHEDULERS[0].lock().replace(CoreScheduler {
        current: Task::bootstrap(),
        ready: VecDeque::new(),
    });
    println!("[ok] Scheduler initialized");
}

/// Initialize an AP's scheduler.
///
/// Creates a [`CoreScheduler`] whose current task is the AP's idle loop
/// (wrapped via [`Task::bootstrap()`]). Called from the AP entry function
/// after GDT/TSS/IDT/LAPIC setup but before enabling interrupts.
pub fn init_ap(cpu_index: u32) {
    SCHEDULERS[cpu_index as usize].lock().replace(CoreScheduler {
        current: Task::bootstrap(),
        ready: VecDeque::new(),
    });
}

/// Spawn a new task on the current core's ready queue.
///
/// The task is created in the [`Ready`](TaskState::Ready) state and
/// placed at the back of the queue. It won't run until the current
/// task yields or is preempted.
pub fn spawn(entry_point: fn() -> !) {
    let cpu = arch::smp::current_cpu() as usize;
    let task = Task::new(entry_point);
    let id = task.id.0;
    SCHEDULERS[cpu]
        .lock()
        .as_mut()
        .expect("scheduler not initialized")
        .ready
        .push_back(task);
    println!("[sched] Spawned task {} on CPU {}", id, cpu);
}

/// The per-core scheduling core — round-robin swap with work stealing.
///
/// Callers must ensure interrupts are disabled before calling. This
/// function does not manage interrupt state — `yield_now()` does
/// `cli`/`sti`, and `on_tick()` relies on the interrupt gate's IF=0
/// and `iretq`'s IF restore.
fn schedule() {
    let cpu = arch::smp::current_cpu() as usize;

    // Try local queue first.
    let switch_targets = {
        let mut guard = SCHEDULERS[cpu].lock();
        let sched = guard.as_mut().expect("scheduler not initialized");

        if let Some(next) = sched.ready.pop_front() {
            Some(prepare_switch(sched, cpu, next))
        } else {
            None
        }
    };

    if let Some((save_to, restore_from)) = switch_targets {
        unsafe { arch::Arch::switch(&mut *save_to, &*restore_from) };
        return;
    }

    // Local queue empty — try to steal from another core.
    if let Some(stolen) = try_steal(cpu) {
        let switch_targets = {
            let mut guard = SCHEDULERS[cpu].lock();
            let sched = guard.as_mut().expect("scheduler not initialized");
            prepare_switch(sched, cpu, stolen)
        };
        let (save_to, restore_from) = switch_targets;
        unsafe { arch::Arch::switch(&mut *save_to, &*restore_from) };
    }
}

/// Swap the current task with `next`, returning raw pointers for the
/// context switch. The old current goes to the back of the ready queue.
///
/// Must be called while holding the scheduler lock.
fn prepare_switch(
    sched: &mut CoreScheduler,
    cpu: usize,
    next: Task,
) -> (*mut TaskContext, *const TaskContext) {
    let mut prev = core::mem::replace(&mut sched.current, next);
    prev.state = TaskState::Ready;
    sched.current.state = TaskState::Running;
    sched.ready.push_back(prev);

    TICKS_REMAINING[cpu].store(PREEMPT_TICKS, Ordering::Relaxed);

    let save_to = &mut sched.ready.back_mut().unwrap().context as *mut TaskContext;
    let restore_from = &sched.current.context as *const TaskContext;
    (save_to, restore_from)
}

/// Try to steal a task from another core's ready queue.
///
/// Scans all other cores in round-robin order starting from the core
/// after `cpu`. Steals one task from the back of the first non-empty
/// queue found. Returns `None` if all queues are empty.
fn try_steal(cpu: usize) -> Option<Task> {
    let count = arch::smp::cpu_count() as usize;
    for i in 1..count {
        let target = (cpu + i) % count;
        let mut guard = SCHEDULERS[target].lock();
        if let Some(sched) = guard.as_mut() {
            if let Some(task) = sched.ready.pop_back() {
                return Some(task);
            }
        }
    }
    None
}

/// Yield the CPU to the next ready task (cooperative scheduling).
///
/// The current task moves to the back of the ready queue and the
/// front task becomes current. If no other tasks are ready, returns
/// immediately.
pub fn yield_now() {
    arch::Arch::disable_interrupts();
    schedule();
    arch::Arch::enable_interrupts();
}

/// Suppress preemptive scheduling on the current core.
///
/// While disabled, [`on_tick()`] still counts ticks but never calls
/// [`schedule()`]. Used by the SYS_YIELD wait loop which needs to
/// call [`poll_once()`] (enabling interrupts briefly) without getting
/// preempted off the syscall kernel stack.
pub fn disable_preemption() {
    let cpu = arch::smp::current_cpu() as usize;
    PREEMPT_DISABLED[cpu].store(true, Ordering::SeqCst);
}

/// Re-enable preemptive scheduling on the current core.
pub fn enable_preemption() {
    let cpu = arch::smp::current_cpu() as usize;
    PREEMPT_DISABLED[cpu].store(false, Ordering::SeqCst);
}

/// Called from the APIC timer handler on every core (1 kHz).
///
/// Decrements this core's tick counter. When it expires, forces a
/// context switch via [`schedule()`] — unless preemption is disabled.
/// The interrupt gate guarantees IF=0, and `iretq` will restore the
/// preempted task's RFLAGS (with IF=1) when it resumes.
pub fn on_tick() {
    let cpu = arch::smp::current_cpu() as usize;
    if TICKS_REMAINING[cpu].fetch_sub(1, Ordering::Relaxed) == 1 {
        if !PREEMPT_DISABLED[cpu].load(Ordering::SeqCst) {
            schedule();
        }
    }
}
