//! Process lifecycle management — identity, state tracking, and process stores.
//!
//! Every userspace program gets a first-class identity through the process
//! table. Each process is assigned a unique [`Pid`], tracked through state
//! transitions (Created → Running → Exited), and backed by a kernel-managed
//! **process store** whose lifecycle fields (`pid`, `name`, `status`,
//! `exit_code`) are observable through the existing [`store::watch()`]
//! mechanism.
//!
//! This replaces the traditional `waitpid`/`SIGCHLD` model with reactive
//! store subscriptions: any kernel task can watch a process's store and
//! react to state changes asynchronously.
//!
//! ## Design
//!
//! PIDs are monotonic `u64` values, never reused — same ABA-avoidance
//! pattern as [`StoreId`] and [`AsyncTaskId`](crate::executor::AsyncTaskId).
//! The process table is a simple `BTreeMap` protected by a spinlock.
//!
//! ## Concurrency
//!
//! The current PID is stored per-core in the [`PerCpu`](crate::arch::smp::PerCpu)
//! struct, accessed via GS-relative addressing. Each core can run a process
//! independently — [`current_pid()`] reads from the calling core's PerCpu,
//! and [`set_current()`] / [`clear_current()`] write to it.
//!
//! ## Lock ordering
//!
//! Process table lock → store registry lock. [`exit()`] acquires the
//! process table to look up the store ID and update state, releases it,
//! *then* calls [`store::set_no_cli`]. No path acquires them in reverse.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;

use crate::store::{self, StoreId};
use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

// ————————————————————————————————————————————————————————————————————————————
// Types
// ————————————————————————————————————————————————————————————————————————————

/// Process identifier — monotonic, never reused.
///
/// Same pattern as [`StoreId`]: a newtype over `u64` to prevent accidental
/// misuse of raw integers as PIDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pid(u64);

impl Pid {
    /// Reconstruct a `Pid` from its raw `u64` representation.
    pub fn from_raw(raw: u64) -> Self {
        Pid(raw)
    }

    /// Extract the raw `u64` value.
    pub fn as_raw(self) -> u64 {
        self.0
    }
}

/// The lifecycle state of a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    /// Spawned but not yet running — store created, resources allocated.
    Created,
    /// Currently executing in Ring 3.
    Running,
    /// Terminated — exit code captured in the process store.
    Exited,
}

/// A tracked process — links a PID to its lifecycle state and store.
struct Process {
    #[allow(dead_code)]
    pid: Pid,
    #[allow(dead_code)]
    name: &'static str,
    state: ProcessState,
    store_id: StoreId,
}

/// Schema for process lifecycle stores.
///
/// Every process gets a store with these four fields, created automatically
/// by [`spawn()`]. External observers can [`store::watch()`] the `status`
/// field to react to lifecycle transitions.
struct ProcessStoreSchema;

impl StoreSchema for ProcessStoreSchema {
    fn name() -> &'static str { "Process" }
    fn fields() -> &'static [FieldDef] {
        &[
            FieldDef { name: "pid", kind: FieldKind::U64 },
            FieldDef { name: "name", kind: FieldKind::Str },
            FieldDef { name: "status", kind: FieldKind::Str },
            FieldDef { name: "exit_code", kind: FieldKind::U64 },
        ]
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Globals
// ————————————————————————————————————————————————————————————————————————————

/// The process table — all live processes keyed by raw PID.
struct ProcessTable {
    processes: BTreeMap<u64, Process>,
    next_pid: u64,
}

/// Global process table, initialized once from `kmain` via [`init()`].
///
/// Uses `spin::Once<spin::Mutex<...>>` — same pattern as the store registry,
/// scheduler, and executor.
static PROCESS_TABLE: spin::Once<spin::Mutex<ProcessTable>> = spin::Once::new();

fn table() -> &'static spin::Mutex<ProcessTable> {
    PROCESS_TABLE.get().expect("process::init() not called")
}

// The PID of the currently executing userspace process is stored per-core
// in the PerCpu struct (accessed via GS base). See `smp::current_pid()`
// and `smp::set_current_pid()`. `u64::MAX` means no process is running.

// ————————————————————————————————————————————————————————————————————————————
// Public API
// ————————————————————————————————————————————————————————————————————————————

/// Initialize the process table. Must be called after [`store::init()`].
pub fn init() {
    PROCESS_TABLE.call_once(|| {
        spin::Mutex::new(ProcessTable {
            processes: BTreeMap::new(),
            next_pid: 1,
        })
    });
    crate::println!("[ok] Process table initialized");
}

/// Spawn a new process — allocate a PID, create its process store, and
/// insert it into the table in the `Created` state.
///
/// The process store is initialized with:
/// - `pid` = the allocated PID
/// - `name` = the provided name
/// - `status` = "created"
/// - `exit_code` = 0
pub fn spawn(name: &'static str) -> Pid {
    // Allocate the process store first (acquires registry lock internally).
    // We need the PID for the store defaults, so allocate it under the
    // process table lock, then release the lock to create the store, then
    // re-acquire to insert the process.
    let pid_raw = {
        let mut pt = table().lock();
        let pid = pt.next_pid;
        pt.next_pid += 1;
        pid
    };

    let store_id = store::create::<ProcessStoreSchema>(&[
        ("pid", Value::U64(pid_raw)),
        ("name", Value::Str(String::from(name))),
        ("status", Value::Str(String::from("created"))),
        ("exit_code", Value::U64(0)),
    ]).expect("create process store");

    let pid = Pid(pid_raw);
    let process = Process {
        pid,
        name,
        state: ProcessState::Created,
        store_id,
    };

    table().lock().processes.insert(pid_raw, process);
    pid
}

/// Transition a process from Created → Running.
///
/// Updates the process store's `status` field to `"running"`.
/// Called just before [`jump_to_ring3`](crate::arch::syscall::jump_to_ring3).
pub fn start(pid: Pid) {
    let store_id = {
        let mut pt = table().lock();
        let proc = pt.processes.get_mut(&pid.0)
            .expect("start: process not found");
        assert_eq!(proc.state, ProcessState::Created, "start: process not in Created state");
        proc.state = ProcessState::Running;
        proc.store_id
    };

    store::set(store_id, &[("status", Value::Str(String::from("running")))])
        .expect("update process store status to running");
}

/// Transition a process to Exited with the given exit code.
///
/// Updates the process store's `status` to `"exited"` and `exit_code`
/// to the provided value. Uses [`store::set_no_cli`] because this runs
/// in the syscall path with IF=0.
///
/// Lock ordering: acquires process table lock to look up store_id and
/// update state, releases it, *then* calls `store::set_no_cli` — never
/// holds both locks simultaneously.
pub fn exit(pid: Pid, exit_code: u64) {
    let store_id = {
        let mut pt = table().lock();
        let proc = pt.processes.get_mut(&pid.0)
            .expect("exit: process not found");
        proc.state = ProcessState::Exited;
        proc.store_id
    };

    // IF=0 in the syscall path — use set_no_cli to avoid re-enabling interrupts.
    store::set_no_cli(store_id, &[
        ("status", Value::Str(String::from("exited"))),
        ("exit_code", Value::U64(exit_code)),
    ]).expect("update process store on exit");
}

/// Read the PID of the currently executing userspace process on this core.
pub fn current_pid() -> Pid {
    Pid(crate::arch::smp::current_pid())
}

/// Set the current process PID on this core before entering Ring 3.
pub fn set_current(pid: Pid) {
    crate::arch::smp::set_current_pid(pid.0);
}

/// Clear the current process PID on this core after returning from Ring 3.
pub fn clear_current() {
    crate::arch::smp::set_current_pid(u64::MAX);
}

/// Look up the store ID backing a process's lifecycle store.
pub fn store_id(pid: Pid) -> Option<StoreId> {
    let pt = table().lock();
    pt.processes.get(&pid.0).map(|p| p.store_id)
}

/// Remove a process from the table and destroy its process store.
///
/// Call this after reading the exit status — the process store becomes
/// inaccessible after destruction.
pub fn destroy(pid: Pid) {
    let store_id = {
        let mut pt = table().lock();
        let proc = pt.processes.remove(&pid.0)
            .expect("destroy: process not found");
        proc.store_id
    };

    store::destroy(store_id).expect("destroy process store");
}
