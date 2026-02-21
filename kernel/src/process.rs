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
use alloc::vec::Vec;

use crate::arch;
use crate::memory::{self, FRAME_SIZE};
use crate::platform::{MemoryManager, PageFlags};
use crate::store::{self, Reducer, StoreAccessor, StoreId};
use oboos_api::{FieldDef, FieldKind, StoreError, StoreSchema, Value};

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

/// Tracks the demand-paged user stack for a process.
///
/// The stack occupies a 256 KiB virtual region growing downward from
/// [`USER_STACK_TOP`]. Only the top page is pre-mapped at spawn time;
/// additional pages are mapped on demand when the page fault handler detects
/// a touch within the stack region. A guard page below the region catches
/// stack overflow.
struct ProcessStack {
    /// Highest virtual address (exclusive) — initial RSP points here.
    top: usize,
    /// Lowest mappable virtual address (inclusive) — `top - MAX_PAGES * 4096`.
    bottom: usize,
    /// Guard page address — one page below `bottom`. Touching this is fatal.
    guard: usize,
    /// Physical frames backing mapped stack pages — `(virt, phys)` pairs,
    /// collected for cleanup on process exit.
    frames: Vec<(usize, usize)>,
}

/// Top of the user stack region — initial RSP.
pub const USER_STACK_TOP: usize = 0x0080_0000;

/// Maximum number of 4 KiB pages the stack can grow to (256 KiB).
const USER_STACK_MAX_PAGES: usize = 64;

/// Lowest mappable address in the stack region.
const USER_STACK_BOTTOM: usize = USER_STACK_TOP - USER_STACK_MAX_PAGES * FRAME_SIZE;

/// Guard page — one page below the stack region. Touching this = stack overflow.
const USER_STACK_GUARD: usize = USER_STACK_BOTTOM - FRAME_SIZE;

/// Tracks the virtual address range and physical frames of a process's heap,
/// grown incrementally via the `MapHeap` mutation.
struct ProcessHeap {
    /// Next virtual address to map at. Starts at [`HEAP_REGION_START`] and
    /// advances by `count * 4096` on each allocation.
    next_virt: usize,
    /// Physical frames backing the heap — collected here for cleanup on exit.
    frames: Vec<usize>,
}

/// Start of the userspace heap region. Grows upward via MUTATE/MapHeap.
const HEAP_REGION_START: usize = 0x0100_0000;

/// A tracked process — links a PID to its lifecycle state and store.
struct Process {
    #[allow(dead_code)]
    pid: Pid,
    #[allow(dead_code)]
    name: &'static str,
    state: ProcessState,
    store_id: StoreId,
    /// Physical address of this process's PML4 (root page table).
    pml4_phys: usize,
    stack: ProcessStack,
    heap: ProcessHeap,
}

/// Schema for process lifecycle stores.
///
/// Every process gets a store with these four fields, created automatically
/// by [`spawn()`]. External observers can [`store::watch()`] the `status`
/// field to react to lifecycle transitions.
pub(crate) struct ProcessStoreSchema;

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

    // Register the reducer so this process store can handle MUTATE syscalls
    // (MapHeap, Exit).
    store::register_reducer::<ProcessStoreSchema>(store_id)
        .expect("register process store reducer");

    let pml4_phys = crate::arch::x86_64::paging::create_user_page_table();

    let pid = Pid(pid_raw);
    let process = Process {
        pid,
        name,
        state: ProcessState::Created,
        store_id,
        pml4_phys,
        stack: ProcessStack {
            top: USER_STACK_TOP,
            bottom: USER_STACK_BOTTOM,
            guard: USER_STACK_GUARD,
            frames: Vec::new(),
        },
        heap: ProcessHeap {
            next_virt: HEAP_REGION_START,
            frames: Vec::new(),
        },
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

/// Look up the PML4 physical address for a process's page table.
pub fn pml4_phys(pid: Pid) -> Option<usize> {
    let pt = table().lock();
    pt.processes.get(&pid.0).map(|p| p.pml4_phys)
}

/// Map the initial top page of the user stack for a newly spawned process.
///
/// Called from [`userspace`](crate::userspace) after [`spawn()`]. Allocates one
/// physical frame and maps it at `USER_STACK_TOP - FRAME_SIZE` with user,
/// writable, no-execute flags. The page is zeroed through the HHDM so we don't
/// need a temporarily writable user mapping.
pub fn init_stack(pid: Pid) {
    use crate::arch::x86_64::memory::phys_to_virt;

    let mut pt = table().lock();
    let proc = pt.processes.get_mut(&pid.0)
        .expect("init_stack: process not found");

    let virt = USER_STACK_TOP - FRAME_SIZE;
    let frame = memory::alloc_frame().expect("out of memory allocating initial stack page");
    let flags = PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::USER | PageFlags::NO_EXECUTE;
    arch::Arch::map_page(virt, frame, flags);

    // Zero through the HHDM — same pattern as the ELF loader.
    unsafe {
        core::ptr::write_bytes(phys_to_virt(frame as u64), 0, FRAME_SIZE);
    }

    proc.stack.frames.push((virt, frame));
    crate::println!("[stack] Pre-mapped initial page at {:#X}", virt);
}

/// Demand-page a stack frame for a user-mode page fault.
///
/// Called from the #PF handler when the fault is a user-mode, page-not-present
/// access. Checks whether `fault_addr` falls within the process's stack region,
/// allocates a frame, maps it, and returns `Ok(())` so the handler can `iretq`
/// back to retry the faulting instruction.
///
/// # Errors
///
/// Returns `Err` if:
/// - `fault_addr` hits the guard page (stack overflow)
/// - `fault_addr` is outside the stack region
pub fn grow_stack(fault_addr: usize) -> Result<(), &'static str> {
    use crate::arch::x86_64::memory::phys_to_virt;
    use crate::arch::x86_64::paging::is_page_mapped;

    let page = fault_addr & !(FRAME_SIZE - 1);

    let pid = current_pid();
    let mut pt = table().lock();
    let proc = pt.processes.get_mut(&pid.0)
        .ok_or("grow_stack: no current process")?;

    if page == proc.stack.guard {
        return Err("stack overflow (hit guard page)");
    }

    if page < proc.stack.bottom || page >= proc.stack.top {
        return Err("fault outside stack region");
    }

    // Safety net: if the page is somehow already mapped, return success
    // to avoid map_page's panic on double-map.
    if is_page_mapped(page) {
        return Ok(());
    }

    let frame = memory::alloc_frame().ok_or("out of memory growing stack")?;
    let flags = PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::USER | PageFlags::NO_EXECUTE;
    arch::Arch::map_page(page, frame, flags);

    // Zero through the HHDM — not through the user mapping.
    unsafe {
        core::ptr::write_bytes(phys_to_virt(frame as u64), 0, FRAME_SIZE);
    }

    proc.stack.frames.push((page, frame));
    crate::println!("[stack] Demand-mapped page at {:#X}", page);

    Ok(())
}

/// Remove a process from the table, free its stack and heap, and destroy its
/// process store.
///
/// Returns the physical address of the process's PML4 so the caller can
/// free intermediate page table frames via [`paging::destroy_user_page_table()`]
/// after switching back to the kernel CR3.
///
/// Call this after reading the exit status — the process store becomes
/// inaccessible after destruction.
pub fn destroy(pid: Pid) -> usize {
    let (store_id, pml4_phys, stack, heap) = {
        let mut pt = table().lock();
        let proc = pt.processes.remove(&pid.0)
            .expect("destroy: process not found");
        (proc.store_id, proc.pml4_phys, proc.stack, proc.heap)
    };

    // Unmap and free all demand-paged stack pages.
    for &(virt, phys) in &stack.frames {
        arch::Arch::unmap_page(virt);
        memory::free_frame(phys);
    }

    // Unmap and free all heap pages.
    for (i, &frame) in heap.frames.iter().enumerate() {
        let virt = HEAP_REGION_START + i * FRAME_SIZE;
        arch::Arch::unmap_page(virt);
        memory::free_frame(frame);
    }

    store::destroy(store_id).expect("destroy process store");

    pml4_phys
}

// ————————————————————————————————————————————————————————————————————————————
// Heap page mapping
// ————————————————————————————————————————————————————————————————————————————

/// Allocate physical frames and map them into the process's heap region.
///
/// Returns the virtual address of the start of the newly mapped region.
/// Called from [`StoreAccessor::map_user_pages`] inside a MUTATE reducer.
///
/// Lock ordering: only touches the frame allocator and page tables — never
/// acquires the store registry lock (which the caller already holds).
pub fn map_heap_pages(pid: Pid, count: usize) -> Result<u64, StoreError> {
    if count == 0 || count > 256 {
        return Err(StoreError::InvalidArg);
    }

    let mut pt = table().lock();
    let proc = pt.processes.get_mut(&pid.0).ok_or(StoreError::NotFound)?;

    let start_virt = proc.heap.next_virt;
    let flags = PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::USER | PageFlags::NO_EXECUTE;

    for i in 0..count {
        let frame = memory::alloc_frame().ok_or(StoreError::InvalidArg)?;
        let virt = proc.heap.next_virt;
        arch::Arch::map_page(virt, frame, flags);

        // Zero the page so userspace doesn't see stale kernel data.
        unsafe {
            core::ptr::write_bytes(virt as *mut u8, 0, FRAME_SIZE);
        }

        proc.heap.frames.push(frame);
        proc.heap.next_virt += FRAME_SIZE;

        // If allocation fails partway, the already-mapped pages remain and
        // will be cleaned up when the process exits. This is simpler than
        // trying to roll back and matches how sbrk() works on real systems.
        let _ = i;
    }

    Ok(start_virt as u64)
}

// ————————————————————————————————————————————————————————————————————————————
// Process store mutations
// ————————————————————————————————————————————————————————————————————————————

/// Mutations supported by the process store — the trust boundary for
/// process-level kernel operations.
pub(crate) enum ProcessMutation {
    /// Allocate and map `pages` physical frames into the process's heap.
    MapHeap { pages: u64 },
    /// Terminate the process with the given exit code.
    Exit { code: u64 },
}

impl Reducer for ProcessStoreSchema {
    type Mutation = ProcessMutation;

    fn reduce(
        store: &mut StoreAccessor,
        mutation: ProcessMutation,
    ) -> Result<u64, StoreError> {
        match mutation {
            ProcessMutation::MapHeap { pages } => {
                let addr = store.map_user_pages(pages as usize)?;
                Ok(addr)
            }
            ProcessMutation::Exit { code } => {
                store.set("status", String::from("exited"))?;
                store.set("exit_code", code)?;
                store.request_process_exit(code);
                Ok(0)
            }
        }
    }

    fn deserialize(id: u8, payload: &[u8]) -> Result<ProcessMutation, StoreError> {
        match id {
            oboos_api::PROCESS_MUTATE_MAP_HEAP => {
                let bytes: [u8; 8] = payload.try_into().map_err(|_| StoreError::InvalidArg)?;
                let pages = u64::from_ne_bytes(bytes);
                Ok(ProcessMutation::MapHeap { pages })
            }
            oboos_api::PROCESS_MUTATE_EXIT => {
                let bytes: [u8; 8] = payload.try_into().map_err(|_| StoreError::InvalidArg)?;
                let code = u64::from_ne_bytes(bytes);
                Ok(ProcessMutation::Exit { code })
            }
            _ => Err(StoreError::InvalidArg),
        }
    }
}
