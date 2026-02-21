//! Symmetric Multiprocessing (SMP) — per-CPU data and AP bringup.
//!
//! On x86_64, the BSP (Bootstrap Processor) is the core that runs `kmain`.
//! Additional cores are called APs (Application Processors). Limine's
//! `MpRequest` handles the hard part of AP bringup — the real-mode
//! trampoline and INIT-SIPI-SIPI sequence that wakes each AP from its
//! post-reset sleep state. We just need to:
//!
//! 1. Set up per-CPU data (GDT, TSS, double-fault stack) for each AP
//! 2. Write each AP's entry function via `goto_address`
//! 3. Wait for all APs to come online
//!
//! Per-CPU data is accessed via the GS segment base register, set to
//! point at the core's [`PerCpu`] struct via `wrmsr(IA32_GS_BASE)`.
//! The first field is a self-pointer so `gs:[0]` always yields the
//! PerCpu address — a common pattern in OS kernels (Linux uses the
//! same trick with its `current_task` pointer).

use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use super::gdt::{self, Tss};

/// Maximum number of CPUs we support. 16 is generous for QEMU testing
/// and early development; can be bumped later for real hardware.
const MAX_CPUS: usize = 16;

/// MSR address for the GS segment base. Writing this MSR sets the base
/// address that the CPU adds to any GS-relative memory access (`gs:[offset]`).
/// Unlike the GDT-based segment base (which is zeroed in long mode for
/// most segments), the GS base MSR is the modern mechanism for per-CPU data.
const IA32_GS_BASE: u32 = 0xC000_0101;

// ————————————————————————————————————————————————————————————————————————————
// Per-CPU data
// ————————————————————————————————————————————————————————————————————————————

/// Per-CPU data structure, one instance per core, accessed via `gs:[offset]`.
///
/// Each core has its own GDT and TSS because the TSS holds per-core stack
/// pointers: RSP0 (for ring 3→0 transitions) and IST entries (for
/// double-fault handling). If cores shared a TSS, a double fault on one
/// core could corrupt the other core's interrupt stack.
///
/// The struct is `#[repr(C)]` so field offsets are predictable for
/// assembly code that accesses them via GS-relative addressing.
#[repr(C)]
pub struct PerCpu {
    /// Self-pointer at offset 0 for fast access: `mov rax, gs:[0]`
    /// gives the PerCpu address without needing to read the MSR.
    self_ptr: *const PerCpu,
    /// LAPIC ID as reported by the firmware. Not necessarily contiguous —
    /// multi-socket systems commonly have gaps (e.g., 0, 2, 4, 6 instead
    /// of 0, 1, 2, 3). Our code never assumes contiguous IDs.
    pub lapic_id: u32,
    /// Sequential index (0 = BSP, 1..N = APs). Used to index into
    /// per-CPU arrays. Always contiguous.
    pub cpu_index: u32,
    /// Saved user RSP during syscall handling. Written by `syscall_entry`
    /// before switching to the kernel stack. Per-core so multiple cores
    /// can handle syscalls simultaneously.
    pub saved_user_rsp: u64,
    /// Kernel stack pointer loaded by `syscall_entry`. Set by
    /// [`set_kernel_rsp`](super::syscall::set_kernel_rsp) before Ring 3.
    pub kernel_rsp: u64,
    /// Saved kernel context for the process exit return path. Layout:
    /// `[rsp, rbx, rbp, r12, r13, r14, r15]` — the callee-saved register
    /// set saved by [`jump_to_ring3`](super::syscall::jump_to_ring3).
    pub return_context: [u64; 7],
    /// PID of the currently executing userspace process on this core.
    /// `u64::MAX` means no process is running (sentinel).
    pub current_pid: u64,
    /// Per-CPU GDT — each core needs its own because the TSS descriptor
    /// embeds the TSS's physical address, which differs per core.
    gdt: [u64; 7],
    /// Per-CPU TSS — holds this core's RSP0 and IST stacks.
    /// `pub(crate)` so `gdt::set_rsp0()` can write RSP0 via raw pointer.
    pub(crate) tss: Tss,
    /// Dedicated 4 KiB double-fault stack for this core's IST1.
    /// If this core's kernel stack overflows, the CPU switches to this
    /// stack before running the double-fault handler.
    double_fault_stack: [u8; 4096],
    /// Set to `true` once this core has finished initialization.
    pub online: AtomicBool,
}

// PerCpu contains a raw pointer (self_ptr) which opts it out of Send/Sync
// by default. This is safe because each PerCpu is only written during init
// (single-threaded per core) and then read exclusively by its owning core
// via GS base.
unsafe impl Send for PerCpu {}
unsafe impl Sync for PerCpu {}

/// Storage for per-CPU data. Indexed by cpu_index (0 = BSP).
///
/// Uses `MaybeUninit` because we initialize each entry during SMP bringup
/// via raw pointer writes, not Rust constructors. The BSP initializes
/// index 0, each AP initializes its own index.
static mut PER_CPU_DATA: [MaybeUninit<PerCpu>; MAX_CPUS] =
    [const { MaybeUninit::uninit() }; MAX_CPUS];

/// Number of CPUs that have completed initialization.
/// Starts at 1 (the BSP is always online).
static CPU_COUNT: AtomicU32 = AtomicU32::new(1);

// ————————————————————————————————————————————————————————————————————————————
// PerCpu field offsets for assembly
// ————————————————————————————————————————————————————————————————————————————

/// Byte offset of `saved_user_rsp` in [`PerCpu`], for GS-relative access
/// in the [`syscall_entry`](super::syscall) naked assembly.
pub const PERCPU_SAVED_USER_RSP: usize = core::mem::offset_of!(PerCpu, saved_user_rsp);

/// Byte offset of `kernel_rsp` in [`PerCpu`].
pub const PERCPU_KERNEL_RSP: usize = core::mem::offset_of!(PerCpu, kernel_rsp);

/// Byte offset of `return_context` in [`PerCpu`].
pub const PERCPU_RETURN_CONTEXT: usize = core::mem::offset_of!(PerCpu, return_context);

/// Byte offset of `current_pid` in [`PerCpu`].
pub const PERCPU_CURRENT_PID: usize = core::mem::offset_of!(PerCpu, current_pid);

// ————————————————————————————————————————————————————————————————————————————
// TLB shootdown state
// ————————————————————————————————————————————————————————————————————————————

/// Virtual address to invalidate on remote cores during TLB shootdown.
static TLB_SHOOTDOWN_ADDR: AtomicU64 = AtomicU64::new(0);

/// Number of cores that still need to execute `invlpg`. The initiator
/// spins until this reaches 0.
static TLB_SHOOTDOWN_PENDING: AtomicU32 = AtomicU32::new(0);

// ————————————————————————————————————————————————————————————————————————————
// Public API
// ————————————————————————————————————————————————————————————————————————————

/// Return the number of online CPUs.
pub fn cpu_count() -> u32 {
    CPU_COUNT.load(Ordering::Acquire)
}

/// Return the LAPIC ID for a given CPU index. Used for IPI targeting.
pub fn lapic_id_for_cpu(cpu_index: u32) -> u32 {
    unsafe {
        let percpu = PER_CPU_DATA[cpu_index as usize].as_ptr();
        (*percpu).lapic_id
    }
}

/// Return the calling core's CPU index by reading the PerCpu via GS base.
///
/// Must only be called after [`init_bsp_percpu()`] on the BSP or after
/// AP entry has set GS base. Returns 0..N-1 contiguous indices.
pub fn current_cpu() -> u32 {
    unsafe {
        let percpu: u64;
        core::arch::asm!(
            "mov {}, gs:[0]",
            out(reg) percpu,
            options(nomem, nostack, preserves_flags),
        );
        (*(percpu as *const PerCpu)).cpu_index
    }
}

/// Return a raw pointer to the current core's PerCpu struct.
///
/// Reads the self-pointer from `gs:[0]`. Must only be called after
/// [`init_bsp_percpu()`] on the BSP or after AP entry has set GS base.
pub fn current_percpu() -> *mut PerCpu {
    unsafe {
        let percpu: u64;
        core::arch::asm!(
            "mov {}, gs:[0]",
            out(reg) percpu,
            options(nomem, nostack, preserves_flags),
        );
        percpu as *mut PerCpu
    }
}

/// Read the PID of the currently executing userspace process on this core.
/// Returns `u64::MAX` if no process is running.
pub fn current_pid() -> u64 {
    unsafe { (*current_percpu()).current_pid }
}

/// Set the current process PID for this core.
pub fn set_current_pid(pid: u64) {
    unsafe { (*current_percpu()).current_pid = pid; }
}

/// Invalidate a virtual page's TLB entry on all other cores.
///
/// Sends IPI vector 50 to every online core except the caller, then
/// spins until all have executed `invlpg`. The caller must have already
/// invalidated the page locally before calling this.
///
/// Only meaningful when `cpu_count() > 1`. Safe to call from
/// non-interrupt context (IF can be either 0 or 1 — the sender doesn't
/// need to receive interrupts, only remote cores do).
///
/// # Concurrent shootdown safety
///
/// This function uses a single global `TLB_SHOOTDOWN_ADDR` /
/// `TLB_SHOOTDOWN_PENDING` pair, so concurrent shootdowns from
/// multiple cores would race. This is currently safe because all
/// callers hold the process table lock or run from a single kernel
/// context (page fault handler with IF=0). If concurrent unmap/remap
/// from multiple cores is ever needed, this will need a per-shootdown
/// spinlock or a per-core request queue.
pub fn tlb_shootdown(virt: usize) {
    let count = cpu_count();
    if count <= 1 {
        return;
    }

    let current = current_cpu();

    TLB_SHOOTDOWN_ADDR.store(virt as u64, Ordering::Release);
    TLB_SHOOTDOWN_PENDING.store(count - 1, Ordering::Release);

    // Send the shootdown IPI to every other online core.
    for i in 0..count {
        if i != current {
            let target = lapic_id_for_cpu(i);
            super::lapic::send_ipi(target, super::lapic::TLB_SHOOTDOWN_VECTOR);
        }
    }

    // Spin until all remote cores have invalidated.
    while TLB_SHOOTDOWN_PENDING.load(Ordering::Acquire) > 0 {
        core::hint::spin_loop();
    }
}

/// Handle a TLB shootdown IPI on the receiving core.
///
/// Called from the vector 50 interrupt handler. Reads the target address,
/// executes `invlpg`, and decrements the pending counter.
pub fn handle_tlb_shootdown() {
    let addr = TLB_SHOOTDOWN_ADDR.load(Ordering::Acquire);
    unsafe {
        core::arch::asm!("invlpg [{}]", in(reg) addr, options(nostack, preserves_flags));
    }
    TLB_SHOOTDOWN_PENDING.fetch_sub(1, Ordering::AcqRel);
}

/// Set up the BSP's PerCpu struct and GS base register early.
///
/// Called before `scheduler::init()` so that `current_cpu()` works
/// on the BSP. The LAPIC ID is filled in later by [`init()`] once
/// the MP response is available.
pub fn init_bsp_percpu() {
    unsafe {
        let bsp = PER_CPU_DATA[0].as_mut_ptr();
        core::ptr::write_bytes(bsp as *mut u8, 0, core::mem::size_of::<PerCpu>());
        (*bsp).cpu_index = 0;
        (*bsp).self_ptr = bsp;
        (*bsp).current_pid = u64::MAX;
        (*bsp).online = AtomicBool::new(true);

        // Reload GDT+TSS from PerCpu storage so the BSP uses the same
        // per-core layout as APs. After this, set_rsp0() writes to the
        // PerCpu's TSS instead of the static one in gdt.rs.
        gdt::init_gdt_tss(
            &mut (*bsp).gdt,
            &mut (*bsp).tss,
            &(*bsp).double_fault_stack,
        );

        wrmsr(IA32_GS_BASE, bsp as u64);
    }
}

/// Bring up all APs: assign each an index and start their entry function.
///
/// The BSP's PerCpu LAPIC ID is also filled in from the MP response.
/// Each AP sets up its own GDT/TSS, loads the shared IDT, initializes
/// its LAPIC, creates a per-core scheduler, and enters the idle loop.
///
/// Must be called after `gdt::init()`, `interrupts::init()`,
/// `scheduler::init()`, and `heap::init()`.
pub fn init(mp_response: &limine::response::MpResponse) {
    let bsp_lapic_id = mp_response.bsp_lapic_id();
    let cpus = mp_response.cpus();

    // Fill in BSP's LAPIC ID (PerCpu struct was already created by
    // init_bsp_percpu, but LAPIC ID wasn't known yet).
    unsafe {
        let bsp = PER_CPU_DATA[0].as_mut_ptr();
        (*bsp).lapic_id = bsp_lapic_id;
    }

    crate::println!("[smp] BSP online (LAPIC ID {})", bsp_lapic_id);

    // Bring up APs. Each AP gets an index (1, 2, ...) stored in the
    // Limine `extra` field so the AP entry function can read it.
    let mut ap_index = 1u32;
    for cpu in cpus {
        if cpu.lapic_id == bsp_lapic_id {
            continue; // Skip the BSP
        }

        if ap_index as usize >= MAX_CPUS {
            crate::println!("[smp] Warning: more CPUs than MAX_CPUS ({}), skipping rest", MAX_CPUS);
            break;
        }

        // Store the cpu_index in the `extra` field — the AP reads this
        // to know which PER_CPU_DATA slot to initialize.
        cpu.extra.store(ap_index as u64, Ordering::Relaxed);

        // Write the entry function pointer. Limine uses SeqCst on the
        // store, which acts as a release barrier ensuring the `extra`
        // write is visible to the AP before it starts executing.
        cpu.goto_address.write(ap_entry);

        ap_index += 1;
    }

    // Wait for all APs to come online. Use a spin counter as timeout
    // since the PIT isn't ticking yet (IF=0 during init).
    let expected = ap_index;
    let mut spins = 0u64;
    while CPU_COUNT.load(Ordering::Acquire) < expected {
        spins += 1;
        if spins > 100_000_000 {
            crate::println!(
                "[smp] Timeout waiting for APs ({}/{} online)",
                CPU_COUNT.load(Ordering::Acquire),
                expected,
            );
            break;
        }
        core::hint::spin_loop();
    }

    crate::println!(
        "[ok] SMP: {} CPUs online",
        CPU_COUNT.load(Ordering::Acquire),
    );
}

// ————————————————————————————————————————————————————————————————————————————
// AP entry point
// ————————————————————————————————————————————————————————————————————————————

/// AP entry point — called by Limine when an AP's `goto_address` is written.
///
/// Each AP arrives here in 64-bit long mode with paging enabled (same
/// page tables as the BSP), a small Limine-provided stack, and a minimal
/// GDT (no TSS). We set up the full GDT/TSS, load the shared IDT, set
/// GS base to our PerCpu, and park in `hlt`. Later phases will have APs
/// participate in scheduling and async execution.
///
/// # Safety
///
/// Called by the Limine MP protocol with interrupts disabled. The
/// `cpu_info` reference is valid for the duration of this function.
unsafe extern "C" fn ap_entry(cpu_info: &limine::mp::Cpu) -> ! {
    let cpu_index = cpu_info.extra.load(Ordering::Relaxed) as usize;
    let lapic_id = cpu_info.lapic_id;

    unsafe {
        let percpu = PER_CPU_DATA[cpu_index].as_mut_ptr();

        // Zero the PerCpu to clear the embedded TSS reserved fields and
        // double-fault stack. AtomicBool is safe to zero (false = 0).
        core::ptr::write_bytes(percpu as *mut u8, 0, core::mem::size_of::<PerCpu>());

        // Fill identity fields.
        (*percpu).lapic_id = lapic_id;
        (*percpu).cpu_index = cpu_index as u32;
        (*percpu).self_ptr = percpu;
        (*percpu).current_pid = u64::MAX;

        // Initialize per-CPU GDT+TSS. This loads a full GDT with TSS
        // (replacing Limine's minimal one), reloads all segment registers,
        // and loads the Task Register.
        gdt::init_gdt_tss(
            &mut (*percpu).gdt,
            &mut (*percpu).tss,
            &(*percpu).double_fault_stack,
        );

        // Load the shared IDT — all cores use the same interrupt handlers,
        // just with different stacks (via per-CPU TSS).
        super::interrupts::load_idt();

        // Configure SYSCALL/SYSRET MSRs. These are per-CPU registers —
        // each core needs its own LSTAR, STAR, FMASK, and EFER.SCE so
        // it can handle syscalls independently.
        super::syscall::init_syscall_msrs();

        // Set GS base to point at this core's PerCpu. From here on,
        // `gs:[0]` = self_ptr, `gs:[8]` = lapic_id, etc.
        wrmsr(IA32_GS_BASE, percpu as u64);

        // Initialize this AP's Local APIC and start its timer. The APIC
        // base address and calibrated tick rate are already known from BSP
        // init — each AP just enables its own APIC hardware.
        super::lapic::init_ap();

        // Signal that this AP is online.
        (*percpu).online.store(true, Ordering::Release);
    }

    CPU_COUNT.fetch_add(1, Ordering::AcqRel);

    crate::println!("[smp] CPU {} online (LAPIC ID {})", cpu_index, lapic_id);

    // Create this AP's per-core scheduler and async executor. The
    // bootstrap task wraps the current execution context (the executor's
    // run loop below), so work-stolen tasks can preempt it and return.
    crate::scheduler::init_ap(cpu_index as u32);
    crate::executor::init_ap(cpu_index as u32);

    // Enable interrupts and enter the executor loop. poll_once() drains
    // the wake queue and polls woken futures, then hlt waits for the
    // next interrupt (timer tick, IPI wake, etc.).
    unsafe { core::arch::asm!("sti") };
    crate::executor::run();
}

// ————————————————————————————————————————————————————————————————————————————
// MSR helper
// ————————————————————————————————————————————————————————————————————————————

/// Write a Model-Specific Register.
///
/// MSRs are CPU configuration registers addressed by a 32-bit index.
/// `wrmsr` splits the 64-bit value across EDX:EAX (high:low) — a
/// convention inherited from the 32-bit era.
unsafe fn wrmsr(msr: u32, value: u64) {
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    unsafe {
        core::arch::asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") lo,
            in("edx") hi,
            options(nomem, nostack, preserves_flags),
        );
    }
}
