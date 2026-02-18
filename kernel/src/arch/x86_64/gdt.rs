//! Global Descriptor Table (GDT) and Task State Segment (TSS).
//!
//! The GDT is a table of segment descriptors that the CPU consults on every
//! instruction fetch. In 64-bit long mode, most segmentation fields are ignored
//! (base and limit don't matter for code/data segments), but the GDT still
//! serves three critical purposes:
//!
//! 1. **Mode selection** — the L (Long) bit in the code segment descriptor is
//!    what keeps the CPU in 64-bit mode. Without it, you're in compatibility
//!    mode (32-bit).
//!
//! 2. **Privilege levels** — each segment descriptor specifies a ring level
//!    (DPL). Ring 0 = kernel, Ring 3 = userspace. The CPU checks this on every
//!    instruction to enforce isolation.
//!
//! 3. **TSS pointer** — the GDT is the only place the CPU looks for the Task
//!    State Segment descriptor, which points to the TSS struct containing
//!    interrupt stack pointers.
//!
//! The TSS (Task State Segment) is a CPU structure that holds:
//! - **RSP0–RSP2**: Stack pointers for ring transitions. When a user-mode
//!   program (ring 3) triggers a syscall or interrupt, the CPU reads RSP0
//!   from the TSS to find the kernel stack.
//!
//! - **IST1–IST7**: Interrupt Stack Table entries. When an IDT entry specifies
//!   an IST index, the CPU unconditionally switches to that stack before
//!   pushing the interrupt frame. This is critical for double faults — if
//!   the kernel stack itself is corrupted, the IST gives us a known-good
//!   stack to land on.
//!
//! ## Why replace Limine's GDT?
//!
//! Limine sets up a minimal GDT (null + code + data) to get us into 64-bit
//! mode, but it doesn't include a TSS. Without a TSS, we can't use the IST,
//! which means a double fault will escalate to a triple fault (CPU reset).
//! We need our own GDT so we can add the TSS descriptor and control our
//! segment layout going forward.

use core::mem::size_of;

// ————————————————————————————————————————————————————————————————————————————
// Segment selectors
// ————————————————————————————————————————————————————————————————————————————

/// Segment selectors are 16-bit values that index into the GDT. The format is:
///
/// ```text
/// Bits 15–3:  GDT index (which entry in the table)
/// Bit 2:      Table Indicator — 0 = GDT, 1 = LDT (we never use the LDT)
/// Bits 1–0:   RPL (Requested Privilege Level) — 0 for kernel
/// ```
///
/// So index 1 with RPL 0 = `0b0000_0000_0000_1_0_00` = 0x08.
/// Selectors are always multiples of 8 because the bottom 3 bits are flags.
pub const KERNEL_CODE_SELECTOR: u16 = 0x08; // GDT index 1 — 64-bit kernel code
const KERNEL_DATA_SELECTOR: u16 = 0x10; // GDT index 2 — kernel data
/// User data segment selector (DPL=3). Placed before user code because
/// SYSRET loads SS = STAR[63:48] + 8 and CS = STAR[63:48] + 16 — the
/// Intel/AMD SYSCALL convention requires data before code.
pub const USER_DATA_SELECTOR: u16 = 0x18; // GDT index 3
/// User code segment selector (DPL=3, Long mode). RPL=3 in the selector
/// value (0x20 | 3 = 0x23) tells the CPU "running at ring 3."
pub const USER_CODE_SELECTOR: u16 = 0x20; // GDT index 4
const TSS_SELECTOR: u16 = 0x28; // GDT index 5 — TSS descriptor (moved from 0x18)

// ————————————————————————————————————————————————————————————————————————————
// Task State Segment (TSS)
// ————————————————————————————————————————————————————————————————————————————

/// The TSS as defined by the Intel/AMD 64-bit architecture.
///
/// Despite the name "Task State Segment," modern OSes don't use it for
/// hardware task switching (that's a 32-bit legacy feature). In 64-bit mode,
/// the TSS exists purely as a container for stack pointers:
///
/// - `rsp[0..3]` (RSP0–RSP2): used on ring transitions (e.g., user → kernel)
/// - `ist[0..7]` (IST1–IST7): emergency stacks for specific interrupt vectors
///
/// The struct layout is mandated by the CPU — every field must be at exactly
/// the right offset, hence `#[repr(C, packed)]`. The reserved fields exist
/// because this struct evolved from the 32-bit TSS, which had fields for
/// hardware task switching that are now vestigial.
#[repr(C, packed)]
struct Tss {
    /// Reserved (was "link to previous TSS" in 32-bit hardware task switching).
    _reserved0: u32,

    /// Privilege-level stack pointers. RSP0 is loaded into RSP when the CPU
    /// transitions from ring 3 → ring 0 (e.g., syscall or interrupt from
    /// user mode). RSP1 and RSP2 are for rings 1 and 2, which nobody uses.
    /// We'll set RSP0 when we have user-mode tasks.
    rsp: [u64; 3],

    /// Reserved (was CR3 and EIP in the 32-bit TSS).
    _reserved1: u64,

    /// Interrupt Stack Table. When an IDT entry has a non-zero IST field
    /// (1–7), the CPU loads the corresponding IST pointer into RSP before
    /// pushing the interrupt frame. This happens *unconditionally* — even
    /// if we're already in ring 0 — which is what makes it useful for
    /// double faults where the current stack might be trashed.
    ///
    /// We use IST[0] (= IST1 in Intel terminology, they 1-index it) for
    /// the double-fault handler's stack.
    ist: [u64; 7],

    /// Reserved.
    _reserved2: u64,

    /// Reserved.
    _reserved3: u16,

    /// Offset to the I/O permission bitmap, relative to the TSS base.
    /// Setting this to `size_of::<Tss>()` means "no I/O bitmap" — the
    /// bitmap would start past the end of the TSS, so it effectively
    /// doesn't exist. (An I/O bitmap controls which ports user-mode code
    /// can access via `in`/`out` instructions — we don't need one yet.)
    iomap_base: u16,
}

// ————————————————————————————————————————————————————————————————————————————
// Static storage
// ————————————————————————————————————————————————————————————————————————————

/// The double-fault handler's dedicated stack.
///
/// When a double fault fires, the CPU switches to this stack (via IST1)
/// before executing the handler. This is critical: if a stack overflow
/// caused the original fault, we can't use the same stack to handle it.
///
/// 4 KiB is plenty — the double-fault handler just prints an error and
/// halts. Stacks grow downward on x86, so IST1 will point to the *end*
/// of this array (address of element [4095] + 1).
static mut DOUBLE_FAULT_STACK: [u8; 4096] = [0; 4096];

/// The TSS instance. Filled in by `init()`.
static mut TSS: Tss = Tss {
    _reserved0: 0,
    rsp: [0; 3],
    _reserved1: 0,
    ist: [0; 7],
    _reserved2: 0,
    _reserved3: 0,
    iomap_base: 0,
};

/// The GDT. 7 entries × 8 bytes = 56 bytes.
///
/// Entry layout:
/// ```text
/// [0] 0x00  Null descriptor       — CPU requires this, faults if you use it
/// [1] 0x08  Kernel code (64-bit)  — CS points here (DPL=0)
/// [2] 0x10  Kernel data           — DS/ES/SS point here (DPL=0)
/// [3] 0x18  User data             — user-mode SS (DPL=3)
/// [4] 0x20  User code (64-bit)    — user-mode CS (DPL=3, L=1)
/// [5] 0x28  TSS low 8 bytes   ─┐
/// [6] 0x30  TSS high 8 bytes  ─┘  Together form a 16-byte system descriptor
/// ```
///
/// The STAR MSR (used by SYSCALL/SYSRET) constrains the ordering: SYSRET
/// loads CS = STAR[63:48] + 16 and SS = STAR[63:48] + 8, so user data must
/// come before user code. Kernel code/data at indices 1–2 are similarly
/// constrained by SYSCALL (CS = STAR[47:32], SS = STAR[47:32] + 8).
///
/// The TSS descriptor is 16 bytes because it needs to hold a full 64-bit
/// base address. Regular segment descriptors only have room for a 32-bit
/// base (which is ignored in long mode anyway), but system descriptors
/// (like TSS) still use the base field, so the CPU extends it to 16 bytes.
static mut GDT: [u64; 7] = [0; 7];

/// The GDT Register (GDTR) — loaded via `lgdt`.
///
/// This tells the CPU where the GDT lives and how big it is.
/// `limit` is the byte size minus 1 (so 39 for a 40-byte table).
/// `base` is the virtual address of the first byte.
#[repr(C, packed)]
struct GdtRegister {
    limit: u16,
    base: u64,
}

// ————————————————————————————————————————————————————————————————————————————
// Initialization
// ————————————————————————————————————————————————————————————————————————————

/// Set up and load the GDT + TSS.
///
/// After this function returns:
/// - CS points to our 64-bit kernel code segment (selector 0x08)
/// - DS/ES/SS point to our kernel data segment (selector 0x10)
/// - User data (0x18) and user code (0x20) segments are ready for Ring 3
/// - The CPU knows about our TSS (loaded via `ltr`, selector 0x28)
/// - IST1 in the TSS points to a dedicated 4 KiB double-fault stack
///
/// This must be called before setting up the IDT, because the
/// double-fault IDT entry will reference IST1, which lives in the TSS.
pub fn init() {
    // Safety: we're in single-threaded kernel init, no other code is
    // touching these statics yet. After init, the GDT and TSS are
    // read-only from the CPU's perspective (it reads them, we don't
    // write them again until we set RSP0 for user-mode tasks later).
    unsafe {
        // ── Step 1: Fill the TSS ──────────────────────────────────────
        //
        // Point IST[0] (= IST1 in Intel's 1-indexed naming) to the top
        // of our double-fault stack. "Top" = highest address, because
        // x86 stacks grow downward (push decrements RSP).
        let stack_base = &raw const DOUBLE_FAULT_STACK as u64;
        let stack_top = stack_base + size_of::<[u8; 4096]>() as u64;
        (*(&raw mut TSS)).ist[0] = stack_top;

        // "No I/O bitmap" — set the offset past the end of the TSS.
        (*(&raw mut TSS)).iomap_base = size_of::<Tss>() as u16;

        // ── Step 2: Build the GDT entries ─────────────────────────────
        //
        // Entry 0: Null descriptor (required by the CPU).
        let gdt = &raw mut GDT;
        (*gdt)[0] = 0;

        // Entry 1: Kernel code segment (selector 0x08).
        //
        // Bit-by-bit breakdown of 0x00AF_9A00_0000_FFFF:
        //
        //   Limit[15:0]  = 0xFFFF   (ignored in 64-bit mode, but convention)
        //   Base[15:0]   = 0x0000   (ignored in 64-bit mode)
        //   Base[23:16]  = 0x00     (ignored in 64-bit mode)
        //   Access byte  = 0x9A:
        //     P  (Present)     = 1  — segment is valid
        //     DPL              = 00 — ring 0 (kernel privilege)
        //     S  (Descriptor)  = 1  — this is a code/data segment, not a system descriptor
        //     Type             = 0xA (1010):
        //       Code           = 1  — this is a code segment (executable)
        //       Conforming     = 0  — don't allow lower-privilege code to call in
        //       Readable       = 1  — code can also be read (needed for some instructions)
        //       Accessed       = 0  — CPU sets this on first use, we leave it clear
        //   Flags nibble = 0xA:
        //     G  (Granularity) = 1  — limit is in 4 KiB pages (irrelevant in 64-bit)
        //     D/B              = 0  — must be 0 when L=1
        //     L  (Long mode)   = 1  — THIS IS THE BIT THAT KEEPS US IN 64-BIT MODE
        //     AVL              = 0  — available for OS use, we don't use it
        //   Limit[19:16] = 0xF     (ignored in 64-bit mode)
        //   Base[31:24]  = 0x00    (ignored in 64-bit mode)
        (*gdt)[1] = 0x00AF_9A00_0000_FFFF;

        // Entry 2: Kernel data segment (selector 0x10).
        //
        // 0x00CF_9200_0000_FFFF:
        //   Access byte = 0x92:
        //     P=1, DPL=00, S=1, Type=0x2 (0010):
        //       Code       = 0  — data segment (not executable)
        //       Direction  = 0  — grows up (conventional)
        //       Writable   = 1  — can write to data through this segment
        //       Accessed   = 0
        //   Flags nibble = 0xC:
        //     G=1, D/B=1 (32-bit stack ops), L=0 (L only applies to code), AVL=0
        (*gdt)[2] = 0x00CF_9200_0000_FFFF;

        // Entry 3: User data segment (selector 0x18).
        //
        // 0x00CF_F200_0000_FFFF:
        //   Access byte = 0xF2:
        //     P=1, DPL=11 (ring 3), S=1, Type=0x2 (data, writable)
        //   Flags nibble = 0xC:
        //     G=1, D/B=1, L=0 (L only applies to code), AVL=0
        //
        // Must come before user code because SYSRET loads SS from
        // STAR[63:48]+8 (this slot) and CS from STAR[63:48]+16 (next slot).
        (*gdt)[3] = 0x00CF_F200_0000_FFFF;

        // Entry 4: User code segment (selector 0x20).
        //
        // 0x00AF_FA00_0000_FFFF:
        //   Access byte = 0xFA:
        //     P=1, DPL=11 (ring 3), S=1, Type=0xA (code, readable)
        //   Flags nibble = 0xA:
        //     G=1, D/B=0, L=1 (64-bit mode!), AVL=0
        //
        // The L bit is what keeps user code in 64-bit long mode — same
        // role as in the kernel code segment, just with DPL=3.
        (*gdt)[4] = 0x00AF_FA00_0000_FFFF;

        // Entries 5–6: TSS descriptor (16 bytes = two u64 slots).
        //
        // System descriptors (like TSS) are different from code/data descriptors.
        // In 64-bit mode they're extended to 16 bytes so they can hold a full
        // 64-bit base address. The layout is:
        //
        // Low 8 bytes (GDT[5]):
        //   Bits  0–15:  Limit[15:0]       — TSS size minus 1
        //   Bits 16–39:  Base[23:0]        — low 24 bits of TSS address
        //   Bits 40–47:  Access byte       — 0x89 = Present, Type 0x9 (64-bit TSS, available)
        //   Bits 48–51:  Limit[19:16]      — upper nibble of limit
        //   Bits 52–55:  Flags             — 0 for system descriptors
        //   Bits 56–63:  Base[31:24]       — next 8 bits of TSS address
        //
        // High 8 bytes (GDT[6]):
        //   Bits  0–31:  Base[63:32]       — upper 32 bits of TSS address
        //   Bits 32–63:  Reserved (must be zero)
        let tss_addr = &raw const TSS as u64;
        let tss_limit = (size_of::<Tss>() - 1) as u64;

        let mut low: u64 = 0;
        low |= tss_limit & 0xFFFF; // Limit[15:0]
        low |= (tss_addr & 0x00FF_FFFF) << 16; // Base[23:0]
        low |= 0x89 << 40; // Access: present, 64-bit TSS available
        low |= ((tss_limit >> 16) & 0xF) << 48; // Limit[19:16]
        low |= ((tss_addr >> 24) & 0xFF) << 56; // Base[31:24]
        (*gdt)[5] = low;

        let high: u64 = tss_addr >> 32; // Base[63:32]
        (*gdt)[6] = high;

        // ── Step 3: Load the GDT ─────────────────────────────────────
        //
        // `lgdt` takes a pointer to a 10-byte structure: 2-byte limit
        // (table size minus 1) followed by an 8-byte base address.
        let gdtr = GdtRegister {
            limit: (size_of::<[u64; 7]>() - 1) as u16, // 55
            base: &raw const GDT as u64,
        };

        core::arch::asm!("lgdt [{}]", in(reg) &gdtr, options(readonly, nostack, preserves_flags));

        // ── Step 4: Reload CS (code segment register) ────────────────
        //
        // After loading a new GDT, CS still points at the OLD GDT's code
        // segment entry. We need to reload it to reference our new one.
        //
        // Problem: you can't `mov cs, <value>` — the CPU forbids it.
        // CS can only change via instructions that simultaneously update
        // CS and RIP (instruction pointer): far jumps, far calls, far
        // returns, or interrupts.
        //
        // The trick: push the new CS selector and a return address onto
        // the stack, then execute `retfq` (far return). The CPU pops
        // RIP and CS together, atomically switching to our new segment.
        core::arch::asm!(
            "push {sel}",       // push new CS selector (0x08)
            "lea {tmp}, [rip + 2f]", // compute address of label '2'
            "push {tmp}",       // push return address
            "retfq",            // far return: pop RIP, pop CS
            "2:",               // we land here with CS = 0x08
            sel = in(reg) KERNEL_CODE_SELECTOR as u64,
            tmp = lateout(reg) _,
            options(preserves_flags),
        );

        // ── Step 5: Reload data segment registers ────────────────────
        //
        // DS, ES, and SS need to point to our kernel data segment.
        // Unlike CS, these CAN be loaded with a regular `mov`.
        // FS and GS are set to 0 (null) — we don't use them yet.
        // (Later, FS/GS can be used for thread-local storage or
        // per-CPU data.)
        core::arch::asm!(
            "mov ds, {sel:x}",
            "mov es, {sel:x}",
            "mov ss, {sel:x}",
            "mov fs, {zero:x}",
            "mov gs, {zero:x}",
            sel = in(reg) KERNEL_DATA_SELECTOR,
            zero = in(reg) 0u16,
            options(nostack, preserves_flags),
        );

        // ── Step 6: Load the TSS ─────────────────────────────────────
        //
        // `ltr` (Load Task Register) tells the CPU where the TSS is.
        // The operand is a GDT selector (0x28 = index 5), not an address.
        // After this, the CPU can read IST entries from our TSS when
        // handling interrupts.
        core::arch::asm!(
            "ltr {sel:x}",
            sel = in(reg) TSS_SELECTOR,
            options(nostack, preserves_flags),
        );
    }

    crate::println!("[ok] GDT loaded with TSS");
}

/// Set the Ring 0 stack pointer in the TSS.
///
/// When the CPU transitions from Ring 3 to Ring 0 (via interrupt or
/// exception — SYSCALL doesn't use the TSS), it loads RSP from TSS.RSP0.
/// This must point to a valid kernel stack before entering user mode.
///
/// # Safety
///
/// `rsp0` must point to the top of a valid, mapped kernel stack with
/// enough space for interrupt frames. The caller must ensure this stack
/// remains valid for the entire duration of Ring 3 execution.
pub unsafe fn set_rsp0(rsp0: u64) {
    unsafe {
        (*(&raw mut TSS)).rsp[0] = rsp0;
    }
}
