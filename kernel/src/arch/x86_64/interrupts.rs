//! Interrupt Descriptor Table (IDT) and exception handlers.
//!
//! The IDT is a 256-entry table that tells the CPU what function to call when
//! an interrupt or exception occurs. Each entry is a 16-byte "gate descriptor"
//! containing a handler address, a code segment selector, and flags.
//!
//! ## How interrupts work on x86_64
//!
//! When the CPU encounters an exception (e.g., divide-by-zero) or receives a
//! hardware interrupt (e.g., keyboard), it:
//!
//! 1. Looks up the vector number in the IDT (e.g., vector 0 = divide-by-zero)
//! 2. If the entry specifies an IST index, switches to that stack from the TSS
//! 3. Pushes an **interrupt stack frame** onto the (possibly new) stack:
//!    SS, RSP, RFLAGS, CS, RIP — and for some exceptions, an error code
//! 4. Loads CS and RIP from the IDT entry (jumping to our handler)
//! 5. If the gate is an interrupt gate (not trap gate), clears IF to disable
//!    further maskable interrupts
//!
//! The handler runs, and returns via `iretq` which pops the stack frame and
//! restores the previous execution state.
//!
//! ## Interrupt gate vs trap gate
//!
//! Both are identical except for one thing: an **interrupt gate** automatically
//! clears the IF (Interrupt Flag) when the handler is entered, preventing
//! nested interrupts. A **trap gate** leaves IF unchanged. We use interrupt
//! gates for everything — if we ever need nested interrupts, we'll re-enable
//! them explicitly inside the handler.
//!
//! ## Error codes
//!
//! Some exceptions push an error code onto the stack between the interrupt
//! frame and the handler. This changes the handler's signature. The
//! `extern "x86-interrupt"` calling convention handles this automatically —
//! handlers with an error code parameter get the right stack adjustment.
//!
//! Exceptions that push error codes: Double Fault (8, always 0),
//! Invalid TSS (10), Segment Not Present (11), Stack Segment Fault (12),
//! General Protection Fault (13), Page Fault (14), Alignment Check (17).

use super::{gdt, pic};
use core::mem::size_of;

// ————————————————————————————————————————————————————————————————————————————
// IDT entry (gate descriptor)
// ————————————————————————————————————————————————————————————————————————————

/// A single IDT gate descriptor (16 bytes).
///
/// The handler address is split across three fields — a legacy of the 32-bit
/// format that was extended for 64-bit mode. The CPU reassembles them into a
/// single 64-bit address when looking up a handler.
///
/// ```text
/// Offset  Size  Field
/// 0       2     handler_low      — address bits 0–15
/// 2       2     selector         — GDT code segment selector (0x08)
/// 4       1     ist              — bits 0–2: IST index (0 = none, 1–7 = use IST stack)
///                                  bits 3–7: reserved (zero)
/// 5       1     type_attr        — Present | DPL | gate type
/// 6       2     handler_mid      — address bits 16–31
/// 8       4     handler_high     — address bits 32–63
/// 12      4     _reserved        — must be zero
/// ```
#[derive(Clone, Copy)]
#[repr(C, packed)]
struct IdtEntry {
    handler_low: u16,
    selector: u16,
    ist: u8,
    type_attr: u8,
    handler_mid: u16,
    handler_high: u32,
    _reserved: u32,
}

impl IdtEntry {
    /// A blank (not-present) IDT entry. The CPU ignores entries where the
    /// Present bit is clear — triggering that vector will cause a fault
    /// (or double-fault if the missing entry *is* the fault handler).
    const MISSING: Self = Self {
        handler_low: 0,
        selector: 0,
        ist: 0,
        type_attr: 0,
        handler_mid: 0,
        handler_high: 0,
        _reserved: 0,
    };

    /// Build a present interrupt gate entry.
    ///
    /// - `handler`: the function address (will be split across three fields)
    /// - `ist`: IST index (0 for normal stack, 1 for double-fault stack)
    ///
    /// The type_attr byte is 0x8E:
    ///   bit 7     = 1 (Present)
    ///   bits 5–6  = 00 (DPL 0 — only kernel can trigger via `int` instruction)
    ///   bit 4     = 0 (system segment)
    ///   bits 0–3  = 0xE (interrupt gate — clears IF on entry)
    fn new(handler: u64, ist: u8) -> Self {
        Self {
            handler_low: handler as u16,
            selector: gdt::KERNEL_CODE_SELECTOR,
            ist,
            type_attr: 0x8E, // present, ring 0, interrupt gate
            handler_mid: (handler >> 16) as u16,
            handler_high: (handler >> 32) as u32,
            _reserved: 0,
        }
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Interrupt stack frame
// ————————————————————————————————————————————————————————————————————————————

/// The state the CPU pushes onto the stack before calling an interrupt handler.
///
/// When an interrupt or exception fires, the CPU saves the interrupted code's
/// context so it can be resumed later (via `iretq`). The fields are pushed in
/// reverse order (SS first, then RSP, etc.) so that RIP ends up at the lowest
/// address — matching the struct layout with `#[repr(C)]`.
///
/// For exceptions that push an error code (double fault, GPF, page fault, etc.),
/// the error code sits between this frame and the handler — the
/// `extern "x86-interrupt"` calling convention handles that automatically.
#[repr(C)]
pub struct InterruptStackFrame {
    /// Instruction pointer — where the fault occurred (or where to resume).
    pub rip: u64,
    /// Code segment selector at the time of the interrupt.
    pub cs: u64,
    /// CPU flags (carry, zero, interrupt enable, etc.) at the time of interrupt.
    pub rflags: u64,
    /// Stack pointer at the time of the interrupt.
    pub rsp: u64,
    /// Stack segment selector at the time of the interrupt.
    pub ss: u64,
}

// ————————————————————————————————————————————————————————————————————————————
// Static storage
// ————————————————————————————————————————————————————————————————————————————

/// The IDT — 256 entries, one per possible interrupt vector.
///
/// Vectors 0–31 are reserved by the CPU for exceptions (divide-by-zero,
/// page fault, etc.). Vectors 32–255 are available for hardware interrupts
/// and software-defined interrupts. We'll fill in hardware IRQ handlers
/// in Phase 1c when we set up the PIC.
static mut IDT: [IdtEntry; 256] = [IdtEntry::MISSING; 256];

/// The IDT Register (IDTR) — same pattern as the GDTR.
/// Loaded with `lidt`. Tells the CPU where the IDT is and how big it is.
#[repr(C, packed)]
struct IdtRegister {
    /// Byte size of the IDT minus 1.
    limit: u16,
    /// Virtual address of the first entry.
    base: u64,
}

// ————————————————————————————————————————————————————————————————————————————
// IRQ handler table
// ————————————————————————————————————————————————————————————————————————————

/// Registered handlers for hardware IRQs 0–15.
///
/// Each slot holds an optional function pointer. When a hardware interrupt
/// fires, the corresponding IRQ stub below checks this table and calls the
/// handler if one is registered. `None` means nobody has claimed that IRQ
/// yet — we just acknowledge it and move on.
///
/// This is `static mut` because it's written during driver initialization
/// (single-threaded, interrupts disabled) and read from interrupt handlers
/// (which can't race with each other since our interrupt gates disable IF).
static mut IRQ_HANDLERS: [Option<fn()>; 16] = [None; 16];

/// Register a handler for a hardware IRQ and enable that IRQ line.
///
/// After this call, the PIC will deliver interrupts for the given IRQ,
/// and our IDT stub will call `handler` each time it fires.
///
/// # Panics
///
/// Panics if `irq` is not in 0–15.
pub fn set_irq_handler(irq: u8, handler: fn()) {
    assert!(irq < 16, "IRQ number must be 0-15, got {}", irq);
    unsafe {
        IRQ_HANDLERS[irq as usize] = Some(handler);
    }
    pic::unmask(irq);
}

/// Generate 16 IRQ handler stubs — one per hardware interrupt line.
///
/// Each stub is an `extern "x86-interrupt"` function that:
/// 1. Sends EOI to the PIC before calling the handler
/// 2. Looks up the registered handler in [`IRQ_HANDLERS`]
/// 3. Calls it if present (otherwise the IRQ is silently ignored)
///
/// EOI is sent *before* the handler because a handler may context-switch
/// away (e.g., the PIT tick triggers preemptive scheduling). If we sent
/// EOI after the handler, a context switch would delay it until the task
/// cycles back — blocking all further interrupts on that IRQ line. This
/// is safe because IF=0 (interrupt gate) and PIT/keyboard are
/// edge-triggered.
///
/// We use a macro because the only difference between the 16 handlers is
/// the IRQ number, and writing them by hand would be tedious and error-prone.
macro_rules! irq_handler {
    ($name:ident, $irq:expr) => {
        extern "x86-interrupt" fn $name(_frame: InterruptStackFrame) {
            pic::acknowledge($irq);
            unsafe {
                if let Some(handler) = IRQ_HANDLERS[$irq] {
                    handler();
                }
            }
        }
    };
}

irq_handler!(irq_handler_0,  0);
irq_handler!(irq_handler_1,  1);
irq_handler!(irq_handler_2,  2);
irq_handler!(irq_handler_3,  3);
irq_handler!(irq_handler_4,  4);
irq_handler!(irq_handler_5,  5);
irq_handler!(irq_handler_6,  6);
irq_handler!(irq_handler_7,  7);
irq_handler!(irq_handler_8,  8);
irq_handler!(irq_handler_9,  9);
irq_handler!(irq_handler_10, 10);
irq_handler!(irq_handler_11, 11);
irq_handler!(irq_handler_12, 12);
irq_handler!(irq_handler_13, 13);
irq_handler!(irq_handler_14, 14);
irq_handler!(irq_handler_15, 15);

/// All 16 IRQ handler function pointers, indexed by IRQ number.
/// Used during IDT setup to register vectors 32–47.
const IRQ_STUBS: [extern "x86-interrupt" fn(InterruptStackFrame); 16] = [
    irq_handler_0,  irq_handler_1,  irq_handler_2,  irq_handler_3,
    irq_handler_4,  irq_handler_5,  irq_handler_6,  irq_handler_7,
    irq_handler_8,  irq_handler_9,  irq_handler_10, irq_handler_11,
    irq_handler_12, irq_handler_13, irq_handler_14, irq_handler_15,
];

// ————————————————————————————————————————————————————————————————————————————
// Exception handlers
// ————————————————————————————————————————————————————————————————————————————
//
// These use the `extern "x86-interrupt"` calling convention, which tells the
// Rust compiler:
// - Save ALL registers on entry (not just the callee-saved ones)
// - Return with `iretq` instead of `ret`
// - For handlers with an error_code parameter, adjust the stack to account
//   for the extra value the CPU pushed
//
// This is a nightly feature (`#![feature(abi_x86_interrupt)]`) but it's been
// stable in practice for years. The alternative is writing naked assembly
// stubs that save registers manually and call Rust functions — doable but
// more error-prone for no real benefit at this stage.

/// Helper to print the interrupt stack frame in a readable format.
fn print_frame(frame: &InterruptStackFrame) {
    crate::println!("  RIP:    {:#018X}", frame.rip);
    crate::println!("  CS:     {:#06X}", frame.cs);
    crate::println!("  RFLAGS: {:#018X}", frame.rflags);
    crate::println!("  RSP:    {:#018X}", frame.rsp);
    crate::println!("  SS:     {:#06X}", frame.ss);
}

/// Vector 0: Divide by Zero (#DE).
///
/// Triggered by `div` or `idiv` when the divisor is zero, or when the
/// quotient is too large for the destination register. This is what our
/// `trigger_test_fault()` fires.
///
/// No error code. RIP points to the faulting instruction.
extern "x86-interrupt" fn divide_by_zero_handler(frame: InterruptStackFrame) {
    crate::println!();
    crate::println!("EXCEPTION: Divide by Zero (#DE)");
    print_frame(&frame);
    panic!("divide by zero");
}

/// Vector 3: Breakpoint (#BP).
///
/// Triggered by the `int3` instruction (opcode 0xCC). Unlike other
/// exceptions, this is NOT fatal — debuggers use it to set breakpoints.
/// RIP points to the instruction *after* `int3`, so we can just return.
///
/// No error code.
extern "x86-interrupt" fn breakpoint_handler(frame: InterruptStackFrame) {
    crate::println!();
    crate::println!("EXCEPTION: Breakpoint (#BP)");
    print_frame(&frame);
    crate::println!("  (resuming execution)");
}

/// Vector 6: Invalid Opcode (#UD).
///
/// The CPU encountered an instruction it doesn't recognize. This usually
/// means a jump went to the wrong place (executing data as code), or
/// the code uses an instruction not supported by this CPU.
///
/// No error code. RIP points to the faulting instruction.
extern "x86-interrupt" fn invalid_opcode_handler(frame: InterruptStackFrame) {
    crate::println!();
    crate::println!("EXCEPTION: Invalid Opcode (#UD)");
    print_frame(&frame);
    panic!("invalid opcode");
}

/// Vector 8: Double Fault (#DF).
///
/// A double fault occurs when an exception fires while the CPU is already
/// trying to handle a previous exception. Common cause: a page fault
/// during a page fault handler (e.g., the handler's stack overflowed).
///
/// This handler uses IST1 — a dedicated stack from the TSS — so it works
/// even if the kernel stack is completely trashed. Without the IST, a
/// double fault on a broken stack would triple-fault (CPU reset).
///
/// The error code is always 0. A double fault is always fatal.
extern "x86-interrupt" fn double_fault_handler(
    frame: InterruptStackFrame,
    error_code: u64,
) {
    crate::println!();
    crate::println!("EXCEPTION: Double Fault (#DF)");
    crate::println!("  Error code: {}", error_code);
    print_frame(&frame);
    panic!("double fault — this is unrecoverable");
}

/// Vector 13: General Protection Fault (#GP).
///
/// The kitchen-sink exception — fires for a wide range of protection
/// violations: loading an invalid segment selector, writing to a
/// read-only segment, exceeding segment limits, accessing system-reserved
/// I/O ports from user mode, and many more.
///
/// The error code is the segment selector that caused the fault, or 0
/// if it wasn't a segment-related error.
extern "x86-interrupt" fn general_protection_fault_handler(
    frame: InterruptStackFrame,
    error_code: u64,
) {
    crate::println!();
    crate::println!("EXCEPTION: General Protection Fault (#GP)");
    crate::println!("  Error code: {:#06X}", error_code);
    print_frame(&frame);
    panic!("general protection fault");
}

/// Vector 14: Page Fault (#PF).
///
/// The CPU tried to access a virtual address that isn't mapped, or
/// violated the page's permission flags (e.g., writing to a read-only
/// page, executing a non-executable page, user-mode accessing a
/// kernel page).
///
/// The faulting virtual address is in the CR2 register. The error code
/// tells us what kind of access failed:
///   bit 0: 0 = page not present, 1 = protection violation
///   bit 1: 0 = read access, 1 = write access
///   bit 2: 0 = supervisor mode, 1 = user mode
///   bit 4: 0 = not instruction fetch, 1 = instruction fetch (NX violation)
extern "x86-interrupt" fn page_fault_handler(
    frame: InterruptStackFrame,
    error_code: u64,
) {
    // CR2 holds the virtual address that caused the page fault.
    // It's set by the CPU before calling the handler.
    let faulting_address: u64;
    unsafe {
        core::arch::asm!("mov {}, cr2", out(reg) faulting_address, options(nomem, nostack));
    }

    crate::println!();
    crate::println!("EXCEPTION: Page Fault (#PF)");
    crate::println!("  Faulting address: {:#018X}", faulting_address);
    crate::println!("  Error code: {:#06X} ({}{}{}{})",
        error_code,
        if error_code & 1 != 0 { "protection violation" } else { "page not present" },
        if error_code & 2 != 0 { ", write" } else { ", read" },
        if error_code & 4 != 0 { ", user mode" } else { "" },
        if error_code & 16 != 0 { ", instruction fetch (NX)" } else { "" },
    );
    print_frame(&frame);
    panic!("page fault");
}

// ————————————————————————————————————————————————————————————————————————————
// Initialization
// ————————————————————————————————————————————————————————————————————————————

/// Set up the IDT with exception handlers and load it.
///
/// After this function returns, CPU exceptions produce readable panic
/// messages instead of silent triple-faults. Must be called after
/// `gdt::init()` (the double fault handler needs IST1 from the TSS).
pub fn init() {
    unsafe {
        let idt = &raw mut IDT;

        // Vector 0: Divide by Zero — no error code, no IST
        (*idt)[0] = IdtEntry::new(divide_by_zero_handler as *const () as u64, 0);

        // Vector 3: Breakpoint — no error code, no IST
        (*idt)[3] = IdtEntry::new(breakpoint_handler as *const () as u64, 0);

        // Vector 6: Invalid Opcode — no error code, no IST
        (*idt)[6] = IdtEntry::new(invalid_opcode_handler as *const () as u64, 0);

        // Vector 8: Double Fault — has error code, uses IST1
        // IST1 = the dedicated double-fault stack we set up in gdt.rs.
        // This is the whole reason we built the TSS in Phase 1a.
        (*idt)[8] = IdtEntry::new(double_fault_handler as *const () as u64, 1);

        // Vector 13: General Protection Fault — has error code, no IST
        (*idt)[13] = IdtEntry::new(general_protection_fault_handler as *const () as u64, 0);

        // Vector 14: Page Fault — has error code, no IST
        (*idt)[14] = IdtEntry::new(page_fault_handler as *const () as u64, 0);

        // Vectors 32–47: Hardware IRQ handlers (one per PIC line).
        // These are registered now but won't fire until the PIC is
        // initialized and individual IRQs are unmasked by device drivers.
        for i in 0..16u8 {
            let vector = (pic::IRQ_BASE + i) as usize;
            (*idt)[vector] = IdtEntry::new(IRQ_STUBS[i as usize] as *const () as u64, 0);
        }

        // Load the IDT register — same pattern as lgdt in gdt.rs.
        let idtr = IdtRegister {
            limit: (size_of::<[IdtEntry; 256]>() - 1) as u16,
            base: &raw const IDT as u64,
        };

        core::arch::asm!("lidt [{}]", in(reg) &idtr, options(readonly, nostack, preserves_flags));
    }

    crate::println!("[ok] IDT loaded (6 exception handlers, 16 IRQ vectors)");

    // Now that the IDT has entries for vectors 32–47, remap the PIC so
    // hardware IRQs actually route to those vectors instead of colliding
    // with CPU exceptions.
    pic::init();
}
