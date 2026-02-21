//! Model-Specific Register (MSR) access helpers.
//!
//! MSRs are per-CPU configuration registers addressed by a 32-bit index.
//! They control features like SYSCALL/SYSRET, APIC base, PAT, and many
//! more. The `rdmsr`/`wrmsr` instructions read and write them, with the
//! 64-bit value split across EDX:EAX — a convention from the 32-bit era
//! that persists in 64-bit mode.
//!
//! These helpers are used by [`lapic`](super::lapic) (to read the APIC
//! base address) and [`syscall`](super::syscall) (to configure the
//! SYSCALL MSRs).

/// Read a Model-Specific Register.
///
/// # Safety
///
/// The caller must ensure `msr` is a valid MSR index for the current CPU.
/// Reading a non-existent MSR triggers a #GP fault.
pub(super) unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") msr,
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack, preserves_flags),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// Write a Model-Specific Register.
///
/// # Safety
///
/// The caller must ensure `msr` is a valid MSR index and `value` is
/// appropriate for that register. Writing an invalid value can misconfigure
/// the CPU or trigger a #GP fault.
pub(super) unsafe fn wrmsr(msr: u32, value: u64) {
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
