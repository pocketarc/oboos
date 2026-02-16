//! Page table management for x86_64.
//!
//! x86_64 uses a 4-level page table hierarchy to translate 48-bit virtual
//! addresses to physical addresses. Each level is a 4 KiB table of 512
//! entries (9 bits of the virtual address per level):
//!
//! ```text
//! Virtual address bits:  [63..48 sign-ext] [47..39 PML4] [38..30 PDPT] [29..21 PD] [20..12 PT] [11..0 offset]
//! ```
//!
//! The CPU walks PML4 → PDPT → PD → PT, ANDing permission bits at each
//! level. This means intermediate entries must be *at least* as permissive
//! as the leaf PTE — a non-writable PD entry blocks writes even if the PT
//! entry allows them.
//!
//! Limine has already set up page tables with the kernel mapped at
//! `0xFFFFFFFF80000000` and the HHDM at `0xFFFF800000000000`. We modify
//! these tables in place — no new hierarchy from scratch until Phase 2
//! per-process address spaces.

use super::memory::phys_to_virt;
use crate::memory::{self, FRAME_SIZE};

// ————————————————————————————————————————————————————————————————————————————
// Page table entry
// ————————————————————————————————————————————————————————————————————————————

/// A single 64-bit page table entry, used at every level of the hierarchy.
///
/// The hardware defines a fixed bit layout — we use a newtype to keep raw
/// bit manipulation contained and provide meaningful accessors.
#[derive(Clone, Copy)]
#[repr(transparent)]
struct PageTableEntry(u64);

impl PageTableEntry {
    // Hardware-defined bits. These are the same at every level.
    const PRESENT: u64 = 1 << 0;
    const WRITABLE: u64 = 1 << 1;
    const USER: u64 = 1 << 2;
    /// Bit 7 at the PD level (or bit 7 at PDPT for 1 GiB pages) marks a
    /// "huge page" — the translation stops here instead of descending to
    /// the next level. We don't create huge pages but need to detect them
    /// during walks so we don't misinterpret a huge-page entry as a table
    /// pointer.
    const HUGE_PAGE: u64 = 1 << 7;
    const NO_EXECUTE: u64 = 1 << 63;
    /// Bits 51:12 hold the physical address of the next-level table (or
    /// the final frame). The low 12 bits are flags, and bits 63:52 are
    /// reserved/NX. This mask extracts just the 40-bit physical address,
    /// shifted into its natural position.
    const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

    /// An empty (not-present) entry. The CPU ignores all other bits when
    /// bit 0 is clear.
    const fn empty() -> Self {
        Self(0)
    }

    /// Create a new entry pointing to `phys_addr` with the given flags.
    ///
    /// `phys_addr` must be 4 KiB-aligned (low 12 bits zero) — the hardware
    /// stores flags in those bits.
    const fn new(phys_addr: u64, flags: u64) -> Self {
        Self((phys_addr & Self::ADDR_MASK) | flags)
    }

    fn is_present(self) -> bool {
        self.0 & Self::PRESENT != 0
    }

    fn is_huge(self) -> bool {
        self.0 & Self::HUGE_PAGE != 0
    }

    /// Extract the physical address this entry points to (next-level table
    /// or final frame).
    fn physical_addr(self) -> u64 {
        self.0 & Self::ADDR_MASK
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Page table
// ————————————————————————————————————————————————————————————————————————————

/// A page table — 512 entries, 4 KiB total, naturally aligned.
///
/// The same struct represents all four levels (PML4, PDPT, PD, PT). The
/// hardware doesn't distinguish them structurally; the *meaning* of each
/// entry depends on which level you're at.
#[repr(C, align(4096))]
struct PageTable {
    entries: [PageTableEntry; 512],
}

// ————————————————————————————————————————————————————————————————————————————
// Virtual address index extraction
// ————————————————————————————————————————————————————————————————————————————
//
// Each of these pulls out a 9-bit index (0–511) from the virtual address,
// selecting which entry to follow at that level of the page table walk.

fn pml4_index(virt: usize) -> usize {
    (virt >> 39) & 0x1FF
}

fn pdpt_index(virt: usize) -> usize {
    (virt >> 30) & 0x1FF
}

fn pd_index(virt: usize) -> usize {
    (virt >> 21) & 0x1FF
}

fn pt_index(virt: usize) -> usize {
    (virt >> 12) & 0x1FF
}

// ————————————————————————————————————————————————————————————————————————————
// Hardware operations
// ————————————————————————————————————————————————————————————————————————————

/// Read the CR3 register, which holds the physical address of the PML4 table.
///
/// The low 12 bits of CR3 contain flags (PCID, write-through, cache-disable)
/// that we mask off — we only want the table address.
fn read_cr3() -> u64 {
    let cr3: u64;
    unsafe {
        core::arch::asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack));
    }
    cr3 & PageTableEntry::ADDR_MASK
}

/// Invalidate the TLB entry for a single virtual address.
///
/// The CPU caches virtual-to-physical translations in the TLB. After we
/// remove or change a mapping, we must flush the stale entry or the CPU
/// will keep using the old translation. `invlpg` is the surgical version —
/// it only flushes one page, unlike reloading CR3 which flushes everything.
fn invlpg(virt: usize) {
    unsafe {
        core::arch::asm!("invlpg [{}]", in(reg) virt, options(nostack, preserves_flags));
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Page table traversal helpers
// ————————————————————————————————————————————————————————————————————————————

/// Get a mutable reference to the page table at the given physical address.
///
/// Uses the HHDM to convert the physical address to a virtual pointer.
/// Returns a `'static` reference because page tables live for the lifetime
/// of the kernel.
fn table_at_phys(phys_addr: u64) -> &'static mut PageTable {
    let virt = phys_to_virt(phys_addr) as *mut PageTable;
    unsafe { &mut *virt }
}

/// Follow an existing page table entry to the next level, or allocate a
/// new empty table if the entry is not present.
///
/// When allocating, the new entry is installed with permissive flags
/// (PRESENT | WRITABLE | USER). This is deliberate: x86_64 ANDs permission
/// bits across all four levels, so a restrictive intermediate entry would
/// silently deny access that the leaf PTE permits. The real permissions
/// live in the leaf (PT) entry.
fn get_or_create_next_table(entry: &mut PageTableEntry) -> &'static mut PageTable {
    if entry.is_present() {
        return table_at_phys(entry.physical_addr());
    }

    // Allocate a fresh frame for the new table.
    let frame_phys = memory::alloc_frame().expect("out of memory allocating page table");
    let table = table_at_phys(frame_phys as u64);

    // Zero the new table — all entries start as not-present.
    table.entries = [PageTableEntry::empty(); 512];

    // Install the new table with permissive intermediate flags.
    *entry = PageTableEntry::new(
        frame_phys as u64,
        PageTableEntry::PRESENT | PageTableEntry::WRITABLE | PageTableEntry::USER,
    );

    table
}

/// Walk the page table hierarchy for `virt`, allocating intermediate tables
/// as needed. Returns the final PT (level-1 table) where the leaf entry
/// for this virtual address lives.
///
/// # Panics
///
/// Panics if a huge page (2 MiB or 1 GiB) is encountered — we don't
/// support splitting or mapping those yet.
fn walk_or_create(virt: usize) -> &'static mut PageTable {
    let pml4 = table_at_phys(read_cr3());
    let pdpt_entry = &mut pml4.entries[pml4_index(virt)];
    let pdpt = get_or_create_next_table(pdpt_entry);

    let pd_entry = &mut pdpt.entries[pdpt_index(virt)];
    assert!(
        !pd_entry.is_present() || !pd_entry.is_huge(),
        "walk_or_create: encountered 1 GiB huge page at {:#X}",
        virt
    );
    let pd = get_or_create_next_table(pd_entry);

    let pt_entry = &mut pd.entries[pd_index(virt)];
    assert!(
        !pt_entry.is_present() || !pt_entry.is_huge(),
        "walk_or_create: encountered 2 MiB huge page at {:#X}",
        virt
    );
    get_or_create_next_table(pt_entry)
}

/// Walk the page table hierarchy for `virt` without allocating. Returns
/// `None` if any intermediate table is missing (the address has never been
/// mapped at any level).
///
/// Used by `unmap_page` — we don't want to create empty tables just to
/// discover there's nothing to unmap.
fn walk_existing(virt: usize) -> Option<&'static mut PageTable> {
    let pml4 = table_at_phys(read_cr3());
    let pdpt_entry = pml4.entries[pml4_index(virt)];
    if !pdpt_entry.is_present() {
        return None;
    }

    let pdpt = table_at_phys(pdpt_entry.physical_addr());
    let pd_entry = pdpt.entries[pdpt_index(virt)];
    if !pd_entry.is_present() {
        return None;
    }

    let pd = table_at_phys(pd_entry.physical_addr());
    let pt_entry = pd.entries[pd_index(virt)];
    if !pt_entry.is_present() {
        return None;
    }

    Some(table_at_phys(pt_entry.physical_addr()))
}

// ————————————————————————————————————————————————————————————————————————————
// Public API
// ————————————————————————————————————————————————————————————————————————————

/// Convert [`PageFlags`] to the hardware bit layout.
///
/// The HAL flags happen to match the x86_64 PTE bit positions (by design),
/// so this is just an extraction of the inner `u64`. If we ever add an
/// architecture where they diverge, the conversion goes here.
fn translate_flags(flags: crate::platform::PageFlags) -> u64 {
    flags.0
}

/// Map a 4 KiB virtual page to a physical frame.
///
/// Walks the page table hierarchy, allocating intermediate tables as needed,
/// and installs a leaf PTE. No TLB flush — new mappings can't have stale
/// TLB entries.
///
/// # Panics
///
/// - If `virt` or `phys` are not page-aligned.
/// - If the virtual address is already mapped (catches accidental overwrites).
pub fn map_page(virt: usize, phys: usize, flags: crate::platform::PageFlags) {
    assert_eq!(
        virt % FRAME_SIZE,
        0,
        "map_page: virtual address {:#X} is not page-aligned",
        virt
    );
    assert_eq!(
        phys % FRAME_SIZE,
        0,
        "map_page: physical address {:#X} is not page-aligned",
        phys
    );

    let pt = walk_or_create(virt);
    let idx = pt_index(virt);
    let existing = pt.entries[idx];

    assert!(
        !existing.is_present(),
        "map_page: virtual address {:#X} is already mapped to {:#X}",
        virt,
        existing.physical_addr()
    );

    let hw_flags = translate_flags(flags) | PageTableEntry::PRESENT;
    pt.entries[idx] = PageTableEntry::new(phys as u64, hw_flags);
}

/// Unmap a 4 KiB virtual page and flush the TLB entry.
///
/// Does NOT free the physical frame — the caller is responsible for that.
/// The frame might be MMIO-mapped or shared between multiple virtual addresses.
///
/// # Panics
///
/// - If `virt` is not page-aligned.
/// - If the virtual address is not currently mapped.
pub fn unmap_page(virt: usize) {
    assert_eq!(
        virt % FRAME_SIZE,
        0,
        "unmap_page: virtual address {:#X} is not page-aligned",
        virt
    );

    let pt = walk_existing(virt).expect("unmap_page: no page table for address");
    let idx = pt_index(virt);

    assert!(
        pt.entries[idx].is_present(),
        "unmap_page: virtual address {:#X} is not mapped",
        virt
    );

    pt.entries[idx] = PageTableEntry::empty();
    invlpg(virt);
}
