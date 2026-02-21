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
//! `0xFFFFFFFF80000000` and the HHDM at `0xFFFF800000000000`. Per-process
//! page tables share the kernel half (PML4 entries 256–511) via shallow
//! copy — the same physical PDPT/PD/PT frames are referenced from every
//! process's PML4, so kernel mappings are automatically visible everywhere.

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
// Kernel PML4 — saved at boot, used as the template for per-process tables
// ————————————————————————————————————————————————————————————————————————————

/// Physical address of the boot-time (kernel) PML4, saved by [`init()`].
///
/// Every per-process page table copies entries 256–511 from this PML4 so
/// the kernel half of the address space is shared across all address spaces.
static KERNEL_PML4: spin::Once<usize> = spin::Once::new();

/// Save the current CR3 as the kernel PML4 physical address.
///
/// Must be called once from `kmain()` after `Arch::init()` and before any
/// process is spawned. The saved address is used by [`create_user_page_table()`]
/// to shallow-copy the kernel half into new page tables.
pub fn init() {
    KERNEL_PML4.call_once(|| read_cr3() as usize);
    crate::println!("[ok] Kernel PML4 saved ({:#X})", kernel_pml4());
}

/// Return the physical address of the kernel (boot-time) PML4.
pub fn kernel_pml4() -> usize {
    *KERNEL_PML4.get().expect("paging::init() not called")
}

/// Allocate a fresh PML4 for a new process.
///
/// The lower half (entries 0–255, user space) starts empty. The upper half
/// (entries 256–511, kernel space) is shallow-copied from the kernel PML4 —
/// both page tables point to the same physical PDPT/PD/PT frames, so any
/// kernel mapping changes at levels below PML4 are immediately visible in
/// all address spaces.
pub fn create_user_page_table() -> usize {
    let frame = memory::alloc_frame().expect("out of memory allocating user PML4");
    let new_pml4 = table_at_phys(frame as u64);
    let kern_pml4 = table_at_phys(kernel_pml4() as u64);

    // Zero the user half.
    for i in 0..256 {
        new_pml4.entries[i] = PageTableEntry::empty();
    }

    // Shallow-copy the kernel half — same physical intermediate tables.
    for i in 256..512 {
        new_pml4.entries[i] = kern_pml4.entries[i];
    }

    frame
}

/// Load a new PML4 into CR3, switching address spaces.
///
/// Writing CR3 implicitly flushes the entire TLB — no manual `invlpg` needed.
///
/// # Safety
///
/// `pml4_phys` must be the physical address of a valid, 4 KiB-aligned PML4
/// whose kernel half (entries 256–511) maps the currently executing code.
pub fn switch_page_table(pml4_phys: usize) {
    unsafe {
        core::arch::asm!("mov cr3, {}", in(reg) pml4_phys as u64, options(nostack, preserves_flags));
    }
}

/// Free all intermediate page table frames in the user half of a PML4.
///
/// Walks entries 0–255 and recursively frees PT → PD → PDPT frames
/// bottom-up, then frees the PML4 frame itself. Does NOT touch entries
/// 256–511 (kernel half — those point to shared intermediate tables).
///
/// All leaf PTEs in the user half should already have been cleared by
/// `unmap_page()` calls (from `process::destroy()`) before calling this.
/// This function only frees the *intermediate* page table frames, not the
/// data frames the leaf PTEs pointed to.
pub fn destroy_user_page_table(pml4_phys: usize) {
    let pml4 = table_at_phys(pml4_phys as u64);

    // Walk only the user half (entries 0–255).
    for i in 0..256 {
        let pml4e = pml4.entries[i];
        if !pml4e.is_present() {
            continue;
        }

        let pdpt = table_at_phys(pml4e.physical_addr());
        for j in 0..512 {
            let pdpte = pdpt.entries[j];
            if !pdpte.is_present() || pdpte.is_huge() {
                continue;
            }

            let pd = table_at_phys(pdpte.physical_addr());
            for k in 0..512 {
                let pde = pd.entries[k];
                if !pde.is_present() || pde.is_huge() {
                    continue;
                }

                // Free the PT frame (level 1).
                memory::free_frame(pde.physical_addr() as usize);
            }

            // Free the PD frame (level 2).
            memory::free_frame(pdpte.physical_addr() as usize);
        }

        // Free the PDPT frame (level 3).
        memory::free_frame(pml4e.physical_addr() as usize);
    }

    // Free the PML4 frame itself.
    memory::free_frame(pml4_phys);
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

/// Walk the page table hierarchy starting from an explicit PML4, allocating
/// intermediate tables as needed. Returns the final PT (level-1 table)
/// where the leaf entry for this virtual address lives.
///
/// # Panics
///
/// Panics if a huge page (2 MiB or 1 GiB) is encountered — we don't
/// support splitting or mapping those yet.
fn walk_or_create_at(pml4_phys: u64, virt: usize) -> &'static mut PageTable {
    let pml4 = table_at_phys(pml4_phys);
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

/// Walk the page table hierarchy for `virt` using the current CR3.
fn walk_or_create(virt: usize) -> &'static mut PageTable {
    walk_or_create_at(read_cr3(), virt)
}

/// Walk the page table hierarchy starting from an explicit PML4 without
/// allocating. Returns `None` if any intermediate table is missing.
fn walk_existing_at(pml4_phys: u64, virt: usize) -> Option<&'static mut PageTable> {
    let pml4 = table_at_phys(pml4_phys);
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

/// Walk the page table hierarchy for `virt` using the current CR3, without
/// allocating. Returns `None` if any intermediate table is missing.
fn walk_existing(virt: usize) -> Option<&'static mut PageTable> {
    walk_existing_at(read_cr3(), virt)
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

    // Flush the TLB entry on all other cores.
    super::smp::tlb_shootdown(virt);
}

/// Check whether a virtual address has a present leaf mapping.
///
/// Walks the page table hierarchy without allocating. Returns `true` if all
/// four levels are present and the leaf PTE is marked present, `false`
/// otherwise. Used by [`grow_stack`](crate::process::grow_stack) as a safety
/// net before calling [`map_page`] (which panics on double-map).
pub fn is_page_mapped(virt: usize) -> bool {
    walk_existing(virt)
        .map(|pt| pt.entries[pt_index(virt)].is_present())
        .unwrap_or(false)
}

/// Map a 4 KiB virtual page in a specific address space (explicit PML4).
///
/// Like [`map_page`], but operates on the given PML4 instead of the current
/// CR3. Use this when mapping pages into a process's address space that may
/// not be the currently active one.
///
/// # Panics
///
/// - If `virt` or `phys` are not page-aligned.
/// - If the virtual address is already mapped in the target address space.
pub fn map_page_at(pml4_phys: usize, virt: usize, phys: usize, flags: crate::platform::PageFlags) {
    assert_eq!(virt % FRAME_SIZE, 0,
        "map_page_at: virtual address {:#X} is not page-aligned", virt);
    assert_eq!(phys % FRAME_SIZE, 0,
        "map_page_at: physical address {:#X} is not page-aligned", phys);

    let pt = walk_or_create_at(pml4_phys as u64, virt);
    let idx = pt_index(virt);
    let existing = pt.entries[idx];

    assert!(!existing.is_present(),
        "map_page_at: virtual address {:#X} is already mapped to {:#X}",
        virt, existing.physical_addr());

    let hw_flags = translate_flags(flags) | PageTableEntry::PRESENT;
    pt.entries[idx] = PageTableEntry::new(phys as u64, hw_flags);
}

/// Unmap a 4 KiB virtual page in a specific address space (explicit PML4).
///
/// Like [`unmap_page`], but operates on the given PML4. Does NOT flush
/// the TLB — the caller must handle TLB invalidation if this is the
/// current CR3 or if other cores may have cached the mapping.
pub fn unmap_page_at(pml4_phys: usize, virt: usize) {
    assert_eq!(virt % FRAME_SIZE, 0,
        "unmap_page_at: virtual address {:#X} is not page-aligned", virt);

    let pt = walk_existing_at(pml4_phys as u64, virt)
        .expect("unmap_page_at: no page table for address");
    let idx = pt_index(virt);

    assert!(pt.entries[idx].is_present(),
        "unmap_page_at: virtual address {:#X} is not mapped", virt);

    pt.entries[idx] = PageTableEntry::empty();
}

/// Check whether a virtual address has a present leaf mapping in a specific
/// address space (explicit PML4).
pub fn is_page_mapped_at(pml4_phys: usize, virt: usize) -> bool {
    walk_existing_at(pml4_phys as u64, virt)
        .map(|pt| pt.entries[pt_index(virt)].is_present())
        .unwrap_or(false)
}

/// Ensure a device MMIO physical address is mapped through the HHDM.
///
/// Limine's HHDM covers RAM regions from the memory map, but not necessarily
/// MMIO regions like the Local APIC (`0xFEE00000`) or I/O APIC (`0xFEC00000`).
/// This maps a single 4 KiB page at the HHDM virtual address corresponding
/// to `phys`, with cache-disabled flags appropriate for device registers.
///
/// Safe to call multiple times — skips if the page is already mapped.
pub fn ensure_mmio_mapped(phys: usize) {
    let virt = phys_to_virt(phys as u64) as usize;

    // Check if Limine already mapped this through the HHDM.
    if let Some(pt) = walk_existing(virt) {
        if pt.entries[pt_index(virt)].is_present() {
            return;
        }
    }

    // Map with PRESENT | WRITABLE | PWT (bit 3) | PCD (bit 4).
    // PCD (Page Cache Disable) and PWT (Page Write-Through) are essential
    // for MMIO — without them the CPU may cache device register reads and
    // coalesce/reorder writes, producing stale data or lost updates.
    let flags = PageTableEntry::PRESENT | PageTableEntry::WRITABLE | (1 << 3) | (1 << 4);
    let pt = walk_or_create(virt);
    pt.entries[pt_index(virt)] = PageTableEntry::new(phys as u64, flags);
}
