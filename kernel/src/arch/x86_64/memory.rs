//! Physical memory discovery via the Limine boot protocol.
//!
//! The Limine bootloader sets up a Higher Half Direct Map (HHDM) — a linear
//! mapping of all physical RAM starting at a high virtual address (typically
//! `0xFFFF800000000000`). This means we can access any physical address by
//! simply adding the HHDM offset: `virt = phys + hhdm_offset`.
//!
//! This module reads two Limine responses at boot:
//! 1. **HHDM** — tells us the offset so we can build `phys_to_virt()`
//! 2. **Memory map** — tells us which physical regions are usable RAM vs.
//!    reserved (MMIO, ACPI, framebuffer, etc.)
//!
//! It converts the Limine-specific types into our arch-independent
//! [`MemoryRegion`](crate::memory::MemoryRegion) format and hands them to the
//! frame allocator in [`crate::memory`].

use core::sync::atomic::{AtomicU64, Ordering};

use limine::memory_map::EntryType;
use limine::request::{HhdmRequest, MemoryMapRequest};

use crate::memory::{MemoryRegion, MemoryRegionKind};

// Limine requests — the bootloader scans our binary for these magic structs
// and fills in response pointers before jumping to kmain.

#[used]
#[unsafe(link_section = ".requests")]
static MEMORY_MAP: MemoryMapRequest = MemoryMapRequest::new();

#[used]
#[unsafe(link_section = ".requests")]
static HHDM: HhdmRequest = HhdmRequest::new();

/// HHDM offset, set once during init. All physical memory is mapped at
/// `virtual = physical + HHDM_OFFSET` by the bootloader's page tables.
///
/// Atomic because `phys_to_virt` is called from all cores after SMP bringup.
/// Written once during single-threaded init, then read-only.
static HHDM_OFFSET: AtomicU64 = AtomicU64::new(0);

/// Convert a physical address to a virtual pointer via the HHDM.
///
/// This works as long as the bootloader's page tables are active (which they
/// are until we build our own). The HHDM maps all of physical RAM linearly,
/// so this is just pointer arithmetic.
///
/// # Safety
///
/// The caller must ensure `phys` is a valid physical address within mapped
/// RAM. Passing an address beyond physical memory will produce a pointer
/// into unmapped virtual space, causing a page fault on dereference.
pub fn phys_to_virt(phys: u64) -> *mut u8 {
    (phys + HHDM_OFFSET.load(Ordering::Relaxed)) as *mut u8
}

/// Initialize physical memory management.
///
/// Reads the Limine HHDM and memory map responses, prints the physical memory
/// layout to serial, and initializes the frame allocator.
///
/// # Panics
///
/// Panics if the bootloader didn't provide HHDM or memory map responses
/// (shouldn't happen with a correctly built Limine binary).
pub fn init() {
    // --- HHDM offset ---
    let hhdm_response = HHDM
        .get_response()
        .expect("Limine did not provide HHDM response");
    let offset = hhdm_response.offset();

    HHDM_OFFSET.store(offset, Ordering::Relaxed);
    crate::println!("[mem] HHDM offset: {:#018X}", offset);

    // --- Memory map ---
    let mmap_response = MEMORY_MAP
        .get_response()
        .expect("Limine did not provide memory map response");
    let entries = mmap_response.entries();

    crate::println!("[mem] Physical memory map ({} entries):", entries.len());

    // Convert Limine entries to our arch-independent format.
    // 64 entries is generous — real machines typically report 10–20 regions.
    let mut regions = [MemoryRegion {
        base: 0,
        length: 0,
        kind: MemoryRegionKind::Reserved,
    }; 64];
    let mut region_count = 0;
    let mut total_usable: u64 = 0;

    for entry in entries {
        let kind = match entry.entry_type {
            EntryType::USABLE => MemoryRegionKind::Usable,
            EntryType::RESERVED => MemoryRegionKind::Reserved,
            EntryType::ACPI_RECLAIMABLE => MemoryRegionKind::AcpiReclaimable,
            EntryType::ACPI_NVS => MemoryRegionKind::AcpiNvs,
            EntryType::BAD_MEMORY => MemoryRegionKind::BadMemory,
            EntryType::BOOTLOADER_RECLAIMABLE => MemoryRegionKind::BootloaderReclaimable,
            EntryType::EXECUTABLE_AND_MODULES => MemoryRegionKind::KernelAndModules,
            EntryType::FRAMEBUFFER => MemoryRegionKind::Framebuffer,
            _ => MemoryRegionKind::Reserved,
        };

        // Pretty-print each entry. The format shows the range, human-readable
        // size, and type — really helpful when debugging memory layout issues.
        let end = entry.base + entry.length;
        let size_kib = entry.length / 1024;
        let kind_str = match kind {
            MemoryRegionKind::Usable => "Usable",
            MemoryRegionKind::Reserved => "Reserved",
            MemoryRegionKind::AcpiReclaimable => "ACPI Reclaimable",
            MemoryRegionKind::AcpiNvs => "ACPI NVS",
            MemoryRegionKind::BadMemory => "Bad Memory",
            MemoryRegionKind::BootloaderReclaimable => "Bootloader Reclaimable",
            MemoryRegionKind::KernelAndModules => "Kernel & Modules",
            MemoryRegionKind::Framebuffer => "Framebuffer",
        };

        if size_kib >= 1024 {
            crate::println!(
                "[mem]   {:#018X} .. {:#018X} ({:>5} MiB) {}",
                entry.base,
                end,
                size_kib / 1024,
                kind_str
            );
        } else {
            crate::println!(
                "[mem]   {:#018X} .. {:#018X} ({:>5} KiB) {}",
                entry.base,
                end,
                size_kib,
                kind_str
            );
        }

        if kind == MemoryRegionKind::Usable {
            total_usable += entry.length;
        }

        if region_count < regions.len() {
            regions[region_count] = MemoryRegion {
                base: entry.base,
                length: entry.length,
                kind,
            };
            region_count += 1;
        }
    }

    crate::println!(
        "[mem] Total usable: {} MiB",
        total_usable / 1024 / 1024
    );

    // Hand off to the arch-independent frame allocator.
    crate::memory::init_frame_allocator(&regions[..region_count], phys_to_virt);
}
