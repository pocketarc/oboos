//! Minimal ELF64 loader for static executables.
//!
//! Parses an ELF64 binary embedded in the kernel, maps its PT_LOAD segments
//! into the current address space with appropriate page flags, and returns
//! the entry point address. This is enough to load our userspace Rust
//! programs, which are simple static executables with a handful of segments.
//!
//! ## What we support
//!
//! - ELF64, little-endian, x86_64, ET_EXEC (static, non-relocatable)
//! - PT_LOAD segments only (the only segment type that needs memory mapping)
//! - Segments that aren't page-aligned (the loader aligns down and copies
//!   at the correct offset within the page)
//!
//! ## What we don't support (yet)
//!
//! - ET_DYN (PIE/shared objects) — would need a base address randomizer
//! - Dynamic linking, relocations, or symbol resolution
//! - Program interpreter (PT_INTERP)
//! - TLS (PT_TLS) — no thread-local storage in userspace yet
//!
//! ## Memory management
//!
//! Segment pages are allocated from the frame allocator and mapped with
//! USER permissions. The file data is copied through the HHDM (Higher Half
//! Direct Map) rather than through the user-space mapping, which avoids
//! needing to temporarily set WRITABLE on read-only text segments.
//!
//! [`LoadedElf::unload()`] unmaps and frees all allocated pages, making
//! cleanup straightforward for the test harness.

use crate::arch;
use crate::memory;
use crate::platform::{MemoryManager, PageFlags};
use crate::println;

// ————————————————————————————————————————————————————————————————————————————
// ELF64 structures
// ————————————————————————————————————————————————————————————————————————————

/// ELF64 file header (Ehdr). 64 bytes at the start of every ELF binary.
///
/// The first 16 bytes (`e_ident`) contain the magic number and format flags.
/// The rest describes the file type, target architecture, entry point, and
/// the location of the program header table (which tells us what to load).
#[repr(C, packed)]
struct Elf64Header {
    e_ident: [u8; 16],
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

/// ELF64 program header (Phdr). 56 bytes per entry.
///
/// Each program header describes a segment: a contiguous range of the file
/// that maps to a contiguous range of virtual memory. PT_LOAD segments are
/// the ones we actually need to map — they contain code and data.
#[repr(C, packed)]
struct Elf64Phdr {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

// ELF constants.
const ELFMAG: [u8; 4] = [0x7F, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1; // little-endian
const ET_EXEC: u16 = 2;
const EM_X86_64: u16 = 62;
const PT_LOAD: u32 = 1;

// ELF segment permission flags.
const PF_X: u32 = 1;
const PF_W: u32 = 2;

// ————————————————————————————————————————————————————————————————————————————
// Loaded ELF result
// ————————————————————————————————————————————————————————————————————————————

/// A successfully loaded ELF binary: entry point + list of mapped pages
/// for cleanup.
pub struct LoadedElf {
    /// Virtual address of the ELF entry point (`_start`).
    pub entry: u64,
    /// All (virt_addr, phys_addr) pairs allocated for PT_LOAD segments.
    /// Used by [`unload()`] to clean up.
    pages: [(usize, usize); 32],
    page_count: usize,
}

impl LoadedElf {
    /// Unmap and free all pages allocated for this ELF binary.
    pub fn unload(self) {
        for i in 0..self.page_count {
            let (virt, phys) = self.pages[i];
            arch::Arch::unmap_page(virt);
            memory::free_frame(phys);
        }
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Loader
// ————————————————————————————————————————————————————————————————————————————

/// Load an ELF64 binary into the current address space.
///
/// Validates the ELF header, then maps each PT_LOAD segment with appropriate
/// permissions (USER + page flags derived from the ELF segment flags).
///
/// # Panics
///
/// Panics if the binary is not a valid ELF64 x86_64 executable, or if
/// frame allocation fails during loading.
pub fn load_elf(binary: &[u8]) -> LoadedElf {
    assert!(binary.len() >= core::mem::size_of::<Elf64Header>(), "ELF binary too small for header");

    // Safety: we've verified the binary is large enough, and the struct is
    // repr(C, packed) so alignment doesn't matter.
    let header = unsafe { &*(binary.as_ptr() as *const Elf64Header) };

    // Validate the ELF header. Fields are copied to locals before comparing
    // because assert_eq! takes references, and you can't take a reference to
    // a field of a packed struct (it might be misaligned).
    assert_eq!(&header.e_ident[0..4], &ELFMAG, "bad ELF magic");
    assert_eq!(header.e_ident[4], ELFCLASS64, "not ELF64");
    assert_eq!(header.e_ident[5], ELFDATA2LSB, "not little-endian");

    let e_type = header.e_type;
    let e_machine = header.e_machine;
    assert_eq!(e_type, ET_EXEC, "not ET_EXEC");
    assert_eq!(e_machine, EM_X86_64, "not x86_64");

    let phoff = header.e_phoff as usize;
    let phentsize = header.e_phentsize as usize;
    let phnum = header.e_phnum as usize;
    let entry = header.e_entry;

    assert!(phentsize >= core::mem::size_of::<Elf64Phdr>(), "phentsize too small");
    assert!(phoff + phnum * phentsize <= binary.len(), "program headers extend past binary");

    println!("[elf] Loading ELF binary ({} bytes, {} program headers, entry={:#X})", binary.len(), phnum, entry);

    let mut pages: [(usize, usize); 32] = [(0, 0); 32];
    let mut page_count = 0;
    let mut segments_loaded = 0;

    for i in 0..phnum {
        let phdr_offset = phoff + i * phentsize;
        let phdr = unsafe { &*(binary.as_ptr().add(phdr_offset) as *const Elf64Phdr) };

        if phdr.p_type != PT_LOAD {
            continue;
        }

        let vaddr = phdr.p_vaddr as usize;
        let memsz = phdr.p_memsz as usize;
        let filesz = phdr.p_filesz as usize;
        let offset = phdr.p_offset as usize;
        let flags = phdr.p_flags;

        // Translate ELF segment flags to page table flags.
        // All user segments need PRESENT | USER.
        let mut page_flags = PageFlags::PRESENT | PageFlags::USER;
        if flags & PF_W != 0 {
            page_flags = page_flags | PageFlags::WRITABLE;
        }
        if flags & PF_X == 0 {
            page_flags = page_flags | PageFlags::NO_EXECUTE;
        }

        // Calculate page-aligned virtual address range.
        let page_start = vaddr & !0xFFF;
        let page_end = (vaddr + memsz + 0xFFF) & !0xFFF;
        let num_pages = (page_end - page_start) / memory::FRAME_SIZE;

        println!("[elf]   PT_LOAD: vaddr={:#X}, memsz={:#X}, filesz={:#X}, pages={}", vaddr, memsz, filesz, num_pages);

        for p in 0..num_pages {
            let page_virt = page_start + p * memory::FRAME_SIZE;

            // Check if this page was already allocated by a previous segment.
            // This happens when two segments share a page boundary (e.g. .rodata
            // and .got can land on the same 4K page). In that case, reuse the
            // existing frame — just copy additional data into it.
            let existing_phys = pages[..page_count]
                .iter()
                .find(|&&(v, _)| v == page_virt)
                .map(|&(_, phys)| phys);

            let frame = match existing_phys {
                Some(phys) => phys,
                None => {
                    let f = memory::alloc_frame().expect("alloc frame for ELF segment");

                    // Zero new frames through the HHDM.
                    let ptr = arch::memory::phys_to_virt(f as u64);
                    unsafe { core::ptr::write_bytes(ptr, 0, memory::FRAME_SIZE); }

                    // Map the new page and track it for cleanup.
                    arch::Arch::map_page(page_virt, f, page_flags);
                    assert!(page_count < pages.len(), "too many ELF pages (max 32)");
                    pages[page_count] = (page_virt, f);
                    page_count += 1;

                    f
                }
            };

            // Copy file data into the frame through the HHDM.
            let hhdm_ptr = arch::memory::phys_to_virt(frame as u64);
            let seg_start = vaddr;
            let seg_file_end = vaddr + filesz;
            let page_lo = page_virt;
            let page_hi = page_virt + memory::FRAME_SIZE;

            let copy_lo = page_lo.max(seg_start);
            let copy_hi = page_hi.min(seg_file_end);

            if copy_lo < copy_hi {
                let copy_len = copy_hi - copy_lo;
                let dst_offset = copy_lo - page_virt;
                let src_offset = offset + (copy_lo - seg_start);

                unsafe {
                    core::ptr::copy_nonoverlapping(
                        binary.as_ptr().add(src_offset),
                        hhdm_ptr.add(dst_offset),
                        copy_len,
                    );
                }
            }
        }

        segments_loaded += 1;
    }

    println!("[elf] Mapped {} PT_LOAD segments, entry at {:#X}", segments_loaded, entry);

    LoadedElf {
        entry,
        pages,
        page_count,
    }
}
