//! Physical memory frame allocator.
//!
//! This module manages physical RAM at page-frame granularity (4 KiB frames).
//! It's architecture-independent — the arch layer discovers the memory map from
//! the bootloader and feeds it here as [`MemoryRegion`] slices.
//!
//! The allocator uses a bitmap: one bit per physical frame in the system. A set
//! bit (1) means "allocated / unavailable", a clear bit (0) means "free". We
//! start with every bit set (everything allocated) and then explicitly free the
//! regions the bootloader told us are usable. This "default allocated" approach
//! is safe — anything we forgot about (MMIO holes, kernel image, framebuffer,
//! ACPI tables, the bitmap itself) stays marked allocated without needing an
//! explicit reservation for each one.
//!
//! The bitmap itself is carved out of the first usable region large enough to
//! hold it — the classic bootstrap trick used by Linux and Windows. For 128 MiB
//! of RAM in QEMU, the bitmap is only 4 KiB. For 96 GiB it's ~3 MiB. Any
//! usable region can hold it.

/// Size of a physical page frame. x86_64 (and most architectures) use 4 KiB
/// as the base page size. Every physical address the allocator hands out is
/// aligned to this boundary.
pub const FRAME_SIZE: usize = 4096;

/// A region of physical memory reported by the bootloader.
///
/// The arch layer converts bootloader-specific structures into these
/// arch-independent descriptors before passing them to [`init_frame_allocator`].
#[derive(Debug, Clone, Copy)]
pub struct MemoryRegion {
    /// Physical base address of the region (byte-aligned, not necessarily
    /// page-aligned for non-usable regions).
    pub base: u64,
    /// Length in bytes.
    pub length: u64,
    /// What this region is used for.
    pub kind: MemoryRegionKind,
}

/// Classification of a physical memory region, mirroring the Limine memory map
/// entry types. The bootloader probes the firmware (BIOS E820 / UEFI memory map)
/// and reports these categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegionKind {
    /// Free RAM — safe to allocate.
    Usable,
    /// Firmware-reserved (MMIO, ROM, etc.) — do not touch.
    Reserved,
    /// ACPI tables that can be reclaimed after the OS parses them.
    AcpiReclaimable,
    /// ACPI Non-Volatile Storage — must be preserved.
    AcpiNvs,
    /// Physically defective memory.
    BadMemory,
    /// Bootloader data structures (page tables, response structs). Can be
    /// reclaimed once we build our own page tables and no longer reference
    /// bootloader responses.
    BootloaderReclaimable,
    /// The kernel binary and any loaded modules.
    KernelAndModules,
    /// Video framebuffer memory.
    Framebuffer,
}

/// Bitmap-based physical frame allocator.
///
/// Each bit in the bitmap corresponds to one 4 KiB frame of physical memory.
/// Bit N covers the frame at physical address `N * 4096`. A set bit means the
/// frame is in use; a clear bit means it's free.
struct BitmapFrameAllocator {
    /// The bitmap itself, living in physical memory accessed via the HHDM.
    bitmap: &'static mut [u8],
    /// Number of frames the bitmap covers (based on highest physical address).
    total_frames: usize,
    /// Number of frames currently marked free — useful for diagnostics.
    free_frames: usize,
    /// Hint for where to start scanning. Avoids re-scanning the low region
    /// that's typically allocated early and stays allocated.
    next_free_hint: usize,
}

impl BitmapFrameAllocator {
    /// Allocate a single physical frame.
    ///
    /// Scans the bitmap starting from `next_free_hint`, skipping fully-allocated
    /// bytes (0xFF) for speed. Returns the physical address of the allocated
    /// frame, or `None` if memory is exhausted.
    fn alloc(&mut self) -> Option<usize> {
        let start_byte = self.next_free_hint / 8;
        let bitmap_len = self.bitmap.len();

        // Scan from hint to end, then wrap around from 0 to hint.
        for offset in 0..bitmap_len {
            let byte_idx = (start_byte + offset) % bitmap_len;
            let byte = self.bitmap[byte_idx];

            // Skip fully-allocated bytes — this is the main speedup over
            // checking individual bits. In practice most of the bitmap is
            // either all-allocated (low memory) or all-free (high memory).
            if byte == 0xFF {
                continue;
            }

            // Found a byte with at least one free bit. Find which one.
            for bit in 0..8 {
                if byte & (1 << bit) == 0 {
                    let frame_index = byte_idx * 8 + bit;
                    if frame_index >= self.total_frames {
                        return None;
                    }

                    // Mark allocated.
                    self.bitmap[byte_idx] |= 1 << bit;
                    self.free_frames -= 1;
                    self.next_free_hint = frame_index + 1;

                    return Some(frame_index * FRAME_SIZE);
                }
            }
        }

        None
    }

    /// Return a physical frame to the free pool.
    ///
    /// # Panics
    ///
    /// Panics on double-free (freeing a frame that's already marked free) or
    /// if the address is out of range. These indicate allocator bugs.
    fn free(&mut self, phys_addr: usize) {
        assert!(
            phys_addr % FRAME_SIZE == 0,
            "free: address {:#X} is not frame-aligned",
            phys_addr
        );

        let frame_index = phys_addr / FRAME_SIZE;
        assert!(
            frame_index < self.total_frames,
            "free: frame index {} out of range (total: {})",
            frame_index,
            self.total_frames
        );

        let byte_idx = frame_index / 8;
        let bit = frame_index % 8;

        assert!(
            self.bitmap[byte_idx] & (1 << bit) != 0,
            "free: double-free of frame at {:#X}",
            phys_addr
        );

        self.bitmap[byte_idx] &= !(1 << bit);
        self.free_frames += 1;

        // Move the hint back so the freed frame gets reused quickly.
        if frame_index < self.next_free_hint {
            self.next_free_hint = frame_index;
        }
    }
}

/// Global frame allocator instance.
///
/// `spin::Once` ensures one-time initialization (like `std::sync::Once`).
/// `spin::Mutex` protects concurrent access — we're single-core today but
/// interrupt handlers could theoretically call into the allocator, and the
/// mutex makes that safe.
static FRAME_ALLOCATOR: spin::Once<spin::Mutex<BitmapFrameAllocator>> = spin::Once::new();

/// Initialize the frame allocator from the bootloader's memory map.
///
/// This is called once during boot by the arch layer after it has parsed
/// the bootloader's memory map into [`MemoryRegion`] slices.
///
/// # Arguments
///
/// * `regions` — the memory map entries from the bootloader
/// * `phys_to_virt` — converts a physical address to a virtual pointer via
///   the HHDM (Higher Half Direct Map). We need this to access the bitmap
///   which lives in physical memory.
///
/// # Panics
///
/// Panics if no usable region is large enough to hold the bitmap, or if
/// called more than once.
pub fn init_frame_allocator(
    regions: &[MemoryRegion],
    phys_to_virt: fn(u64) -> *mut u8,
) {
    // Step 1: Find the highest physical address that's actually usable.
    // We only need the bitmap to cover memory we might allocate from. Huge
    // reserved regions above physical RAM (e.g. MMIO at 0xFD00000000) would
    // bloat the bitmap to tens of megabytes for no reason.
    let highest_addr = regions
        .iter()
        .filter(|r| r.kind == MemoryRegionKind::Usable)
        .map(|r| r.base + r.length)
        .max()
        .expect("no usable memory regions");

    let total_frames = (highest_addr as usize + FRAME_SIZE - 1) / FRAME_SIZE;
    // Round up to whole bytes — one bit per frame.
    let bitmap_bytes = (total_frames + 7) / 8;

    crate::println!(
        "[mem] Bitmap: {} bytes for {} frames ({} MiB address space)",
        bitmap_bytes,
        total_frames,
        highest_addr / 1024 / 1024
    );

    // Step 2: Find a usable region large enough to hold the bitmap.
    // We'll carve the bitmap from the start of this region.
    let mut bitmap_phys: Option<u64> = None;
    for region in regions {
        if region.kind == MemoryRegionKind::Usable && region.length >= bitmap_bytes as u64 {
            bitmap_phys = Some(region.base);
            break;
        }
    }
    let bitmap_phys = bitmap_phys.expect("no usable region large enough for bitmap");

    // Step 3: Create the bitmap slice via the HHDM.
    let bitmap_virt = phys_to_virt(bitmap_phys);
    let bitmap: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(bitmap_virt, bitmap_bytes) };

    // Step 4: Mark everything as allocated (set all bits to 1).
    // Anything we don't explicitly free stays allocated — safe default.
    bitmap.fill(0xFF);

    // Step 5: Free the usable regions, being careful to exclude the bitmap's
    // own memory so we don't hand it out.
    let bitmap_end = bitmap_phys + bitmap_bytes as u64;
    let mut free_frames = 0usize;

    for region in regions {
        if region.kind != MemoryRegionKind::Usable {
            continue;
        }

        let mut region_start = region.base;
        let region_end = region.base + region.length;

        // If the bitmap lives inside this region, skip past it.
        if region_start < bitmap_end && region_end > bitmap_phys {
            region_start = bitmap_end;
        }

        // Align start up to frame boundary, end down.
        let frame_start = ((region_start as usize + FRAME_SIZE - 1) / FRAME_SIZE) * FRAME_SIZE;
        let frame_end = (region_end as usize / FRAME_SIZE) * FRAME_SIZE;

        // Clear bits for each free frame in this region.
        let mut addr = frame_start;
        while addr < frame_end {
            let frame_index = addr / FRAME_SIZE;
            let byte_idx = frame_index / 8;
            let bit = frame_index % 8;
            bitmap[byte_idx] &= !(1 << bit);
            free_frames += 1;
            addr += FRAME_SIZE;
        }
    }

    let allocator = BitmapFrameAllocator {
        bitmap,
        total_frames,
        free_frames,
        next_free_hint: 0,
    };

    crate::println!(
        "[mem] Free frames: {} ({} MiB)",
        free_frames,
        (free_frames * FRAME_SIZE) / 1024 / 1024
    );

    FRAME_ALLOCATOR.call_once(|| spin::Mutex::new(allocator));

    crate::println!("[ok] Frame allocator initialized");
}

/// Allocate a single 4 KiB physical frame.
///
/// Returns the physical address of the frame, or `None` if out of memory.
pub fn alloc_frame() -> Option<usize> {
    FRAME_ALLOCATOR
        .get()
        .expect("frame allocator not initialized")
        .lock()
        .alloc()
}

/// Free a previously allocated physical frame.
///
/// # Panics
///
/// Panics on double-free or unaligned address.
pub fn free_frame(phys_addr: usize) {
    FRAME_ALLOCATOR
        .get()
        .expect("frame allocator not initialized")
        .lock()
        .free(phys_addr)
}

/// Return the number of free frames available for allocation.
pub fn free_frame_count() -> usize {
    FRAME_ALLOCATOR
        .get()
        .expect("frame allocator not initialized")
        .lock()
        .free_frames
}
