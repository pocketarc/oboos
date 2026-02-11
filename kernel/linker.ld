/* OBOOS kernel linker script for x86_64 with Limine bootloader */

OUTPUT_FORMAT(elf64-x86-64)
ENTRY(kmain)

/* The kernel lives in the "higher half" of virtual memory.
   Limine maps our kernel here and sets up the page tables for us.
   The lower half (0x0000...) is reserved for userspace later. */
KERNEL_OFFSET = 0xFFFFFFFF80000000;

/* Define the ELF program headers (segments) explicitly.
   Each LOAD segment gets mapped into memory by the bootloader.
   Separating them lets us set different permissions later
   (text = execute, rodata = read-only, data = read-write). */
PHDRS {
    text     PT_LOAD;
    rodata   PT_LOAD;
    data     PT_LOAD;
}

SECTIONS {
    . = KERNEL_OFFSET;

    .text : {
        *(.text .text.*)
    } :text

    /* Page-align between sections so they can have different permissions. */
    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .rodata : {
        *(.rodata .rodata.*)
    } :rodata

    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .data : {
        *(.data .data.*)

        /* Limine request structs go in .data — they're mutable because
           the bootloader writes response pointers into them. The start/end
           markers let Limine know where to scan. */
        KEEP(*(.requests_start_marker))
        KEEP(*(.requests))
        KEEP(*(.requests_end_marker))
    } :data

    /* BSS = uninitialized globals. Zeroed on boot. */
    .bss : {
        *(.bss .bss.*)
        *(COMMON)
    } :data

    /* Throw away unneeded sections. */
    /DISCARD/ : {
        *(.eh_frame*)
        *(.note*)
        *(.comment*)
    }
}
