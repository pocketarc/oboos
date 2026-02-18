/* Userspace linker script — places the program at virtual address 0x400000. */
/* This address is in the lower half of the canonical address space (user     */
/* space). The kernel's ELF loader reads the PT_LOAD segments and maps them   */
/* at these addresses with USER page flags.                                   */

ENTRY(_start)

SECTIONS
{
    . = 0x400000;

    .text : ALIGN(4K)
    {
        *(.text .text.*)
    }

    .rodata : ALIGN(4K)
    {
        *(.rodata .rodata.*)
    }

    .data : ALIGN(4K)
    {
        *(.data .data.*)
    }

    .bss : ALIGN(4K)
    {
        *(.bss .bss.*)
    }

    /DISCARD/ :
    {
        *(.eh_frame)
        *(.note*)
        *(.comment)
    }
}
