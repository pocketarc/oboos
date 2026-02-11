.PHONY: all clean run

KERNEL_ELF = target/x86_64-unknown-none/debug/oboos-kernel
ISO = oboos.iso
ISO_ROOT = iso_root
LIMINE_DIR = limine

all: $(ISO)

# Build the kernel. Cargo handles cross-compilation via .cargo/config.toml.
$(KERNEL_ELF): kernel/src/**/*.rs kernel/Cargo.toml
	cargo build

# Clone the Limine bootloader if not present.
$(LIMINE_DIR):
	git clone https://github.com/limine-bootloader/limine.git --branch=v8.x-binary --depth=1
	$(MAKE) -C $(LIMINE_DIR)

# Assemble the bootable ISO.
$(ISO): $(KERNEL_ELF) $(LIMINE_DIR) limine.conf
	rm -rf $(ISO_ROOT)
	mkdir -p $(ISO_ROOT)/boot
	cp $(KERNEL_ELF) $(ISO_ROOT)/boot/kernel
	cp limine.conf $(ISO_ROOT)/
	cp $(LIMINE_DIR)/limine-bios.sys $(ISO_ROOT)/boot/
	cp $(LIMINE_DIR)/limine-bios-cd.bin $(ISO_ROOT)/boot/
	cp $(LIMINE_DIR)/limine-uefi-cd.bin $(ISO_ROOT)/boot/
	xorriso -as mkisofs \
		-b boot/limine-bios-cd.bin \
		-no-emul-boot \
		-boot-load-size 4 \
		-boot-info-table \
		--efi-boot boot/limine-uefi-cd.bin \
		-efi-boot-part \
		--efi-boot-image \
		--protective-msdos-label \
		$(ISO_ROOT) -o $(ISO)
	./$(LIMINE_DIR)/limine bios-install $(ISO)

# Run in QEMU. Serial output goes to your terminal via -serial stdio.
run: $(ISO)
	qemu-system-x86_64 \
		-cdrom $(ISO) \
		-serial stdio \
		-no-reboot \
		-no-shutdown \
		-m 128M

clean:
	cargo clean
	rm -rf $(ISO_ROOT) $(ISO)
