.PHONY: all clean run test iso

KERNEL_ELF = target/x86_64-unknown-none/debug/oboos-kernel
ISO = oboos.iso
ISO_ROOT = iso_root
LIMINE_DIR = limine

QEMU_FLAGS = -cdrom $(ISO) -serial stdio -no-reboot -no-shutdown -m 128M
TEST_TIMEOUT ?= 5

all: iso

# Clone the Limine bootloader if not present.
$(LIMINE_DIR):
	git clone https://github.com/limine-bootloader/limine.git --branch=v8.x-binary --depth=1
	$(MAKE) -C $(LIMINE_DIR)

# Assemble a bootable ISO from whatever kernel ELF is currently built.
# Declared .PHONY so it always runs — Cargo's own fingerprinting decides
# whether to recompile, and we always re-pack the ISO to match.
iso: $(LIMINE_DIR) limine.conf
	cargo build
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
run: iso
	qemu-system-x86_64 $(QEMU_FLAGS)

# Build with smoke tests enabled and boot in QEMU.
# Cargo tracks features in its fingerprint, so switching between
# `make run` and `make test` always triggers the right recompile.
test: $(LIMINE_DIR) limine.conf
	cargo build --features smoke-test
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
	timeout $(TEST_TIMEOUT) qemu-system-x86_64 $(QEMU_FLAGS) || true

clean:
	cargo clean
	rm -rf $(ISO_ROOT) $(ISO)
