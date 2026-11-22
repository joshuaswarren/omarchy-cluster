# Boot layout

The Spark firmware is UEFI. DGX OS already installs a UEFI boot entry; the
Omarchy install adds a second one instead of replacing it.

- Bootloader: `systemd-boot` on the Omarchy ESP. The DGX OS boot entry stays in
  the UEFI menu (`efibootmgr -v`) and on its own ESP. The boot menu or
  `efibootmgr -n` picks between them.
- Kernel: `linux-dgx-spark` (GB10 kernel, NVIDIA's Ubuntu fork patches plus
  the validated config). A mainline kernel does not bring up the ConnectX-7
  Ethernet, USB, or the full GPU on this board.
- Entries: `omarchy-spark.conf` (normal), `omarchy-spark-fallback.conf`
  (fallback initramfs). Replace `<ROOT_UUID>` at install time.
- The `nvidia-drm.modeset=1` option is required for GSP bring-up on GB10.
