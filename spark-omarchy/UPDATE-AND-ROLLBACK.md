# Update and rollback on the Omarchy Spark

The root filesystem is btrfs with subvolumes `@` (root), `@home`, and
`@snapshots`. `snapper` owns the `@` timeline. Two rollback layers exist:

1. **In-Omarchy**: roll the root subvolume back to a pre-update snapshot.
2. **Whole-OS**: boot the untouched DGX OS entry from the UEFI menu.

## Before every `pacman -Syu`

```sh
sudo snapper -c root create --type pre --description "pre-pacman-$(date -Is)"
sudo pacman -Syu
sudo snapper -c root create --type post --description "post-pacman-$(date -Is)"
```

DKMS rebuilds the NVIDIA module inside the transaction when the kernel or
headers change. After the transaction, before rebooting:

```sh
dkms status                       # nvidia/580.178.04 must show installed for the new kernel
ls /boot/vmlinuz-linux-dgx-spark  # kernel image present
```

If `dkms status` shows the module missing for the new kernel, run
`sudo dkms autoinstall`, and if it still fails, do not reboot — roll back
first (below), then investigate.

## In-Omarchy rollback

```sh
sudo snapper -c root list        # find the pre-update snapshot number
sudo snapper -c root rollback <N>
sudo reboot
```

`snapper rollback` makes the snapshot the new default `@`. The boot entries
do not change because they name `rootflags=subvol=@`, not a snapshot path.

To undo only the packages that changed, instead of the whole root:

```sh
sudo snapper -c root undochange <PRE>..<POST>
```

## Kernel rollback

The fallback entry (`omarchy-spark-fallback.conf`) boots the fallback
initramfs of the same kernel. For a regression inside the kernel package
itself, keep the previous `linux-dgx-spark` package in `/var/cache/pacman/pkg`
and `pacman -U` it, then `dkms autoinstall` and reboot.

## Whole-OS rollback (DGX OS)

The DGX OS install and its UEFI entry are never modified by this kit.

- From a booted Omarchy: `sudo efibootmgr -n <DGX_OS_BOOTNUM>` then reboot —
  one boot into DGX OS; the order is unchanged afterwards.
- From firmware: the boot menu (Esc at power-on, per the board docs) lists
  both entries.

Use this when the Omarchy rootfs does not reach a shell, or when the vendor
stack is needed for a comparison run.

## What not to do

- Do not run `pacman -Syu` without the pre/post snapshots. The snapshot pair
  is what makes `undochange` precise.
- Do not edit the DGX OS ESP, its boot entry, or its root partition. It is
  the rollback target, and touching it removes the one guaranteed way back.
- Do not reboot after a kernel or DKMS transaction until `dkms status` shows
  the module built for the kernel you are about to boot.
