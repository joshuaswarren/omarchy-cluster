# Omarchy on the DGX Spark: bring-up kit

This directory is the software side of putting Omarchy (Arch Linux ARM64) on
an NVIDIA DGX Spark (GB10 Grace-Blackwell). It covers what a stock Arch ARM64
install cannot get from its own repositories: the GB10 kernel, the NVIDIA
open kernel module, the CUDA 13 toolkit, the RPC-enabled llama.cpp build, the
network rules, and the update and rollback procedure.

It does not replace the physical install notes. Flashing the spare NVMe,
keeping the DGX OS install untouched as the rollback target, and the first
boot checks are operator steps; this kit is what gets installed once the
Omarchy rootfs is on disk and booting.

## Versions pinned for parity with DGX OS 7.6.0

| Component | Version | Why |
|---|---|---|
| Kernel | `7.0.0-1019-nvidia` family | The GB10 needs NVIDIA's Ubuntu-fork kernel patches (ConnectX-7 Ethernet, USB, full GPU). Mainline does not bring the board up. |
| Open kernel module | `580.178.04` | Same driver build as the vendor OS, so measurements compare. |
| CUDA toolkit | `13.4.2` | CUDA 13 SBSA payload; DGX OS ships 13.0.3, toolkit ≥ driver contract holds. |
| llama.cpp | `v1.0.0` (`2d8a3250`) | Pinned tag; the cluster proof kit requires the same commit's binaries on both hosts. |
| GPU architecture | `sm_121` | GB10 compute capability. |

## Layout

- `pkgbuilds/` — the four packages this install needs:
  - `nvidia-open-spark-dkms` — 580.178.04 open kernel modules via DKMS, so the
    module rebuilds whenever the kernel package changes.
  - `nvidia-spark-utils` — headless userland from the NVIDIA aarch64 runfile:
    libcuda, libnvidia-ml, nvidia-smi, nvidia-persistenced, OpenCL ICD, device
    node permissions. No display stack; the node has no X or Wayland session.
  - `cuda-spark` — CUDA 13.4.2 toolkit subset (nvcc, headers, cudart, cublas,
    nvrtc) under `/opt/cuda`. Driver components in the runfile are skipped.
  - `llamacpp-cuda-rpc` — `llama-server` + `rpc-server` built with
    `GGML_CUDA=ON` and `GGML_RPC=ON` for `sm_121`.
- `boot/` — systemd-boot entry templates and notes on keeping the DGX OS
  UEFI entry as the rollback target.
- `network/` — Avahi/mDNS setup and the UFW rule list (SSH, mDNS, Remnic
  :8095/:8098 scoped to the LAN; the RPC port is never opened permanently).
- `UPDATE-AND-ROLLBACK.md` — the `pacman -Syu` procedure with pre-update
  snapshots, the in-Omarchy rollback path, and the DGX OS fallback path.

## Kernel

The GB10 kernel is not in this kit and not in omarchy-pkgs yet. Use
`linux-dgx-spark` (Arch kernel package building NVIDIA's Launchpad Ubuntu
kernel fork with the validated GB10 config, derived from the nixos-dgx-spark
work). Pin the checkout used for the install in the install receipt; the
package is `linux-dgx-spark`/`linux-dgx-spark-headers`, optionally with the
`-64k` page-size variant. Do not install a mainline kernel and expect the
network or GPU to come up.

## Install order on a booted Omarchy rootfs

1. Kernel first: install `linux-dgx-spark` and `linux-dgx-spark-headers`,
   install `systemd-boot`, write the entries from `boot/`, confirm the DGX OS
   entry is still in `efibootmgr`.
2. Driver: build and install `nvidia-spark-utils`, then
   `nvidia-open-spark-dkms`. Check `dkms status` shows the module built for
   the installed kernel before rebooting.
3. Toolkit: `cuda-spark`, then `llamacpp-cuda-rpc`.
4. Network: `avahi`, `nss-mdns`, `ufw`, then apply `network/ufw.commands`.
5. Reboot into the Omarchy entry. First-boot checks:
   - `uname -r` reports the GB10 kernel.
   - `nvidia-smi` lists the GB10 GPU and driver `580.178.04`.
   - `llama-server --version` reports `v1.0.0`.
   - `avahi-browse -a` sees the node's own service; `ufw status` shows only
     the listed rules.
   - `rpc-server` and `llama-server` present on `PATH` for the proof kit.
6. Restore the Remnic services and model caches from the migration bundle,
   re-open the LAN-scoped :8095/:8098 rules, and re-run the service health
   checks.

## Rollback in one line

The DGX OS install is never touched. If the Omarchy boot fails at any point,
pick the DGX OS entry in the UEFI boot menu (or `efibootmgr -n` from a
working boot) and the machine is back on the vendor stack. In-Omarchy update
rollback uses the snapshot procedure in `UPDATE-AND-ROLLBACK.md`.

---

How-to text in this directory is licensed under CC BY-SA 4.0, credited to
Joshua Warren. Code and PKGBUILDs stay under their own licenses.
