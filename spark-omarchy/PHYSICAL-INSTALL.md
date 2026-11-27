# CL13 physical install checklist — Omarchy ARM64 on the DGX Spark (GN100)

One page, for the person standing at the box. The agent side (pre-cutover
backup, post-install packages, service restore) is operator work; this covers
only the physical steps: boot media, install, verify SSH, prove the DGX OS
fallback.

**Do not start until all three gates are green:**

- [ ] The vendor-EXO baseline receipt is committed (`receipts/milestone1-exo-pair/`) — Spark
  stays on DGX OS until the vendor baseline exists.
  **Gate state 2026-11-26: NO-GO.** No live EXO pair run exists yet — the
  only artifact under `receipts/milestone1-exo-pair/` is the runner's
  dry-run self-test (`run.json` `dry_run: true`, run id
  `20261122T051652Z-mac-m2max-macos-dgx-spark-dgxos`); there is no
  `live-<timestamp>/` directory, so no vendor-EXO baseline has been
  captured. Do not start the install. The gate clears only when a live
  pair run lands under `receipts/milestone1-exo-pair/live-<ts>/`.
- [ ] Off-Spark state backup verified on private storage.
- [ ] The ~30–45 min service cutover window is approved (the local agent
  services on the Spark are down for the duration).

## Hardware layout (why the external drive)

- GN100 has **one internal M.2 slot** — occupied by the 1.9 TB
  DGX OS drive (`nvme0n1`, `TS2TMTE480T-VS1`, 915 GiB models on it).
- Install goes to an **external ≥ 2 TB USB4/USB-C NVMe enclosure**. The
  internal drive is never touched; it stays the rollback target and both
  entries stay switchable via `efibootmgr -n`.
- Needs at the box: monitor, USB keyboard, Ethernet (`enP7s7`), the external
  NVMe enclosure, and the installer USB stick.

## Boot media (make ahead, anywhere)

- [ ] `omarchy-4.5.0-aarch64.iso` downloaded + verified:
  - sha256 `c9a9dd1107c10924eeb1e885226f9884080b1ffbf227d1f7f9ad255e4cebfb17`
  - `gpg --verify omarchy-4.5.0-aarch64.iso.sig` (key `40DFB630FF42BCFFB047046CF0134EE680CAC571`)
- [ ] Flashed to a ≥ 8 GB USB stick: `sudo dd if=omarchy-4.5.0-aarch64.iso of=/dev/sdX bs=4M status=progress oflag=sync` (confirm `sdX` with `lsblk` first).
- [ ] Optional: a second tiny drive labeled `cidata` for headless handoff
  (`authorized_keys` with your SSH keys, credentials JSONs, single-use
  tailscale authkey). Skip it and steps 3–4 below happen at the console.

## At the box

- [ ] Pre-cutover (over SSH): record `efibootmgr -v` + DGX OS
  `BootXXXX` id, `lsblk -f`, `ip -4 addr` on `enP7s7`; run the final
  state-bundle backup off-box.
- [ ] Plug monitor + keyboard, external NVMe enclosure, installer USB.
- [ ] Reboot, tap **Esc**/**Del** → UEFI setup: **Secure Boot → Disabled**;
  Boot Override → the Omarchy installer stick.
- [ ] Installer: pick your username and hostname; **disk: select
  ONLY the external USB NVMe** — confirm the model is not
  `TS2TMTE480T-VS1`/`nvme0n1`; **unencrypted** (headless closet box).
- [ ] Install finishes (~5 min); pull the installer stick (leave the external
  NVMe in), reboot.

## Verify SSH (done-when)

- [ ] Console (skip if `cidata` staged it): `sudo systemctl enable --now sshd`,
  `sudo ufw allow ssh`, your `authorized_keys` into `~/.ssh` (600).
- [ ] From another host: `ssh <spark-ip>` (fresh host key) — same static
  DHCP address as before.
- [ ] `omarchy-hw-platform` prints `aarch64-gb10`.

## Prove the DGX OS fallback

- [ ] `efibootmgr -v` — the DGX OS `BootXXXX` entry is still there.
- [ ] `sudo efibootmgr -n <DGX_BOOTNUM> && sudo reboot` → DGX OS answers SSH
  on `<spark-ip>` (the local agent services come back with it).
- [ ] Reboot once more → lands back on the Omarchy entry; SSH answers.

## Not measured until first physical boot

- Whether GN100 UEFI boots an external USB4 NVMe enclosure. If it refuses:
  fallback is swapping the internal M.2 with a fresh NVMe and keeping the
  DGX OS drive on the desk — works, but remote `efibootmgr -n` switching is
  lost (rollback becomes a physical swap again).
- Exact DMI `product_name` for the post-install GB10 fixups.

Hand back to the operator after the SSH + fallback checks pass; packages, CUDA,
`llamacpp-cuda-rpc`, UFW, and the service restore continue per
`README.md` install order and `UPDATE-AND-ROLLBACK.md`.

---

How-to text in this directory is licensed under CC BY-SA 4.0, credited to
Joshua Warren.
