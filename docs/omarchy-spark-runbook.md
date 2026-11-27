# Omarchy on the DGX Spark — Milestone 2 validation runbook (CL14)

The operator sequence from "Omarchy booted, nothing restored" to "the Spark
is a validated Omarchy cluster node with the memory service back on DGX OS
as the cold rollback". Every section has pass cells; fill them into the
run's receipt as you go.

**Hard rules**

- The DGX OS install and its UEFI entry are never written. It is the
  rollback target for the whole milestone.
- The memory service stays down until the rollback proof in section 5 is
  recorded. No reboot may ever happen with restored services enabled.
- All service ports and hostnames live in the private ops inventory. This
  document names them as placeholders: `<REMNIC_CORE_PORT>`,
  `<RERANKER_PORT>`, `<SPARK_HOST>`, `<MAC_HOST>`.

**Preconditions (all must be true before section 1)**

- [ ] `spark-omarchy/PHYSICAL-INSTALL.md` completed: Omarchy boots from the
  external NVMe, DGX OS entry verified reachable from the firmware menu.
- [ ] The `spark-omarchy/` kit applied: `linux-dgx-spark` kernel, the four
  `pkgbuilds/` packages installed, `boot/` and `network/` pieces in place.
- [ ] The CL11 restore bundle is on reachable private storage and its
  checksum verifies.
- [ ] A service-window approval exists for the restore section (the memory
  service is unavailable for the duration).

## 1. Cold boot

| Check | Command | Pass |
|---|---|---|
| SSH reachable | `ssh <SPARK_HOST> hostname` | returns `gn100` hostname |
| Right OS | `uname -r` | reports the `linux-dgx-spark` kernel version from the install receipt |
| Right root | `findmnt -no SOURCE /` | the external NVMe partition, never `nvme0n1` |
| Internal drive untouched | `lsblk -o NAME,MOUNTPOINT nvme0n1` | no mountpoint on `nvme0n1` |

## 2. GB10 platform

| Check | Command | Pass |
|---|---|---|
| Kernel module | `dkms status` | `nvidia/580.178.04` installed for the running kernel |
| GPU visible | `nvidia-smi -L` | one GB10 (`sm_121`) entry |
| Persistence | `systemctl is-active nvidia-persistenced` | `active` |
| Toolkit | `/opt/cuda/bin/nvcc --version` | CUDA 13.4.2 |

## 3. RPC inference check

Use the smallest GGUF already on the external drive's model tree (mounted
read-only via `ro,noload` from the internal drive per the install notes —
verify checksums before first use).

| Check | Command | Pass |
|---|---|---|
| RPC server | `rpc-server -H 0.0.0.0 -p <RPC_PORT>` in a bounded session | listens, logs the sm_121 device |
| Inference | `llama-server --rpc <SPARK_HOST>:<RPC_PORT> -m <small-gguf> --no-mmap -ngl 99` one greedy 32-token request | tokens decode; server log shows the RPC backend |
| Parity | repeat with `-dev none` | identical token digest as the CUDA arm |

Kill both processes after the check. The RPC port is never opened in UFW
permanently — see `spark-omarchy/network/`.

## 4. Network

| Check | Command | Pass |
|---|---|---|
| mDNS | `avahi-resolve -n <SPARK_HOST>.local` from `<MAC_HOST>` | resolves to the LAN address |
| UFW | `sudo ufw status verbose` | SSH + mDNS + the memory-service pair allowed, RPC closed, default deny |
| LAN throughput | `iperf3 -s` on `<SPARK_HOST>`, `iperf3 -c <SPARK_HOST>` from `<MAC_HOST>` | measured value recorded — feeds the milestone comparison |

## 5. Rollback proof — before any service is restored

This is the gate the whole milestone stands on. Prove both layers while
nothing depends on the node.

| Check | Command | Pass |
|---|---|---|
| In-Omarchy | `sudo snapper -c root list` | pre/post install snapshots exist |
| Whole-OS | `sudo efibootmgr -n <DGX_OS_BOOTNUM> && sudo reboot` | boots into DGX OS; `efibootmgr` shows the order unchanged |
| Return | boot menu -> Omarchy entry | Omarchy comes back clean |

Do the `efibootmgr -n` proof now — before units are enabled — so no reboot
ever happens with restored services live. Procedure detail:
`spark-omarchy/UPDATE-AND-ROLLBACK.md`.

## 6. Update + rollback drill

Run the full `pacman -Syu` procedure from `spark-omarchy/UPDATE-AND-ROLLBACK.md`
on the live node exactly once:

- [ ] `snapper` pre/post snapshots created around the transaction
- [ ] `dkms status` still shows the module for the new kernel before reboot
- [ ] After reboot: sections 1–2 re-pass
- [ ] Rehearse `snapper undochange` on a scratch package (not the root
  timeline) so the operator has run the path once

## 7. Memory-service restore

Restore from the CL11 bundle. The ABI gate first: if `ldd` on the restored
binaries shows unresolved libraries against the new userland, rebuild from
the pinned source instead of transplanting.

- [ ] Bundle restored; model tree verified by checksum against the internal
  drive copy
- [ ] `systemctl --user enable --now` the memory-service units; linger on
- [ ] Health poll `<REMNIC_CORE_PORT>` for 120 s — steady responses
- [ ] `<RERANKER_PORT>` health check green
- [ ] UFW rules for the pair confirmed scoped to the LAN

## 8. Sign-off

| Cell | Value |
|---|---|
| Kernel / module | `uname -r` / `dkms status` output |
| GPU | `nvidia-smi -L` line |
| Inference parity | both digests, arm named |
| Rollback | snapshot numbers + `efibootmgr` transcript |
| Update drill | package delta + post state |
| Service | health-poll duration + status codes |

Commit the filled cells as the CL14 receipt alongside this file's merge.
