# Run one inference request across an Omarchy Mac and a DGX Spark

This is an application runbook for an **already booted** M2 Max on Omarchy Linux (Borealis Vulkan, AGX counters, active UFW) and DGX Spark (GB10, CUDA) on a private IPv4 LAN. It installs the cluster package in a throwaway virtual environment on each host. It does not install, update, or recover the OS on either host. In particular, this run did not validate Omarchy boot, updates, or rollback on Spark.

## Install and run the proof

1. On both hosts, use a [checkout of the pinned proof kit](https://github.com/joshuaswarren/omarchy-cluster/tree/06069de09e9d50c3701ab6d88e2bf06a29cdc093/receipts/two-node-proof-kit). Check out `06069de09e9d50c3701ab6d88e2bf06a29cdc093`, set `SOURCE` to that checkout's absolute path, and run `sha256sum -c --strict kit.sha256` from `receipts/two-node-proof-kit/`. The kit installs source commit `10a47b69df53f60b1e9b3f67aa5092fdff2c4562` from the checkout without changing the system Python or installing a persistent agent.
2. Follow the kit README for binary, model and certificate setup: compatible RPC-enabled llama.cpp builds (Vulkan on Mac, CUDA on Spark), the exact `Qwen3-1.7B-Q4_K_M.gguf` checksum, synced clocks, and `socat` with OpenSSL support on both hosts. Check `socat -V | grep WITH_OPENSSL`, `Vulkan0` and AGX counters on the Mac, and `CUDA0` with per-process `nvidia-smi --query-compute-apps` on Spark. Reserve an exclusive Mac GPU window and approved Spark cache-drop window. Use narrow, operator-approved noninteractive sudo for the kit's UFW and cache-drop operations. Do not grant passwordless sudo broadly.
3. Generate a fresh dedicated CA and separate disposable per-host key/cert directories as specified by the kit README. Do not install the CA system-wide or start the manual relays as well. The kit keeps raw RPC loopback-only, verifies the peers, and manages one temporary UFW rule for the Spark IP. Start each command from its own kit directory with a fresh `OUT` and that host's absolute paths:

   ```bash
   # Mac: the worker stays up for at most four minutes after READY.
   bash two-node-proof.sh mac "$SOURCE" "$GGUF" "$OUT" "$LLAMA_SERVER" "$TLS" "$M2_ADDR" "$RPC_SERVER" "$SPARK_ADDR"

   # Spark: run once after READY.
   bash two-node-proof.sh spark "$SOURCE" "$GGUF" "$OUT" "$LLAMA_SERVER" "$TLS" "$M2_ADDR"
   ```

4. When Spark exits, stop the Mac command with Ctrl-C and let it clean up. Bring both output directories to one host and run `python3 check-proof.py combine "$MAC_OUT" "$SPARK_OUT" > combine.json` from the kit directory. A pass includes both GPUs and RPC traffic during the shared request, all 64 token IDs matching across the three arms, and UFW restored. See [#124](https://github.com/joshuaswarren/omarchy-cluster/issues/124) for what to share. Do not publish raw network details, keys, or weights.

## Spark: `cudaSetDevice(0)` reports out of memory

A shared attempt in the lab negotiated TLS and enumerated the Mac RPC device, then CUDA failed before the HTTP server came up. `MemAvailable` includes reclaimable cache and did not predict launch success in this run. The kit runs `sync` and `sudo -n sysctl -q vm.drop_caches=3` on Spark **immediately before each** Spark-only and shared CUDA launch. It records `MemFree`, `MemAvailable`, and `Cached` from `/proc/meminfo` and requires **at least 6 GiB `MemFree`** for this model. Resident CUDA allocations are recorded, not used to reject the launch. This is a floor for the proof, not a capacity estimate for other checkpoints.

The cache drop is host-wide and may slow other workloads while their caches refill. Let the kit do it during an approved window; do not stop unrelated services or add a permanent sysctl. For an equivalent manual retry during that window, run the same `sync` and cache drop, record the three fields, and check the floor just before starting the engine. If the floor is missed or `cudaSetDevice` still fails above it, keep the three values and first engine error and stop. Do not keep dropping caches or assume the diagnosis applies to other NVIDIA hosts.

The passing Spark receipts recorded these `/proc/meminfo` values just before launch (in KiB):

| CUDA arm | `MemFree` | `MemAvailable` | `Cached` |
|---|---:|---:|---:|
| Spark only | 8,859,656 | 24,937,260 | 17,665,236 |
| Shared | 10,273,420 | 24,765,636 | 16,083,612 |

Both launches succeeded without stopping the resident services. This does not show that every CUDA allocation failure is caused by file cache.

## Worker loss, restart and cleanup

There is no automatic failover. In a separate live drill, stopping only the test-owned Mac RPC worker made the next request return HTTP 502. Restarting that worker and the test gateway, then replaying the same request, produced the same 64 IDs at 9.62 decode tokens/s. This was manual service recovery, not a host reboot or an in-flight retry.

For a failed kit run, stop the test commands normally and let the exit handlers close the test processes and relay sockets and remove the credentials. Check both `exit-code` receipts and byte-compare the Mac's `ufw-before.txt` and `ufw-after.txt`. If receipts are missing or cleanup failed, inspect live state before retrying. Use [the secure RPC teardown checks](secure-rpc.md#tear-down): stop only the test processes; remove only the labelled, source-scoped rule after verifying it matches; verify the reserved listeners are gone and the UFW snapshot is restored; remove the disposable credentials. Do not reset UFW or kill unrelated GPU processes. To retry, use new output directories and a fresh certificate set, start the Mac, wait for `READY`, then start Spark and verify the new receipt.

## Measured result and limits

[Reference receipt](https://github.com/joshuaswarren/omarchy-cluster/blob/06069de09e9d50c3701ab6d88e2bf06a29cdc093/receipts/two-node-proof-kit/reference-combine.json) (November 20, 2026 CST, SHA-256 `f42663ebff292225da75008b45c7d22bcc402d694e316105e6bd029010903197`). Same 43-token prompt, greedy decode, 64 generated IDs, one request at a time. Prefill comes from the per-arm engine timings; request wall time comes from the kit's start/end timestamps around the full local HTTP response, excluding model load. Engine prefill is not client time to first token.

| Arm | Engine prefill (ms) | Engine decode (tokens/s) | Request wall time (s) |
|---|---:|---:|---:|
| Mac only, Vulkan | 500.32 | 67.4 | 1.45 |
| Spark only, CUDA | 50.22 | 26.86 | 2.44 |
| Shared, CUDA + remote Vulkan | 247.21 | 9.41 | 6.99 |

The shared request split layers over llama.cpp RPC; it was not routed to just one machine. The Mac recorded +587 AGX jobs and 215.95 ms GPU busy time; Spark's engine recorded CUDA0 and RPC0, a 399.89 MiB CUDA model buffer, and 673 MiB for the test PID in each of seven in-request samples. The RPC socket returned 524,864 bytes. All 64 token IDs matched and UFW was restored.

Shared decode was slower than either solo arm; this co-loaded run shows the split works, not a speedup or an isolated scaling comparison. Exact IDs are a check for this prompt, not a general quality test. Direct client TTFT, peak system memory, whole-link traffic and energy were not measured here. Other hardware, models, larger loads, external reproduction, and Spark OS lifecycle are unverified for this recipe. The mutual TLS link protects cross-host RPC only, not application HTTP or SSH; do not expose these services to the internet.
