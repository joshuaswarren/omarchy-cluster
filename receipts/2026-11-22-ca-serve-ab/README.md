# C-a cross-machine serve A/B: three-arm medians (2026-11-22)

One spec, one runner (`omarchy_cluster/serve_ab.py`, spec
`docs/serve-ab-ca.md`), one pinned GGUF
(`lmstudio-community/Qwen3-4B-Instruct-2507-GGUF` at revision
`4edb920b6f14e3b9284d4502a6485103d72cde05`, file
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, sha256 `8cdb57cb…82641fc`), three arms
measured fresh on 2026-11-22. Three rounds of three fixed prompts, 128 greedy
tokens per run, TTFT and decode medians per prompt, token-id digests identical
across rounds on every arm.

| Arm | Host chip | Backend | Decode tok/s (p1/p2/p3) | TTFT ms (p1/p2/p3) | run digest | llama-server |
| --- | --- | --- | --- | --- | --- | --- |
| rtx5060ti-cuda | apple,t6001 (16-inch M1 Max) | CUDA | 98.01 / 98.00 / 97.91 | 14 / 15 / 34 | `e2e785b607fbaec3` | 0.8.0-dev 29c720ea, `-dev CUDA0` |
| b60-vulkan | apple,t6001 (16-inch M1 Max) | Vulkan | measured; numbers kept private for now | — | `9892fa5ad6d01e4d` | 0.9.0-dev 8e28dfa8, `-dev Vulkan0` |
| m2max-apple-vulkan | apple,t6021 (14-inch M2 Max) | Vulkan | 42.75 / 42.60 / 42.62 | 323 / 316 / 431 | `66ce5ed30680e97e` | 0.9.0-dev cb44fdb, single-device host |

The B60 arm was measured under the same spec (round-identical token-id
digests); its numbers are kept private for now and live off-repo.

## Spark arm: not measurable

The DGX Spark (GB10, CUDA) arm is recorded as not measurable on this date: the
host's production Remnic serving holds about 98 GiB of the 121 GiB unified
memory, and the runner's `cudaSetDevice` fails with an out-of-memory status at
context init. The attempt record with the exact failure, memory numbers and
co-tenant inventory is `spark-cuda-attempt-1-oom.json`. The CUDA-13 build and
provisioning are complete and parked, so the arm can run when the memory
ruling changes.

## Audit notes

- The runner pins the GPU with `-dev <name>` (commit 31747a0). The first B60
  run predated the pin on the three-device host and was superseded by a pinned
  rerun; both receipts live off-repo.
- The M2 arm ran the pre-pin runner (a826210), sound on a single-device host;
  `pinned_dev` is absent from that receipt by design.
- Two RTX rows refused with exit 3 (missing ready file, then a CUDA runtime
  path the relink fixed) before the recorded run; refusals write no receipt.
- An unrelated lane CPU build overlapped the first 45 seconds of the first B60
  run at nice 19 and stopped before its timed rounds; noted for the trail.
- The two 16-inch M1 Max arms ran serialized under the guard's GPU lock on the
  same boot; the RTX arm served with `pinned_dev CUDA0`, the B60 arm with
  `pinned_dev Vulkan0`.

Per-arm receipts with full runs, settings, binary shas and device listings are
the JSON files beside this README, except the private B60 receipt. The GGUF
bytes were hash-verified on every host before serving.
