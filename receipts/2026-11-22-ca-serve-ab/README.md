# C-a cross-machine serve A/B: four-arm medians (2026-11-22, Spark arm 2026-11-23)

One spec, one runner (`omarchy_cluster/serve_ab.py`, spec
`docs/serve-ab-ca.md`), one pinned GGUF
(`lmstudio-community/Qwen3-4B-Instruct-2507-GGUF` at revision
`4edb920b6f14e3b9284d4502a6485103d72cde05`, file
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, sha256 `8cdb57cb…82641fc`), four arms:
three measured fresh on 2026-11-22, the Spark arm on 2026-11-23 in a bounded
serving window. Three rounds of three fixed prompts, 128 greedy tokens per
run, TTFT and decode medians per prompt.

| Arm | Host chip | Backend | Decode tok/s (p1/p2/p3) | TTFT ms (p1/p2/p3) | run digest | llama-server |
| --- | --- | --- | --- | --- | --- | --- |
| rtx5060ti-cuda | apple,t6001 (16-inch M1 Max) | CUDA | 98.01 / 98.00 / 97.91 | 14 / 15 / 34 | `e2e785b607fbaec3` | 0.8.0-dev 29c720ea, `-dev CUDA0` |
| spark-cuda | NVIDIA DGX Spark, GB10 | CUDA | 70.29 / 69.76 / 67.57 | 21.1 / 19.6 / 29.3 | `d1fba08d5425949c` | 0.7.0-dev c7fc509, single-GPU host |
| m2max-apple-vulkan | apple,t6021 (14-inch M2 Max) | Vulkan | 42.75 / 42.60 / 42.62 | 323 / 316 / 431 | `66ce5ed30680e97e` | 0.9.0-dev cb44fdb, single-device host |
| b60-vulkan | apple,t6001 (16-inch M1 Max) | Vulkan | 8.25 / 8.23 / 8.24 | 579 / 596 / 950 | `9892fa5ad6d01e4d` | 0.9.0-dev 8e28dfa8, `-dev Vulkan0` |

Token-id digests are identical across rounds on every arm except where
GPU-side greedy nondeterminism produced per-round differences (p2/p3 on the
Spark arm); those digests are recorded as emitted in the JSON.

## Spark arm

Measured 2026-11-23 in a bounded window: the host's production serving pool
was drained through its documented drain/join controller (~87 GiB freed) and
rejoined afterward. The runner's `build-cuda` llama-server from the 2026-11-22
provisioning could not execute on GB10 (`no kernel image is available for
execution on the device` at every context size); the arm ran the host's
`build/bin/llama-server` at commit `c7fc509` — the same commit the production
short-preempt build uses — at ctx 4096. The earlier out-of-memory attempt
record is retained as `spark-cuda-attempt-1-oom.json`.

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
the JSON files beside this README. The GGUF bytes were hash-verified on every
host before serving.
