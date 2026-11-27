# CL09 — joint-request A/B: Spark prefill → mac-desktop decode

**Verdict: joint mechanism proven end-to-end with bit-identical output, but joint
loses the A/B at this pairing.** The KV hop costs more wall time than the entire
solo request, so the milestone's "lower TTFT/total vs Mac solo" criterion is
falsified for Qwen3-1.7B @ 8,192 tokens on a ~1 GbE link — and the phases show it
cannot flip at this model size even with instant remote prefill.

## Arms

- **joint**: `omarchy-cluster serve --engine joint` on mac-desktop. Prefill on
  spark's llama-server (`--slot-save-path ~/.local/state/omarchy-cluster/kv`,
  port 8194, `-np 1`), slot save → KV file over the node agent (`/v1/kv`, LAN, TLS-pinned, `X-Cluster-Token`) → slot restore → decode on
  mac-desktop's llama-server (127.0.0.1:8132). Engine fronted on :8131.
- **solo**: same model, same llama.cpp build, same `-c 16384 -np 1` on
  mac-desktop, cold slot, `/completion` directly.

Both llama-servers build `1.1.0-dev (commit 171527522, tag b12774)` — version
match is required for slot save/restore; spark's binary was built for this run
from tag `b12774` with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`.

## Request

- `prompt-8192.txt`: exactly 8,192 tokens under the model tokenizer (verified
  via `/tokenize`), sha256
  `65cadbf91d47a3b578de712144ab56d0d2b5ef15c1bcfbbda9a69a99a29bb292`
- `n_predict = 64`, `temperature = 0` (greedy) both arms
- model: `Qwen3-1.7B-Q4_K_M.gguf` (1,282,439,584 B), same file content both hosts

## Results

| arm | run | TTFT | total | digest (sha256 of token ids) |
|---|---|---|---|---|
| joint | cold | 71.36 s | 71.36 s | `1c48810a…4bf43` |
| joint | warm | 10.88 s | 10.88 s | `1c48810a…4bf43` |
| solo | cold | ≈5.07 s | 6.19 s | `1c48810a…4bf43` |

**Identical token IDs across all runs** — the remote-KV handoff produces
bit-identical decode. `decode_prompt_n: 1` in both joint runs proves the decode
server did not re-evaluate the prompt; the restored slot KV did the work.
(`joint-warm` re-prefilled only 2 tokens because the prefill slot still held the
cached prefix; it is included to show the transport floor, not as a second
independent request.)

Phase breakdown, joint cold (ms): prefill 55,522 (8,192 tokens, CPU fallback —
see blocker) → slot save 5,993 (939,655,892 B KV) → transport 8,539 @ 880.31
Mbps → slot restore 251 → decode 627 @ 100.4 tok/s.

Phase breakdown, joint warm (ms): prefill 144 → save 1,043 → transport 8,390 @
895.93 Mbps → restore 207 → decode 631 @ 99.9 tok/s.

Solo cold (ms): prompt eval 5,050 (8,192 @ 1,622 tok/s, Metal) → decode 1,144 @
55.1 tok/s.

## Why joint loses here — and when it would win

The slot KV for 8,192 tokens of Qwen3-1.7B is **939.7 MB**. Over the measured
~890 Mbps LAN the transfer alone costs ~8.5 s — more than the whole solo request
(6.19 s). Joint wins only when

```
remote_prefill + slot_save + kv_bytes/LAN_rate + slot_restore < local_prompt_eval
```

For this model/prompt pair the right side is ~5 s and `kv_bytes/LAN_rate` is
~8.5 s, so joint cannot win on this LAN at any remote prefill speed. It would
flip with a much slower local prefill (bigger model on a weak decode host), a
much smaller KV (shorter prompts, small-KV architectures), or a faster link
(10 GbE ⇒ transfer ≈ 0.75 s ⇒ warm-path joint ≈ 2.4 s < 6.19 s solo).

## Blocker hit during setup (recorded, not worked around silently)

spark's CUDA context creation OOMs: remnic-core's resident llama-server holds
86.7 GiB of GPU memory (`nvidia-smi --query-compute-apps`; 262,144-token ctx ×8
slots) plus 5.3 GiB for a reranker — the unified-memory carve-out leaves nothing
for a new context (`NV_ERR_NO_MEMORY` in `kgrctxAllocMainCtxBuffer`, even at
`-ngl 0`). The joint cold run therefore prefilled on **CPU**
(`CUDA_VISIBLE_DEVICES=""`), at ~148 tok/s vs 1,622 tok/s on Metal — which is
why the cold TTFT is 71 s. The GPU-prefill number was not measured; per the
inequality above it would only move joint cold to ≈11 s, still behind solo.

Note: telemetry `memory.prefill.gpu_backend: "cuda"` names the binary's compiled
backend, not the live path — the run was CPU-only.

## Artifacts

- `joint-cold.json` — full `/generate` reply + telemetry, first joint request
- `joint-warm.json` — second joint request (prefix-cached prefill)
- `solo-cold.json` — solo `/completion` timings + digest
- `prompt-8192.txt` — the shared prompt

Lane wB2, 2026-11-27. Hosts: spark (GB10, `gn100-7f30`) prefill + agent :8025,
mac-desktop decode + joint engine :8131 / gateway :8130.
