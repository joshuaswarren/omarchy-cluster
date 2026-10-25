# Measured results

Every number here is greedy (temperature 0), on real machines, with the run recorded under
`receipts/`. Machines: the Studio is a Mac Studio M1 Ultra 128 GB on macOS (Metal); the
laptops are MacBook Pros on Omarchy M+ Linux (Vulkan through omarchy-mlx): a 16-inch M1 Max
64 GB, a 14-inch M2 Max 96 GB and a 13-inch M1 16 GB.

"Agree" means rank 1 logged the same text SHA-256 as the text the gateway returned.
"Same text" means two requests returned byte-identical text. All runs used the prompt
"Write a haiku about shared memory." unless a receipt says otherwise.

## Capacity: models that do not fit on one machine

| Model | Weights | Nodes | Layers per node | Decode | Prompt time |
|---|---|---|---|---|---|
| GLM-5.3 UD-IQ1_S (llama.cpp RPC, 79 layers), 2026-10-07 | 216.7 GB | 13-inch M1 (Omarchy, CPU, host) + x86 laptop Quadro M1200 4 GB (Omarchy, CUDA + CPU) + M1 Max (Omarchy, Vulkan) + Mac Studio (Metal) + M2 Max (Omarchy, CPU + Vulkan) | 11 + 1 + 8 + 14 + 15 + 8 + 22, from the layer ranges in the run record | 0.34 tok/s cold; median 0.39 tok/s (0.36 to 0.40, n = 5), 64 tokens each; texts differ, see [record](../receipts/2026-10-07-glm53-full-7device) | median 28.8 s for 20 tokens (25.1 to 34.4, n = 6, uncached) |
| GLM-5.3-Flash UD-IQ4_XS (llama.cpp RPC, 46 layers), 2026-10-07 | 156.8 GB | Mac Studio + M2 Max + M1 Max (Vulkan) + 13-inch M1 (CPU), 2.5 Gb Ethernet | 14 + 17 + 12 + 3 | 1.36 tok/s (first request, cold) and 1.54 tok/s, 64 tokens each, same text, [receipt](../receipts/2026-10-07-glm53-flash-4node) | 13.6 s for 20 tokens (first request; the second hit the prompt cache) |
| GLM-4.5-Air-4bit (glm4_moe, 46 layers), 2026-10-06 | 60.1 GB | Mac Studio + M1 Max, Thunderbolt | 31 + 15 | 0.87 and 0.86 tok/s, 64 tokens, agree; recorded in [docs/demo.cast](demo.cast) | 15.5 s, 13.2 s, 16.1 s |
| Muse Spark Open 816B-A42B 2-bit mixed MLX (62 layers), 2026-10-16 | ~237 GB | Mac Studio + M2 Max (two rank processes) + M1 Max + 13-inch M1, five MLX ranks | 24 + 12 + 12 + 12 + 2 | 2.00 to 2.23 tok/s over 64-token replies; TTFT about 26 s (159-token prompt), [receipt](../receipts/2026-10-16-muse-spark-mlx-full62) | not benchmarked beyond the captures |
| Muse Spark Open UD-IQ1_S GGUF (62 layers, llama.cpp RPC), 2026-10-16 | 175.6 GB | Mac Studio (gateway, CPU, mmap) + M2 Max (two servers) + M1 Max, 2.5 Gb Ethernet + Thunderbolt | 13 + 15 + 15 + 19 | 3.04 tok/s over a 107-token answer (3.17 tok/s over 300), [receipt](../receipts/2026-10-16-muse-spark-llamacpp-rpc) | first token 72.3 s cold, 25.5 to 26.8 s after |

Load times: GLM-5.3 2724 s; GLM-5.3-Flash 1055 s; Muse MLX full-62 10 min 48 s; Muse GGUF 857 s.
The GLM-4.5-Air recording shows 0.83 tok/s.

The GLM-5.3 layer map records the last block as 57-78. Its model has an MTP
layer at 78 that llama.cpp does not load. The counts above describe that
recorded map, not a claim that the MTP layer ran.

## Speed: a model that fits on one machine (DeepSeek-Coder-V2-Lite-Instruct-4bit, 27 layers)

In these runs, the split was slower than the machine that could hold the
model alone. These rows are reference points, not a universal speed claim.

| Machines | Link | Layers per node | Decode |
|---|---|---|---|
| Studio alone | | 27 | 68.2 and 67.6 tok/s |
| M1 Max alone | | 27 | 2.45 tok/s (earlier omarchy-mlx build) |
| M1 Max alone, 2026-10-11, current build | | 27 | median 15.73 tok/s (15.70 to 15.74, n = 3), 22-token reply; [receipt](../receipts/2026-10-11-single-host-baseline) |
| M1 13-inch 16 GB alone, 2026-10-11 | | 27 | does not load: out of GPU device memory ([receipt](../receipts/2026-10-11-single-host-baseline)) |
| M1 16 GB + M1 Max, 2026-10-06, cluster `c95baa0`, earlier omarchy-mlx build | 2.4 Gb/s wired | 1 + 26, chosen automatically | 1.79 and 1.80 tok/s, agree (no receipt directory) |
| Studio + M1 Max | Thunderbolt, 12 Gb/s | 26 + 1, chosen automatically | 46.4 and 48.0 tok/s, agree |
| Studio + M2 Max, 2026-10-05 | 2.2 Gb/s wired | 26 + 1, chosen automatically | median 52.5 tok/s (52.2 to 55.4), agree |
| same two machines | same | 14 + 13, a naive even split | 5.6 to 9.6 tok/s |
| same two machines | same | 24 + 3 | 28.6 to 31.0 tok/s |

## Where a token's time goes

Per token in the GLM-4.5-Air split: the laptop's 15 layers take about 1.1 s and the Studio's
31 layers about 45 ms; the omarchy-mlx Vulkan path runs a GLM-4.5-Air layer at about 74 ms
and Metal at about 1.5 ms, so the Omarchy machine sets the speed. In the Muse MLX full-62
run the Studio rank took 44.9 ms per token and the four laptop ranks combined 450.6 ms.

The ring exchange between machines cost under 0.5 ms per token on the measured links
(planner boundary estimate 0.342 ms/token on the 12 Gb/s Thunderbolt route in
[docs/demo.cast](demo.cast)); almost all of a token's time is the layers on the slower node,
which is why the planner puts as few layers there as memory allows. What one llama.cpp RPC
hop costs per token is measured in [receipts/2026-10-07-rpc-hop-cost](../receipts/2026-10-07-rpc-hop-cost)
and profiled in [receipts/2026-10-07-rpc-protocol-profile](../receipts/2026-10-07-rpc-protocol-profile).

## Determinism

Only rank 0 samples, so ranks agree on every token: a split on one backend gives the same
text as one machine. Two Omarchy Linux Macs produced byte-identical text to the M1 Max alone.
The text of a Metal plus Vulkan split is not always the text of a single Mac: hidden states
differ between the Metal and Vulkan paths by up to 1 bf16 ulp on the prompt we checked, which
can flip a near-tie token. In the runs above the Mac Studio alone wrote "Threads in code, ...",
the split wrote "Threads in harmony, ...". Each backend is deterministic run to run; the Muse
full-62 run's repeat pass was byte-identical.

## GPU contention

Other GPU work on a Mac (another model server, a browser, a chat app) can slow that node's
decode several-fold. The contention watch in `GET /status` reports it. During the 2026-10-05
runs a model server on the Mac Studio cut the small-model split from about 46 to about 22 tok/s
while it was busy.

## Driver and build provenance (2026-10-11 rows)

The Mesa driver build recorded `6543eeb7df7` (commit
`6543eeb7df702b6bcaa4e02a18425571305d4e05`) sat in the pinned ICD path on both laptops on
2026-10-11, with a byte-identical Honeykrisp ICD library, sha256
`3546bcafe8ed3995d091debcc2fb00803087e0074a88d7ebfe681071c58e57d0`, checked after the runs and
logged by the 13-inch M1 weight-load run. The runs' own logs record the pinned ICD, not the
library bytes. The runs used the cluster rank venv with omarchy-mlx wheel
`0.32.4.dev202610102049+93131d1b`, mlx-lm `0.31.4.dev132`, and the omarchy-cluster tree
`3c82f0a` or a later docs-only commit. Details:
[receipts/2026-10-11-driver-and-build](../receipts/2026-10-11-driver-and-build),
[receipts/2026-10-11-single-host-baseline](../receipts/2026-10-11-single-host-baseline).

## iPhone as a llama.cpp RPC node

Not part of the MLX pipeline; uses `serve MODEL --engine llamacpp` with the app from
[ios/](../ios/). An iPhone 15 Pro Max (A17 Pro, 8 GB, iOS 27) ran llama.cpp's `rpc-server`
as an app; an M1 Max on Omarchy sent model layers over USB; built, signed and installed from
Omarchy Linux with no Mac and no Xcode. Full data:
[receipts/2026-10-06-iphone-rpc-node](../receipts/2026-10-06-iphone-rpc-node).

The phone is a capacity node, not a speedup: each phone layer costs about 0.8 ms per token
against about 0.5 ms on the laptop CPU, so no split beats the laptop alone for a model the
laptop holds. Prefill is the phone's strength: with all 28 layers of Qwen3-1.7B on the phone
GPU it processes a 128-token prompt at 424 tok/s against the laptop CPU's 349.

Qwen3-1.7B Q4_K_M, 64-token decode, lm_head on the laptop:

| Setup | Decode tok/s |
|---|---:|
| Laptop alone (CPU) | 71.1 |
| Phone GPU, 7 of 28 layers on the phone | 35.5 |
| Phone GPU, 14 layers | 31.6 |
| Phone GPU, 14 layers, `OMP_WAIT_POLICY=ACTIVE` (separate run) | 45.0 |
| Phone GPU, 28 layers | 21.1 |
| Phone GPU, 28 layers, `OMP_WAIT_POLICY=ACTIVE` (separate run) | 35.1 |
| Phone CPU, 14 layers | 30.0 |

With the phone on its CPU, greedy output is byte-identical to the laptop alone at 7, 14 and
28 phone layers. On the phone GPU, output matches at 7 layers; at full offload the mean KL
divergence against the laptop is 0.008 and the top token agrees 96 percent of the time.
About 2.2 GB of layers ran well on the 8 GB phone; 4.4 GB pushed iOS into memory pressure.
Speculative decoding with the draft model on the phone was slower (6.6 tok/s) than with the
draft on the laptop (25.1 tok/s).

## Related receipts

Placement and KV sizing measurements (no GPU, planner math only):
[2026-10-07-measured-placement](../receipts/2026-10-07-measured-placement),
[2026-10-15-hybrid-kv-reservation](../receipts/2026-10-15-hybrid-kv-reservation),
[2026-10-15-swa-kv-reservation](../receipts/2026-10-15-swa-kv-reservation).
CPU thread defaults for the llama.cpp host:
[2026-10-07-perf-core-threads](../receipts/2026-10-07-perf-core-threads).
