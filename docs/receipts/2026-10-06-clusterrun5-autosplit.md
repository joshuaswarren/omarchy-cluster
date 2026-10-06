# ClusterRun5 — automatic pipeline split from measured ms/layer (2026-10-06)

Same pair and prompt as `2026-10-06-clusterrun5-speed.md` (macstudio rank 0,
Metal; jw14m2-linux rank 1, Vulkan; DeepSeek-Coder-V2-Lite-Instruct-4bit,
"Write a haiku about shared memory.", greedy, max_tokens 64). M2 GPU held by a
gpu-turn ticket 01:26:26-01:46:07Z; every session `--gpu-turn 0` inside it.

## What changed (`edd0c57`)
- No `--split`: each rank times 1-token decode steps through 1 and 4 of the
  model's last layers in a child process (alternating, fastest of 8), keeps the
  fastest ms/layer its node has seen (`~/.local/state/omarchy-cluster/layer-ms.json`),
  all_gathers (ms/layer, layer cap from free memory), and every rank runs
  `planner.choose_split`: one layer per rank, the rest to the cheapest ranks up
  to their caps. `--split` still overrides.
- `place`/`serve` no longer print or pass the free-memory-proportional layer
  ranges (never what ran). The planner checks that each stage holds at least
  one layer and all stages together hold the model, and prints each cap.
- The even-split fallback is gone: counts are always explicit or measured.
- `python3 -m pytest -q`: 28 passed.

## Measurements
Per-node calibration, child process (12 starts: s9 r14-r20, s10 x2, runner x3):
macstudio 0.39-3.08 ms/layer, plus 9.72 once (r18, below); jw14m2-linux
7.77-8.42 ms/layer. Chosen split: `[26, 1]` (the best manual split) in 11 of
12 starts. Text `fc080ce9…` on both ranks in every `[26, 1]` job.

Interleaved auto vs manual `--split 26,1` sessions (lab
`artifacts/ClusterRun5/m2/sessions/s9-*`, `s10-*`). Both hosts have
load bursts that do not depend on the split: macstudio Metal at 7-10
ms/layer instead of 0.67, and the M2's 1 layer at 8-11 ms instead of
3.8-4.4. Sessions where both timing jobs show neither state:

| | plain-job tok/s | median |
|---|---|---|
| auto (r9 in-process calibration; r14, r16 child) | 53.12, 52.19, 52.07, 52.81, 51.13, 53.68 | 52.5 |
| manual 26,1 (r2, r3, r4, r5, r15, r19) | 27.48, 53.66, 48.74, 8.47, 29.48, 54.62, 9.51, 67.85, 53.69, 53.68, 53.09, 53.22, 49.89, 52.67, 49.54, 52.46 | 52.9 |

Auto is 0.7 % below manual 26,1: within 10 %. (24,3, uncontended r2: 30.1-30.6 tok/s.)
Production runner, no SPLIT, `GPU_TURN=0`: 4.16 / 33.82 / 34.23 tok/s, split
`[26, 1]`, text `fc080ce9…` on both ranks, peer-stop-proof 0 bytes
(macstudio `runs/20261006T014121Z`, `…014135Z`, `…014552Z`).

## Two problems found and fixed during the proof
1. Calibrating inside the rank process left rank 0's Metal decode about 12x
   slower: 9 of 13 auto sessions ran at 6.9-9.4 ms/layer, against 0 of 13
   interleaved manual sessions (r1-r13). Calibrating in a child process: 1 of 7
   auto sessions slow (r18, a burst that also slowed both neighbouring manual
   sessions) and 3 of 7 manual (r14-r20). The `edd0c57` message says "0 of 7";
   1 of 7 is the correct count. Root cause of the in-process after-effect not
   found; a standalone macstudio probe (`calib_aftereffect.py`) showed the same
   slow state with and without calibration.
2. r18: macstudio measured 9.72 ms/layer during a burst, above the M2's 7.97,
   and the split flipped to `[1, 26]` (4.6 tok/s; text `f9bc076f…` on both
   ranks). Fixed by keeping each node's fastest measured ms/layer
   (`_fastest_layer_ms`, unit test). Verified live: s10 created the files
   (macstudio 2.83, then 0.39; M2 8.18, then 7.87); the third runner start
   measured 0.89 on macstudio, used the stored 0.39 and chose `[26, 1]`.

## Limits
- Single-stream decode only: `choose_split` minimises one sequence's step
  time, so it fills the fastest node. Throughput with concurrent requests
  would want a balanced split.
- The fastest-seen memory never forgets: after a real slowdown (driver,
  thermals, new venv under the same python path), delete `layer-ms.json`.
- `--no-decode MacStudio` still means "rank 0" in practice: the planner's
  decode-tail ordering predates mlx-lm's pipeline, where rank 0 samples.
