# ClusterRun5 — contention watch, and whether to re-split under contention (2026-10-06)

Context: `2026-10-06-clusterrun5-macstudio-gpu-contention.md`. Bursts of the
Claude desktop app's GPU process slow macstudio Metal decode from 0.67 to
8-10 ms/layer. macstudio was observed only; nothing on it was changed.

## 1. Live check (`86d9fc0`, `eba911c`)
Rank 0 forces evals on one decode step in 8, and only after the first token.
It splits that step into its own compute (`rank0`) and its wait on ranks 1..
(`rank1+`). A part trips when it is above 3x its reference and at least 5 ms
over it for 4 sampled steps in a row. The reference is the lower of the
calibration (ms/layer x layers) and the fastest step seen; with `--split`,
rank 0 uses its node's fastest-seen ms/layer from `layer-ms.json`. A trip
logs `rank 0 contention: ...` and is kept in `GET /status` (engine :8031,
gateway :8020). /status holds the split, the calibrated ms/layer, per-part
last/ref/ratio/contended, and the last 20 events. Every `/generate` result
carries `watch`, and the runner saves `status.json` in its receipt (`18b8d26`).

Proof (macstudio two-rank loopback, `--split 26,1`, a 25 s synthetic GPU
contender, lab `loopback/loop-watch6`): 50 jobs.
- Jobs 0-25: rank 0 15.6-23.5 ms/step, ratio 1.2-2.3 against 10.08 ms
  (26 x 0.388).
- Jobs 26-35: 86-109 ms/step (8.5-10.8x). The watch tripped at 04:00:08Z
  (`rank0 101.3 ms/step, 10.1x its reference 10.1 ms for 4 sampled steps`),
  `contended: true` for jobs 27-35.
- Job 36 onward: cleared.
- Text `aa1061fa…` on both ranks in all 50 jobs.

A 26-1 run started inside contention without the `layer-ms.json` reference
(`loop-watch5`) never tripped, because its fastest step was itself slow;
`eba911c` fixes that and adds a test.

Bug found and fixed on the way: the first version forced rank 0's evals on
the chunked-prefill forward. There it ran an all_gather that rank 1 never
evaluates, and the ring hung (two loopback repros). Prefill forwards are no
longer forced in any mode. `python3 -m pytest -q`: 32 passed.

## 2. Re-split under contention: not justified, not coded
Per-layer costs measured in the same sessions, from rank 0's and rank 1's
step marks (lab `m2/sessions/`, 15 burst sessions, 01:07-01:44Z):
- macstudio Metal during a burst: 7.9-9.5 ms/layer (quiet: 0.54-0.70).
- jw14m2-linux Vulkan: about 8.2 ms per MoE layer (13/7/3/1-layer sweep);
  the M2 is not affected by macstudio bursts.
- Measured tok/s during bursts: `[26,1]` 4.1-5.4 (12 sessions), `[24,3]`
  4.2-4.9 (2), `[1,26]` 4.6 (1).
- Same-burst pair r18: `[26,1]` 5.00-5.15 vs `[1,26]` 4.61-4.64. Moving 25
  layers to the M2 was 9 % slower.

During a burst every split costs about 27 x 8.3 ms per step. Even at the
worst burst seen (9.5 ms/layer), putting all layers on the M2 would gain at
most 13 %. Bursts last 10-60 s, and a re-split reloads layers on both nodes
(calibration plus `sharded_load`, about 5-10 s). It would also have to switch
back to `[26,1]` the moment the burst ends: quiet, `[26,1]` runs about 52
tok/s against 16 for `[20,7]` and 9 for an even split. So no per-request
re-split. The watch makes bursts visible; the useful fix is on macstudio
(Claude.app), which is the operator's call.

Not done: a fresh M2 run of `[20,7]` during a burst. Main cancelled my M2
hold for higher-priority work. Claude Helper also read 0-18 GPU ms/s from
03:10 to 03:14Z, so a burst was not guaranteed to occur.
