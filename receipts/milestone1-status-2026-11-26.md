# Milestone 1 receipt inventory — CL04 / CL05 / CL06 — 2026-11-26

Per-milestone evidence audit for the README run matrix (SUP-191). Each row
says what exists, what is staged but never executed, and what was never run.
Host paths name the machine by role only; private-ops scratch paths are listed
as staged, not copied here.

## CL04 — Spark DGX OS solo baseline — ran, evidence on the Spark host only

Ran 2026-11-22 on the DGX Spark (solo EXO node, `omarchy-m1-exo` namespace,
stock EXO `a60b95698ca8`, GPU shared with the co-resident model server under
per-round activity logging). The `--min-nodes 1` path of the CL02 runner
produced `cl02-baseline/1` output.

| Artifact | State |
|---|---|
| `~/cl02-out/llama8b/run.json` + `exo-pair-details.json` (cold run) | exists on the Spark host |
| `~/cl02-out/llama8b-warm/repeat-{1,2,3}/run.json` | exists on the Spark host |
| `~/cl02-out/qwen17b/run.json` + `exo-pair-details.json` (cold run) | exists on the Spark host |
| `~/cl02-out/qwen17b-warm/repeat-{1,2,3}/run.json` | exists on the Spark host |
| `~/cl02-out/{llama8b,qwen17b}.cos401.{before,after}` — co-resident server memory/activity logs per round | exists on the Spark host |
| a copy under `receipts/` in this repo | missing |

Results (from the eight `run.json` files): Llama-3.1-8B-Q4 cold TTFT 49.07 s,
warm medians TTFT 14.96 s, decode 14.98 tok/s; Qwen3-1.7B-4bit cold TTFT
43.77 s, warm medians TTFT 4.26 s, decode 24.78 tok/s. Token digests
`3cd8e00e67750471` (Llama) and `6aef0ba54d8a5bc8` (Qwen3) are identical across
cold and warm runs — these are the parity targets for CL05/CL06.

## CL05 — macOS solo baseline — claimed, staged, never ran

The macOS solo baseline (jwm2 on macOS, or the Mac Studio fallback) never
executed: the 11-22 macOS window closed before CL04 landed, and no later
window opened. Claimed in the lane log by w9C (14:02 CST 2026-11-22); no
`-done` marker and no output.

| Artifact | State |
|---|---|
| `omp-studio-local:~/scratch/main/cl05/COMMANDS.md` — exact command sheet, pins matching CL04 (EXO `a60b95698ca8`, same models/revisions, seed 42, powermetrics probe) | staged off-repo (2026-11-22 00:54 CST, wAY) |
| `receipts/milestone1-jwm2-macos-solo/run.json` or any live output | missing — never ran |

To run it: the staged sheet needs a jwm2 macOS boot window; `--token-id-cmd`
copied from the CL04 runs; digest parity targets above.

## CL06 — vendor-OS EXO pair (jwm2 macOS + Spark DGX OS) — staged, one failed arm, never ran

Both sides were staged on 2026-11-22; the pair never formed. The Spark side
armed at 21:07 UTC and timed out at 23:06 UTC waiting for node 2 — jwm2
returned to Omarchy Linux before the macOS window opened, and no second arm
was attempted (Spark-side logs: `~/cl06/arm.log`, `exo.log`, `watchdog.log`).

| Artifact | State |
|---|---|
| `receipts/milestone1-exo-pair/run-exo-pair.sh` + `omarchy_cluster/exo_pair.py` (PR #134) | committed |
| `receipts/milestone1-exo-pair/dry-run/` (`run.json` schema-validated `cl02-baseline/1`, `dry-run.log`) | committed (offline self-test) |
| Spark-side stage: `~/cl06/{start-exo,watchdog,run-cl06,cl06-arm}.sh`, `~/cl06/dry/{run.json,exo-pair-details.json}` (on-host dry run), arm + watchdog logs | staged on the Spark host (w85) |
| lane-side copy of the Spark stage | staged off-repo (`~/scratch/w85/cl06/`) |
| `receipts/milestone1-exo-pair/live-*/run.json` | missing — never ran |

Blocker chain: CL06 needs jwm2 on macOS (the CL05 window) *and* the Spark
armed at the same time. EXO pairing is wire-gated — both nodes must match on
exo `a60b95698ca8`, mlx 0.32.4, and mlx-lm `0.32.3.dev16+g6b572ab1b`.

## Downstream reads

- CL09 (joint-request A/B) is sequenced after CL04 and uses the staged run
  sheet at `omp-studio-local:~/scratch/main/CL09-run-sheet.md`.
- CL13/CL14/CL15 inherit the missing CL05/CL06 baselines as comparators.
- The CL04 digests above are the token-parity targets every later milestone
  must reproduce.
