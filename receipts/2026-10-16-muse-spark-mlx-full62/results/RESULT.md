Muse Spark Open 2-bit, 62 of 62 layers, 5 ranks on 4 Apple Silicon machines

Result of attempt 2, 2026-10-16. Real weights, every layer on an Apple Silicon GPU through MLX, no expert pruned, no 3-bit, no CPU-hosted layer.
Text status: 64-token answers read by reviewer 1 (verdict GOOD on 391, Canberra, is_prime; repeat pass byte-identical). 128/256-token captures NOT DONE. Peak memory and prefill tok/s NOT MEASURED.

## Placement (rank order)

| Rank | Machine | Layers | Weights (rank log) | MLX limit |
|---|---|---|---|---|
| 0 | M1 Ultra Mac Studio | 38-61 + lm_head (24) | 92.372 GB | 130.57 GB |
| 1 | 14-inch M2 Max (process A) | 26-37 (12) | 45.514 GB | 57.98 GB |
| 2 | 14-inch M2 Max (process B) | 14-25 (12) | 45.514 GB | 57.98 GB |
| 3 | 16-inch M1 Max | 2-13 (12) | 45.514 GB | 53.1 GB |
| 4 | 13-inch M1 | 0-1 + embedding (2) | 8.567 GB | 9.9 GB |

Sum 24 + 12 + 12 + 12 + 2 = 62 layers. Pipeline prune on, `MLX_DISABLE_COMPILE=1`, default kernels, raw `/v1/completions`, temperature 0.

## Measured (gateway timings, 159-token prompt)

| Prompt | Pass | TTFT | Decode | Tokens | Output |
|---|---|---|---|---|---|
| 17 x 23 | 1 | 26.506 s | 3.86 tok/s | 2 | 391 |
| 17 x 23 | 2 | 26.681 s | 3.95 tok/s | 2 | 391 |
| capital of Australia | 1 | 26.099 s | 2.23 tok/s | 8 | "The capital of Australia is Canberra." |
| capital of Australia | 2 | 26.215 s | 2.18 tok/s | 8 | same |
| is_prime | 1 | 25.927 s | 2.00 tok/s | 64 (cut at limit) | python def is_prime |
| is_prime | 2 | 25.981 s | 2.03 tok/s | 64 (cut at limit) | same |

Warm-up (11-token prompt, 8 tokens): prefill 11.835 s, 2.22 tok/s. Pass 2 output text equals pass 1 for all three prompts (md5 of output.txt below the banner).
Per-token split, gateway status at the capital prompt: rank 0 (Studio) 44.9 ms, ranks 1-4 combined 450.6 ms. The ring path through the Linux ranks sets the decode speed.

## Load and memory

- Launch 12:14:31Z (driver start), serve start 12:14:36Z, gateway healthy 12:25:19Z. The Studio run.log says healthy after 643 s (10 min 43 s) from serve start; driver start to healthy is 10 min 48 s. Do not quote 1481 s: that was the poll loop of the abandoned first attempt finding this gateway.
- Studio swap (monitor, 5 s samples, 12:14-12:26Z): 14.29 GB at start, 10.92 GB minimum, never above the start value. No guard stop in attempt 2.
- Studio free percent minimum 47, MemAvailable minimum 32.7 GB, internal free minimum 46.2 GB.
- Minimum MemAvailable during load: M2 6.11 GB, 16-inch 15.2 GB, 13-inch 5.26 GB.
- Guard (the memory-guard watch, router off): per-minute swap line 14,000 MB, +14,000 MB over baseline 13,624 MB, internal floor 20 GB, NVMe floor 25 GB.

## History of this attempt (plain)

1. Attempt 1 (line 8,000 MB/min) was stopped by the watch at 12:03:47Z: swap +10,173 MB in 20 s during the 24-layer Studio load. The ladder at 24 layers had jumped +6,969 MB. The other ranks died from the lost ring peer.
2. The memory-guard operator stopped the remaining tenants and raised the line to 14,000 MB/min. Attempt 2 loaded with no swap growth.
3. The driver script died after the gateway came up. I ran the rest by hand (`pF-cont.sh`: limits, warm-up, 64-token captures; `pF-more.sh`: repeat pass, 128/256 try). The captures are real and unedited.
4. 12:46:45Z, during the first 128-token capture, the 13-inch GPU hung: `Vulkan timeline counter failed to advance for 10000 ms (last observed=4245, target=4246)`. The 13-inch kernel log shows a display session restart (lab-session-guard, apple-dcp re-init) in the same second. The M2 and 16-inch ranks then logged `[ring] connection to a peer was lost`. One event; the display restart as cause is a correlation, not proven. Evidence: `failure-12-46Z/`.

## Not measured / open

128 and 256-token captures (the 128 canary folder is partial and must not be used), peak memory, prefill tok/s, reviewer 1's 5-prompt coherence check, any quality score.
The 13-inch holds 8.567 GB of 9.9 GB; the same chip hung once under load in this run and once in the N=56 run.

Files: `capture/` (3 captures at 64 tokens + partial 128), `capture-repeat64/`, `limits-vs-weights.txt`, `rank-load-lines.txt`, `gateway-status.json`, `capture.log`, terminal recordings inside each capture folder.
