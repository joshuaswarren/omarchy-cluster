# Muse Spark Open on MLX: full 62 layers on 4 Apple Silicon machines (2026-10-16)

Model: [joshuaswarren/Muse-Spark-Open-816B-A42B-MLX-2bit-mixed](https://huggingface.co/joshuaswarren/Muse-Spark-Open-816B-A42B-MLX-2bit-mixed)
(2-bit mixed MLX build: experts 2-bit g128, attention and embedding 4-bit g64, lm_head 6-bit g64; ~237 GB; converted from AMD's Quark FP8 release; Meta Muse license).

All 62 layers, real weights, every layer on an Apple Silicon GPU through MLX (omarchy-mlx on Omarchy Linux ranks plus the Mac Studio). No expert pruning, no 3-bit, no CPU-hosted layers, nothing substituted.

## Placement (5 MLX ranks, 4 machines, rank order)

| Rank | Machine | Layers | Weights (rank log) | MLX limit |
|---|---|---|---|---|
| 0 | M1 Ultra Mac Studio | 38-61 + lm_head (24) | 92.372 GB | 130.57 GB |
| 1 | 14-inch M2 Max, process A (Omarchy) | 26-37 (12) | 45.514 GB | 57.98 GB |
| 2 | 14-inch M2 Max, process B (Omarchy) | 14-25 (12) | 45.514 GB | 57.98 GB |
| 3 | 16-inch M1 Max (Omarchy) | 2-13 (12) | 45.514 GB | 53.1 GB |
| 4 | 13-inch M1 (Omarchy) | 0-1 + embedding (2) | 8.567 GB | 9.9 GB |

## Settings

- `MLX_OMARCHY_PIPELINE_PRUNE=1` (omarchy-mlx PR 199): each rank replaces the embedding / lm_head it never reads with a same-shape placeholder. Pipeline-stage tensor dropping only; no layer or expert is removed.
- `MLX_DISABLE_COMPILE=1`, default kernels, raw `/v1/completions`, temperature 0 (greedy).
- Gateway: `http://<studio>:18020`. Run scripts, launch files and result summaries are in `scripts/` and `results/`; see `FILES.md`.

## Measured (gateway timings, 159-token prompt, 64-token cap)

| Prompt | TTFT | Decode | Output |
|---|---|---|---|
| 17 x 23 | 26.5 s | 3.86 tok/s (2 tokens, not a speed number) | 391 |
| capital of Australia | 26.1 s | 2.23 tok/s | "The capital of Australia is Canberra." |
| is_prime | 25.9 s | 2.00 tok/s | correct `def is_prime(n)` (cut at 64 tokens) |

A repeat pass was byte-identical. Decode speed is **about 2.0-2.2 tok/s**, and TTFT is **about 26 s**.
Per token: Studio rank ~45 ms, the four laptop ranks combined ~450 ms, so the Linux ring path sets the speed.
Load: launched 12:14:31Z, gateway healthy 12:25:19Z (10 min 48 s).

Not yet measured: peak memory, prefill tok/s, 128/256-token captures. Known issue: the 13-inch GPU hung once at 12:46Z during a 128-token capture (Vulkan timeline timeout), and the ring dropped.
