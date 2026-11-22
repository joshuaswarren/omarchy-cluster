# Milestone 1 EXO Pair Runner (macOS + DGX OS)

Reproducible runner for the Milestone 1 vendor-OS EXO pair baseline (`CL03` / `CL06`), matching EXO's 8,192-token Llama-3.1-8B comparison shape (8,192-token prompt, greedy decode, 32 generated tokens; author-reported `2.32 s` combined vs `6.42 s` Mac solo / `4.34 s` Spark solo) and emitting the `cl02-baseline/1` 8-metric JSON schema.

## Pinned Versions

- **EXO repository**: `https://github.com/exo-explore/exo`
- **EXO commit**: `a60b95698ca81737d367443674dd5563cbc9559b` (`v0.3.70`, `2026-11-06`)
- **Models**:
  - `Llama-3.1-8B-Q4`: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (`241a666dad6cb93c8ff213d39a7f34a36bf26db4`), 8,192-token prompt, 32 generated tokens
  - `Llama-3.1-8B-BF16`: `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16` (`f8311090f9ee47782b6f094984a20c856eb841d6`), 8,192-token prompt, 32 generated tokens
  - `Qwen3-1.7B-Q4`: `mlx-community/Qwen3-1.7B-4bit` (`3b1b1768f8f8cf8351c712464f906e86c2b8269e`), 43-token prompt, 64 generated tokens
- **Cluster namespace**: `omarchy-m1-exo` (ensures macOS and DGX OS nodes running the same source checkout join only each other)

## Telemetry Guarantees

- **Token-ID digest (`metrics.token_digest`)**: Strictly hashes ordered integer token IDs. Stock EXO `a60b95698ca8` does not emit `token_id` on `/bench/chat/completions` SSE chunks (`exo/api/adapters/chat_completions.py`), so live runs must either pass `--token-id-cmd` (backed by the pinned model tokenizer) or run a build that emits integer `token_id`. Otherwise the runner refuses to substitute text chunks, leaves `metrics.token_digest` empty, and records an explicit error in `metrics.errors`.
- **Peak memory (`metrics.ram_peak_mb`, `metrics.vram_peak_mb`)**: `ram_peak_mb` comes strictly from EXO's `generation_stats.peak_memory_usage` (never host-wide OS `nodeMemory` snapshots); `vram_peak_mb` comes strictly from in-request peak device sampling (`nvidia-smi`) or `generation_stats.peak_vram_usage`, otherwise `null`.
- **Power provenance (`metrics.power_source`)**: Only reports `nvidia-smi` or `powermetrics` when that host probe actually ran and produced samples during the request window; EXO `/state/nodeSystem` `sysPower` readings stay under `exo_node_system_power` in `exo-pair-details.json`.
- **Teardown**: Instance cleanup wraps creation, readiness await, and inference so a failed or timed-out `/instance/await` still deletes the created instance.

## Node Setup

Print exact setup commands for both vendor OSes:

```bash
bash receipts/milestone1-exo-pair/run-exo-pair.sh --print-setup
```

## Dry-Run Self-Test

```bash
bash receipts/milestone1-exo-pair/run-exo-pair.sh \
  --dry-run \
  --out-dir receipts/milestone1-exo-pair/dry-run
```

## Live Pair Run (`CL06`)

```bash
bash receipts/milestone1-exo-pair/run-exo-pair.sh \
  --exo-url http://<coordinator-ip>:52415 \
  --exo-repo ~/src/exo \
  --model Llama-3.1-8B-Q4 \
  --iface <lan-iface> \
  --power-source nvidia-smi \
  --token-id-cmd '<tokenizer-cmd>' \
  --out-dir receipts/milestone1-exo-pair/live-$(date -u +%Y%m%dT%H%M%SZ)
```

Outputs written to `--out-dir`:
- `run.json` — strictly conforming `cl02-baseline/1` 8-metric JSON object
- `exo-pair-details.json` — two-node placement, per-node metadata snapshots, EXO `/state/nodeSystem` power telemetry, pinned EXO commit, and raw `generation_stats`
