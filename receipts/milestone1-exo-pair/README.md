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

## Node Setup

Print exact setup commands for both vendor OSes:

```bash
bash receipts/milestone1-exo-pair/run-exo-pair.sh --print-setup
```

- **macOS node (Apple Silicon, Metal MLX)**:
  ```bash
  git clone https://github.com/exo-explore/exo ~/src/exo
  git -C ~/src/exo checkout a60b95698ca81737d367443674dd5563cbc9559b
  cd ~/src/exo && uv sync --extra mlx
  uv run exo --namespace omarchy-m1-exo --api-port 52415
  ```
- **DGX Spark node (DGX OS / Ubuntu ARM64, CUDA 13)**:
  ```bash
  git clone https://github.com/exo-explore/exo ~/src/exo
  git -C ~/src/exo checkout a60b95698ca81737d367443674dd5563cbc9559b
  cd ~/src/exo && uv sync --extra mlx-cuda13
  uv run exo --namespace omarchy-m1-exo --api-port 52415
  ```

## Dry-Run Self-Test

Runs the full runner end-to-end against an in-process loopback mock EXO two-node server (`mac-m2max-macos` + `dgx-spark-dgxos`), checks two-node placement, streaming SSE metrics, teardown cleanup, and `cl02-baseline/1` schema validation:

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
  --power-source powermetrics \
  --out-dir receipts/milestone1-exo-pair/live-$(date -u +%Y%m%dT%H%M%SZ)
```

Outputs written to `--out-dir`:
- `run.json` — strictly conforming `cl02-baseline/1` 8-metric JSON object
- `exo-pair-details.json` — two-node placement, per-node metadata, pinned EXO commit, and raw `generation_stats`
