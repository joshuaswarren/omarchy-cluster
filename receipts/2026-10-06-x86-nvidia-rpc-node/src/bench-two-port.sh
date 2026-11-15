#!/usr/bin/env bash
# bench-two-port.sh — run on mbp16. mbp16 + BOTH x86-node RPC nodes:
# CUDA (Quadro M1200) on :50052 and CPU/RAM on :50053.
set -euo pipefail

LLAMA="$HOME/src/llama.cpp/build-linux/bin"
MODEL="$HOME/models/Qwen3-1.7B-Q4_K_M.gguf"
RPC="<x86-node>:50052,<x86-node>:50053"
OUT="$HOME/rpc-x86-bench"
PROMPT="The three primary colors are"
mkdir -p "$OUT"

exec 9>/tmp/m1-gpu.lock
flock -w 1800 9

echo "=== llama-bench: mbp16 + BOTH x86-node nodes ($RPC)" | tee "$OUT/bench-mbp16-plus-x86node-2port.txt"
"$LLAMA/llama-bench" -m "$MODEL" -ngl 99 --rpc "$RPC" 2>&1 | tee -a "$OUT/bench-mbp16-plus-x86node-2port.txt"

"$LLAMA/llama-cli" -m "$MODEL" -ngl 99 --rpc "$RPC" -st \
  -p "$PROMPT" -n 48 --temp 0 --seed 42 \
  >"$OUT/gen-2port.txt" 2>"$OUT/gen-2port.log" || true

echo "=== token match: 2-port split vs mbp16 alone (greedy, seed 42, n=48)"
if diff -u "$OUT/gen-alone.txt" "$OUT/gen-2port.txt" >"$OUT/gen-2port-diff.txt"; then
  echo "TOKENS MATCH (2-port): yes"
else
  echo "TOKENS MATCH (2-port): no (diff in gen-2port-diff.txt)"
  grep -m5 '^[+-]' "$OUT/gen-2port-diff.txt"
fi | tee "$OUT/token-match-2port.txt"
echo "=== done: $OUT"
