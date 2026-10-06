#!/usr/bin/env bash
# bench-x86-nvidia-split.sh — run on jw16.
# Baseline (jw16 alone, CPU-only build) vs split (jw16 + 4 GB Maxwell CUDA RPC node),
# plus a fixed-prompt greedy token-match check between the two configs.
set -euo pipefail

LLAMA="$HOME/src/llama.cpp/build-linux/bin"
MODEL="$HOME/models/Qwen3-1.7B-Q4_K_M.gguf"
RPC=192.168.3.191:50052
OUT="$HOME/rpc-x86-bench"
PROMPT="The three primary colors are"
mkdir -p "$OUT"

exec 9>/tmp/m1-gpu.lock
flock -w 1800 9

echo "=== llama-bench: jw16 alone" | tee "$OUT/bench-jw16-alone.txt"
"$LLAMA/llama-bench" -m "$MODEL" -ngl 0 2>&1 | tee -a "$OUT/bench-jw16-alone.txt"

echo "=== llama-bench: jw16 + RPC CUDA node ($RPC)" | tee "$OUT/bench-jw16-plus-esper-rpc.txt"
"$LLAMA/llama-bench" -m "$MODEL" -ngl 99 --rpc "$RPC" 2>&1 | tee -a "$OUT/bench-jw16-plus-esper-rpc.txt"

for cfg in alone:0 "rpc:99:$RPC"; do
  name=${cfg%%:*}
  rest=${cfg#*:}
  ngl=${rest%%:*}
  val=${rest#*:}
  flag=()
  [[ "$ngl" != "$rest" ]] && flag=(--rpc "$val")
  "$LLAMA/llama-cli" -m "$MODEL" -ngl "$ngl" "${flag[@]}" -st \
    -p "$PROMPT" -n 48 --temp 0 --seed 42 \
    >"$OUT/gen-$name.txt" 2>"$OUT/gen-$name.log" || true
done

echo "=== token match (greedy, seed 42, n=48)"
if diff -u "$OUT/gen-alone.txt" "$OUT/gen-rpc.txt" >"$OUT/gen-token-diff.txt"; then
  echo "TOKENS MATCH: yes"
else
  echo "TOKENS MATCH: no (diff saved)"
  grep -m5 '^[+-]' "$OUT/gen-token-diff.txt"
fi | tee "$OUT/token-match.txt"
echo "=== done: $OUT"
