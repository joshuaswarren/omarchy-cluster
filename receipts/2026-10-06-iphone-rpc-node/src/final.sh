#!/usr/bin/env bash
# Final table in the m1max-linux quiet window. Model Qwen3-1.7B Q4_K_M; lm_head pinned to m1max-linux for every split.
set -uo pipefail
cd "$(dirname "$0")"
B=~/src/llama.cpp/build-linux-aligned/bin
M=~/models/Qwen3-1.7B-Q4_K_M.gguf
OUT=~/src/phone-node/runs/final-$(date -u +%H%M)
mkdir -p "$OUT"
OT="^output\.weight=CPU"
P="The capital of France is"
exec 9>/tmp/m1-gpu.lock; flock -w 600 9
uptime | tee "$OUT/uptime-start.txt"
bench() { "$B/llama-bench" -m "$M" -t 8 -p 128 -n 64 -r 3 -o md "$@" 2>/dev/null | grep qwen; }
ident() { # $1 tag, $2 ngl
  "$B/llama-completion" -m "$M" -p "$P" -n 64 -c 4096 --temp 0 --seed 42 -t 8 -no-cnv \
    --rpc 127.0.0.1:50052 -ngl "$2" -ot "$OT" > "$OUT/$1.txt" 2> "$OUT/$1.log"
  cmp -s "$OUT/ref.txt" "$OUT/$1.txt" && echo "$1 TOKENS-IDENTICAL" || echo "$1 TOKENS-DIVERGE $(cmp "$OUT/ref.txt" "$OUT/$1.txt")"
}
echo "== m1max-linux alone"; bench | tee "$OUT/bench-ref.md"
"$B/llama-completion" -m "$M" -p "$P" -n 64 -c 4096 --temp 0 --seed 42 -t 8 -no-cnv > "$OUT/ref.txt" 2> "$OUT/ref.log"
for mode in metal cpu; do
  flock -u 9
  KLD=0 bash phase.sh "$mode" "" | tail -1
  flock -w 600 9
  echo "== phone $mode"; bench --rpc 127.0.0.1:50052 -ngl 7,14,28 -ot "$OT" | tee "$OUT/bench-$mode.md"
  for n in 7 14 28; do ident "$mode-ngl$n" "$n"; done
  flock -u 9; bash trace-run.sh "final-$mode-14" -ngl 14 -ot "$OT" | grep -E "tokens=|median"; flock -w 600 9
done
flock -u 9
KLD=0 bash phase.sh metal "" | tail -1
flock -w 600 9
echo "== speculative (phone metal draft)"; bash spec-bench.sh "$OUT/spec"
uptime | tee "$OUT/uptime-end.txt"
echo "OUT=$OUT"
