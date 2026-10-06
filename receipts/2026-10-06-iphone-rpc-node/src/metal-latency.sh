#!/usr/bin/env bash
# Phone Metal per-token cost: fixed part (1-layer trace), phone-alone decode with lm_head on the host
# vs on the phone, and on-phone (backend) sampling. Host OpenMP workers stay awake (ACTIVE).
set -uo pipefail
cd "$(dirname "$0")"
source env.sh
export OMP_WAIT_POLICY=ACTIVE
B=~/src/llama.cpp/build-linux-aligned/bin
M=~/models/Qwen3-1.7B-Q4_K_M.gguf
OT="^output\.weight=CPU"
OUT=~/src/phone-node/runs/metal-lat-$(date -u +%H%M)
mkdir -p "$OUT"
KLD=0 bash phase.sh metal "" | tail -1
uptime | tee "$OUT/uptime.txt"
for n in 1 28; do echo "== trace metal ngl $n, lm_head host"; bash trace-run.sh "lat$n" -ngl "$n" -ot "$OT" | grep -E "eval time|median"; done
echo "== trace metal ngl 99 (lm_head phone), backend sampling"
bash trace-run.sh lat99bs -ngl 99 -bs | grep -E "eval time|tokens=|median"
echo "== bench (tg64 -r 5)"
flock -w 600 /tmp/m1-gpu.lock "$B/llama-bench" -m "$M" -t 8 -p 128 -n 64 -r 5 -o md --rpc 127.0.0.1:50052 -ngl 28 -ot "$OT" \
  2>/dev/null | grep qwen | tee "$OUT/bench-ngl28-hostlmhead.md"
flock -w 600 /tmp/m1-gpu.lock "$B/llama-bench" -m "$M" -t 8 -p 128 -n 64 -r 5 -o md --rpc 127.0.0.1:50052 -ngl 99 \
  2>/dev/null | grep qwen | tee "$OUT/bench-ngl99.md"
echo "== greedy identity, phone alone"
greedy() { # $1 tag, rest: placement args
  local tag=$1; shift
  flock -w 600 /tmp/m1-gpu.lock "$B/llama-completion" -m "$M" -p "The capital of France is" -n 64 -c 4096 --temp 0 \
    --seed 42 -t 8 -no-cnv --rpc 127.0.0.1:50052 "$@" > "$OUT/$tag.txt" 2> "$OUT/$tag.log"
  echo "$tag $(grep -E ' eval time' "$OUT/$tag.log" | grep -oE '[0-9.]+ tokens per second') $(sha256sum < "$OUT/$tag.txt" | cut -c1-12)"
}
greedy ngl28-hostlmhead -ngl 28 -ot "$OT"
greedy ngl99 -ngl 99
greedy ngl99-bs -ngl 99 -bs
echo "OUT=$OUT"
