#!/usr/bin/env bash
# One traced greedy run through rpc-trace.py. Usage: trace-run.sh <tag> <llama-completion args...>
set -euo pipefail
cd "$(dirname "$0")"
tag=$1; shift
B=~/src/llama.cpp/build-linux-aligned/bin
M=${MODEL:-~/models/Qwen3-1.7B-Q4_K_M.gguf}
T=/tmp/rpc-trace-$tag.tsv
python3 rpc-trace.py 127.0.0.1:50060 127.0.0.1:50052 "$T" & proxy=$!
trap 'kill $proxy' EXIT
sleep 0.5
flock -w 900 /tmp/m1-gpu.lock "$B/llama-completion" -m "$M" -p "The capital of France is" -n 48 -c 4096 \
  --temp 0 --seed 42 -t 8 -no-cnv --rpc 127.0.0.1:50060 "$@" 2>&1 | grep -E " eval time"
python3 trace-summary.py "$T"
