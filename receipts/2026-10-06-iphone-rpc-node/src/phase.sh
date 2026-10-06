#!/usr/bin/env bash
# Relaunch the phone rpc-server with a backend (cpu|metal), then run KLD gate and traced runs.
# Usage: BUNDLE=<xtool bundle id, XTL-<team>.dev.omarchy.phoneinference> phase.sh <cpu|metal> "<ngl list>"
set -euo pipefail
cd "$(dirname "$0")"
mode=$1; ngls=$2
PM3=(~/venvs/pm3/bin/python -m pymobiledevice3)
dev=(); [[ $mode == cpu ]] && dev=(-d CPU)
timeout 60 "${PM3[@]}" developer dvt launch --kill-existing \
  "${BUNDLE:?set BUNDLE} -H 127.0.0.1 -p 50052 -t 4 ${dev[*]}" 2>&1 | grep -v WARN | tail -1
for _ in $(seq 1 30); do sleep 2; timeout 5 python3 rpc-ping.py 127.0.0.1:50052 20 2>/dev/null && break; done
timeout 30 ~/src/llama.cpp/build-linux-aligned/bin/llama-cli --rpc 127.0.0.1:50052 --list-devices 2>&1 | grep RPC0
if [[ ${KLD:-1} == 1 ]]; then
  flock -w 900 /tmp/m1-gpu.lock ~/src/llama.cpp/build-linux-aligned/bin/llama-perplexity \
    -m ~/models/Qwen3-1.7B-Q4_K_M.gguf --kl-divergence-base ~/src/phone-node/kld/base-m1max-linux.bin --kl-divergence \
    -c 512 --chunks 8 -t 8 --rpc 127.0.0.1:50052 -ngl 99 -f ~/src/llama.cpp/docs/build.md \
    > ~/src/phone-node/kld/$mode-ngl99.log 2>&1
  grep -E "Mean PPL\(Q\)/|Mean    KLD|99.0%   KLD|Same top p" ~/src/phone-node/kld/$mode-ngl99.log
fi
for n in $ngls; do echo "== $mode ngl $n"; bash trace-run.sh "$mode$n" -ngl "$n"; done
