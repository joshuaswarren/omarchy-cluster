#!/usr/bin/env bash
# Runs inside a gpu-turn hold on omarchy-m1: loopback Vulkan rpc-server on :50071, then the hop-cost configs.
set -u
cd "$HOME/.local/share/omarchy-bench"
"$HOME/.local/bin/rpc-vulkan" -H 127.0.0.1 -p 50071 > /tmp/hop-loop-rpc.log 2>&1 &
LOOP=$!
sleep 4
timeout 780 python3 hopcost.py local-vulkan loopback-rpc-vulkan m2-wired mac-wired mac-thunderbolt \
  mac-wired-no-graph-reuse chain2-m2-mac chain3-m2-mac-loop
rc=$?
kill "$LOOP" 2>/dev/null
wait "$LOOP" 2>/dev/null
echo "hoprun rc=$rc $(date -u +%T)"
