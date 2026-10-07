#!/usr/bin/env bash
# Inside one gpu-turn ticket on omarchy-m1: loopback Vulkan rpc-server :50071, recording proxy :50072 -> :50071,
# then the timing configs (direct) and the proxied configs (command log). Everything on 127.0.0.1.
set -u
cd "$HOME/.local/share/omarchy-bench"
"$HOME/.local/bin/rpc-vulkan" -H 127.0.0.1 -p 50071 > /tmp/prof-rpc.log 2>&1 &
RPC=$!
python3 rpcproxy.py 50072 127.0.0.1 50071 /tmp/rpcprof.jsonl &
PX=$!
sleep 4
OUT=$HOME/.local/share/omarchy-bench/rpcprof-timing.jsonl timeout 200 python3 rpcprof.py \
  local-vulkan loopback-rpc-vulkan loopback-rpc-outhost
WARM=1 OUT=$HOME/.local/share/omarchy-bench/rpcprof-proxy.jsonl timeout 120 python3 rpcprof.py proxy-rpc proxy-rpc-outhost
kill "$PX" "$RPC" 2>/dev/null
wait "$PX" "$RPC" 2>/dev/null
echo "rpcprof done $(date -u +%T)"
