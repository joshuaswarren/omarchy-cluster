#!/usr/bin/env bash
# Runs ON omarchy-host (llama.cpp host): GLM-5.3 UD-IQ1_S over Mac Studio (Metal) + x86-laptop (CUDA + CPU) + omarchy-m1 (Vulkan)
# + M2 (CPU + Vulkan), first HOSTL layers on this host's CPU (mmapped). Device order = layer order.
# Env: MACGB (measured on the Mac right before the run), ESP, M2IP, budgets below.
set -euo pipefail
MODEL=${MODEL:-$HOME/models/GLM-5.3-UD-IQ1_S/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf}
ESP=${ESP:-10.10.0.20}; M2IP=${M2IP:-10.10.0.14}; M2NAME=${M2NAME:-omarchy-m2}
HOSTL=${HOSTL:-11}; ESGPU=${ESGPU:-2.75}; ESCPU=${ESCPU:-22}; M1GB=${M1GB:-38}; M2CPU=${M2CPU:-24}; M2GB=${M2GB:-62}
OC=$HOME/.local/bin/omarchy-cluster
fail() { echo "FAILED: $*" >&2; $OC stop 2>&1 | tail -6; exit 1; }
say() { printf '\n\033[1m$ %s\033[0m\n' "$*"; }
[ -n "${MACGB:-}" ] || { echo "set MACGB (measured on the Mac Studio right before the run)"; exit 2; }
say "omarchy-cluster status"
$OC status 2>&1 | sed -n 1,8p
say "omarchy-cluster serve $(basename "$MODEL") --engine llamacpp --host-layers $HOSTL --rpc-node x86-gpu=$ESGPU --rpc-node x86-cpu=$ESCPU --rpc-node omarchy-m1=$M1GB --rpc-node mac-ultra=$MACGB --rpc-node m2-cpu=$M2CPU --rpc-node $M2NAME=$M2GB"
t0=$(date +%s)
$OC serve "$MODEL" --engine llamacpp --llama-server "$HOME/src/llama.cpp/build-cpu/bin/llama-server" --host-layers "$HOSTL" \
  --rpc-node "$ESP:50052=$ESGPU" --rpc-node "$ESP:50053=$ESCPU" --rpc-node "omarchy-m1=$M1GB" \
  --rpc-node "mac-ultra=$MACGB" --rpc-node "$M2IP:50061=$M2CPU" --rpc-node "$M2NAME=$M2GB" \
  --rpc-binary "mac-ultra=~/.local/bin/rpc-nocache" \
  --rpc-binary "omarchy-m1=~/.local/bin/rpc-vulkan" \
  --rpc-binary "$M2NAME=~/.local/bin/rpc-vulkan" --ctx 2048 || fail "serve"
echo "loaded in $(( $(date +%s) - t0 )) s"
say "measure.py: 1 cold + 5 warm requests straight to llama-server, cache_prompt false"
"$HOME/measure.py" || fail "measure"
