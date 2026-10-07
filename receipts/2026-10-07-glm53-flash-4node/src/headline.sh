#!/usr/bin/env bash
# Runs ON omarchy-host (llama.cpp host): GLM-5.3-Flash over Mac Studio (Metal) + M2 + omarchy-m1 (Vulkan) + omarchy-host (CPU).
# Env: MODEL (first shard), M2NAME (discovered name of the M2), M2GB, M1GB, HOSTGB, MACGB (required).
set -euo pipefail
MODEL=${MODEL:-$HOME/models/GLM-5.3-Flash-UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf}
M2NAME=${M2NAME:-omarchy-m2}; M2GB=${M2GB:-62}; M1GB=${M1GB:-45}; HOSTGB=${HOSTGB:-8}
OC=$HOME/.local/bin/omarchy-cluster
fail() { echo "FAILED: $*" >&2; $OC stop 2>&1 | tail -6; exit 1; }
say() { printf '\n\033[1m$ %s\033[0m\n' "$*"; }
[ -n "${MACGB:-}" ] || { echo "set MACGB (measured on the Mac Studio right before the run)"; exit 2; }
say "omarchy-cluster status"
$OC status 2>&1 | sed -n 1,8p
say "omarchy-cluster serve $(basename "$MODEL") --engine llamacpp --rpc-node mac-ultra=$MACGB --rpc-node $M2NAME=$M2GB --rpc-node omarchy-m1=$M1GB --rpc-node omarchy-host=$HOSTGB"
t0=$(date +%s)
$OC serve "$MODEL" --engine llamacpp --llama-server "$HOME/src/llama.cpp/build-cpu/bin/llama-server" \
  --rpc-node "mac-ultra=$MACGB" --rpc-node "$M2NAME=$M2GB" --rpc-node "omarchy-m1=$M1GB" --rpc-node "omarchy-host=$HOSTGB" \
  --rpc-binary "mac-ultra=~/.local/bin/rpc-nocache" \
  --rpc-binary "$M2NAME=~/.local/bin/rpc-vulkan" \
  --rpc-binary "omarchy-m1=~/.local/bin/rpc-vulkan" \
  --rpc-binary "omarchy-host=$HOME/.local/bin/rpc-cpu" --ctx 4096 || fail "serve"
echo "loaded in $(( $(date +%s) - t0 )) s"
ask() {
  curl -sS --max-time 3600 http://127.0.0.1:8020/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Write a haiku about shared memory."}],"max_tokens":64,"temperature":0}' |
    python3 -c 'import json,sys,hashlib; d=json.load(sys.stdin)
if "error" in d: sys.exit("ERROR %s" % d)
t=d["choices"][0]["message"]["content"]; tm=d["timings"]
print(t); print("\n%d tokens, %.2f tok/s decode, prompt %.1f s, text sha256 %s" % (d["usage"]["completion_tokens"], tm["tokens_per_s"], tm["prefill_ms"]/1000, hashlib.sha256(t.encode()).hexdigest()[:16]))'
}
say "curl -s localhost:8020/v1/chat/completions -d '{... \"Write a haiku about shared memory.\", \"temperature\":0}'"
ask || fail "request"
say "curl -s localhost:8020/v1/chat/completions   # same request again"
ask || fail "request"
say "curl -s localhost:8020/status"
curl -s --max-time 10 http://127.0.0.1:8020/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k: d.get(k) for k in ("engine","tensor_split")}))' || fail "status"
say "omarchy-cluster stop"
$OC stop 2>&1 | tail -6
