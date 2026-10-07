#!/usr/bin/env bash
# Inside one gpu-turn ticket on omarchy-m1: stock Vulkan rpc-server :50071 vs patched :50074, loopback, same session.
set -u
cd "$HOME/.local/share/omarchy-bench" || exit 1
HK_SYSMEM=57000000000 "$HOME/src/llama.cpp/build-vulkan/bin/ggml-rpc-server" -H 127.0.0.1 -p 50071 > /tmp/ab-stock.log 2>&1 &
S=$!
HK_SYSMEM=57000000000 "$HOME/src/llama.cpp-rpcasync/build-vk/bin/ggml-rpc-server" -H 127.0.0.1 -p 50074 > /tmp/ab-async.log 2>&1 &
A=$!
sleep 4
OUT=$HOME/.local/share/omarchy-bench/rpcasync-ab.jsonl timeout 330 python3 rpcprof.py \
  local-vulkan loopback-rpc-vulkan loopback-rpc-async loopback-rpc-vulkan loopback-rpc-async
kill "$S" "$A" 2>/dev/null; wait "$S" "$A" 2>/dev/null
# correctness: the greedy text of every patched request must equal the stock text
python3 - "$HOME/.local/share/omarchy-bench/rpcasync-ab.jsonl" <<'PY'
import hashlib, json, sys
seen = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    m = r["response"]["choices"][0]["message"]
    t = (m.get("reasoning_content") or "") + (m.get("content") or "")
    if not t:
        sys.exit("empty answer in %s run %s: correctness check would be vacuous" % (r["config"], r["run"]))
    seen.setdefault(r["config"], set()).add(hashlib.sha256(t.encode()).hexdigest()[:16])
for k, v in seen.items():
    print("digest %-22s %s" % (k, sorted(v)))
stock, patched = seen.get("loopback-rpc-vulkan", set()), seen.get("loopback-rpc-async", set())
ok = len(stock) == 1 and stock == patched
print("CORRECTNESS", "PASS: patched greedy text equals stock" if ok else "FAIL")
sys.exit(0 if ok else 4)
PY
echo "rpcasync-ab done $(date -u +%T)"
