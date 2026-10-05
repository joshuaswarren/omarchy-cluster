#!/usr/bin/env bash
# Run a complete 2-node omarchy-cluster pipeline completion through :8020.
#
# rank 0 = macstudio (/tmp/od-mac-venv313, stock MLX)
# rank 1 = jw14m2-linux (Omarchy wheel MLX 0.32.4.dev, via gpu-turn)
# model   = mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
#
# Prior manual output reference: c7506c8ff8e0c4bd4c3a21f2f5ede4405de7c103c198bfebccb8c74bbc3e8c01.
# Cookbook: `mlx-omarchy` branch `agent/omarchy-distributed`, commit 41d93a1ee.
# It states equality with an M2 single-host baseline but includes no SHA.
set -euo pipefail
if [ "$#" -ne 0 ]; then
    printf "usage: %s\n" "$0" >&2
    exit 2
fi

MODEL='mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit'
PROMPT='Write a haiku about shared memory.'
MAX_TOKENS=64
MAC_PY="/tmp/od-mac-venv313/bin/python"
M2_PY="/var/tmp/shared-omarchy-venv/bin/python"
REPO_DIR="${OMARCHY_CLUSTER_DIR:-$HOME/src/omarchy-cluster}"

run_dir="$HOME/.local/state/omarchy-cluster"
receipt="$HOME/.local/share/omarchy-cluster/runs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir" "$receipt"
cleanup_on_exit() {
    if [ -f "$run_dir/serve.json" ]; then
        cd "$REPO_DIR"
        PYTHONPATH=. omarchy-cluster stop >>"$receipt/stop.txt" 2>&1 || true
    fi
}
trap cleanup_on_exit EXIT

echo "=== Step 1: agents up on :8025 (verified before this script ran) ==="
ssh macstudio  'curl -fsS http://127.0.0.1:8025/v1/health'
echo
ssh jw14m2-linux 'curl -fsS http://127.0.0.1:8025/v1/health'
echo

echo "=== Step 2: status (nodes + links) ==="
cd "$REPO_DIR"
PYTHONPATH=. omarchy-cluster status --links "$run_dir/links.json" 2>&1 | tee "$receipt/status.txt"

echo "=== Step 3: serve (places + launches ranks + gateway on :8020) ==="
# The CLI runs in this foreground process; gateway is forked as a detached child.
# Rank 0 (engine host) will be on whichever node has the decode tail.
PYTHONPATH=. \
    omarchy-cluster serve "$MODEL" \
        --stages 2 \
        --no-decode jwm1 \
        --python-linux "$M2_PY" \
        --python-mac   "$MAC_PY" \
        --rank-pythonpath /home/joshuawarren/.local/share/omarchy-cluster/mlx-lm-pkgs \
        --port 8020 --engine-port 8031 \
    >"$receipt/gateway.log" 2>&1 &
GW_PID=$!
echo "gateway supervisor pid=$GW_PID"

# Wait for the gateway to actually accept
gateway_up=0
for i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8020/health >/dev/null 2>&1; then
        echo "gateway up after ${i}s"
        gateway_up=1
        break
    fi
    sleep 1
done
if [ "$gateway_up" -ne 1 ]; then
    echo "gateway failed to become ready on :8020" >&2
    exit 1
fi

echo "=== Step 4: greedy OpenAI completion through :8020 ==="
curl -fsS --max-time 620 -X POST http://127.0.0.1:8020/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL" --arg p "$PROMPT" --argjson t $MAX_TOKENS '{
        model:$m,
        messages:[{role:"user",content:$p}],
        max_tokens:$t,
        temperature:0
    }')" \
    -o "$receipt/chat.json" -w '%{time_total}' >"$receipt/request_seconds"
cat "$receipt/chat.json" | python3 -m json.tool
text=$(python3 -c "import json,sys; d=json.load(open('$receipt/chat.json')); print(d['choices'][0]['message']['content'])")
text_sha=$(printf '%s' "$text" | shasum -a 256 | cut -d' ' -f1)
timings_text=$(python3 -c "import json; d=json.load(open('$receipt/chat.json')); print(json.dumps(d['timings']))")
tokens_per_s=$(python3 -c "import json; print(json.load(open('$receipt/chat.json'))['timings']['tokens_per_s'])")
echo "completion text_sha256=$text_sha"
echo "tokens_per_s=$tokens_per_s"
printf '%s\n' "$text"
echo "timings=$timings_text"
echo "$text_sha" > "$receipt/text_sha256"
echo "$tokens_per_s" > "$receipt/tokens_per_s"
echo "$timings_text" > "$receipt/timings"

echo "=== Step 5: stop (kill gateway + both rank agents) ==="
PYTHONPATH=. omarchy-cluster stop 2>&1 | tee "$receipt/stop.txt"

# Wait for supervisor
wait $GW_PID 2>/dev/null || true

echo "=== Step 6: prove nothing is left ==="
echo "--- jw14m2-linux processes ---"
ssh jw14m2-linux 'ps -ef | grep -E "omarchy_cluster|gpu-turn" | grep -v grep | wc -l'
echo "--- macstudio processes ---"
ssh macstudio    'ps -ef | grep -E "omarchy_cluster|gateway" | grep -v grep | wc -l'
echo "--- gateway port 8020 ---"
ss -tln 2>/dev/null | grep ':8020' || echo "port :8020 not listening"
echo "--- engine port 8031 ---"
ss -tln 2>/dev/null | grep ':8031' || echo "port :8031 not listening"

echo "=== receipt dir contents ==="
ls -la "$receipt"