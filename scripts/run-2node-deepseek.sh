#!/usr/bin/env bash
# Run a complete two-node completion through the OpenAI-compatible gateway.
# rank 0: MacStudio (stock MLX); rank 1: jw16-linux (Omarchy MLX, gpu-turn).
set -euo pipefail
if [ "$#" -ne 0 ]; then
    printf 'usage: %s\n' "$0" >&2
    exit 2
fi

MODEL='mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit'
PROMPT='Write a haiku about shared memory.'
MAX_TOKENS=64
MAC_PY=/tmp/od-mac-venv313/bin/python
LINUX_PY=/home/joshuawarren/.local/share/mlx-omarchy/golden-main-venv/bin/python
REPO_DIR="${OMARCHY_CLUSTER_DIR:-/tmp/omarchy-cluster-run}"
JW16_IP="${JW16_IP:-169.254.30.193}"
JW16_SSH=(ssh -o IdentitiesOnly=yes -i "$HOME/.ssh/id_rsa_2025" -o HostKeyAlgorithms=ssh-ed25519 "joshuawarren@$JW16_IP")

run_cli() {
    PYTHONPATH="$REPO_DIR" "$MAC_PY" -c 'from omarchy_cluster import discover; discover.browse=lambda *a, **k: {}; discover.read_hostfile=lambda path=None: {"MacStudio":"127.0.0.1", "jw16-linux":"'"$JW16_IP"'"}; from omarchy_cluster.cli import main; main()' "$@"
}

run_dir="$HOME/.local/state/omarchy-cluster"
receipt="$HOME/.local/share/omarchy-cluster/runs/$(date -u +%Y%m%dT%H%M%SZ)"
links_path="$run_dir/links.json"
route_profile="${ROUTE_PROFILE:?set ROUTE_PROFILE to a measured links.json route profile}"
mkdir -p "$run_dir" "$receipt"
if [ -f "$links_path" ]; then
    cp "$links_path" "$receipt/links.before.json"
    had_links=1
else
    had_links=0
fi
restore_links() {
    if [ "$had_links" -eq 1 ]; then cp "$receipt/links.before.json" "$links_path"; else rm -f "$links_path"; fi
}
capture_rank_logs() {
    cp "$HOME/.local/share/omarchy-cluster/rank0.log" "$receipt/rank0.log" 2>/dev/null || true
    "${JW16_SSH[@]}" 'cat ~/.local/share/omarchy-cluster/rank1.log' >"$receipt/rank1.log" 2>&1 || true
}
cleanup_on_exit() {
    if [ -f "$run_dir/serve.json" ]; then run_cli stop >>"$receipt/stop.txt" 2>&1 || true; fi
    restore_links
}
trap 'capture_rank_logs; cleanup_on_exit' EXIT
cp "$route_profile" "$links_path"

printf 'route profile: %s\n' "$route_profile" | tee "$receipt/route.txt"
printf 'MacStudio health: ' | tee -a "$receipt/health.txt"
curl -fsS --max-time 5 http://127.0.0.1:8025/v1/health | tee -a "$receipt/health.txt"
printf '\nJW16 health: ' | tee -a "$receipt/health.txt"
"${JW16_SSH[@]}" 'curl -fsS --max-time 5 http://127.0.0.1:8025/v1/health' | tee -a "$receipt/health.txt"
printf '\n' | tee -a "$receipt/health.txt"
run_cli status --links "$route_profile" 2>&1 | tee "$receipt/status.txt"

run_cli serve "$MODEL" --stages 2 --no-decode MacStudio \
    --python-mac "$MAC_PY" --python-linux "$LINUX_PY" \
    --port 8020 --engine-port 8031 --gpu-turn 20 \
    >"$receipt/gateway.log" 2>&1 &
GW_PID=$!
printf 'gateway supervisor pid=%s\n' "$GW_PID" | tee "$receipt/supervisor.txt"

gateway_up=0
for i in $(seq 1 300); do
    if curl -fsS --max-time 2 http://127.0.0.1:8020/health >/dev/null 2>&1; then
        printf 'gateway ready after %ss\n' "$i" | tee "$receipt/gateway-ready.txt"
        gateway_up=1
        break
    fi
    sleep 1
done
if [ "$gateway_up" -ne 1 ]; then
    capture_rank_logs
    cat "$receipt/gateway.log" "$receipt/rank0.log" "$receipt/rank1.log" >&2
    exit 1
fi

curl -fsS --max-time 620 -X POST http://127.0.0.1:8020/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL" --arg p "$PROMPT" --argjson t "$MAX_TOKENS" '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$t,temperature:0}')" \
    -o "$receipt/chat.json" -w '%{time_total}' >"$receipt/request_seconds"
text=$(python3 -c "import json; print(json.load(open('$receipt/chat.json'))['choices'][0]['message']['content'])")
printf '%s' "$text" | shasum -a 256 | cut -d' ' -f1 >"$receipt/text_sha256"
python3 -c "import json; d=json.load(open('$receipt/chat.json')); print(json.dumps(d['timings']))" >"$receipt/timings"
cat "$receipt/chat.json" | python3 -m json.tool
printf 'completion SHA256: '; cat "$receipt/text_sha256"
printf 'elapsed seconds: '; cat "$receipt/request_seconds"; printf '\n'
capture_rank_logs

run_cli stop 2>&1 | tee "$receipt/stop.txt"
wait "$GW_PID" 2>/dev/null || true
if [ -f "$run_dir/serve.json" ]; then echo 'serve state remains' >&2; exit 1; fi
if lsof -nP -iTCP:8020 -sTCP:LISTEN; then echo ':8020 still listening' >&2; exit 1; else echo ':8020 clear'; fi
if lsof -nP -iTCP:8031 -sTCP:LISTEN; then echo ':8031 still listening' >&2; exit 1; else echo ':8031 clear'; fi
"${JW16_SSH[@]}" 'ps -ef | grep -E "omarchy_cluster|mlx_lm|gpu-turn" | grep -v grep || true; ss -tlnp | grep -E ":(52100|8031)\\b" || true' | tee "$receipt/jw16-stop-proof.txt"
printf 'private receipt: %s\n' "$receipt"
