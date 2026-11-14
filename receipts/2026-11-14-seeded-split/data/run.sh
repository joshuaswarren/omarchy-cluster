#!/usr/bin/env bash
# #82 seeded-sampling run. Runs on the 16-inch M1 Max (node-a), gateway there. Arms: whole (placed on the 14-inch M2 Max), split over both.
# Starts serve only when neither Mac's GPU queue has a holder, so no rank holds a GPU idle while the other waits.
set -u
W=$HOME/scratch/w7G-p711
M=$W/model-q25-1p5b
L=$HOME/links.json
OUT=$W/out-$(date -u +%Y%m%dT%H%MZ)
mkdir -p "$OUT"
PROMPT='Write a short story opening about a lighthouse keeper.'
busy() { ~/bin/gpu-turn --status 2>/dev/null | grep -q 'pid='; }
rbusy() { ssh node-b '~/bin/gpu-turn --status 2>/dev/null | grep -q "pid="'; }
for arm in whole split; do
  echo "$(date -u +%FT%TZ) $arm: waiting for both GPU queues idle" | tee -a "$OUT/log.txt"
  for _ in $(seq 1 600); do
    if ! busy && ! rbusy; then break; fi
    sleep 3
  done
  extra=""
  [ $arm = split ] && extra="--stages 2 --ring-ip node-a=ring-ip-a --ring-ip node-b=ring-ip-b"
  echo "$(date -u +%FT%TZ) $arm: serve $extra" | tee -a "$OUT/log.txt"
  # shellcheck disable=SC2086
  omarchy-cluster serve "$M" --links "$L" --nodes node-a,node-b $extra \
    --python-linux "$W/rankpy.sh" --rank-pythonpath "$W/mlxlm-0.32.2" --gpu-turn 30 >"$OUT/$arm.serve.txt" 2>&1
  echo "$(date -u +%FT%TZ) $arm: serve rc=$?" | tee -a "$OUT/log.txt"
  ready=0
  for _ in $(seq 1 240); do
    curl -sf -m 5 http://127.0.0.1:8020/health >/dev/null && ready=1 && break
    sleep 5
  done
  echo "$(date -u +%FT%TZ) $arm: ready=$ready" | tee -a "$OUT/log.txt"
  if [ $ready = 1 ]; then
    # name:temperature:top_p:seed ; greedy first as the control
    for spec in greedy:0:1:none s7a:0.8:0.95:7 s7b:0.8:0.95:7 s8:0.8:0.95:8; do
      IFS=: read -r name temp topp seed <<<"$spec"
      seedfield=""
      [ "$seed" != none ] && seedfield=",\"seed\":$seed"
      curl -s -m 300 http://127.0.0.1:8020/v1/completions -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"max_tokens\":48,\"temperature\":$temp,\"top_p\":$topp$seedfield}" >"$OUT/$arm.$name.json"
      echo "$(date -u +%FT%TZ) $arm $name: $(wc -c <"$OUT/$arm.$name.json") bytes" | tee -a "$OUT/log.txt"
    done
  fi
  cp ~/.local/share/omarchy-cluster/rank0.log "$OUT/$arm.rank0.log" 2>/dev/null
  omarchy-cluster stop >"$OUT/$arm.stop.txt" 2>&1
  echo "$(date -u +%FT%TZ) $arm: stopped" | tee -a "$OUT/log.txt"
done
sha256sum "$W/src/omarchy_cluster/rank.py" | tee -a "$OUT/log.txt"
echo "$OUT"
