#!/bin/bash
# Runs ON the Studio. FULL RUN: all 62 layers of Muse Spark Open 2-bit (real weights, parts 1..63), five ranks:
# Studio rank 0 (the last layers + lm_head), M2 ranks 1 and 2, 16-inch rank 3, 13-inch rank 4 (layers 0-2 + embed). The router leg is stopped by its owner for this run and restarted by its owner afterwards.
# Usage: muse-pF-run.sh 62 SPLIT   (full model; SPLIT in rank order, for example 24,12,12,12,2)
set -u
N="${1:?N}"; SPLIT="${2:?split}"
D="$HOME/muse-run"
OUT="$D/muse-full$N-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/run.log"; }
CL="$D/mlx-venv/bin/python -m omarchy_cluster.cli"
export PYTHONPATH="$D/cluster-n"
# a second agent started from this ssh session: the launchd agent's Python 3.9 cannot open files on the external volume
if ! curl -sf -m 3 http://127.0.0.1:8026/v1/health > /dev/null; then
  nohup "$D/mlx-venv/bin/python" -m omarchy_cluster.agent --port 8026 --name ${STUDIO_NODE} >> "$D/agent-8026.log" 2>&1 < /dev/null &
  sleep 5
fi
curl -sf -m 3 http://127.0.0.1:8026/v1/health > /dev/null || { log "agent on 8026 is not up"; exit 6; }
log "agent 8026 up"
T0=$(date +%s)
$CL serve local/muse-2bit --nodes ${STUDIO_NODE},${M2_NODE},${M2_NODE},${LAPTOP16_NODE},${LAPTOP13_NODE} --split "$SPLIT" --no-fit-check \
  --ring-ip ${STUDIO_NODE}=${STUDIO_IP} --ring-ip ${M2_NODE}=${M2_IP} --ring-ip ${LAPTOP16_NODE}=${LAPTOP16_IP} --ring-ip ${LAPTOP13_NODE}=${LAPTOP13_IP} \
  --python-mac "$D/rankpy.sh" --python-linux "${LINUX_HOME}/muse-run/rankpy.sh" --port 18020 --engine-port 18031 --ctx 4096 \
  > "$OUT/serve.log" 2>&1 &
log "serve pid $!"
for _ in $(seq 1 600); do
  curl -sf -m 3 http://127.0.0.1:18020/health > /dev/null && break
  sleep 5
done
if ! curl -sf -m 3 http://127.0.0.1:18020/health > /dev/null; then
  log "gateway not healthy after $(( $(date +%s) - T0 )) s; see $OUT/serve.log"
  tail -n 20 "$OUT/serve.log" | tee -a "$OUT/run.log"
  tail -n 6 "$HOME/.local/share/omarchy-cluster/rank0.log" | tee -a "$OUT/run.log"
  exit 4
fi
log "gateway healthy after $(( $(date +%s) - T0 )) s (load time)"
log "ready for captures (engine left running)"
