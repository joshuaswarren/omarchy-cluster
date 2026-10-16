#!/bin/bash
# CT driver for the FULL 62-layer Muse run: five ranks on four Apple Silicon machines, GPU only, no CPU-hosted layers, nothing pruned.
# Usage: pF-driver.sh SPLIT   (rank order Studio, M2, M2, 16-inch, 13-inch; for example 24,12,12,12,2)
# The Studio script runs detached and is polled, so a held ssh channel cannot hang the driver. Ranks stay up after the captures; stop with pN-stop.sh.
set -u
SPLIT="${1:?split}"
M2=(-o ConnectTimeout=10 -o HostName=${M2_IP} ${M2_HOST})
JW=(-o ConnectTimeout=10 -o HostName=${LAPTOP16_IP} ${LAPTOP16_HOST})
M1=(-o ConnectTimeout=10 -o HostName=${LAPTOP13_IP} ${LAPTOP13_HOST})
ST=(-o ConnectTimeout=10 ${NODE_USER}@${STUDIO_IP})
LABEL="Muse Spark Open 2-bit, 62 of 62 layers, 5 ranks on 4 Apple Silicon machines; text NOT CHECKED for coherence; quality NOT MEASURED"
TS=$(date -u +%Y%m%dT%H%M%SZ)
ART="${RUN_ARTIFACT:-$HOME/muse-full62-run-$TS}"
mkdir -p "$ART"
cat > "$ART/README.md" <<EOF
$LABEL

Muse Spark Open (2-bit mixed build: experts 2-bit g128, attention and embedding 4-bit g64, lm_head 6-bit g64), all 62 layers, real weights, no pruning of
experts, no 3-bit, nothing substituted. Split in rank order $SPLIT: rank 0 Studio (the last layers + lm_head), ranks 1 and 2 two processes on the M2,
rank 3 the 16-inch, rank 4 the 13-inch (the first layers + the embedding). Every layer is on an Apple Silicon GPU through MLX; no layer is on a CPU.
Pipeline prune (PR 199) on, MLX_DISABLE_COMPILE=1, default kernels, raw completion route, greedy (the gateway accepts temperature 0 only).
TTFT is the gateway's prefill_ms and decode tokens per second is the gateway's tokens_per_s. One throw-away request runs before the captures.
Memory lines come from this run's rank logs only (files newer than the start marker). Files: studio-run.log, limits-vs-weights.txt, hk-sysmem.txt,
gateway-status.json, capture/, capture.log, rank-load-lines.txt. A killed driver leaves the prune flag file on the nodes; it is harmless.
EOF
each() {
  for t in ST M2 JW M1; do
    case $t in ST) g=("${ST[@]}") ;; M2) g=("${M2[@]}") ;; JW) g=("${JW[@]}") ;; M1) g=("${M1[@]}") ;; esac
    echo "== $t"; ssh "${g[@]}" "$1" 2>&1
  done
}
chk() {  # name, minimum kB, ssh args...
  local name="$1" min="$2"; shift 2
  local v; v=$(ssh "$@" "awk '/MemAvailable/{print \$2}' /proc/meminfo")
  echo "$name MemAvailable ${v} kB (need $min)"
  if [ -z "$v" ] || [ "$v" -lt "$min" ]; then echo "STOP: $name MemAvailable under the plan plus margin"; exit 5; fi
}
chk "M2" 95000000 "${M2[@]}"
chk "16-inch" 49500000 "${JW[@]}"
chk "13-inch" 12500000 "${M1[@]}"
{ echo "$LABEL"; echo "hk-sysmem files (bytes; a missing file means the default heap):"
  for t in M2 JW M1 ST; do
    case $t in ST) g=("${ST[@]}") ;; M2) g=("${M2[@]}") ;; JW) g=("${JW[@]}") ;; M1) g=("${M1[@]}") ;; esac
    echo "$t: $(ssh "${g[@]}" 'cat ~/muse-run/hk-sysmem 2>/dev/null || echo none')"
  done; } | tee "$ART/hk-sysmem.txt"
if ! grep -qE "^JW: (5[89]|[6-9][0-9])[0-9]{9}$" "$ART/hk-sysmem.txt" || ! grep -qE "^M1: (9|1[0-4])[0-9]{9}$" "$ART/hk-sysmem.txt"; then echo "STOP: hk-sysmem file missing or out of range on the 16-inch or 13-inch"; exit 7; fi
each "rm -f ~/muse-run/compile-on ~/muse-run/sub0 ~/muse-run/memlimit; touch ~/muse-run/prune ~/muse-run/pF-start; ls ~/muse-run/prune"
echo "PRUNE FLAGS SET, START MARKER TAKEN $(date -u +%T)"
scp -q $(dirname "$0")/muse-pF-run.sh "${ST[2]}:muse-run/muse-pF-run.sh"
ssh "${ST[@]}" "nohup bash ~/muse-run/muse-pF-run.sh 62 $SPLIT >> ~/muse-run/pF-run.out 2>&1 < /dev/null & echo started"
{ echo "$LABEL"; echo "Studio script started $(date -u +%T)"; } | tee "$ART/studio-run.log"
for _ in $(seq 1 720); do
  if ssh "${ST[@]}" 'curl -sf -m 3 http://127.0.0.1:18020/health > /dev/null'; then break; fi
  if ssh "${ST[@]}" 'tail -n 40 ~/muse-run/pF-run.out | grep -q "not healthy"'; then echo "STOP: the Studio script reports an unhealthy gateway"; break; fi
  sleep 5
done
ssh "${ST[@]}" 'tail -n 8 ~/muse-run/pF-run.out' 2>&1 | tee -a "$ART/studio-run.log"
if ! ssh "${ST[@]}" 'curl -sf -m 3 http://127.0.0.1:18020/health > /dev/null'; then
  echo "STOP: gateway not healthy; last lines of each rank log:"
  each 'for f in ~/.local/share/omarchy-cluster/rank*.log; do echo "-- $f"; tail -3 "$f" | cut -c1-200; done' | tee "$ART/rank-tails.txt"
  echo "ranks are left loaded; run pN-stop.sh"
  exit 4
fi
echo "GATEWAY HEALTHY $(date -u +%T)"
GW=http://${STUDIO_IP}:18020
{ echo "$LABEL"; echo "weights per rank (this run's rank logs) against each node's MLX memory limit:"; each "grep -h 'sharded_load done: local/muse-2bit ' ~/.local/share/omarchy-cluster/rank*.log | cut -c1-170; ~/muse-run/rankpy.sh ~/muse-run/memlimit-check.py 2>&1 | grep -o 'previous memory limit [0-9]* bytes [0-9.]* GB'"; } | tee "$ART/limits-vs-weights.txt"
{ echo "$LABEL"; grep -h "sharded_load done" "$ART/limits-vs-weights.txt"; } > "$ART/rank-load-lines.txt"
curl -s -m 10 "$GW/status" > "$ART/gateway-status.json"
curl -s -m 600 "$GW/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"local/muse-2bit","prompt":"<|begin_of_text|><|start|>user<|message|>hi<|eot|><|start|>assistant to=user<|message|>","raw":true,"max_tokens":8,"temperature":0}' \
  > "$ART/warmup-NOT-AN-ANSWER.json"
if [ ! -s "$ART/warmup-NOT-AN-ANSWER.json" ]; then echo "WARNING: the warm-up request returned nothing; ranks left loaded; look at the stacks before anything else" | tee -a "$ART/README.md"; exit 8; fi
echo "Guard for this run (attempt 2): Studio per-minute swap line 14,000 MB (approved 07:20 CDT); total swap line 32,000 MB; +14,000 MB over the baseline the watch read at its start; internal floor 20 GB; NVMe floor 25 GB. Attempt 1 (line 8,000) was stopped at 12:03:47Z at +10,173 MB in 20 s; the ladder at 24 layers jumped +6,969 MB. The 22-layer Studio share was the last one that ran inside the original 3,072 line; the 24-layer share ran only under a raised line." | tee -a "$ART/README.md"
echo "Studio swap at launch: $(ssh "${ST[@]}" 'sysctl -n vm.swapusage')" | tee -a "$ART/README.md"
PLACEMENT='M1 Ultra Mac Studio: layers 38-61 and lm_head (24 layers); 14-inch M2 Max: two ranks, layers 26-37 and 14-25 (12 + 12); 16-inch M1 Max: layers 2-13 (12); 13-inch M1: layers 0-1 and the embedding (2)'
CAPTURE_LENGTHS='64' bash ${CAPTURE_TOOLS}/capture-full62.sh "$GW" "$ART/capture" local/muse-2bit 5 4 "$PLACEMENT" 2>&1 | tail -60 | { echo "$LABEL"; cat; } | tee "$ART/capture.log"
echo "NEXT after reviewer 1 coherence read: CAPTURE_LENGTHS='128 256' capture-full62.sh $GW $ART/capture local/muse-2bit 5 4 '<placement>'" | tee -a "$ART/README.md"
echo "RANKS LEFT RUNNING for the coherence check; stop with pN-stop.sh" | tee -a "$ART/README.md"
echo "PF DRIVER DONE $(date -u +%T) art=$ART"
