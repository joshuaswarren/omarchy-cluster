#!/bin/bash
# Stop the cluster run: ranks and gateway through the cluster stop, the second Studio agent, the prune flag files; then list what is left
# on each node. Run this BEFORE the router owner restarts the router (it needs about 15 GB on the Studio). The bracket form in each pgrep pattern keeps
# the search from matching its own shell.
set -u
M2=(-o ConnectTimeout=10 -o HostName=${M2_IP} ${M2_HOST})
JW=(-o ConnectTimeout=10 -o HostName=${LAPTOP16_IP} ${LAPTOP16_HOST})
M1=(-o ConnectTimeout=10 -o HostName=${LAPTOP13_IP} ${LAPTOP13_HOST})
ST=(-o ConnectTimeout=10 ${NODE_USER}@${STUDIO_IP})
ssh "${ST[@]}" "cd ~/muse-run; PYTHONPATH=~/muse-run/cluster-n ~/muse-run/mlx-venv/bin/python -m omarchy_cluster.cli stop" 2>&1 | tail -8
ssh "${ST[@]}" 'for p in $(pgrep -f "omarchy_cluster[.]agent --port 8026"); do kill $p; done; sleep 1; echo "agent8026 left: $(pgrep -f "omarchy_cluster[.]agent --port 8026" | tr "\n" " ")"'
for t in ST M2 JW M1; do
  case $t in ST) g=("${ST[@]}") ;; M2) g=("${M2[@]}") ;; JW) g=("${JW[@]}") ;; M1) g=("${M1[@]}") ;; esac
  echo "== $t"
  ssh "${g[@]}" 'rm -f ~/muse-run/prune; left=$(pgrep -f "omarchy_cluster[.]rank --model" | tr "\n" " "); echo "rank pids left: [${left}]"'
done
echo "STOPPED $(date -u +%T)"
