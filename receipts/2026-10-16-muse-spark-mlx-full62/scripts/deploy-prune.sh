#!/bin/bash
# Deploy the current omarchy-cluster tree and PR 199's adapter __init__ (prune off by default) to one node; backups kept beside the originals.
set -euo pipefail
node="$1"
case "$node" in
  studio) host=${NODE_USER}@${STUDIO_IP}; ssh_opts=(-o ConnectTimeout=15) ;;
  m2)     host=${M2_HOST}; ssh_opts=(-o ConnectTimeout=15 -o HostName=${M2_IP}) ;;
  ${LAPTOP16_HOST})   host=${LAPTOP16_HOST}; ssh_opts=(-o ConnectTimeout=15 -o HostName=${LAPTOP16_IP}) ;;
  *) echo "unknown node $node" >&2; exit 2 ;;
esac
work="$HOME/scratch/run"
tar -C "$work/cluster-main" -cf "$work/cluster-tree.tar" omarchy_cluster
scp -q -l 320000 "${ssh_opts[@]}" "$work/cluster-tree.tar" "$work/prune-init.py.new" "$work/deploy-n/rankpy.sh" "$host:muse-run/"
ssh "${ssh_opts[@]}" "$host" 'cd ~/muse-run && mkdir -p cluster-n && tar -C cluster-n -xf cluster-tree.tar 2>/dev/null; a=omarchy-mlx-serve/mlx_omarchy_muse/__init__.py; [ -f $a.orig ] || cp $a $a.orig; cp prune-init.py.new $a; chmod +x rankpy.sh; grep -c PIPELINE_PRUNE $a; grep -c "peak %.3f" cluster-n/omarchy_cluster/rank.py'
