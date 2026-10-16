#!/bin/bash
# Rank interpreter wrapper: puts the Muse adapter, the branch source, the 2-bit MLX tree (Linux) and mlx-lm main on the path, turns compile off.
D="$HOME/muse-run"
if [ "$(uname)" = Darwin ]; then
  PY="$D/mlx-venv/bin/python"
  EXTRA="$D/cluster-n:$D/omarchy-mlx-serve"
else
  PY="$HOME/.local/share/coreglass/venv-main/bin/python"
  EXTRA="$D/cluster-n:$D/omarchy-mlx-serve:$D/mlxlm-main"
  G3D="$HOME/scratch/mlx-run/g3e-tree"; [ -d "$G3D" ] || G3D="$HOME/scratch/mlx-run/g3d-tree"
  [ -d "$G3D" ] && [ ! -f "$D/no-g3d" ] && EXTRA="$G3D:$EXTRA"
fi
[ -f "$D/rank-cpu" ] && EXTRA="$D/cpu-site:$EXTRA"
[ -f "$D/prune" ] && export MLX_OMARCHY_PIPELINE_PRUNE=1
[ -f "$D/hk-sysmem" ] && export HK_SYSMEM="$(cat "$D/hk-sysmem")"
[ -f "$D/sub0" ] && export MLX_OMARCHY_GATHER_QMM_SUB=0
if [ -f "$D/memlimit" ]; then export W7G_MEMLIMIT="$(cat "$D/memlimit")"; EXTRA="$D/memlimit-site:$EXTRA"; fi
[ -f "$D/compile-on" ] || export MLX_DISABLE_COMPILE=1
export OMARCHY_MLX_SERVE="$D/omarchy-mlx-serve"
export PYTHONPATH="$EXTRA${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$@"
