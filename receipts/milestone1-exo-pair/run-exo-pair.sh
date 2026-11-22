#!/usr/bin/env bash
set -euo pipefail

# Milestone 1 EXO pair (macOS + DGX OS) runner wrapper (CL03).
# Delegates to omarchy_cluster.exo_pair (pure Python 3.9+ stdlib).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m omarchy_cluster.exo_pair "$@"
