"""Milestone 1 vendor-OS solo baseline (CL02).

One host, its vendor OS (macOS or DGX OS), EXO running on that host alone.
This wraps the EXO pair runner with a one-node placement so the solo row
writes the same cl02-baseline/1 run.json as the pair row.

    python -m omarchy_cluster.baseline_solo --model Llama-3.1-8B-Q4 \
        --prompt-tokens 8192 --seed 42 --power-source nvidia-smi --out-dir OUT
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from omarchy_cluster import exo_pair


def solo_argv(argv: Sequence[str]) -> list[str]:
    """Pair-runner arguments for a one-node run; refuse a conflicting --min-nodes."""
    args = list(argv)
    for i, arg in enumerate(args):
        if arg == "--min-nodes" or arg.startswith("--min-nodes="):
            value = arg.split("=", 1)[1] if "=" in arg else (args[i + 1] if i + 1 < len(args) else "")
            if value != "1":
                raise SystemExit("baseline_solo runs one node; drop --min-nodes or set it to 1")
            return args
    return ["--min-nodes", "1", *args]


def main(argv: Sequence[str] | None = None) -> int:
    return exo_pair.main(solo_argv(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
