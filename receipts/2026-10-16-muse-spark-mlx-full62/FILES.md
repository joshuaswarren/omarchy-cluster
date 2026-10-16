# Files in this receipt

Run scripts and launch files of the full 62-layer run, plus the run results. Host names, LAN addresses, user names and home paths are replaced:
shell variables in `scripts/` (`STUDIO_IP`, `M2_IP`, `LAPTOP16_IP`, `LAPTOP13_IP`, `NODE_USER`, `*_HOST`, `*_NODE`, `LINUX_HOME` (home folder on the Linux nodes), `CAPTURE_TOOLS`, `RUN_ARTIFACT`) and
angle-bracket placeholders in `results/`. The local model id is `local/muse-2bit`; the node working folder is `muse-run`. No measured number was changed.

## scripts/

| File | Use |
|---|---|
| `pF-driver.sh SPLIT` | Driver on the control machine. Pre-checks free memory, sets the prune flag files, starts the Studio script, waits for the gateway, then limits, warm-up and 64-token captures. Attempt 2 used `24,12,12,12,2`. |
| `muse-pF-run.sh 62 SPLIT` | Runs on the Studio. Starts the second agent, then `omarchy_cluster.cli serve` with five ranks and `--no-fit-check` (the Studio share is above the planner's 90 percent free-memory rule). |
| `rankpy.sh` | Rank interpreter wrapper: paths, `MLX_DISABLE_COMPILE=1`, prune and heap flag files. |
| `deploy-prune.sh` | Copies the cluster tree and the pipeline-prune adapter to a node. |
| `pF-cont.sh`, `pF-more.sh` | By-hand continuation. The driver died after the gateway came up; these ran limits, warm-up, the 64-token captures, the repeat pass and the 128-token try. |
| `pN-stop.sh` | Stops the cluster and prints rank pids left on each node. |

## results/

| File | Content |
|---|---|
| `RESULT.md` | Placement, GB per rank, load time, swap, history of both attempts, what was not measured. |
| `RUN-README.md` | The run folder's own README, as written by the driver. |
| `rank-load-lines.txt`, `limits-vs-weights.txt` | Rank load lines (active and peak GB) and each node's MLX memory limit. |
| `capture-64tok/` | `summary.md` of the three 64-token captures (pass 1). |
| `capture-64tok-repeat/` | `summary.md` of the identical second pass. |

The 128-token capture that was running when the 13-inch GPU hung produced no output and is left out. Text status of every capture: not checked for coherence by the script; two independent reads judged the 64-token set good (see `RESULT.md`).
