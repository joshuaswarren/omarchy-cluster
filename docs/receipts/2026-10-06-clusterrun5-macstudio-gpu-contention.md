# ClusterRun5 — why macstudio Metal decode sometimes runs ~12x slower (2026-10-06)

**Verdict: the Claude desktop app's GPU process (`Claude Helper --type=gpu-process`, pid 3384, parent Claude.app pid 3287) on macstudio. Whenever it renders at 500+ GPU ms/s, MLX decode there runs at 8.4 ms/layer instead of 0.67. The cause is not oMLX, not the GPU clock and not the GPU power state.**

Observation only: no service was stopped, restarted or reconfigured.

## Method
One persistent MLX process on macstudio (`gpu_probe.py`, stock MLX 0.32.2)
timed 12 decode steps through all 27 layers of
DeepSeek-Coder-V2-Lite-4bit every ~2 s for 40 min (02:08:33-02:48:38Z, 1216
samples). Alongside it ran:
- per-process GPU time every 1 s from `ioreg -c AGXDeviceUserClient` AppUsage
  `accumulatedGPUTime`, which is what Activity Monitor's GPU column uses
  (`gpu_time_sampler.py`);
- `sudo powermetrics --samplers tasks,gpu_power` (GPU residency, frequency,
  power);
- `ps` top CPU and TCP connections to :8002/:8766/:11434/:1234 every 1 s;
- the oMLX server log.

powermetrics' per-process "GPU ms/s" column reads 0.00 for every process on
this macOS. AppUsage does not show MLX compute clients (oMLX, the probe) at
all, so they cannot be ranked there; graphics clients do appear.

## Result (lab `artifacts/ClusterRun5/gpu-contention/gpuobs3/`)
| Claude Helper GPU ms/s during the sample | samples | slow (>3 ms/layer) | median ms/layer |
|---|---|---|---|
| 0-150 | 928 | 0 (0 %) | 0.68 |
| 150-300 | 10 | 2 (20 %) | 0.75 |
| 300-500 | 22 | 15 (68 %) | 4.26 |
| 500+ | 256 | 256 (100 %) | 8.37 |

The per-minute timeline (`minute-timeline.txt`) shows the slow minutes are
the minutes when Claude Helper runs 500-710 ms/s; in quiet minutes it runs
74-88 ms/s and decode stays at 0.67-0.70.

## Ruled out
- **GPU clock/power state.** In slow and fast windows alike the GPU sat at
  1294-1296 MHz (top P-state) with 99-100 % active residency at 50-65 W.
  A warmup or keep-alive cannot help a GPU already pegged at full clock, so I
  did not test one.
- **oMLX :8002.** It is busy all day: 10-13 requests in flight from
  192.168.10.150 during every ClusterRun5 session, slow or fast, with
  `josh-voice-heretic-v4-q4` completions taking 364-1357 s. Its CPU mode lines
  up with a separate, smaller 2.2x step (0.6 vs 1.45 ms/layer, gpuobs1), not
  with the 12x state.
- **mlx-embed :8766, Ollama :11434, LM Studio.** Idle by CPU and GPU through
  every slow window.
- **The rank's idle gaps.** 0, 4 and 20 ms sleeps between decode steps
  (rank 0 waiting on the ring) changed nothing (p50 1.33-1.45 ms/layer).

## Other findings (observed, not acted on)
- The 1Password Autofill extension (pid 93041, running for 3.5 h+) used
  600-930 GPU ms/s throughout, about 1000 ms/s together with Claude Helper.
  It does not slow MLX: fast samples average 894 ms/s of 1Password. It does
  keep the GPU at full power.
- oMLX's log has 1734 `GET /admin/static/<scanner path> -> 404` lines since
  01:30Z (`.php`, `<script>alert('XSS')</script>`, `non-existent-839183143`).
  Something is scanning :8002, which listens on 0.0.0.0. The source IP is not
  logged.
- oMLX completions take 10-20 min each under the current load from
  192.168.10.150 (Proxmox MAC bc:24:11:ec:aa:1b).

## What this means for the cluster
The cluster's speed depends on what the macstudio desktop is rendering. The
measured split is protected (each node keeps its fastest-seen ms/layer), but
decode tok/s is not. INFERENCE, not verified because AppUsage was not
recorded before 02:08Z: the earlier "contended" ClusterRun5 sessions (rank 0
at 7-10 ms/layer) were Claude Helper bursts too, since the 8-10 ms/layer level
is the same. NOT TESTED (observe only): with Claude.app idle, closed or
minimised on macstudio, the effect should disappear. Whether to do that is
an operator choice; no code change follows from it.
