# ClusterRun3 receipt — omarchy-cluster 2-node proof (2026-10-05)

## Commits (all pushed to origin/main, Forgejo)
- `7869c1d` Fix rank /prompt/wait polling to GET; allow macstudio as rank 0 via no_decode
- `37706c5` Regression test: rank polls /prompt/wait via GET
- `b1bb6cd` Delete quiet mode entirely (endpoint, CLI, flag guard, tests)
- `7d4ecb8` Parametrize jw16 route IP in 2-node runner (LAN/TB profiles)
- `e5641…`-series: rank_stop SIGKILL escalation; single-stage loopback ring; single-replica plain load; streamed-text accumulation; cmd_stop double-stop tolerance; runner engine-health wait

## Tests
20 passed (`python3 -m pytest -q`) after quiet-mode deletion (24 before, incl. 4 quiet tests removed with the feature).

## Quiet mode
Deleted: `/v1/quiet` GET+POST endpoints, `quiet_enabled`/`quiet_flag_path`, rank_start guard + 409 branch, `cmd_quiet` + `quiet` subparser, `_node_quiet` gate, `tests/test_quiet.py`. Commit `b1bb6cd`.

## Single-host baseline (macstudio, replica, greedy, max_tokens 64)
Prompt: "Write a haiku about shared memory."
Text: `Threads in code,\nInterlocked exchange,\nShared memory binds.`
sha256 `aa1061fadde8fb577c3881d1c6af82fafcff73f318ce708b18e1722baaf1480c`
16 completion tokens, decode 2744.7 ms, 8.15 tok/s (mlx-lm-reported; first-request warmup — standalone run measured 70.9 tok/s).
Saved at macstudio `~/.local/share/omarchy-cluster/runs/single-host-baseline.json`.

## 2-node (macstudio rank0 + jw16 rank1, DeepSeek-Coder-V2-Lite-Instruct-4bit)
- Measured routes (probe.measure_route, builtin-tcp, 6 s):
  - LAN en0:192.168.10.15 -> enu1:192.168.10.244 — 4.33/4.30 Gb/s, rtt 0.50 ms, decode-ok
  - TB en2:169.254.208.104 -> thunderbolt0:169.254.30.193 — 12.04/12.25 Gb/s, rtt 0.33 ms, decode-ok
  Profiles: macstudio `/tmp/links-lan.json`, `/tmp/links-tb.json`.
- Ring forms, both ranks sharded_load, gateway :8020 healthy, engine :8031 healthy.
- **BLOCKED: completion never returns.** Reproduced 3× (LAN, multiple code states). py-spy:
  rank0 spins at ~99% CPU inside stream_generate; rank1 idle at the poll loop
  (rank.py:69) — it never observes the pending prompt id 1 that rank0's engine
  holds (verified via direct GET /prompt/wait?after=0). Both mlx-lm 0.31.3 / mlx
  0.32.2. Suspects: rank1 GET fetch timing vs pending publication, or
  mlx-lm pipeline collective divergence. NOT a network problem (ring TCP established
  both directions; 4.3/12 Gb/s measured).
- TB profile run not reached (blocked behind the LAN hang).

## Infra bugs found & fixed along the way
- Runner fired the chat request at gateway-ready before the engine bound :8031 →
  instant 502 + set -e teardown. Now waits for engine /health.
- rank_stop SIGTERM-only orphaned ranks stuck in collectives (a stale rank held
  jw16's GPU lock for 16 min). Now escalates to SIGKILL after 3 s. Agent daemons
  restarted on both hosts to pick it up (jw16: systemd user unit).
- Single-stage serve ringed over a link-local IP and mlx ring cannot self-connect;
  replica now does a plain mlx_lm.load.

## State left behind
Both hosts cleaned (no rank/gateway processes, no :8020/:8031/:52100 listeners on
mac; jw16 ring ports free). jw16 gpu-turn queue untouched for other lanes.
