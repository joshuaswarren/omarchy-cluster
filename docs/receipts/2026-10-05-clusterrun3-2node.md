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
- **BLOCKED (resolved by ClusterRun4 below): completion never returned.** Reproduced 3× (LAN, multiple code states). py-spy:
  rank0 spins at ~99% CPU inside stream_generate; rank1 idle at the poll loop
  (rank.py:69) — it never observes the pending prompt id 1 that rank0's engine
  holds (verified via direct GET /prompt/wait?after=0). Both mlx-lm 0.31.3 / mlx
  0.32.2. NOT a network problem (ring TCP established both directions; 4.3/12 Gb/s measured).
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

# ClusterRun4 — deadlock root cause, fix, live 2-node proof (2026-10-05/06)

## Root cause (two bugs)
1. **Protocol key mismatch (the deadlock).** The engine published
   `{"id", "messages", "max_tokens"}`; rank 1 accepted a job only if it carried
   `"prompt"` (rank.py:71 at abfbab7), so it skipped every job and slept. rank 0
   entered stream_generate alone and parked in the deepseek_v2 pipeline
   `recv_like` from rank 1. Bounded repro on abfbab7 (macstudio, both ranks on a
   loopback ring): `/generate` hung past 40 s; rank0 99.6% CPU, rank1 0.0%; the
   exact URL rank 1 polls (`/prompt/wait`) returned
   `{"id": 1, "messages": [...], "max_tokens": 64}`; macOS `sample` of rank 0:
   CPU stream thread in `mlx::core::distributed::ring::RingGroup::recv`, ring
   `SocketThread::worker` spinning in `__recvfrom` (2309/2433 samples); rank 1
   main thread in `time.sleep`. The URL was correct; this was not a reachability
   or start-order problem.
2. **mlx-lm 0.31.3 drops a pipeline layer.** `PipelineMixin.pipeline` starts
   rank r at `(size-r-1) * its own layer count`; with 27 layers on 2 ranks,
   rank 0 runs 14-26 and rank 1 runs 0-12, so layer 13 never runs. With bug 1
   fixed, the loopback 2-rank text was "Trees grow tall and wide,…" instead of
   the single-host "Threads in code,…". Upstream fix ml-explore/mlx-lm
   6c5d3a298 (#1816) is not in any release yet.

## Fix (omarchy-cluster main)
- `37f4b7c`: rank 0 publishes its exact prompt token ids as `"prompt"`; rank 1
  generates from them (no second tokenizer, so differing transformers versions
  cannot desync the ranks). Both ranks install the #1816 split before
  `sharded_load`. rank 1 logs its poll URL and each job's text sha256.
  Tests: real Engine publication -> HTTP -> rank 1 poll (fails on abfbab7),
  and every-layer-exactly-once split for 27/2, 27/4, 26/2, 5/4. 21 passed.
- `9c5c18f`: dead `write_hostfile`/`RING_BASE_PORT`/`_post_engine` deleted.
- `1c07d1d`, `d39224e`, `30212ed`: runner rank-1 python/peer/key overrides;
  one ready wait bounded by `ENGINE_WAIT_S` (gateway `/health` proxies the
  engine, so the old 300-iteration gateway wait aborted runs whose rank 1 was
  still queued in gpu-turn).
- macstudio loopback after the fix: 200 in 2.6 s, text "Threads in code,
  Interlocked exchange, Shared memory binds.", sha256 `aa1061fa…480c` = the
  single-host baseline, and rank 1 logged the same sha.
- `d9f1187`: `rank_stop` signals every pid in the rank's session. GNU
  `timeout` inside gpu-turn moves the rank into its own process group, so the
  old `killpg(wrapper)` left the M2 rank running after stop (seen live below).
  Regression test fails on the old agent, passes on the new; 22 passed.
  Agents on macstudio and jw14m2-linux redeployed and restarted.

## Live 2-node completion (macstudio rank 0 + jw14m2-linux rank 1, LAN)
jw16 was down (wedged test-kernel boot), so Main moved the live proof to the M2.
- Route en0 192.168.10.15 -> enu1 192.168.10.218, measured 2.163/2.168 Gb/s,
  rtt 0.41 ms. rank 1 python `/var/tmp/OmarchyDistributed-venv` (mlx-omarchy
  0.32.4.dev202610042317+b8af62c with the ring GPU transport, mlx-lm 0.31.3);
  rank 0 `/tmp/od-mac-venv313` (stock MLX 0.32.2). rank 1 waited ~10 min in the
  M2 gpu-turn FIFO while rank 0 sat in the ring accept.
- `POST :8020/v1/chat/completions` -> 200 in 4.14 s: 15 prompt / 17 completion
  tokens, 9.0 tok/s, text "Threads in harmony,\nShared memory binds,\nCreativity
  abounds." sha256 `f9bc076f…12b8`; rank 1 logged the same sha.
- Greedy text vs single host: differs from macstudio single host ("Threads in
  code,…", `aa1061fa…`) at token 3 and from the M2 single host ("Threads in
  harmony,\nShared memory binds,\nCode and data unite.") at line 3. The two
  single hosts already disagree with each other at token 3; the Metal+Metal
  loopback pipeline equals the macstudio single host exactly. Inference: Vulkan
  vs Metal numeric drift across the split, not a pipeline bug (not quantified).
- Stop: `:8020`/`:8031` clear on macstudio; the M2 rank survived the old stop
  (fixed in `d9f1187`, killed by hand ~20 s later). Final state: no rank,
  gateway, or :8020/:8031/:52100 listener on either host.
- Thunderbolt run: not done; it needs jw16 (down until ~14:00Z).
