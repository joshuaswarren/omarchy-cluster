# CL08 test inventory — loopback/mock coverage and gaps (SUP-172)

What CL08's phase separation needs from tests: a joint request runs prefill →
slot-KV save → KV transport → restore → decode across two llama-servers on
different hosts, and the reply carries the card's eight telemetry metrics.
The tests below are the whole loopback/mock suite — every test that runs
without cluster hardware (in-process HTTP fakes on 127.0.0.1, monkeypatched
subprocess/ssh, fixture files, tmp dirs). Counts are `def test` functions,
not parametrized expansions. Suite total: 532.

## CL08-facing tests

| file | tests | stands in for | what it covers |
|---|---|---|---|
| `tests/test_joint_engine.py` | 11 | two fake llama-servers (Spark prefill / Mac decode) + slot-save dirs | 5-phase happy path in order; all 8 telemetry keys; prefill error propagates; missing `--slot-save-path` named; missing restore file; LAN transport counted; requests serialized under the module lock; sampling fields reach both servers; dual `/health`; `serve joint` CLI spawns decode then engine; required-arg exits |
| `tests/test_agent_kv.py` | 12 | in-process agent `Handler` + tmp KV dir | `/v1/kv` basename validation (traversal refused), streamed PUT with rename, short-body refusal (400, nothing written), GET roundtrip, missing file 404, token required, `_agent_push` byte-count checks |
| `tests/test_llamacpp.py` | 37 | fake llama-server (`/completion`, `/health`, model files) | the plain llamacpp engine's request path the joint engine reuses: completion routing, health, sampling, spawn, `--slot-save-path` wiring |
| `tests/test_gateway.py` | 30 | fake engine + in-process gateway | the `/generate` contract the joint engine answers unchanged |

## Other loopback/mock tests (cluster mechanics, planner, ops)

| file | tests | scope |
|---|---|---|
| `tests/test_adapters.py` | 3 | `OMARCHY_MLX_SERVE` adapter registration before rank load |
| `tests/test_agent_auth.py` | 5 | agent token auth on the control endpoints |
| `tests/test_app.py` | 23 | app-level HTTP handler behavior |
| `tests/test_baseline_solo.py` | 2 | solo argv forces one node |
| `tests/test_bootstrap.py` | 18 | bootstrap (recorded subprocess/ssh fixtures) |
| `tests/test_cluster.py` | 18 | thread pinning — llama.cpp skips efficiency cores |
| `tests/test_concurrency.py` | 11 | DH-P7-03 serial queue, k>1 gate floor, `/status` count |
| `tests/test_discover_trust.py` | 11 | discovery never sends the cluster token to a non-member (#87) |
| `tests/test_drill.py` | 9 | CL16 node-loss drill in-process on localhost (CL16) |
| `tests/test_exo_pair.py` | 12 | EXO pair runner loopback dry-run (CL03) |
| `tests/test_heartbeat.py` | 4 | heartbeat bookkeeping (Mock hub) |
| `tests/test_hub.py` | 13 | hub endpoints (Mock) |
| `tests/test_hub_model_info.py` | 6 | hub model-info planner input |
| `tests/test_hybrid_kv.py` | 11 | KV reservation for hybrid checkpoints (attention-layer counting) |
| `tests/test_join.py` | 15 | `join` with a fake ssh (recorded fixture, DH-P7-01) |
| `tests/test_layer_timing_mlx.py` | 3 | config-only timing copy uses quantized operators (#96) |
| `tests/test_leave.py` | 19 | `leave` undoes a join; nameless node refused (#88, #89) |
| `tests/test_link_annotation.py` | 9 | link annotation on facts |
| `tests/test_link_labels.py` | 8 | probe link-media labels |
| `tests/test_live_env.py` | 2 | rank live-env file handling |
| `tests/test_model_path.py` | 3 | planner model-path resolution |
| `tests/test_models.py` | 33 | model list/pull endpoints (fake HTTP) |
| `tests/test_pkgbuild.py` | 5 | Arch package builds offline |
| `tests/test_place_hub.py` | 23 | hub placement |
| `tests/test_placement.py` | 21 | per-layer bytes from GGUF headers (GLM-5.3 UD-IQ1_S) |
| `tests/test_placement_disk.py` | 20 | planner counts model-volume free space (#81) |
| `tests/test_rank.py` | 19 | rank lifecycle (fake servers) |
| `tests/test_rank_sampling*.py` | 15 | sampling transport/wiring everywhere; MLX legs skip without mlx |
| `tests/test_repl.py` | 16 | repl/CLI |
| `tests/test_route_ip.py` | 7 | multi-address node picks a reachable route |
| `tests/test_serve_ab.py` | 10 | serve A/B arm runner, pure parts (no network) |
| `tests/test_serve_ranks.py` | 14 | `--nodes`/`--split` rank ordering, ring addresses |
| `tests/test_stop.py` | 13 | `stop` kills gateway + remote rank agents |
| `tests/test_sushi_engine.py` | 5 | sushi engine loopback |
| `tests/test_tls_pin.py` | 11 | token inside TLS only to a join-pinned key (#98) |
| `tests/test_two_node_proof_kit.py` | 25 | proof-kit preflight/unit legs |
| `tests/manual/two_rank_cpu_toy.py` | — | manual two-rank MLX ring hop, same machine (not collected) |

## Gaps against CL08's phase-separation needs

1. **No gateway→joint-engine hop test.** The joint engine fronts both
   llama-servers as its own HTTP process answering `/generate`; every joint
   test calls `joint_generate` in-process. No loopback test posts through the
   engine's own `do_POST`/`do_GET` surface — malformed JSON (400), non-`/generate`
   paths (404), `/status` body, `/health` 503 when a side is down — and none
   runs gateway→joint end-to-end.
2. **Real `/slots` semantics unproven by mocks.** The fakes copy files; the real
   llama-server needs `--slot-save-path`, does slot-id accounting, and makes the
   decode-side prompt eval free through prefix reuse of the one warm token
   (`PREFILL_N_PREDICT`). Request shapes are asserted; server semantics are
   CL09's live-run job — named here so nobody reads the mocks as proof.
3. **Decode-phase failure untested.** A prefill error and a transport error each
   propagate with named phases; a decode `/completion` failure after a
   successful restore — and the slot/KV state it orphans — has no mock test.
4. **Partial transport on the local arm untested.** `transport_kv` removes a
   truncated same-host copy on `OSError`; there is no mock exercising that path
   or a failed `_agent_push` leaving a short file on the decode host.
5. **KV size realism.** Fake KV is ~20 bytes; a real 8,192-token slot file is
   GB-scale through the `_CHUNK` agent paths. `test_agent_kv` streams correctly
   but no test feeds a multi-megabyte body or a mid-stream disconnect.
6. **Power metric has no seam.** Metric 8 is `power_watts: null,
   power_source: "none"` by construction — no probe interface exists for a
   mock or real sampler to fill it, and no test pins the null contract against
   future change.
7. **Memory-probe edges.** Per-node memory comes from `probes` callables; the
   mock pair covers the populated path but not a probe that raises or returns
   garbage — the telemetry shape on probe failure is unpinned.
8. **Model/context identity between the two servers.** The engine requires the
   same model and context on both sides plus exclusive slot use; nothing —
   mock or code — checks it. A `/props` compare at serve start would make it a
   testable seam; today a mismatched pair fails only as a wrong-token decode
   the fakes cannot represent.
9. **Cancel and timeout semantics absent.** `_post` has a fixed 600 s timeout
   and no request-cancel path; the lock serializes but a second request's wait
   is unbounded. No mock asserts timeout or cancel behavior, and a >600 s
   prefill is only reachable live.
10. **The `>32 GB` sharding OR-arm is unbuilt.** CL08 allowed phase separation
    OR large-model sharding; the joint engine was chosen, so no sharding-mode
    tests exist — correct as long as the arm stays unclaimed, recorded here so
    reopening it starts from zero coverage, not assumed coverage.

What the suite already guarantees: the phase chain runs in order, the eight
telemetry keys ship, both halves of the KV transport (local copy, agent
pull/push) move bytes correctly with basename and auth enforcement, and the
CLI spawns and gates the engine correctly. The gaps above are either
mock-unreachable by construction (2, 9) or real coverage holes a reviewer
should want closed (1, 3–8).
