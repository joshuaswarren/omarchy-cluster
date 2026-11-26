# The joint-request path (CL08)

Where one `/v1/chat/completions` goes when the service runs `--engine joint`, and where the KV
state crosses between hosts. Sources: `omarchy_cluster/gateway.py`, `planner.py`,
`joint_engine.py`, `agent.py`, `cli.py` (`_serve_joint`), at merge `4853d2ce` (PR #133).

## Sequence

```
client            gateway :8020          joint engine :8031        prefill llama-server      decode llama-server     node agents
  |  /v1/chat/completions |                    |                         |                          |                      |
  |---------------------->|                    |                         |                          |                      |
  |                       | auth, body check,  |                         |                          |                      |
  |                       | model name check,  |                         |                          |                      |
  |                       | Slots.acquire      |                         |                          |                      |
  |                       |  POST /generate    |                         |                          |                      |
  |                       |------------------->| _REQUEST_LOCK           |                          |                      |
  |                       |                    | 1. POST /completion     |                          |                      |
  |                       |                    |    n_predict=1 -------->| (warm slot KV)           |                      |
  |                       |                    | 2. POST /slots/N?save   |                          |                      |
  |                       |                    | ----------------------->| write slot-KV file       |                      |
  |                       |                    |                  to --slot-save-path dir           |                      |
  |                       |                    | 3. transport_kv() ==== THE KV-TRANSFER SEAM ================================>
  |                       |                    |    local copy           |                          |              GET/POST /v1/kv
  |                       |                    |    or agent pull/push   |                          |              (token-authed,
  |                       |                    | <========================================= basename-scoped, temp+rename =====|
  |                       |                    | 4. POST /slots/N?restore ---------------------------------------->| load KV
  |                       |                    | 5. POST /completion     |                          |                      |
  |                       |                    |    full prompt -------- | ------------------------>| prefix reuse (free)  |
  |                       |                    |    n_predict=N          |                          | every generated token
  |                       |                    |    computed HERE        |                          |                      |
  |                       |                    | POST /tokenize -------->|                          |                      |
  |                       |  reply (JSON or    | assemble telemetry      |                          |                      |
  |  <--------------------|  SSE pass-through) |                         |                          |                      |
```

1. **Gateway** (`gateway.py` `do_POST`): token auth, JSON body → the engine's `payload` contract
   (`prompt|messages`, `max_tokens`, `timing`, `raw`, `stream`, `greedy` or `sampling`), the
   asked-model check against engine `/status` (issue 71), then `Slots.acquire` admission — the
   ticket-queued permit pool whose limit can shift under parked waiters. Then `POST
   {engine}/generate`; a `stream` request becomes an SSE pass-through.

2. **Planner is a boot path, not a request path.** `planner.plan_placement`/`choose_split` run once
   inside `cli.py serve` to pick ranks and the layer split for the RPC-sharded engine. Per request
   they do nothing. `--engine joint` skips the planner entirely — `_serve_joint` takes
   `--prefill-url` and `--prefill-kv` verbatim, spawns the local decode llama-server
   (`--slot-save-path`, own GPU backend, no `-dev none`), checks the prefill server's `/props`
   model basename matches, then starts the joint engine and the same gateway.

3. **Joint engine** (`joint_engine.py` `joint_generate`, under the module `_REQUEST_LOCK` because
   the slot file and both servers' slot state are shared) runs five phases and records each one's
   ms in `telemetry.phases`:
   - **prefill** — `POST {prefill_url}/completion` with `n_predict=1`, `id_slot`, `cache_prompt`.
     The one decoded token is past the prompt prefix; decode-side prefix reuse drops it.
   - **save** — `POST {prefill_url}/slots/<slot>?action=save {filename}` serializes the slot's KV
     into that server's `--slot-save-path` dir.
   - **transfer — the KV-transfer seam.** `transport_kv(name, src, dst)` between `LocalKv` /
     `AgentKv` endpoints: `shutil.copyfile` when both dirs are local (`path: "loopback"`), else
     `client._agent_pull` (`GET /v1/kv?name=`) into a tempfile and/or `_agent_push` (`POST
     /v1/kv?name=`) — `path: "lan"`, possibly two hops. The agent side (`agent.py`) serves only
     basenames matching `[A-Za-z0-9._-]{1,128}` under `~/.local/state/omarchy-cluster/kv` (no
     traversal), writes via `mkstemp`+`rename` (a reader never sees half a file), and is under the
     same `_authed` cluster-token check as every agent route. Bytes and seconds feed
     `telemetry.kv_bytes`, `phases.transfer_ms`, `lan_mbps`.
   - **restore** — `POST {decode_url}/slots/<slot>?action=restore` loads the file on the decode
     host's server.
   - **decode** — `POST {decode_url}/completion` with the same full prompt on the same slot:
     llama-server reuses the restored prefix, so its prompt evaluation is free and every generated
     token is computed on the decode host.
   - Metrics tail: `/tokenize` on the decode server for `token_ids_sha256`; per-side memory probes
     (`default_probe` — agent facts remote, local facts here). Failures raise `JointError(phase)`
     and land in `telemetry.errors` with `error` on the reply.

4. **Reply** — gateway relays the joint engine's `/generate` JSON (or streams SSE), so a client
   sees the same OpenAI shape as the sharded engine, plus `telemetry`.

## The seam, for the card

- The KV moves exactly once per request, between `save` and `restore`, through
  `transport_kv` — not per layer and not continuously. That is the CL08 difference from arm B
  (RPC sharding), where layers sit on `rpc-server` endpoints and only activations cross during
  each step.
- Its bounds: both llama-servers need the same model and context and exclusive use of the
  configured slot (`-np 1`, or keep other clients off it); the module lock already serializes
  joint requests.
- Failure surface is plain: `JointError.phase` is one of `prefill|save|transport|restore|decode`
  and shows up verbatim in `telemetry.errors`.
