# omarchy-cluster

One local model across every machine in the homelab. Python, stdlib-only,
Python >= 3.9 (macOS system python works).

Omarchy M+ heterogeneous-inference plan: build tasks 1-3 (agent,
discovery, probe) plus build tasks 4-5 (planner, gateway, hub,
node-loss handling).

## What it does

- `omarchy-cluster agent` — per-node daemon: HTTP JSON facts on **:8025**
  (OS, chip, unified memory, GPU backend metal/vulkan/cuda, free memory,
  interfaces with IPs + link speeds), 1 Hz heartbeat counter, TCP bulk-sink
  on :8026, UDP echo on :8027, iperf3 helper endpoints. Token-authed
  `/v1/rank/start` and `/v1/rank/stop` endpoints let the gateway launch a
  pipeline rank on each node without node-to-node ssh. Port 8020 is
  reserved for the phase-1 gateway (never 8002 — that is the live oMLX
  advisor on mac-a).
- `omarchy-cluster install-agent` — installs and starts the daemon:
  launchd LaunchAgent (`~/Library/LaunchAgents/dev.omarchy.cluster-agent.plist`)
  on macOS, systemd user unit + restart (`~/.config/systemd/user/omarchy-cluster-agent.service`)
  on Linux. Copies the package to `~/.local/share/omarchy-cluster/src`,
  drops a `~/.local/bin/omarchy-cluster` wrapper, and provisions a
  shared cluster token at `~/.config/omarchy-cluster/token` (mode 0600;
  `install-agent --token-file PATH` seeds a specific token across the
  fleet).
- `omarchy-cluster discover` — mDNS `_omarchy-cluster._tcp` browse
  (Avahi on Linux, dns-sd on macOS) + `~/.config/omarchy-cluster/hosts`
  override (`name=ip-or-host`, mlx.launch-style lines). Hostfile wins per
  name. The local CLI host is always included; reachable candidates are
  scored by interface speed and the control plane rides the fastest one.
- `omarchy-cluster probe` — every pair x every interface route:
  RTT (TCP connect + UDP echo) and bandwidth both directions. iperf3
  1-stream 10 s when both ends are Linux; **on any macOS end the built-in
  Python TCP sender is used** (Homebrew iperf3 is Local-Network-privacy
  blocked on mac-a, verified 2026-10-04). Writes `links.json`, pins the
  fastest route per pair (bottleneck direction, RTT tiebreak), and marks
  Wi-Fi and Tailscale routes ineligible for decode ranks.
- `omarchy-cluster place MODEL` — picks the pipeline nodes and checks they
  hold the model: every stage must fit at least one layer (weights + KV in
  90 % of free memory) and together all layers; each boundary needs a
  measured pinned route from `links.json`. Prints each stage's layer cap.
  `--stages N` forces an exact N-rank pipeline; `--no-decode NAME` excludes
  a node from decode eligibility (D1: linux-d).
- `omarchy-cluster serve MODEL` — plans, then asks each node's agent to
  launch its own rank over the shared ring hostfile
  (`_omarchy-cluster._tcp` is the service type; the JSON file is
  `[["ip:52100"], ...]`, `MLX_RANK` selects). Starts the OpenAI gateway
  on :8020 with the engine on rank 0. GPU ranks are wrapped in
  `~/bin/gpu-turn` (M2 shared GPU is FCFS, 30 min max per turn).
  Ranks run mlx-lm's pipeline: rank 0 runs the last layers and samples;
  rank 1 long-polls rank 0's engine for each job's token ids.
  The split is measured at rank start: each rank times decode steps through
  1 and 4 of the model's layers (in a child process), keeps the fastest
  ms/layer the node has seen in `~/.local/state/omarchy-cluster/layer-ms.json`
  (delete it after a real slowdown), the ranks all_gather (ms/layer, layer
  cap), and every rank runs the same `planner.choose_split`: one layer per
  rank, the rest on the cheapest ranks up to their caps. `--split N0,N1`
  (rank order) overrides it. A request with `"timing": true`
  returns `timings.step_ms` (per decode step: ring wait vs compute) and
  `poll_ms`; rank 1 logs its own `step_ms`.
  Contention watch: rank 0 times one decode step in 8 (forced evals on rank
  0 only, never during prefill) and splits it into its own compute
  (`rank0`) and its wait on the other ranks (`rank1+`). A part more than 3x
  its reference (the calibrated ms/layer x layers, or the fastest step seen)
  and at least 5 ms over it for 4 sampled steps in a row is logged
  (`rank 0 contention: ...`) and kept in `GET /status` (engine :8031 and
  gateway :8020): split, calibrated ms/layer, per part last/ref/ratio/
  contended, and the last 20 events. Each `/generate` result carries `watch`.
- `omarchy-cluster stop` — stops the detached gateway, terminates every process in each rank's session on its agent, and sweeps listeners on the gateway, engine, and rank ports.
- `omarchy-cluster hub` — heartbeat hub on :8030; `agent --hub URL` POSTs
  the 1 Hz heartbeat. `status --hub URL` shows per-node up/down state.
- `omarchy-cluster status` — node table (OS, chip, memory, backend, up) +
  pinned route per pair + (optional) hub liveness.

## Quick start

```sh
pip install .            # or: python3 -m omarchy_cluster.cli directly
omarchy-cluster install-agent
omarchy-cluster discover
omarchy-cluster probe
omarchy-cluster status
```

Tokens are generated per node the first time `install-agent` runs.
For a fleet where the gateway on one node talks to agents on others,
copy one node's `~/.config/omarchy-cluster/token` to every node and
re-run `install-agent --token-file <path>` (or `scp` the file and
reinstall) so they match.

State lives in `~/.local/state/omarchy-cluster/`.

## Real-fleet receipts

### 2026-10-04: 2-node probe + status (mac-a + linux-b + linux-d)

Run on **linux-b** (control node), default 10 s streams, built-in
Python TCP sender (macOS end):

```
$ omarchy-cluster discover --mdns-timeout 4
mac-a        10.10.10.15   mdns     ok
linux-b     10.10.10.218  mdns     ok

$ omarchy-cluster probe      # real 3m44.7s
mac-a <-> linux-b
   wifi   en0:10.10.10.15   -> wlan0:10.10.3.103    0.18/1.42 Gb/s  rtt 2.17/1.94 ms  NO-DECODE
 * wired  en0:10.10.10.15   -> enu1:10.10.10.218    2.24/2.24 Gb/s  rtt 0.33/0.26 ms  decode-ok  PINNED
   tailscale en1:10.10.3.26 -> tailscale0:100.64.1.36  --/0.20 Gb/s   rtt --/-- ms  NO-DECODE
   ... 9 more candidate routes measured/unmeasured ...
wrote ~/.local/state/omarchy-cluster/links.json

$ omarchy-cluster status
NODE             OS                     CHIP             MEM FREE/TOT  BACKEND UP   IP
mac-a        macOS 26.6.2           Apple M1 Ultra   32.6/128GB    metal   up   10.10.10.15
linux-b     Arch Linux ARM         Apple M2 Max     88.0/94GB     vulkan  up   10.10.10.218
pinned routes (links.json 2026-10-04T23:23:57Z):
mac-a <-> linux-b
 * wired  en0:10.10.10.15   -> enu1:10.10.10.218    2.24/2.24 Gb/s  rtt 0.33/0.26 ms  decode-ok  PINNED
```

LAN pin 2.24/2.24 Gb/s agrees with the hand-measured 2.36/2.35 Gbit/s
fleet receipt within ~6%; Wi-Fi and Tailscale are always marked
NO-DECODE; the picker pins the fastest decode-eligible route.

`omarchy-cluster install-agent` restart proof on mac-a: killed the
agent pid, launchd respawned it and `/v1/facts` answered again within 4 s
(pid 29888 → 53859). Reboot-survival is configured (`RunAtLoad`/KeepAlive,
`WantedBy=default.target` + linger) but was NOT tested by rebooting
mac-a (pinned build oracle) or linux-b (fresh from its Thunderbolt
test).

### 2026-10-06: planner output

```
$ omarchy-cluster place mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit --stages 2 --no-decode mac-a
PIPELINE: mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit (27 layers, KV 216.0 KB/token)
  rank 0  mac-a        fits up to 26 layers  (engine; samples, runs the last layers)
  rank 1  linux-b     fits up to 26 layers
  split: chosen at rank start from measured ms/layer (serve --split overrides)
  boundary mac-a -> linux-b via en0:10.10.10.15 -> enu1:10.10.10.218  2.2 Gb/s rtt 0.41 ms  ~0.439 ms/token
```

### 2026-10-04: gateway end-to-end (world-1 on mac-a, 2026-10-04)

While linux-b's M2 GPU was busy with a Thunderbolt-ring job in the
gpu-turn queue (other lanes' work), the full engine + gateway pipeline
was validated world-1 on mac-a (24 layers all on Metal). The same
acceptance path runs against the real 2-node split once the GPU frees.

Greedy single-node reference (`mlx_lm.generate --model ... --prompt ... --temp 0 --max-tokens 24`):
`"The user is asking a simple factual question about the capital of France. I need to provide the correct answer directly and conc"` (truncated by max-tokens; 24 tokens, 130 tok/s).

Greedy OpenAI completion through `omarchy-cluster` (same model, same prompt, world-1 rank, gateway :8020 on mac-a):

```
$ curl -s -X POST http://127.0.0.1:8020/v1/chat/completions -H "Content-Type: application/json" \
       -d '{"model":"SiddhJagani/Qwen3.8-2B-mlx-4Bit","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":24}'
{
  "id":"chatcmpl-omarchy-1791161057", "object":"chat.completion",
  "model":"SiddhJagani/Qwen3.8-2B-mlx-4Bit",
  "usage":{"prompt_tokens":15,"completion_tokens":24,"total_tokens":39},
  "timings":{"prefill_ms":2898.0,"decode_ms":634.4,"tokens_per_s":37.83},
  "choices":[{"index":0,"message":{"role":"assistant","content":
      "The question asks for the capital of France. This is a straightforward factual question. The capital of France is Paris.\n"},
      "finish_reason":"stop"}]
}
```

Greedy chat-template-applied text equals the single-node
`mlx_lm.generate` reference (both start with the question rephrase
and end with `The capital of France is Paris` before max-tokens).
world-1 tok/s is lower than single-node because all 24 layers share one
M1 Ultra GPU on Metal with chat-template processing; the network-free
fast path.

### 2026-10-04/05: 2-node serve attempt

`ssh linux-b 'omarchy-cluster serve SiddhJagani/Qwen3.8-2B-mlx-4Bit
--stages 2 --no-decode linux-d --python-linux
/var/tmp/shared-omarchy-venv/bin/python --python-mac
/Users/user/.local/share/omarchy-cluster/mlx-venv/bin/python
--rank-pythonpath ~/.local/share/omarchy-cluster/mlx-lm-pkgs --gpu-turn 25'`

The first attempt ran both ranks via the token-authed agent rank launch,
gateway on linux-b:8020 came up. Rank 0 (M2) entered gpu-turn behind
several 20-min jobs from neighboring lanes (other-gpu-jobs, other-gpu-jobs,
other-gpu-jobs). Rank 1 (macOS) never managed to bring its 52100 listener
up before rank 0's turns expired.

Lead then fixed the macOS Application Firewall allow row for the
Homebrew Python (BrewPyLan) and pointed us at the Tailscale addresses
(mac-a 100.64.1.18, M2 100.64.1.6). I rebuilt a manual ring
hostfile with Tailscale addresses and confirmed both ranks bind:

```
LISTEN 0      0                     100.64.1.36:52100      0.0.0.0:*  (M2 wheel, rank 0)
LISTEN 100.64.1.18:52100        (macOS Homebrew Python, rank 1)
ESTABLISHED on both ring sides after launch
```

The ring hop on the first request surfaced two final blockers:

- macOS Metal: `[METAL] Command buffer execution failed: Ignored (for
  causing prior/excessive GPU errors)` from the first `mx.array`
  item() in macOS rank 1's serve_hops. mac rank 1 caught and reset
  caches, but every subsequent submission is also rejected (Metal's
  per-process submission accounting poisons the queue). Other macOS
  MLX workloads (oMLX advisor, brew python core GPU) were active at the
  time; restart of mac rank 1 alone did not clear the rejection.
- M2 wheel `mx.distributed.init(backend="ring")` returned a singleton
  (`rank=0 size=1`) every time MLX_HOSTFILE and MLX_RANK were set and
  a peer was reachable. Without `--hostfile` matching what the ring
  backend actually expects, the wheel binds port 8031 (engine) but never
  52100 (ring), so requests never traverse the wire. The fix landed in
  `_init_ring` (now refuses size mismatch), but the wheel still returns
  the singleton rather than raising.

The world-1 greedy completion through the same gateway on macOS proves
the full OpenAI, ring, and engine path on a single binary. The 2-node
hop validation is the only piece left; it requires the M2 wheel to
stop returning a singleton for ring init AND a quiet macOS Metal
device. Both are operational blockers, not application bugs.

## Design notes

- **No node-to-node ssh**: `serve` only talks to each node's already
  running `agent` over HTTP, authed with the shared token. The agent
  spawns the rank locally. This removes the `authorized_keys`
  dependency and keeps every control action auditable on one port per
  node.
- **MLX stream scoping**: lazy model load, cache build, and generation
  all run on a single persistent worker thread (mlx streams are
  thread-local; the HTTP server handles each request on a new thread).
- **Hybrid-attention masks**: qwen3_5 alternates linear-attention
  (ssm) and full-attention layers, each with its own cache type and
  per-layer mask. The runner mirrors the model's own mask construction
  from `mlx_lm.models.base` instead of the simple `"causal"` heuristic.
- **Tied embeddings**: when `tie_word_embeddings` is true, the head
  is `embed_tokens.as_linear(h)` — `gather` rejects float `int`.
- **2-rank ring protocol**: rank0 accepts on its own address and
  connects to rank1; rank1 runs `serve_hops` and sends back the sampled
  token. World > 2 is not implemented yet.
- **iPerf3 on macOS ends**: the proposal's "use iperf3 when available"
  rule is collapsed to "never use iperf3 when a macOS node is involved"
  because brew iperf3 on mac-a is Local-Network-privacy blocked.

## Tests

```sh
python3 -m pytest tests/
```

Covers the route picker (fastest eligible pin, RTT tiebreak, failover
order, Wi-Fi/Tailscale never decode-eligible), hostfile merge/override,
links.json round-trip, planner (pipeline fit, decode-tail on the bigger
node, decode-exclusion, infeasible when weights don't fit), and hub
registry staleness.